import hashlib
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from world import (
    CheckpointReference,
    ManifestError,
    ManifestRecoveryRequiredError,
    ManifestStatus,
    WorldCausalManifest,
    WorldCausalManifestStore,
)


def manifest(*, tick=0, refs=None, last_sequence=None, digest="", **overrides):
    references = (checkpoint(),) if refs is None else tuple(refs)
    values = {
        "world_id": "world-a",
        "causal_tick": tick,
        "presence_ids": (),
        "checkpoint_references": references,
        "journal_last_sequence": last_sequence,
        "journal_prefix_sha256": digest,
        "commit_id": f"transition-{tick}" if tick else "genesis:world-a",
        "registered_owner_ids": tuple(sorted(
            (ref.owner, ref.owner_identity) for ref in references if ref.owner != "world-journal"
        )),
    }
    values.update(overrides)
    return WorldCausalManifest(**values)


def checkpoint(owner="organism", identity="organism-a", locator="sha256:abc", schema="1"):
    return CheckpointReference(
        owner=owner,
        owner_identity=identity,
        checkpoint_id=f"{owner}-checkpoint-1",
        schema_version=schema,
        locator=locator,
        sha256=hashlib.sha256(locator.encode()).hexdigest(),
    )


def test_manifest_requires_inclusive_durable_journal_boundary():
    assert manifest().journal_last_sequence is None
    assert manifest(last_sequence=4, digest="a" * 64).journal_last_sequence == 4
    with pytest.raises(ValueError, match="empty journal"):
        manifest(digest="a" * 64)
    with pytest.raises(ValueError, match="SHA-256"):
        manifest(last_sequence=4, digest="next-ordinal")


def test_manifest_rejects_duplicate_owner_identity_reference():
    ref = checkpoint()
    with pytest.raises(ValueError, match="duplicate owner identity/schema"):
        manifest(refs=(ref, ref))


def test_manifest_allows_distinct_checkpoint_schemas_for_one_owner_identity():
    refs = (
        checkpoint("embodiment", "body-1", "physical", "physical-v2"),
        checkpoint("embodiment", "body-1", "reserve", "reserve-v2"),
        checkpoint("embodiment", "body-1", "binding", "binding-v1"),
    )
    assert manifest(refs=refs, registered_owner_ids=(("embodiment", "body-1"),)).checkpoint_references == refs


def test_manifest_rejects_duplicate_active_presence_ids():
    with pytest.raises(ValueError, match="duplicate presence ids"):
        manifest(presence_ids=("organism-a", "organism-a"))


def test_manifest_store_has_no_supported_public_mutation_api():
    store = WorldCausalManifestStore("unused.json")

    assert callable(store.read)
    for method in (
        "initialize",
        "require_active",
        "begin_transition",
        "record_failure",
        "commit_transition",
        "complete_recovery",
    ):
        assert not hasattr(store, method), f"{method} must not be a supported public store API"


def test_durable_pending_transition_blocks_ordinary_execution(tmp_path):
    store = WorldCausalManifestStore(tmp_path / "world.json")
    store._initialize(manifest())

    pending = store._begin_transition("transition-1")

    assert pending.status is ManifestStatus.PREPARING
    assert pending.causal_tick == 0
    assert pending.pending_tick == 1
    assert store.read() == pending
    with pytest.raises(ManifestRecoveryRequiredError, match="transition-1"):
        store._require_active()
    assert store.read().status is ManifestStatus.RECOVERY_REQUIRED


def test_only_one_cross_process_transition_can_prepare_at_a_time(tmp_path):
    store = WorldCausalManifestStore(tmp_path / "world.json")
    store._initialize(manifest())
    barrier = Barrier(2)

    def begin(transition_id):
        barrier.wait()
        try:
            store._begin_transition(transition_id)
            return "prepared"
        except ManifestRecoveryRequiredError:
            return "blocked"

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(begin, ("transition-a", "transition-b")))

    assert sorted(outcomes) == ["blocked", "prepared"]
    assert store.read().status is ManifestStatus.PREPARING


def test_successful_causal_cut_publication_verifies_dependency_closure(tmp_path):
    store = WorldCausalManifestStore(tmp_path / "world.json")
    store._initialize(manifest())
    store._begin_transition("transition-1")
    ref = checkpoint()
    committed = manifest(
        tick=1, refs=(ref,), last_sequence=0, digest="b" * 64, commit_id="transition-1"
    )

    result = store._commit_transition(
        committed,
        dependency_verifier=lambda refs: refs == (ref,),
    )

    assert result.status is ManifestStatus.COMMITTED
    assert store.read() == committed
    assert store._require_active() == committed


def test_manifest_store_rejects_journal_only_initial_cut_and_commit(tmp_path):
    store = WorldCausalManifestStore(tmp_path / "world.json")
    journal = checkpoint(owner="world-journal", identity="world-a")
    journal_only_initial = manifest(refs=(journal,))

    with pytest.raises(ManifestError, match="at least one durable owner checkpoint"):
        store._initialize(journal_only_initial)
    assert not store.path.exists()

    store._initialize(manifest())
    pending = store._begin_transition("transition-1")
    journal_only_candidate = manifest(
        tick=1,
        refs=(journal,),
        last_sequence=0,
        digest="d" * 64,
    )
    with pytest.raises(ManifestError, match="at least one durable owner checkpoint"):
        store._commit_transition(journal_only_candidate, dependency_verifier=lambda _refs: True)
    assert store.read() == pending


@pytest.mark.parametrize(
    "candidate_refs",
    [
        (checkpoint(owner="organism", identity="organism-a"),),  # remove one owner
        (
            checkpoint(owner="organism", identity="organism-a"),
            checkpoint(owner="embodiment", identity="body-a"),
            checkpoint(owner="environment", identity="environment-a"),
        ),  # add an owner
        (
            checkpoint(owner="organism", identity="organism-b"),
            checkpoint(owner="embodiment", identity="body-a"),
        ),  # substitute an identity
    ],
)
def test_manifest_store_rejects_owner_membership_change_on_commit(tmp_path, candidate_refs):
    store = WorldCausalManifestStore(tmp_path / "world.json")
    initial_refs = (
        checkpoint(owner="organism", identity="organism-a"),
        checkpoint(owner="embodiment", identity="body-a"),
    )
    store._initialize(manifest(refs=initial_refs))
    pending = store._begin_transition("transition-1")
    candidate = manifest(tick=1, refs=candidate_refs, last_sequence=0, digest="e" * 64)

    with pytest.raises(ManifestError, match="participant roster differs"):
        store._commit_transition(candidate, dependency_verifier=lambda _refs: True)

    assert store.read() == pending


def test_unverified_checkpoint_dependency_leaves_durable_recovery_required(tmp_path):
    store = WorldCausalManifestStore(tmp_path / "world.json")
    store._initialize(manifest())
    pending = store._begin_transition("transition-1")
    ref = checkpoint()
    candidate = manifest(tick=1, refs=(ref,), last_sequence=0, digest="c" * 64)

    with pytest.raises(ManifestError, match="dependency closure"):
        store._commit_transition(candidate, dependency_verifier=lambda _: False)

    assert store.read() == pending
    with pytest.raises(ManifestRecoveryRequiredError):
        store._require_active()


def test_failure_detail_is_durable_and_does_not_clear_recovery_state(tmp_path):
    store = WorldCausalManifestStore(tmp_path / "world.json")
    store._initialize(manifest())
    store._begin_transition("transition-1")

    failed = store._record_failure("transition-1", "synthetic pre-publication failure")

    assert failed.failure == "synthetic pre-publication failure"
    assert store.read() == failed
    with pytest.raises(ManifestRecoveryRequiredError):
        store._require_active()


def test_cannot_initialize_existing_manifest_or_commit_without_pending_transition(tmp_path):
    store = WorldCausalManifestStore(tmp_path / "world.json")
    store._initialize(manifest())
    with pytest.raises(FileExistsError):
        store._initialize(manifest())
    with pytest.raises(ValueError, match="pending transition"):
        store._commit_transition(manifest(tick=1), dependency_verifier=lambda _: True)


def test_initialize_is_atomic_no_overwrite_for_competing_initializers(tmp_path):
    store = WorldCausalManifestStore(tmp_path / "world.json")
    barrier = Barrier(2)

    def initialize(world_id):
        barrier.wait()
        try:
            store._initialize(manifest(world_id=world_id))
            return "created"
        except FileExistsError:
            return "exists"

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(initialize, ("world-a", "world-b")))

    assert sorted(results) == ["created", "exists"]
    assert store.read().world_id in {"world-a", "world-b"}
