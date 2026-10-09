import hashlib

import pytest

from world import (
    CheckpointReference,
    ManifestError,
    ManifestRecoveryRequiredError,
    ManifestStatus,
    WorldCausalManifest,
    WorldCausalManifestStore,
)


def manifest(*, tick=0, refs=(), last_sequence=None, digest="", **overrides):
    values = {
        "world_id": "world-a",
        "causal_tick": tick,
        "checkpoint_references": tuple(refs),
        "journal_last_sequence": last_sequence,
        "journal_prefix_sha256": digest,
    }
    values.update(overrides)
    return WorldCausalManifest(**values)


def checkpoint(owner="organism", identity="organism-a", locator="sha256:abc"):
    return CheckpointReference(
        owner=owner,
        owner_identity=identity,
        checkpoint_id=f"{owner}-checkpoint-1",
        schema_version="1",
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
    with pytest.raises(ValueError, match="duplicate owner identity"):
        manifest(refs=(ref, ref))


def test_durable_pending_transition_blocks_ordinary_execution(tmp_path):
    store = WorldCausalManifestStore(tmp_path / "world.json")
    store.initialize(manifest())

    pending = store.begin_transition("transition-1")

    assert pending.status is ManifestStatus.RECOVERY_REQUIRED
    assert pending.causal_tick == 0
    assert pending.pending_tick == 1
    assert store.read() == pending
    with pytest.raises(ManifestRecoveryRequiredError, match="transition-1"):
        store.require_active()


def test_successful_causal_cut_publication_verifies_dependency_closure(tmp_path):
    store = WorldCausalManifestStore(tmp_path / "world.json")
    store.initialize(manifest())
    store.begin_transition("transition-1")
    ref = checkpoint()
    committed = manifest(tick=1, refs=(ref,), last_sequence=0, digest="b" * 64)

    result = store.commit_transition(
        committed,
        dependency_verifier=lambda refs: refs == (ref,),
    )

    assert result.status is ManifestStatus.COMPLETE
    assert store.read() == committed
    assert store.require_active() == committed


def test_unverified_checkpoint_dependency_leaves_durable_recovery_required(tmp_path):
    store = WorldCausalManifestStore(tmp_path / "world.json")
    store.initialize(manifest())
    pending = store.begin_transition("transition-1")
    ref = checkpoint()
    candidate = manifest(tick=1, refs=(ref,), last_sequence=0, digest="c" * 64)

    with pytest.raises(ManifestError, match="dependency closure"):
        store.commit_transition(candidate, dependency_verifier=lambda _: False)

    assert store.read() == pending
    with pytest.raises(ManifestRecoveryRequiredError):
        store.require_active()


def test_failure_detail_is_durable_and_does_not_clear_recovery_state(tmp_path):
    store = WorldCausalManifestStore(tmp_path / "world.json")
    store.initialize(manifest())
    store.begin_transition("transition-1")

    failed = store.record_failure("transition-1", "synthetic pre-publication failure")

    assert failed.failure == "synthetic pre-publication failure"
    assert store.read() == failed
    with pytest.raises(ManifestRecoveryRequiredError):
        store.require_active()


def test_cannot_initialize_existing_manifest_or_commit_without_pending_transition(tmp_path):
    store = WorldCausalManifestStore(tmp_path / "world.json")
    store.initialize(manifest())
    with pytest.raises(FileExistsError):
        store.initialize(manifest())
    with pytest.raises(ValueError, match="pending transition"):
        store.commit_transition(manifest(tick=1), dependency_verifier=lambda _: True)
