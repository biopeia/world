import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import world.durable_runtime as durable_runtime_module
import world.manifest as manifest_module
from world import (
    CausalRecord,
    CheckpointReference,
    DurableJournalSnapshotStore,
    DurableWorldRuntime,
    EventJournal,
    ManifestError,
    ManifestRecoveryRequiredError,
    ManifestStatus,
    OwnerCausalRecord,
    PresenceLifecycleTransition,
    PresenceId,
    PresenceOwnerLink,
    RecoveryRequiredError,
    WorldCausalManifest,
    WorldCausalManifestStore,
    WorldGenesisSpec,
    WorldId,
)


def _initialize(
    world_id, presence_id, manifest_store, journal_store, *,
    initial_owner_references, registered_owner_ids, owner_checkpoint_verifier,
    owner_checkpoint_contracts, presence_owner_links=None,
):
    owner_ids = tuple(registered_owner_ids)
    if presence_owner_links is None:
        if len(owner_ids) != 1:
            raise AssertionError("tests with multiple owners must declare the presence owner explicitly")
        owner, identity = owner_ids[0]
        presence_owner_links = (PresenceOwnerLink(PresenceId(presence_id), owner, identity),)
    genesis = WorldGenesisSpec(
        world_id=world_id,
        initial_presence_ids=(presence_id,),
        registered_owner_ids=tuple(registered_owner_ids),
        initial_owner_references=tuple(initial_owner_references),
        owner_checkpoint_contracts=owner_checkpoint_contracts,
        presence_owner_links=presence_owner_links,
    )
    return DurableWorldRuntime.initialize(
        genesis, presence_id, manifest_store, journal_store,
        owner_checkpoint_verifier=owner_checkpoint_verifier,
    )


def _checkpoint(path: Path, *, owner: str = "organism") -> CheckpointReference:
    payload = path.read_bytes()
    return CheckpointReference(
        owner=owner,
        owner_identity=f"{owner}-identity",
        checkpoint_id=f"{owner}-checkpoint",
        schema_version="v1",
        locator=str(path),
        sha256=hashlib.sha256(payload).hexdigest(),
    )


def test_journal_v3_appends_fixed_size_records_and_recovers_exact_prefix(tmp_path):
    world_id = WorldId("world-journal-v3-linear")
    store = DurableJournalSnapshotStore(tmp_path / "journal", world_id)
    journal = EventJournal()
    journal.append(CausalRecord("record-0000", world_id.value, 0, "test.event", payload={"v": 0}))
    reference = store.write(journal, causal_tick=0)
    journal_path = Path(reference.locator)
    initial_size = journal_path.stat().st_size
    assert journal_path.read_bytes().startswith(b'{"schema":"event-journal-v3"')
    assert reference.schema_version == "event-journal-v3"

    marginal_sizes = []
    for tick in range(1, 10):
        journal.append(CausalRecord(f"record-{tick:04d}", world_id.value, tick, "test.event", payload={"v": tick}))
        previous_size = journal_path.stat().st_size
        reference = store.write(journal, causal_tick=tick, previous_reference=reference)
        marginal_sizes.append(journal_path.stat().st_size - previous_size)

    assert initial_size > marginal_sizes[0]
    assert len(set(marginal_sizes)) == 1
    assert [item.record_id for item in store.read(reference).replay()] == [
        f"record-{tick:04d}" for tick in range(10)
    ]
    assert store.verify(reference)

    legacy_reference = CheckpointReference(
        owner=reference.owner,
        owner_identity=reference.owner_identity,
        checkpoint_id="journal-prefix-9",
        schema_version="event-journal-v2",
        locator=reference.locator,
        sha256=reference.sha256,
    )
    assert not store.verify(legacy_reference)
    with pytest.raises(ManifestError, match="another World"):
        store.read(legacy_reference)

    # Full-prefix restore detects tampering to an earlier committed record.
    original_log = journal_path.read_bytes()
    encoded = bytearray(original_log)
    marker = encoded.index(b'"v":1')
    encoded[marker + len(b'"v":')] = ord("9")
    journal_path.write_bytes(encoded)
    with pytest.raises(ManifestError, match="integrity"):
        store.read(reference)

    # V3 is closed: even a same-length field rename is rejected instead of
    # being silently ignored while validating a normalized record.
    lines = original_log.splitlines()
    lines[-1] = lines[-1].replace(b'"record_id"', b'"record_iX"', 1)
    journal_path.write_bytes(b"\n".join((*lines, b"")))
    with pytest.raises(ManifestError, match="closed v3 schema"):
        store.read(reference)


def test_corrupted_earlier_committed_prefix_blocks_next_causal_commit(tmp_path):
    world_id = WorldId("world-prefix-integrity-before-commit")
    manifest_store = WorldCausalManifestStore(tmp_path / "manifest.json")
    journal_store = DurableJournalSnapshotStore(tmp_path / "journal", world_id)
    initial_owner = tmp_path / "organism-0.json"
    initial_owner.write_bytes(b"organism at tick zero")
    runtime = _initialize(
        world_id,
        "presence-a",
        manifest_store,
        journal_store,
        initial_owner_references=(_checkpoint(initial_owner),),
        registered_owner_ids=(("organism", "organism-identity"),),
        owner_checkpoint_verifier=_verifier,
        owner_checkpoint_contracts=_contracts(("organism", "organism-identity")),
    )

    def write_owner(tick: int):
        state = tmp_path / f"organism-{tick}.json"
        state.write_bytes(f"organism at tick {tick}".encode())
        return (_checkpoint(state),)

    runtime.tick("transition-1", write_owner)
    runtime.tick("transition-2", write_owner)
    committed_manifest = runtime.manifest
    journal_ref = next(ref for ref in committed_manifest.checkpoint_references if ref.owner == "world-journal")

    # Preserve line length while invalidating an earlier committed record.
    journal_path = Path(journal_ref.locator)
    corrupted = bytearray(journal_path.read_bytes())
    marker = corrupted.index(b"transition-1") + len(b"transition-")
    corrupted[marker] = ord("9")
    journal_path.write_bytes(corrupted)

    with pytest.raises(ManifestError, match="previous committed World journal reference is invalid"):
        runtime.tick("transition-3", write_owner)

    assert runtime.status is ManifestStatus.RECOVERY_REQUIRED
    assert runtime.manifest.causal_tick == committed_manifest.causal_tick == 2
    assert not journal_store.verify(journal_ref)


def test_non_lf_committed_record_terminator_blocks_append_without_mutation(tmp_path):
    world_id = WorldId("world-prefix-framing-before-commit")
    manifest_store = WorldCausalManifestStore(tmp_path / "manifest.json")
    journal_store = DurableJournalSnapshotStore(tmp_path / "journal", world_id)
    initial_owner = tmp_path / "organism-0.json"
    initial_owner.write_bytes(b"organism at tick zero")
    runtime = _initialize(
        world_id,
        "presence-a",
        manifest_store,
        journal_store,
        initial_owner_references=(_checkpoint(initial_owner),),
        registered_owner_ids=(("organism", "organism-identity"),),
        owner_checkpoint_verifier=_verifier,
        owner_checkpoint_contracts=_contracts(("organism", "organism-identity")),
    )

    def write_owner(tick: int):
        state = tmp_path / f"organism-{tick}.json"
        state.write_bytes(f"organism at tick {tick}".encode())
        return (_checkpoint(state),)

    runtime.tick("transition-1", write_owner)
    runtime.tick("transition-2", write_owner)
    committed_manifest = runtime.manifest
    journal_ref = next(ref for ref in committed_manifest.checkpoint_references if ref.owner == "world-journal")
    journal_path = Path(journal_ref.locator)
    valid_log = journal_path.read_bytes()
    offset = DurableJournalSnapshotStore._position(journal_ref)[2]
    prior_record_ids = [record.record_id for record in journal_store.read(journal_ref).replay()]
    assert valid_log[offset - 1:offset] == b"\n"

    # JSON accepts trailing spaces, but the journal format requires LF framing.
    # Keep the byte length unchanged to target framing rather than truncation.
    corrupted = bytearray(valid_log)
    corrupted[offset - 1] = ord(" ")
    corrupted_log = bytes(corrupted)
    journal_path.write_bytes(corrupted_log)

    assert not journal_store.verify(journal_ref)
    with pytest.raises(ManifestError, match="not LF-terminated"):
        journal_store.read(journal_ref)

    with pytest.raises(ManifestError, match="previous committed World journal reference is invalid"):
        runtime.tick("transition-3", write_owner)

    assert journal_path.read_bytes() == corrupted_log
    assert runtime.status is ManifestStatus.RECOVERY_REQUIRED
    assert runtime.manifest.causal_tick == committed_manifest.causal_tick == 2

    # The rejected append did not mutate the bytes. Restoring the test's saved
    # valid prefix demonstrates that the prior committed cut remains readable.
    journal_path.write_bytes(valid_log)
    restored = journal_store.read(journal_ref)
    assert [record.record_id for record in restored.replay()] == prior_record_ids


def _verifier(references: tuple[CheckpointReference, ...]) -> bool:
    return all(
        Path(reference.locator).is_file()
        and hashlib.sha256(Path(reference.locator).read_bytes()).hexdigest() == reference.sha256
        for reference in references
    )


class _OwnerContract:
    def __init__(self, valid=True):
        self.valid = valid

    def validate_checkpoint(self, encoded, owner_identity, schema_version):
        return self.valid and bool(encoded) and bool(owner_identity) and bool(schema_version)

    def verify_restoration(self, encoded, owner_identity, schema_version):
        return self.validate_checkpoint(encoded, owner_identity, schema_version)


def _contracts(*keys):
    return {key: _OwnerContract() for key in keys}


def _presence_lifecycle(transition_type, presence_id):
    return PresenceLifecycleTransition(
        transition_type,
        PresenceId(presence_id),
        {"source": "test-request", "request_id": f"{transition_type.lower()}-{presence_id}"},
    )


def test_durable_tick_publishes_journal_and_owner_checkpoint_before_complete_manifest(tmp_path):
    world_id = WorldId("world-durable")
    manifest_store = WorldCausalManifestStore(tmp_path / "manifest.json")
    journal_store = DurableJournalSnapshotStore(tmp_path / "journal", world_id)
    initial_owner = tmp_path / "organism-0.json"
    initial_owner.write_bytes(b"organism at tick zero")
    runtime = _initialize(
        world_id,
        "presence-a",
        manifest_store,
        journal_store,
        initial_owner_references=(_checkpoint(initial_owner),),
        registered_owner_ids=(("organism", "organism-identity"),),
        owner_checkpoint_verifier=_verifier,
        owner_checkpoint_contracts=_contracts(("organism", "organism-identity")),
    )

    def write_tick_one(tick: int):
        assert tick == 1
        state = tmp_path / "organism-1.json"
        state.write_bytes(b"organism at tick one")
        return (_checkpoint(state),)

    record = runtime.tick("transition-1", write_tick_one)

    assert runtime.status is ManifestStatus.COMMITTED
    assert runtime.causal_tick == 1
    assert tuple(item for item in runtime.records if item.record_type == "world.causal_tick.committed") == (record,)
    assert tuple(item.record_type for item in runtime.records) == (
        "world.causal_cut.committed",
        "world.causal_tick.committed",
        "world.causal_cut.committed",
    )
    assert runtime.manifest.journal_last_sequence == 2
    assert len(runtime.manifest.checkpoint_references) == 2
    assert manifest_store._require_active() == runtime.manifest


def test_second_process_can_observe_preparing_manifest_while_writer_callback_is_open(tmp_path):
    """A separate process can read the pending cut without joining the write.

    This is a single-logical-writer observation test: the writer pauses in its
    owner callback after publishing PREPARING, a child process reads that
    manifest, and only then does the original writer continue and commit. It
    does not establish multi-writer coordination or cross-process recovery.
    """
    world_id = WorldId("world-second-process-observation")
    manifest_path = tmp_path / "manifest.json"
    manifest_store = WorldCausalManifestStore(manifest_path)
    journal_store = DurableJournalSnapshotStore(tmp_path / "journal", world_id)
    initial_owner = tmp_path / "owner-0.json"
    initial_owner.write_bytes(b"owner state at tick zero")
    contracts = _contracts(("organism", "organism-identity"))
    runtime = _initialize(
        world_id,
        "presence-a",
        manifest_store,
        journal_store,
        initial_owner_references=(_checkpoint(initial_owner),),
        registered_owner_ids=(("organism", "organism-identity"),),
        owner_checkpoint_verifier=_verifier,
        owner_checkpoint_contracts=contracts,
    )

    reader = r"""
import json
import sys
from world import WorldCausalManifestStore

manifest = WorldCausalManifestStore(sys.argv[1]).read()
print(json.dumps({
    "status": manifest.status.value,
    "causal_tick": manifest.causal_tick,
    "pending_transition_id": manifest.pending_transition_id,
    "pending_tick": manifest.pending_tick,
}))
"""

    def write_owner_checkpoint(tick: int):
        assert tick == 1
        result = subprocess.run(
            [sys.executable, "-c", reader, str(manifest_path)],
            cwd=Path(__file__).resolve().parents[1],
            env={
                **os.environ,
                "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
            },
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        observed = json.loads(result.stdout)
        assert observed == {
            "status": ManifestStatus.PREPARING.value,
            "causal_tick": 0,
            "pending_transition_id": "transition-visible-to-second-process",
            "pending_tick": 1,
        }

        state = tmp_path / "owner-1.json"
        state.write_bytes(b"owner state at tick one")
        return (_checkpoint(state),)

    record = runtime.tick(
        "transition-visible-to-second-process",
        write_owner_checkpoint,
    )

    assert record.causal_tick == 1
    assert runtime.status is ManifestStatus.COMMITTED
    assert manifest_store.read().commit_id == "transition-visible-to-second-process"


def test_durable_tick_stage_sequence_transitions_from_preparing_to_committed_or_recovery_required(tmp_path):
    """Enforce stage ordering: PREPARING before owner writer runs, then COMMITTED on success, or RECOVERY_REQUIRED on failure.

    This test prevents regression of the stage lifecycle guidance:
    1. During owner checkpoint writer execution, the durable manifest is PREPARING (not RECOVERY_REQUIRED).
    2. Upon successful completion of writer callbacks, dependency verification, and journal flush, it transitions to COMMITTED.
    3. If an owner writer fails or an interrupted prepare is detected, it transitions to RECOVERY_REQUIRED and requires explicit recovery.
    """
    world_id = WorldId("world-stage-sequence")
    manifest_store = WorldCausalManifestStore(tmp_path / "manifest.json")
    journal_store = DurableJournalSnapshotStore(tmp_path / "journal", world_id)
    initial_owner = tmp_path / "owner-0.json"
    initial_owner.write_bytes(b"initial owner state")
    contracts = _contracts(("organism", "organism-identity"))
    runtime = _initialize(
        world_id,
        "presence-a",
        manifest_store,
        journal_store,
        initial_owner_references=(_checkpoint(initial_owner),),
        registered_owner_ids=(("organism", "organism-identity"),),
        owner_checkpoint_verifier=_verifier,
        owner_checkpoint_contracts=contracts,
    )

    observed_stages = []

    def writer_success(tick: int):
        observed = manifest_store.read()
        observed_stages.append(observed.status)
        assert observed.status is ManifestStatus.PREPARING
        assert observed.pending_tick == tick
        assert observed.causal_tick == 0
        state = tmp_path / "owner-1.json"
        state.write_bytes(b"owner state tick 1")
        return (_checkpoint(state),)

    runtime.tick("transition-stage-success", writer_success)
    assert observed_stages == [ManifestStatus.PREPARING]
    assert runtime.status is ManifestStatus.COMMITTED
    assert manifest_store.read().status is ManifestStatus.COMMITTED

    # On post-prepare callback failure, the stage must latch to RECOVERY_REQUIRED,
    # refusing subsequent ticks until explicit recover_previous_cut is performed.
    def writer_failure(tick: int):
        observed = manifest_store.read()
        observed_stages.append(observed.status)
        assert observed.status is ManifestStatus.PREPARING
        raise RuntimeError("simulated owner writer failure")

    with pytest.raises(RuntimeError, match="simulated owner writer failure"):
        runtime.tick("transition-stage-fail", writer_failure)

    assert observed_stages == [ManifestStatus.PREPARING, ManifestStatus.PREPARING]
    assert manifest_store.read().status is ManifestStatus.RECOVERY_REQUIRED
    assert runtime.status is ManifestStatus.RECOVERY_REQUIRED


def test_owner_causal_record_survives_later_tick_and_restore(tmp_path):
    world_id = WorldId("world-owner-record-history")
    manifest_store = WorldCausalManifestStore(tmp_path / "manifest.json")
    journal_store = DurableJournalSnapshotStore(tmp_path / "journal", world_id)
    initial_owner = tmp_path / "owner-0.json"
    initial_owner.write_bytes(b"owner zero")
    contracts = _contracts(("organism", "organism-identity"))
    runtime = _initialize(
        world_id,
        "presence-a",
        manifest_store,
        journal_store,
        initial_owner_references=(_checkpoint(initial_owner),),
        registered_owner_ids=(("organism", "organism-identity"),),
        owner_checkpoint_verifier=_verifier,
        owner_checkpoint_contracts=contracts,
    )
    record_payload = {"checkpoint_sha256": "a" * 64, "nested": {"value": 3}}

    def write_owner_records(tick, references):
        assert tick == 1
        assert references[0].owner == "organism"
        return (
            OwnerCausalRecord(
                "organism",
                "organism-identity",
                "action-1",
                record_payload,
            ),
        )

    first_state = tmp_path / "owner-1.json"
    first_state.write_bytes(b"owner one")
    runtime.tick(
        "transition-1",
        lambda _tick: (_checkpoint(first_state),),
        write_owner_records=write_owner_records,
    )
    record_payload["nested"]["value"] = 99

    second_state = tmp_path / "owner-2.json"
    second_state.write_bytes(b"owner two")
    runtime.tick("transition-2", lambda _tick: (_checkpoint(second_state),))

    restored = DurableWorldRuntime.restore(
        world_id,
        "presence-a",
        manifest_store,
        journal_store,
        owner_checkpoint_verifier=_verifier,
        owner_checkpoint_contracts=contracts,
        owner_checkpoint_restorer=lambda _references: None,
    )
    owner_records = tuple(
        item for item in restored.records if item.record_type == "world.owner_record.committed"
    )
    assert restored.causal_tick == 2
    assert len(owner_records) == 1
    assert owner_records[0].causal_tick == 1
    assert owner_records[0].payload["owner_id"] == "organism"
    assert owner_records[0].payload["owner_record_id"] == "action-1"
    assert owner_records[0].payload["data"]["nested"]["value"] == 3


def test_owner_causal_record_survives_recovery_after_later_failed_tick(tmp_path):
    world_id = WorldId("world-owner-record-recovery")
    manifest_store = WorldCausalManifestStore(tmp_path / "manifest.json")
    journal_store = DurableJournalSnapshotStore(tmp_path / "journal", world_id)
    initial_owner = tmp_path / "owner-0.json"
    initial_owner.write_bytes(b"owner zero")
    contracts = _contracts(("organism", "organism-identity"))
    runtime = _initialize(
        world_id,
        "presence-a",
        manifest_store,
        journal_store,
        initial_owner_references=(_checkpoint(initial_owner),),
        registered_owner_ids=(("organism", "organism-identity"),),
        owner_checkpoint_verifier=_verifier,
        owner_checkpoint_contracts=contracts,
    )

    first_state = tmp_path / "owner-1.json"
    first_state.write_bytes(b"owner one")
    runtime.tick(
        "transition-1",
        lambda _tick: (_checkpoint(first_state),),
        write_owner_records=lambda _tick, _refs: (
            OwnerCausalRecord("organism", "organism-identity", "action-1", {"v": 1}),
        ),
    )

    with pytest.raises(RuntimeError, match="tick callback failed"):
        runtime.tick(
            "transition-2-failed",
            lambda _tick: (_ for _ in ()).throw(RuntimeError("tick callback failed")),
            write_owner_records=lambda _tick, _refs: (
                OwnerCausalRecord("organism", "organism-identity", "action-2", {"v": 2}),
            ),
        )
    assert manifest_store.read().status is ManifestStatus.RECOVERY_REQUIRED

    recovered = DurableWorldRuntime.recover_previous_cut(
        world_id,
        "presence-a",
        manifest_store,
        journal_store,
        owner_checkpoint_verifier=_verifier,
        owner_checkpoint_contracts=contracts,
        owner_checkpoint_restorer=lambda _refs: None,
    )

    owner_records = tuple(
        item for item in recovered.records if item.record_type == "world.owner_record.committed"
    )
    assert recovered.status is ManifestStatus.COMMITTED
    assert recovered.causal_tick == 1
    assert [item.payload["owner_record_id"] for item in owner_records] == ["action-1"]


def test_process_termination_after_preparing_reopens_prior_cut_as_recovery_required(tmp_path):
    """A killed writer must not make its uncommitted tick look committed.

    This exercises one process-termination boundary on the current local
    filesystem: the child exits after World durably publishes PREPARING from
    inside the transition callback, before it writes candidate owner state or
    a candidate journal/COMMITTED manifest. It is not a power-loss test or a
    general filesystem crash-consistency claim.
    """
    world_id = WorldId("world-process-restart")
    manifest_path = tmp_path / "manifest.json"
    journal_path = tmp_path / "journal"
    manifest_store = WorldCausalManifestStore(manifest_path)
    journal_store = DurableJournalSnapshotStore(journal_path, world_id)
    initial_owner = tmp_path / "owner-0.json"
    initial_owner.write_bytes(b"committed owner state at tick zero")
    contracts = _contracts(("organism", "organism-identity"))
    _initialize(
        world_id,
        "presence-a",
        manifest_store,
        journal_store,
        initial_owner_references=(_checkpoint(initial_owner),),
        registered_owner_ids=(("organism", "organism-identity"),),
        owner_checkpoint_verifier=_verifier,
        owner_checkpoint_contracts=contracts,
    )
    committed_before = manifest_store.read()
    initial_journals = (journal_path / "world-causal-journal-v3.jsonl",)
    assert committed_before.status is ManifestStatus.COMMITTED
    assert committed_before.causal_tick == 0
    assert len(initial_journals) == 1

    # Restore in a separate writer process, then terminate exactly from the
    # owner callback, which DurableWorldRuntime invokes after publishing
    # PREPARING and before it writes the candidate journal/COMMITTED cut.
    child = r"""
import os
import sys
from pathlib import Path
from world import (
    DurableJournalSnapshotStore,
    DurableWorldRuntime,
    WorldCausalManifestStore,
    WorldId,
)

root = Path(sys.argv[1])
world_id = WorldId("world-process-restart")
manifest_store = WorldCausalManifestStore(root / "manifest.json")

def verify(references):
    import hashlib
    return all(
        Path(ref.locator).is_file()
        and hashlib.sha256(Path(ref.locator).read_bytes()).hexdigest() == ref.sha256
        for ref in references
    )

class Contract:
    def validate_checkpoint(self, encoded, owner_identity, schema_version):
        return bool(encoded) and owner_identity == "organism-identity" and schema_version == "v1"
    def verify_restoration(self, encoded, owner_identity, schema_version):
        return self.validate_checkpoint(encoded, owner_identity, schema_version)

runtime = DurableWorldRuntime.restore(
    world_id,
    "presence-a",
    manifest_store,
    DurableJournalSnapshotStore(root / "journal", world_id),
    owner_checkpoint_verifier=verify,
    owner_checkpoint_contracts={("organism", "organism-identity"): Contract()},
    owner_checkpoint_restorer=lambda _refs: None,
)

def terminate_after_prepare(_tick):
    pending = manifest_store.read()
    if (
        pending.status.value == "PREPARING"
        and pending.causal_tick == 0
        and pending.commit_id == "genesis:world-process-restart"
    ):
        os._exit(73)
    os._exit(74)

runtime.tick("transition-interrupted", terminate_after_prepare)
os._exit(74)  # Defensive: the callback must terminate before tick returns.
"""
    result = subprocess.run(
        [sys.executable, "-c", child, str(tmp_path)],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")},
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 73, result.stderr

    preparing = manifest_store.read()
    assert preparing.status is ManifestStatus.PREPARING
    assert preparing.causal_tick == committed_before.causal_tick
    assert preparing.commit_id == committed_before.commit_id
    assert preparing.checkpoint_references == committed_before.checkpoint_references
    assert not (tmp_path / "owner-1.json").exists()
    assert tuple(journal_path.glob("world-causal-journal-v3.jsonl")) == initial_journals

    with pytest.raises(ManifestRecoveryRequiredError):
        DurableWorldRuntime.restore(
            world_id,
            "presence-a",
            manifest_store,
            journal_store,
            owner_checkpoint_verifier=_verifier,
            owner_checkpoint_contracts=contracts,
            owner_checkpoint_restorer=lambda _refs: None,
        )

    pending = manifest_store.read()
    assert pending.status is ManifestStatus.RECOVERY_REQUIRED
    assert pending.causal_tick == committed_before.causal_tick
    assert pending.commit_id == committed_before.commit_id
    assert pending.checkpoint_references == committed_before.checkpoint_references

    recovered = DurableWorldRuntime.recover_previous_cut(
        world_id,
        "presence-a",
        manifest_store,
        journal_store,
        owner_checkpoint_verifier=_verifier,
        owner_checkpoint_contracts=contracts,
        owner_checkpoint_restorer=lambda references: references == tuple(
            ref for ref in committed_before.checkpoint_references if ref.owner != "world-journal"
        ) or pytest.fail("explicit recovery did not restore the prior owner cut"),
    )
    assert recovered.status is ManifestStatus.COMMITTED
    assert recovered.causal_tick == committed_before.causal_tick
    assert recovered.manifest.commit_id == committed_before.commit_id
    assert recovered.manifest.presence_owner_links == committed_before.presence_owner_links

    owner_one = tmp_path / "owner-1.json"
    owner_one.write_bytes(b"owner state after explicit recovery")
    resumed = recovered.tick("transition-after-recovery", lambda _tick: (_checkpoint(owner_one),))
    assert resumed.causal_tick == 1
    assert recovered.status is ManifestStatus.COMMITTED

def test_process_termination_during_partial_owner_checkpoint_write_requires_recovery(tmp_path):
    """An interrupted owner write cannot become the committed next cut.

    The writer exits after emitting only a prefix of a candidate owner
    checkpoint from the callback. World has already published PREPARING, but
    the callback never returns a reference and no candidate journal or
    COMMITTED manifest is published. This is process-termination evidence on
    the exercised local filesystem, not proof of owner-file durability or
    power-loss behavior.
    """
    world_id = WorldId("world-partial-owner-checkpoint-write")
    manifest_path = tmp_path / "manifest.json"
    journal_path = tmp_path / "journal"
    manifest_store = WorldCausalManifestStore(manifest_path)
    journal_store = DurableJournalSnapshotStore(journal_path, world_id)
    initial_owner = tmp_path / "owner-0.json"
    initial_owner.write_bytes(b"committed owner state at tick zero")
    contracts = _contracts(("organism", "organism-identity"))
    _initialize(
        world_id,
        "presence-a",
        manifest_store,
        journal_store,
        initial_owner_references=(_checkpoint(initial_owner),),
        registered_owner_ids=(("organism", "organism-identity"),),
        owner_checkpoint_verifier=_verifier,
        owner_checkpoint_contracts=contracts,
    )
    committed_before = manifest_store.read()
    initial_journals = (journal_path / "world-causal-journal-v3.jsonl",)
    assert committed_before.status is ManifestStatus.COMMITTED
    assert committed_before.causal_tick == 0
    assert len(initial_journals) == 1

    child = r"""
import os
import sys
from pathlib import Path
from world import (
    DurableJournalSnapshotStore,
    DurableWorldRuntime,
    WorldCausalManifestStore,
    WorldId,
)

root = Path(sys.argv[1])
world_id = WorldId("world-partial-owner-checkpoint-write")
manifest_store = WorldCausalManifestStore(root / "manifest.json")

def verify(references):
    import hashlib
    return all(
        Path(ref.locator).is_file()
        and hashlib.sha256(Path(ref.locator).read_bytes()).hexdigest() == ref.sha256
        for ref in references
    )

class Contract:
    def validate_checkpoint(self, encoded, owner_identity, schema_version):
        return bool(encoded) and owner_identity == "organism-identity" and schema_version == "v1"
    def verify_restoration(self, encoded, owner_identity, schema_version):
        return self.validate_checkpoint(encoded, owner_identity, schema_version)

runtime = DurableWorldRuntime.restore(
    world_id,
    "presence-a",
    manifest_store,
    DurableJournalSnapshotStore(root / "journal", world_id),
    owner_checkpoint_verifier=verify,
    owner_checkpoint_contracts={("organism", "organism-identity"): Contract()},
    owner_checkpoint_restorer=lambda _refs: None,
)

def write_partial_owner_checkpoint(tick):
    assert tick == 1
    payload = b"candidate owner state at tick one"
    fd = os.open(root / "owner-1.json", os.O_CREAT | os.O_TRUNC | os.O_WRONLY, 0o600)
    os.write(fd, payload[:12])
    os._exit(73)

runtime.tick("transition-interrupted-mid-owner-write", write_partial_owner_checkpoint)
os._exit(74)
"""
    result = subprocess.run(
        [sys.executable, "-c", child, str(tmp_path)],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")},
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 73, result.stderr

    candidate_owner = tmp_path / "owner-1.json"
    assert candidate_owner.read_bytes() == b"candidate ow"
    preparing = manifest_store.read()
    assert preparing.status is ManifestStatus.PREPARING
    assert preparing.causal_tick == committed_before.causal_tick
    assert preparing.commit_id == committed_before.commit_id
    assert preparing.checkpoint_references == committed_before.checkpoint_references
    assert tuple(journal_path.glob("world-causal-journal-v3.jsonl")) == initial_journals

    with pytest.raises(ManifestRecoveryRequiredError):
        DurableWorldRuntime.restore(
            world_id,
            "presence-a",
            manifest_store,
            journal_store,
            owner_checkpoint_verifier=_verifier,
            owner_checkpoint_contracts=contracts,
            owner_checkpoint_restorer=lambda _refs: None,
        )

    pending = manifest_store.read()
    assert pending.status is ManifestStatus.RECOVERY_REQUIRED
    assert pending.causal_tick == committed_before.causal_tick
    assert pending.commit_id == committed_before.commit_id
    assert pending.checkpoint_references == committed_before.checkpoint_references

    recovered = DurableWorldRuntime.recover_previous_cut(
        world_id,
        "presence-a",
        manifest_store,
        journal_store,
        owner_checkpoint_verifier=_verifier,
        owner_checkpoint_contracts=contracts,
        owner_checkpoint_restorer=lambda references: references == tuple(
            ref for ref in committed_before.checkpoint_references if ref.owner != "world-journal"
        ) or pytest.fail("explicit recovery did not restore the prior owner cut"),
    )
    assert recovered.status is ManifestStatus.COMMITTED
    assert recovered.causal_tick == committed_before.causal_tick
    assert recovered.manifest.commit_id == committed_before.commit_id
    assert recovered.manifest.checkpoint_references == committed_before.checkpoint_references


def test_journal_append_fsync_failure_keeps_previous_committed_prefix(tmp_path, monkeypatch):
    """An appended candidate beyond the manifest reference is not committed."""
    world_id = WorldId("world-journal-append-fsync-failure")
    manifest_store = WorldCausalManifestStore(tmp_path / "manifest.json")
    journal_path = tmp_path / "journal"
    journal_store = DurableJournalSnapshotStore(journal_path, world_id)
    initial_owner = tmp_path / "owner-0.json"
    initial_owner.write_bytes(b"committed owner state at tick zero")
    contracts = _contracts(("organism", "organism-identity"))
    runtime = _initialize(
        world_id, "presence-a", manifest_store, journal_store,
        initial_owner_references=(_checkpoint(initial_owner),),
        registered_owner_ids=(("organism", "organism-identity"),),
        owner_checkpoint_verifier=_verifier, owner_checkpoint_contracts=contracts,
    )
    committed_before = manifest_store.read()
    previous_reference = next(ref for ref in committed_before.checkpoint_references if ref.owner == "world-journal")
    committed_records = journal_store.read(previous_reference).replay()
    journal_file = Path(previous_reference.locator)
    previous_size = journal_file.stat().st_size
    original_fsync = os.fsync
    attempts = []

    def fail_journal_fsync(fd):
        try:
            fd_path = Path(os.readlink(f"/proc/self/fd/{fd}"))
        except OSError:
            fd_path = None
        if fd_path == journal_file:
            attempts.append(fd)
            raise OSError(5, "injected journal append fsync failure")
        return original_fsync(fd)

    monkeypatch.setattr(os, "fsync", fail_journal_fsync)

    def write_candidate_owner(tick):
        assert tick == 1
        candidate_owner = tmp_path / "owner-1.json"
        candidate_owner.write_bytes(b"candidate owner state at tick one")
        return (_checkpoint(candidate_owner),)

    with pytest.raises(OSError, match="injected journal append fsync failure"):
        runtime.tick("transition-journal-append-fsync-failure", write_candidate_owner)

    assert attempts
    pending = manifest_store.read()
    assert pending.status is ManifestStatus.RECOVERY_REQUIRED
    assert pending.causal_tick == committed_before.causal_tick
    assert pending.checkpoint_references == committed_before.checkpoint_references
    assert journal_file.stat().st_size > previous_size
    assert journal_store.read(previous_reference).replay() == runtime.records

    recovered = DurableWorldRuntime.recover_previous_cut(
        world_id, "presence-a", manifest_store, journal_store,
        owner_checkpoint_verifier=_verifier, owner_checkpoint_contracts=contracts,
        owner_checkpoint_restorer=lambda _references: None,
    )
    assert recovered.status is ManifestStatus.COMMITTED
    assert recovered.causal_tick == committed_before.causal_tick
    assert journal_store.read(previous_reference).replay() == runtime.records

    # The next successful writer truncates only the uncommitted physical tail
    # and publishes a new committed prefix; owner checkpoint bytes remain
    # outside World journal cleanup.
    monkeypatch.setattr(os, "fsync", original_fsync)
    next_owner = tmp_path / "owner-after-recovery.json"
    next_owner.write_bytes(b"owner state after explicit recovery")
    recovered.tick("transition-after-recovery", lambda _tick: (_checkpoint(next_owner),))
    latest_reference = next(
        ref for ref in recovered.manifest.checkpoint_references if ref.owner == "world-journal"
    )
    latest_records = journal_store.read(latest_reference).replay()
    assert latest_records == recovered.records
    assert sum(
        record.record_type == "world.causal_tick.committed" and record.causal_tick == 1
        for record in latest_records
    ) == 1


def test_process_termination_during_partial_manifest_write_keeps_previous_cut(tmp_path):
    """An interrupted temp-file write must not replace the committed manifest.

    This kills a writer after a prefix of the PREPARING manifest has been
    written to its temporary file, before fsync or atomic replacement. It
    checks process-termination behavior on the current local filesystem only;
    it is not a power-loss or storage-device durability test.
    """
    world_id = WorldId("world-partial-manifest-write")
    manifest_path = tmp_path / "manifest.json"
    journal_path = tmp_path / "journal"
    manifest_store = WorldCausalManifestStore(manifest_path)
    journal_store = DurableJournalSnapshotStore(journal_path, world_id)
    initial_owner = tmp_path / "owner-0.json"
    initial_owner.write_bytes(b"committed owner state at tick zero")
    contracts = _contracts(("organism", "organism-identity"))
    _initialize(
        world_id,
        "presence-a",
        manifest_store,
        journal_store,
        initial_owner_references=(_checkpoint(initial_owner),),
        registered_owner_ids=(("organism", "organism-identity"),),
        owner_checkpoint_verifier=_verifier,
        owner_checkpoint_contracts=contracts,
    )
    committed_before = manifest_store.read()
    initial_journals = (journal_path / "world-causal-journal-v3.jsonl",)
    assert committed_before.status is ManifestStatus.COMMITTED
    assert committed_before.causal_tick == 0
    assert len(initial_journals) == 1

    child = r"""
import os
import sys
from pathlib import Path
import world.manifest as manifest_module
from world import (
    DurableJournalSnapshotStore,
    DurableWorldRuntime,
    WorldCausalManifestStore,
    WorldId,
)

root = Path(sys.argv[1])
world_id = WorldId("world-partial-manifest-write")
manifest_store = WorldCausalManifestStore(root / "manifest.json")

def verify(references):
    import hashlib
    return all(
        Path(ref.locator).is_file()
        and hashlib.sha256(Path(ref.locator).read_bytes()).hexdigest() == ref.sha256
        for ref in references
    )

class Contract:
    def validate_checkpoint(self, encoded, owner_identity, schema_version):
        return bool(encoded) and owner_identity == "organism-identity" and schema_version == "v1"
    def verify_restoration(self, encoded, owner_identity, schema_version):
        return self.validate_checkpoint(encoded, owner_identity, schema_version)

runtime = DurableWorldRuntime.restore(
    world_id,
    "presence-a",
    manifest_store,
    DurableJournalSnapshotStore(root / "journal", world_id),
    owner_checkpoint_verifier=verify,
    owner_checkpoint_contracts={("organism", "organism-identity"): Contract()},
    owner_checkpoint_restorer=lambda _refs: None,
)

original_fdopen = os.fdopen
class InterruptedWriter:
    def __init__(self, fd, mode):
        self.stream = original_fdopen(fd, mode)
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc, traceback):
        return self.stream.__exit__(exc_type, exc, traceback)
    def write(self, payload):
        self.stream.write(payload[:max(1, len(payload) // 2)])
        self.stream.flush()
        os._exit(73)

manifest_module.os.fdopen = lambda fd, mode: InterruptedWriter(fd, mode)
runtime.tick("transition-interrupted-mid-manifest-write", lambda _tick: ())
os._exit(74)
"""
    result = subprocess.run(
        [sys.executable, "-c", child, str(tmp_path)],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")},
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 73, result.stderr

    # Atomic replacement was never reached: the old committed cut remains the
    # authoritative manifest even though an incomplete temporary file remains.
    assert manifest_store.read() == committed_before
    assert manifest_store.read().status is ManifestStatus.COMMITTED
    assert manifest_store.read().causal_tick == 0
    abandoned_temps = tuple(
        path
        for path in tmp_path.glob(".manifest.json.*")
        if path.name != ".manifest.json.writer.lock"
    )
    assert len(abandoned_temps) == 1
    assert 0 < abandoned_temps[0].stat().st_size < manifest_path.stat().st_size
    assert tuple(journal_path.glob("world-causal-journal-v3.jsonl")) == initial_journals

    # A new runtime can still continue from the last committed cut. This does
    # not assert automatic cleanup of the abandoned temporary file.
    restored = DurableWorldRuntime.restore(
        world_id,
        "presence-a",
        manifest_store,
        journal_store,
        owner_checkpoint_verifier=_verifier,
        owner_checkpoint_contracts=contracts,
        owner_checkpoint_restorer=lambda _refs: None,
    )
    assert restored.status is ManifestStatus.COMMITTED
    assert restored.causal_tick == 0


def test_process_termination_after_uncommitted_journal_append_recovers_prior_prefix(tmp_path):
    """A candidate appended past the manifest ref is not part of the cut."""
    world_id = WorldId("world-uncommitted-journal-tail")
    manifest_path = tmp_path / "manifest.json"
    journal_path = tmp_path / "journal"
    manifest_store = WorldCausalManifestStore(manifest_path)
    journal_store = DurableJournalSnapshotStore(journal_path, world_id)
    initial_owner = tmp_path / "owner-0.json"
    initial_owner.write_bytes(b"committed owner state at tick zero")
    contracts = _contracts(("organism", "organism-identity"))
    _initialize(
        world_id, "presence-a", manifest_store, journal_store,
        initial_owner_references=(_checkpoint(initial_owner),),
        registered_owner_ids=(("organism", "organism-identity"),),
        owner_checkpoint_verifier=_verifier, owner_checkpoint_contracts=contracts,
    )
    committed_before = manifest_store.read()
    previous_reference = next(ref for ref in committed_before.checkpoint_references if ref.owner == "world-journal")
    committed_records = journal_store.read(previous_reference).replay()
    committed_size = Path(previous_reference.locator).stat().st_size
    journal_file = Path(previous_reference.locator)

    child = r"""
import os, sys
from pathlib import Path
from world import DurableJournalSnapshotStore, DurableWorldRuntime, WorldCausalManifestStore, WorldId
root = Path(sys.argv[1])
world_id = WorldId("world-uncommitted-journal-tail")
manifest_store = WorldCausalManifestStore(root / "manifest.json")
journal_file = root / "journal" / "world-causal-journal-v3.jsonl"
def verify(refs):
    return all(DurableJournalSnapshotStore.verify(ref) if ref.owner == "world-journal" else Path(ref.locator).is_file() for ref in refs)
class Contract:
    def validate_checkpoint(self, encoded, owner_identity, schema_version):
        return bool(encoded) and owner_identity == "organism-identity" and schema_version == "v1"
    def verify_restoration(self, encoded, owner_identity, schema_version):
        return self.validate_checkpoint(encoded, owner_identity, schema_version)
runtime = DurableWorldRuntime.restore(world_id, "presence-a", manifest_store, DurableJournalSnapshotStore(root / "journal", world_id), owner_checkpoint_verifier=verify, owner_checkpoint_contracts={("organism", "organism-identity"): Contract()}, owner_checkpoint_restorer=lambda _refs: None)
original_fsync = os.fsync
def exit_after_append(fd):
    try:
        if Path(os.readlink(f"/proc/self/fd/{fd}")) == journal_file:
            original_fsync(fd)
            os._exit(73)
    except OSError:
        pass
    return original_fsync(fd)
os.fsync = exit_after_append
def owner_writer(tick):
    assert tick == 1
    path = root / "owner-1.json"
    path.write_bytes(b"owner tick one")
    from hashlib import sha256
    from world import CheckpointReference
    return (CheckpointReference("organism", "organism-identity", "owner-1", "v1", str(path), sha256(path.read_bytes()).hexdigest()),)
runtime.tick("uncommitted-tail", owner_writer)
os._exit(74)
"""
    result = subprocess.run(
        [sys.executable, "-c", child, str(tmp_path)],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")},
        capture_output=True, text=True, timeout=15, check=False,
    )
    assert result.returncode == 73, result.stderr
    pending = manifest_store.read()
    assert pending.status is ManifestStatus.PREPARING
    assert pending.causal_tick == committed_before.causal_tick
    assert pending.checkpoint_references == committed_before.checkpoint_references
    assert journal_file.stat().st_size > committed_size
    assert journal_store.read(previous_reference).replay() == committed_records

    with pytest.raises(ManifestRecoveryRequiredError):
        DurableWorldRuntime.restore(
            world_id, "presence-a", manifest_store, journal_store,
            owner_checkpoint_verifier=_verifier, owner_checkpoint_contracts=contracts,
            owner_checkpoint_restorer=lambda _refs: None,
        )
    recovery = manifest_store.read()
    assert recovery.status is ManifestStatus.RECOVERY_REQUIRED
    assert recovery.causal_tick == committed_before.causal_tick
    assert journal_store.read(previous_reference).replay() == committed_records
    recovered = DurableWorldRuntime.recover_previous_cut(
        world_id, "presence-a", manifest_store, journal_store,
        owner_checkpoint_verifier=_verifier, owner_checkpoint_contracts=contracts,
        owner_checkpoint_restorer=lambda _refs: None,
    )
    assert recovered.status is ManifestStatus.COMMITTED
    assert recovered.causal_tick == committed_before.causal_tick
    assert journal_store.read(previous_reference).replay() == committed_records


def test_duplicate_owner_record_id_is_rejected_across_world_ticks(tmp_path):
    world_id = WorldId("world-owner-record-unique")
    manifest_store = WorldCausalManifestStore(tmp_path / "manifest.json")
    journal_store = DurableJournalSnapshotStore(tmp_path / "journal", world_id)
    initial_owner = tmp_path / "owner-0.json"
    initial_owner.write_bytes(b"owner zero")
    runtime = _initialize(
        world_id,
        "presence-a",
        manifest_store,
        journal_store,
        initial_owner_references=(_checkpoint(initial_owner),),
        registered_owner_ids=(("organism", "organism-identity"),),
        owner_checkpoint_verifier=_verifier,
        owner_checkpoint_contracts=_contracts(("organism", "organism-identity")),
    )

    def commit_action(transition_id):
        state = tmp_path / f"{transition_id}.json"
        state.write_bytes(transition_id.encode())
        return runtime.tick(
            transition_id,
            lambda _tick: (_checkpoint(state),),
            write_owner_records=lambda _tick, _refs: (
                OwnerCausalRecord("organism", "organism-identity", "same-action", {"v": 1}),
            ),
        )

    commit_action("transition-1")
    with pytest.raises(ManifestError, match="already committed"):
        commit_action("transition-2")

    assert manifest_store.read().causal_tick == 1
    assert runtime.causal_tick == 1
    assert len(tuple(item for item in runtime.records if item.record_type == "world.owner_record.committed")) == 1


def test_owner_causal_records_must_be_unique_and_bound_to_participants(tmp_path):
    world_id = WorldId("world-owner-record-boundary")
    manifest_store = WorldCausalManifestStore(tmp_path / "manifest.json")
    journal_store = DurableJournalSnapshotStore(tmp_path / "journal", world_id)
    initial_owner = tmp_path / "owner-0.json"
    initial_owner.write_bytes(b"owner zero")
    runtime = _initialize(
        world_id,
        "presence-a",
        manifest_store,
        journal_store,
        initial_owner_references=(_checkpoint(initial_owner),),
        registered_owner_ids=(("organism", "organism-identity"),),
        owner_checkpoint_verifier=_verifier,
        owner_checkpoint_contracts=_contracts(("organism", "organism-identity")),
    )
    state = tmp_path / "owner-1.json"
    state.write_bytes(b"owner one")

    with pytest.raises(ManifestError, match="participating owner"):
        runtime.tick(
            "transition-unbound",
            lambda _tick: (_checkpoint(state),),
            write_owner_records=lambda _tick, _refs: (
                OwnerCausalRecord("lab", "run-1", "action-1", {}),
            ),
        )
    assert runtime.causal_tick == 0
    assert runtime.status is ManifestStatus.RECOVERY_REQUIRED
    assert not any(item.record_type == "world.causal_tick.committed" for item in runtime.records)


def test_owner_causal_record_rejects_duplicate_ids_within_tick_and_non_json_payload(tmp_path):
    world_id = WorldId("world-owner-record-invalid")
    manifest_store = WorldCausalManifestStore(tmp_path / "manifest.json")
    journal_store = DurableJournalSnapshotStore(tmp_path / "journal", world_id)
    initial_owner = tmp_path / "owner-0.json"
    initial_owner.write_bytes(b"owner zero")
    runtime = _initialize(
        world_id,
        "presence-a",
        manifest_store,
        journal_store,
        initial_owner_references=(_checkpoint(initial_owner),),
        registered_owner_ids=(("organism", "organism-identity"),),
        owner_checkpoint_verifier=_verifier,
        owner_checkpoint_contracts=_contracts(("organism", "organism-identity")),
    )
    state = tmp_path / "owner-1.json"
    state.write_bytes(b"owner one")

    with pytest.raises(ManifestError, match="duplicate owner causal record identities"):
        runtime.tick(
            "transition-duplicate",
            lambda _tick: (_checkpoint(state),),
            write_owner_records=lambda _tick, _refs: (
                OwnerCausalRecord("organism", "organism-identity", "same", {}),
                OwnerCausalRecord("organism", "organism-identity", "same", {}),
            ),
        )
    assert runtime.causal_tick == 0
    assert tuple(item.record_type for item in runtime.records) == ("world.causal_cut.committed",)

    with pytest.raises(TypeError, match="keys must be strings"):
        OwnerCausalRecord("organism", "organism-identity", "invalid", {1: "not JSON object key"})


def test_owner_causal_record_is_not_committed_when_manifest_publication_fails(tmp_path, monkeypatch):
    world_id = WorldId("world-owner-record-publication-failure")
    manifest_store = WorldCausalManifestStore(tmp_path / "manifest.json")
    journal_store = DurableJournalSnapshotStore(tmp_path / "journal", world_id)
    initial_owner = tmp_path / "owner-0.json"
    initial_owner.write_bytes(b"owner zero")
    runtime = _initialize(
        world_id,
        "presence-a",
        manifest_store,
        journal_store,
        initial_owner_references=(_checkpoint(initial_owner),),
        registered_owner_ids=(("organism", "organism-identity"),),
        owner_checkpoint_verifier=_verifier,
        owner_checkpoint_contracts=_contracts(("organism", "organism-identity")),
    )
    state = tmp_path / "owner-1.json"
    state.write_bytes(b"owner one")

    def fail_publication(*_args, **_kwargs):
        raise OSError("simulated manifest publication failure")

    monkeypatch.setattr(manifest_store, "_commit_transition", fail_publication)
    with pytest.raises(OSError, match="simulated manifest publication failure"):
        runtime.tick(
            "transition-publication-failure",
            lambda _tick: (_checkpoint(state),),
            write_owner_records=lambda _tick, _refs: (
                OwnerCausalRecord("organism", "organism-identity", "action-failed", {"v": 1}),
            ),
        )

    assert runtime.status is ManifestStatus.RECOVERY_REQUIRED
    assert runtime.causal_tick == 0
    assert not any(item.record_type == "world.owner_record.committed" for item in runtime.records)
    journal_reference = next(
        ref for ref in manifest_store.read().checkpoint_references if ref.owner == "world-journal"
    )
    assert journal_store.read(journal_reference).tail(1)[0].record_id != f"{world_id.value}:causal-cut:1"


def test_genesis_requires_presence_owner_and_durable_checkpoint_closure(tmp_path):
    world_id = WorldId("world-owner-closure")
    manifest_store = WorldCausalManifestStore(tmp_path / "manifest.json")
    journal_store = DurableJournalSnapshotStore(tmp_path / "journal", world_id)

    with pytest.raises(ManifestError, match="at least one durable owner checkpoint"):
        _initialize(
            world_id,
            "presence-a",
            manifest_store,
            journal_store,
            initial_owner_references=(),
            registered_owner_ids=(("organism", "organism-identity"),),
            owner_checkpoint_verifier=_verifier,
            owner_checkpoint_contracts=_contracts(("organism", "organism-identity")),
        )

    initial_owner = tmp_path / "organism-0.json"
    initial_owner.write_bytes(b"owner at tick zero")
    runtime = _initialize(
        world_id,
        "presence-a",
        manifest_store,
        journal_store,
        initial_owner_references=(_checkpoint(initial_owner),),
        registered_owner_ids=(("organism", "organism-identity"),),
        owner_checkpoint_verifier=_verifier,
        owner_checkpoint_contracts=_contracts(("organism", "organism-identity")),
    )

    # Presence is the World participant identity; owner_identity is a separate
    # resource identity and may differ. The latter must remain in the cut.
    replacement = tmp_path / "replacement-owner.json"
    replacement.write_bytes(b"different owner identity")
    with pytest.raises(ManifestError, match="identities must match"):
        runtime.tick("transition-1", lambda _tick: (_checkpoint(replacement, owner="embodiment"),))

    pending = manifest_store.read()
    assert pending.status is ManifestStatus.RECOVERY_REQUIRED
    assert pending.causal_tick == 0
    assert pending.checkpoint_references[0].owner_identity == "organism-identity"


def test_genesis_spec_is_authoritative_input_for_tick_zero_roster(tmp_path):
    world_id = WorldId("world-genesis-spec")
    store = WorldCausalManifestStore(tmp_path / "manifest.json")
    journal = DurableJournalSnapshotStore(tmp_path / "journal", world_id)
    organism = tmp_path / "organism.json"
    embodiment = tmp_path / "embodiment.json"
    organism.write_bytes(b"organism")
    embodiment.write_bytes(b"embodiment")
    references = (_checkpoint(organism), _checkpoint(embodiment, owner="embodiment"))
    owners = tuple(sorted((("organism", "organism-identity"), ("embodiment", "embodiment-identity"))))
    spec = WorldGenesisSpec(
        world_id=world_id,
        initial_presence_ids=("presence-a", "presence-b"),
        registered_owner_ids=owners,
        initial_owner_references=references,
        owner_checkpoint_contracts=_contracts(*owners),
        presence_owner_links=(
            PresenceOwnerLink(PresenceId("presence-a"), "organism", "organism-identity"),
            PresenceOwnerLink(PresenceId("presence-b"), "embodiment", "embodiment-identity"),
        ),
    )

    runtime = DurableWorldRuntime.initialize(
        spec, "presence-a", store, journal, owner_checkpoint_verifier=_verifier,
    )

    assert runtime.manifest.presence_ids == ("presence-a", "presence-b")
    assert runtime.manifest.registered_owner_ids == owners
    expected_links = (
        ("presence-a", "organism", "organism-identity"),
        ("presence-b", "embodiment", "embodiment-identity"),
    )
    assert runtime.manifest.presence_owner_links == expected_links
    assert store.read().presence_owner_links == expected_links
    old_schema_data = manifest_module._manifest_to_data(runtime.manifest)
    old_schema_data["schema_version"] = 3
    with pytest.raises(ValueError, match="unsupported manifest schema version: 3"):
        manifest_module._manifest_from_data(old_schema_data)
    historical_v3_data = dict(old_schema_data)
    historical_v3_data.pop("presence_owner_links")
    historical_v3_data.pop("pending_presence_owner_links")
    with pytest.raises(ValueError, match="manifest fields do not match schema"):
        manifest_module._manifest_from_data(historical_v3_data)
    assert frozenset((ref.owner, ref.owner_identity) for ref in runtime.manifest.checkpoint_references if ref.owner != "world-journal") == frozenset(owners)

    with pytest.raises(ManifestError, match="accepted World genesis spec"):
        DurableWorldRuntime.initialize(
            spec, "presence-outside-genesis", store, journal,
            owner_checkpoint_verifier=_verifier,
        )


def test_genesis_requires_explicit_unique_links_to_registered_owners():
    world_id = WorldId("world-genesis-presence-links")
    owners = (("organism", "organism-identity"),)
    base = {
        "world_id": world_id,
        "initial_presence_ids": ("presence-a",),
        "registered_owner_ids": owners,
        "initial_owner_references": (),
        "owner_checkpoint_contracts": {},
    }

    with pytest.raises(ValueError, match="cover each genesis presence exactly once"):
        WorldGenesisSpec(**base)
    with pytest.raises(ValueError, match="cover each genesis presence exactly once"):
        WorldGenesisSpec(
            **base,
            presence_owner_links=(
                PresenceOwnerLink(PresenceId("presence-a"), "organism", "organism-identity"),
                PresenceOwnerLink(PresenceId("presence-a"), "organism", "organism-identity"),
            ),
        )
    with pytest.raises(ValueError, match="target a registered genesis owner"):
        WorldGenesisSpec(
            **base,
            presence_owner_links=(
                PresenceOwnerLink(PresenceId("presence-a"), "environment", "environment-identity"),
            ),
        )


def test_world_owner_registry_changes_only_through_explicit_lifecycle_tick(tmp_path):
    world_id = WorldId("world-owner-lifecycle")
    manifest_store = WorldCausalManifestStore(tmp_path / "manifest.json")
    journal_store = DurableJournalSnapshotStore(tmp_path / "journal", world_id)
    organism = tmp_path / "organism-0.json"
    organism.write_bytes(b"organism")
    runtime = _initialize(
        world_id, "presence-a", manifest_store, journal_store,
        initial_owner_references=(_checkpoint(organism),),
        registered_owner_ids=(("organism", "organism-identity"),),
        owner_checkpoint_verifier=_verifier,
        owner_checkpoint_contracts=_contracts(("organism", "organism-identity")),
    )

    body = tmp_path / "embodiment-1.json"
    body.write_bytes(b"body")
    with pytest.raises(ManifestError, match="identities must match"):
        runtime.tick("transition-1", lambda _tick: (_checkpoint(organism), _checkpoint(body, owner="embodiment")))
    assert manifest_store.read().status is ManifestStatus.RECOVERY_REQUIRED
    assert manifest_store.read().registered_owner_ids == (("organism", "organism-identity"),)


def test_new_owner_cannot_register_without_owner_checkpoint_contract(tmp_path):
    world_id = WorldId("world-missing-owner-contract")
    store = WorldCausalManifestStore(tmp_path / "manifest.json")
    journal = DurableJournalSnapshotStore(tmp_path / "journal", world_id)
    owner = tmp_path / "organism.json"
    owner.write_bytes(b"organism")
    runtime = _initialize(
        world_id, "presence-a", store, journal,
        initial_owner_references=(_checkpoint(owner),),
        registered_owner_ids=(("organism", "organism-identity"),),
        owner_checkpoint_verifier=_verifier,
        owner_checkpoint_contracts=_contracts(("organism", "organism-identity")),
    )
    called = False

    def writer(_tick):
        nonlocal called
        called = True
        return (_checkpoint(owner),)

    with pytest.raises(ManifestError, match="requires its owner checkpoint contract"):
        runtime.tick(
            "unverified-owner-registration", writer,
            register_owners=(("embodiment", "embodiment-identity"),),
        )
    assert not called
    assert store._require_active().causal_tick == 0


def test_lifecycle_registration_and_step_participants_are_explicit_and_exact(tmp_path):
    world_id = WorldId("world-participants")
    manifest_store = WorldCausalManifestStore(tmp_path / "manifest.json")
    journal_store = DurableJournalSnapshotStore(tmp_path / "journal", world_id)
    organism = tmp_path / "organism.json"
    organism.write_bytes(b"organism")
    runtime = _initialize(
        world_id, "presence-a", manifest_store, journal_store,
        initial_owner_references=(_checkpoint(organism),),
        registered_owner_ids=(("organism", "organism-identity"),),
        owner_checkpoint_verifier=_verifier,
        owner_checkpoint_contracts=_contracts(("organism", "organism-identity")),
    )

    embodiment = tmp_path / "embodiment.json"
    embodiment.write_bytes(b"embodiment")
    runtime.tick(
        "register-body",
        lambda _tick: (_checkpoint(organism), _checkpoint(embodiment, owner="embodiment")),
        register_owners=(("embodiment", "embodiment-identity"),),
        register_owner_contracts=_contracts(("embodiment", "embodiment-identity")),
    )
    assert runtime.manifest.registered_owner_ids == (
        ("embodiment", "embodiment-identity"),
        ("organism", "organism-identity"),
    )

    modality = tmp_path / "modality.json"
    modality.write_bytes(b"step participant")
    runtime.tick(
        "step-only-modality",
        lambda _tick: (
            _checkpoint(organism),
            _checkpoint(embodiment, owner="embodiment"),
            _checkpoint(modality, owner="modality"),
        ),
        step_participants=(("modality", "modality-identity"),),
        register_owner_contracts=_contracts(("modality", "modality-identity")),
    )
    assert runtime.manifest.step_participant_ids == (("modality", "modality-identity"),)

    runtime.tick(
        "unregister-body",
        lambda _tick: (_checkpoint(organism),),
        unregister_owners=(("embodiment", "embodiment-identity"),),
    )
    assert runtime.manifest.registered_owner_ids == (("organism", "organism-identity"),)


def test_presence_roster_is_published_only_through_world_lifecycle_tick(tmp_path):
    world_id = WorldId("world-presence-roster")
    store = WorldCausalManifestStore(tmp_path / "manifest.json")
    journal = DurableJournalSnapshotStore(tmp_path / "journal", world_id)
    owner = tmp_path / "owner.json"
    owner.write_bytes(b"owner")
    runtime = _initialize(
        world_id, "presence-a", store, journal,
        initial_owner_references=(_checkpoint(owner),),
        registered_owner_ids=(("organism", "organism-identity"),),
        owner_checkpoint_verifier=_verifier,
        owner_checkpoint_contracts=_contracts(("organism", "organism-identity")),
    )
    next_owner = tmp_path / "owner-next.json"
    next_owner.write_bytes(b"owner-next")
    # A presence without an explicitly registered owner is not a valid roster
    # transition, and must be rejected before World publishes PREPARING.
    with pytest.raises(ManifestError, match="exactly one registered causal owner"):
        runtime.tick(
            "presence-join-without-owner",
            lambda _tick: (_checkpoint(next_owner),),
            register_presence=("presence-b",),
            presence_lifecycle=_presence_lifecycle("REGISTER", "presence-b"),
        )
    assert runtime.status is ManifestStatus.COMMITTED
    assert runtime.manifest.presence_ids == ("presence-a",)

    def interrupt_join(_tick):
        raise RuntimeError("presence owner adoption interrupted")

    with pytest.raises(RuntimeError, match="presence owner adoption interrupted"):
        runtime.tick(
            "presence-join-interrupted",
            interrupt_join,
            register_presence=("presence-b",),
            register_owners=(("embodiment", "embodiment-identity"),),
            presence_owner_updates=(PresenceOwnerLink(PresenceId("presence-b"), "embodiment", "embodiment-identity"),),
            register_owner_contracts=_contracts(("embodiment", "embodiment-identity")),
            presence_lifecycle=_presence_lifecycle("REGISTER", "presence-b"),
        )
    pending = store.read()
    assert pending.status is ManifestStatus.RECOVERY_REQUIRED
    assert pending.presence_owner_links == (("presence-a", "organism", "organism-identity"),)
    assert pending.pending_presence_owner_links == (
        ("presence-a", "organism", "organism-identity"),
        ("presence-b", "embodiment", "embodiment-identity"),
    )
    runtime = DurableWorldRuntime.recover_previous_cut(
        world_id,
        "presence-a",
        store,
        journal,
        owner_checkpoint_verifier=_verifier,
        owner_checkpoint_contracts=_contracts(("organism", "organism-identity")),
        owner_checkpoint_restorer=lambda _references: None,
    )
    assert runtime.manifest.presence_owner_links == (("presence-a", "organism", "organism-identity"),)

    body = tmp_path / "body-next.json"
    body.write_bytes(b"body-next")
    runtime.tick(
        "presence-join",
        lambda _tick: (_checkpoint(next_owner), _checkpoint(body, owner="embodiment")),
        register_presence=("presence-b",),
        register_owners=(("embodiment", "embodiment-identity"),),
        presence_owner_updates=(PresenceOwnerLink(PresenceId("presence-b"), "embodiment", "embodiment-identity"),),
        register_owner_contracts=_contracts(("embodiment", "embodiment-identity")),
        presence_lifecycle=_presence_lifecycle("REGISTER", "presence-b"),
    )
    assert runtime.manifest.presence_ids == ("presence-a", "presence-b")
    assert runtime.manifest.presence_owner_links == (
        ("presence-a", "organism", "organism-identity"),
        ("presence-b", "embodiment", "embodiment-identity"),
    )
    # Registering another presence does not silently rebind this runtime's actor.
    runtime.tick(
        "ordinary-after-register",
        lambda _tick: (_checkpoint(next_owner), _checkpoint(body, owner="embodiment")),
    )
    assert runtime.presence_id == "presence-a"
    latest_tick_record = next(
        record for record in reversed(runtime.records)
        if record.record_type == "world.causal_tick.committed"
    )
    assert latest_tick_record.actor_id == "presence-a"
    assert latest_tick_record.payload["presence_id"] == "presence-a"
    with pytest.raises(ManifestError, match="each active presence owner must be in the proposed registered owner roster"):
        runtime.tick(
            "unregister-linked-owner",
            lambda _tick: (_checkpoint(next_owner),),
            unregister_owners=(("embodiment", "embodiment-identity"),),
        )
    assert runtime.manifest.presence_ids == ("presence-a", "presence-b")

    # An owner may own multiple presences when the relationship is explicit.
    runtime.tick(
        "reassign-presence-owner",
        lambda _tick: (_checkpoint(next_owner), _checkpoint(body, owner="embodiment")),
        presence_owner_updates=(PresenceOwnerLink(PresenceId("presence-b"), "organism", "organism-identity"),),
    )
    assert runtime.manifest.presence_owner_links == (
        ("presence-a", "organism", "organism-identity"),
        ("presence-b", "organism", "organism-identity"),
    )
    runtime.tick(
        "presence-leave",
        lambda _tick: (_checkpoint(next_owner), _checkpoint(body, owner="embodiment")),
        unregister_presence=("presence-b",),
        unregister_owners=(("embodiment", "embodiment-identity"),),
        step_participants=(("embodiment", "embodiment-identity"),),
        register_owner_contracts=_contracts(("embodiment", "embodiment-identity")),
        presence_lifecycle=_presence_lifecycle("TERMINATE", "presence-b"),
    )
    assert runtime.manifest.presence_ids == ("presence-a",)
    assert runtime.manifest.presence_owner_links == (("presence-a", "organism", "organism-identity"),)


def test_last_presence_can_terminate_world_remains_active_and_later_registers(tmp_path):
    world_id = WorldId("world-empty-remains-active")
    store = WorldCausalManifestStore(tmp_path / "manifest.json")
    journal = DurableJournalSnapshotStore(tmp_path / "journal", world_id)
    initial = tmp_path / "organism-0.json"
    initial.write_bytes(b"organism initial state")
    contract = _contracts(("organism", "organism-identity"))
    runtime = _initialize(
        world_id, "presence-a", store, journal,
        initial_owner_references=(_checkpoint(initial),),
        registered_owner_ids=(("organism", "organism-identity"),),
        owner_checkpoint_verifier=_verifier,
        owner_checkpoint_contracts=contract,
    )

    departing = tmp_path / "organism-termination.json"
    departing.write_bytes(b"owner state adopted for termination")
    runtime.tick(
        "terminate-presence-a",
        lambda tick: (_checkpoint(departing),),
        unregister_presence=("presence-a",),
        unregister_owners=(("organism", "organism-identity"),),
        step_participants=(("organism", "organism-identity"),),
        register_owner_contracts=contract,
        presence_lifecycle=_presence_lifecycle("TERMINATE", "presence-a"),
    )
    assert runtime.causal_tick == 1
    assert runtime.status is ManifestStatus.COMMITTED
    assert runtime.manifest.presence_ids == ()
    assert runtime.manifest.presence_owner_links == ()
    assert runtime.manifest.registered_owner_ids == ()
    assert runtime.manifest.step_participant_ids == (("organism", "organism-identity"),)
    assert runtime.presence_id is None

    lifecycle = next(
        record for record in runtime.records
        if record.record_type == "world.presence.lifecycle.committed"
    )
    assert lifecycle.actor_id == "world"
    assert lifecycle.payload["authority"] == "world"
    assert lifecycle.payload["trigger_provenance"] == {
        "source": "test-request", "request_id": "terminate-presence-a"
    }
    assert lifecycle.payload["source_tick"] == 0
    assert lifecycle.payload["target_tick"] == 1
    assert lifecycle.payload["resulting_active_presence_ids"] == ()
    assert lifecycle.payload["required_participants"] == (("organism", "organism-identity"),)
    assert lifecycle.payload["owner_checkpoint_references"][0]["sha256"] == hashlib.sha256(
        departing.read_bytes()
    ).hexdigest()

    restored_empty = DurableWorldRuntime.restore(
        world_id, None, store, journal,
        owner_checkpoint_verifier=_verifier,
        owner_checkpoint_contracts=contract,
        owner_checkpoint_restorer=lambda _references: None,
    )
    assert restored_empty.causal_tick == 1
    assert restored_empty.manifest.presence_ids == ()
    assert restored_empty.status is ManifestStatus.COMMITTED
    with pytest.raises(ManifestError, match="at least one participating owner"):
        restored_empty.tick("empty-world-ordinary-cut", lambda _tick: ())
    assert restored_empty.status is ManifestStatus.COMMITTED
    assert restored_empty.causal_tick == 1

    returning = tmp_path / "organism-reentry.json"
    returning.write_bytes(b"organism state for new presence")
    restored_empty.tick(
        "register-presence-b",
        lambda _tick: (_checkpoint(returning),),
        register_presence=("presence-b",),
        register_owners=(("organism", "organism-identity"),),
        presence_owner_updates=(PresenceOwnerLink(PresenceId("presence-b"), "organism", "organism-identity"),),
        register_owner_contracts=contract,
        presence_lifecycle=_presence_lifecycle("REGISTER", "presence-b"),
    )
    assert restored_empty.causal_tick == 2
    assert restored_empty.presence_id == "presence-b"
    assert restored_empty.manifest.presence_ids == ("presence-b",)
    assert restored_empty.records[0].record_type == "world.causal_cut.committed"
    retained_lifecycle = next(
        record for record in restored_empty.records
        if record.record_type == "world.presence.lifecycle.committed"
        and record.payload["lifecycle_transition_id"] == "terminate-presence-a"
    )
    assert retained_lifecycle.payload["owner_checkpoint_references"][0]["sha256"] == hashlib.sha256(
        departing.read_bytes()
    ).hexdigest()
    assert restored_empty.records[-1].causal_tick == 2

    # Reusing a committed lifecycle identity is rejected before PREPARING,
    # even if the retried request names a different otherwise-valid presence.
    callback_called = False

    def should_not_run(_tick):
        nonlocal callback_called
        callback_called = True
        return (_checkpoint(returning),)

    with pytest.raises(ManifestError, match="identity is already committed"):
        restored_empty.tick(
            "terminate-presence-a",
            should_not_run,
            register_presence=("presence-c",),
            presence_owner_updates=(PresenceOwnerLink(PresenceId("presence-c"), "organism", "organism-identity"),),
            presence_lifecycle=_presence_lifecycle("REGISTER", "presence-c"),
        )
    assert not callback_called
    assert restored_empty.status is ManifestStatus.COMMITTED
    assert restored_empty.causal_tick == 2

    restored_empty.tick(
        "register-presence-c",
        lambda _tick: (_checkpoint(returning),),
        register_presence=("presence-c",),
        presence_owner_updates=(PresenceOwnerLink(PresenceId("presence-c"), "organism", "organism-identity"),),
        presence_lifecycle=_presence_lifecycle("REGISTER", "presence-c"),
    )
    assert restored_empty.causal_tick == 3
    assert restored_empty.manifest.presence_ids == ("presence-b", "presence-c")


def test_rejected_or_failed_presence_lifecycle_does_not_publish_next_tick(tmp_path):
    world_id = WorldId("world-lifecycle-failure")
    store = WorldCausalManifestStore(tmp_path / "manifest.json")
    journal = DurableJournalSnapshotStore(tmp_path / "journal", world_id)
    owner = tmp_path / "owner-0.json"
    owner.write_bytes(b"owner")
    contracts = _contracts(("organism", "organism-identity"))
    runtime = _initialize(
        world_id, "presence-a", store, journal,
        initial_owner_references=(_checkpoint(owner),),
        registered_owner_ids=(("organism", "organism-identity"),),
        owner_checkpoint_verifier=_verifier,
        owner_checkpoint_contracts=contracts,
    )
    with pytest.raises(ManifestError, match="explicit lifecycle transition"):
        runtime.tick(
            "untyped-removal", lambda _tick: (_checkpoint(owner),),
            unregister_presence=("presence-a",),
        )
    assert runtime.causal_tick == 0
    assert runtime.status is ManifestStatus.COMMITTED
    assert store.read().presence_ids == ("presence-a",)

    # A successful ordinary tick after the rejected input proves the
    # pre-PREPARING rejection did not latch the runtime.
    continued = tmp_path / "owner-continued.json"
    continued.write_bytes(b"owner after rejected request")
    runtime.tick("ordinary-after-rejection", lambda _tick: (_checkpoint(continued),))
    assert runtime.causal_tick == 1
    assert runtime.status is ManifestStatus.COMMITTED

    def interrupted(_tick):
        raise RuntimeError("owner did not adopt termination state")

    with pytest.raises(RuntimeError, match="did not adopt"):
        runtime.tick(
            "failed-removal", interrupted,
            unregister_presence=("presence-a",),
            unregister_owners=(("organism", "organism-identity"),),
            step_participants=(("organism", "organism-identity"),),
            register_owner_contracts=contracts,
            presence_lifecycle=_presence_lifecycle("TERMINATE", "presence-a"),
        )
    pending = store.read()
    assert pending.status is ManifestStatus.RECOVERY_REQUIRED
    assert pending.causal_tick == 1
    assert pending.presence_ids == ("presence-a",)
    assert pending.pending_presence_ids == ()
    assert not any(
        record.record_type == "world.presence.lifecycle.committed"
        for record in runtime.records
    )
    recovered = DurableWorldRuntime.recover_previous_cut(
        world_id,
        "presence-a",
        store,
        journal,
        owner_checkpoint_verifier=_verifier,
        owner_checkpoint_contracts=contracts,
        owner_checkpoint_restorer=lambda _references: None,
    )
    assert recovered.status is ManifestStatus.COMMITTED
    assert recovered.causal_tick == 1
    assert recovered.manifest.presence_ids == ("presence-a",)


def test_failed_registration_keeps_prior_cut_and_pending_roster_for_recovery(tmp_path):
    world_id = WorldId("world-partial-adoption")
    store = WorldCausalManifestStore(tmp_path / "manifest.json")
    journal = DurableJournalSnapshotStore(tmp_path / "journal", world_id)
    owner = tmp_path / "owner.json"
    owner.write_bytes(b"owner")
    runtime = _initialize(
        world_id, "presence-a", store, journal,
        initial_owner_references=(_checkpoint(owner),),
        registered_owner_ids=(("organism", "organism-identity"),),
        owner_checkpoint_verifier=_verifier,
        owner_checkpoint_contracts=_contracts(("organism", "organism-identity")),
    )
    body = tmp_path / "body.json"
    body.write_bytes(b"partially adopted")

    def fail_after_partial_adoption(_tick):
        raise RuntimeError("owner adoption interrupted")

    with pytest.raises(RuntimeError, match="owner adoption interrupted"):
        runtime.tick(
            "register-body",
            fail_after_partial_adoption,
            register_owners=(("embodiment", "embodiment-identity"),),
            register_owner_contracts=_contracts(("embodiment", "embodiment-identity")),
        )

    pending = store.read()
    assert pending.status is ManifestStatus.RECOVERY_REQUIRED
    assert pending.causal_tick == 0
    assert pending.registered_owner_ids == (("organism", "organism-identity"),)
    assert pending.pending_registered_owner_ids == (
        ("embodiment", "embodiment-identity"),
        ("organism", "organism-identity"),
    )


@pytest.mark.parametrize("validation_result", [False, object()], ids=["reject", "non-boolean-result"])
def test_world_rejects_owner_checkpoint_unless_contract_returns_true(tmp_path, validation_result):
    world_id = WorldId("world-validation")
    store = WorldCausalManifestStore(tmp_path / "manifest.json")
    journal = DurableJournalSnapshotStore(tmp_path / "journal", world_id)
    owner = tmp_path / "owner.json"
    owner.write_bytes(b"owner")
    contract = _OwnerContract()
    runtime = _initialize(
        world_id, "presence-a", store, journal,
        initial_owner_references=(_checkpoint(owner),),
        registered_owner_ids=(("organism", "organism-identity"),),
        owner_checkpoint_verifier=_verifier,
        owner_checkpoint_contracts={("organism", "organism-identity"): contract},
    )
    next_owner = tmp_path / "next-owner.json"
    next_owner.write_bytes(b"next")
    reference = _checkpoint(next_owner)
    contract.validate_checkpoint = lambda *_args: validation_result

    with pytest.raises(ManifestError, match="owner checkpoint validation"):
        runtime.tick("validation-transition", lambda _tick: (reference,))
    assert store.read().status is ManifestStatus.RECOVERY_REQUIRED
    assert store.read().causal_tick == 0


def test_restore_and_recovery_reject_publicly_published_cuts_without_owner_state(tmp_path):
    world_id = WorldId("world-journal-only")
    manifest_store = WorldCausalManifestStore(tmp_path / "manifest.json")
    empty_journal_store = DurableJournalSnapshotStore(tmp_path / "empty-journal", world_id)
    journal_reference = empty_journal_store.write(EventJournal(), causal_tick=0)
    journal_store = DurableJournalSnapshotStore(tmp_path / "journal", world_id)
    owner_state = tmp_path / "owner.json"
    owner_state.write_bytes(b"valid owner")
    valid_runtime = _initialize(
        world_id,
        "presence-a",
        manifest_store,
        journal_store,
        initial_owner_references=(_checkpoint(owner_state),),
        registered_owner_ids=(("organism", "organism-identity"),),
        owner_checkpoint_verifier=_verifier,
        owner_checkpoint_contracts=_contracts(("organism", "organism-identity")),
    )
    valid_data = manifest_module._manifest_to_data(valid_runtime.manifest)
    journal_only_data = dict(valid_data)
    journal_only_data["checkpoint_references"] = [
        {
            "owner": journal_reference.owner,
            "owner_identity": journal_reference.owner_identity,
            "checkpoint_id": journal_reference.checkpoint_id,
            "schema_version": journal_reference.schema_version,
            "locator": journal_reference.locator,
            "sha256": journal_reference.sha256,
        }
    ]
    with pytest.raises(ValueError, match="owner checkpoint references must exactly match"):
        manifest_module._manifest_from_data(journal_only_data)

    # Simulate a pending manifest that has lost its durable owner checkpoint.
    pending_journal_only_data = dict(journal_only_data)
    pending_journal_only_data.update({
        "status": ManifestStatus.RECOVERY_REQUIRED.value,
        "pending_transition_id": "transition-pending",
        "pending_tick": 1,
        "failure": "simulated missing owner state",
        "pending_registered_owner_ids": [["organism", "organism-identity"]],
        "pending_presence_owner_links": [["presence-a", "organism", "organism-identity"]],
        "pending_step_participant_ids": [],
        "pending_presence_ids": ["presence-a"],
    })
    with pytest.raises(ValueError, match="owner checkpoint references must exactly match"):
        manifest_module._manifest_from_data(pending_journal_only_data)

    # The constructor boundary is now strict: an invalid active-presence cut
    # without its one-owner link cannot be represented as a TARGET manifest.
    with pytest.raises(ValueError, match="exactly one owner link per active presence"):
        WorldCausalManifest(
            world_id=world_id.value,
            causal_tick=0,
            presence_ids=("presence-a",),
            checkpoint_references=(journal_reference,),
            journal_last_sequence=None,
            journal_prefix_sha256="",
            commit_id="genesis:world-journal-only",
        )


def test_restore_validates_world_cut_then_delegates_owner_restore(tmp_path):
    world_id = WorldId("world-restore")
    manifest_store = WorldCausalManifestStore(tmp_path / "manifest.json")
    journal_store = DurableJournalSnapshotStore(tmp_path / "journal", world_id)
    owner_state = tmp_path / "organism.json"
    owner_state.write_bytes(b"durable owner state")
    initial = _initialize(
        world_id,
        "presence-a",
        manifest_store,
        journal_store,
        initial_owner_references=(_checkpoint(owner_state),),
        registered_owner_ids=(("organism", "organism-identity"),),
        owner_checkpoint_verifier=_verifier,
        owner_checkpoint_contracts=_contracts(("organism", "organism-identity")),
    )

    def write_tick(tick: int):
        state = tmp_path / f"organism-{tick}.json"
        state.write_bytes(f"owner tick {tick}".encode())
        return (_checkpoint(state),)

    initial.tick("transition-1", write_tick)
    restored_owners = []
    restored = DurableWorldRuntime.restore(
        world_id,
        "presence-a",
        manifest_store,
        journal_store,
        owner_checkpoint_verifier=_verifier,
        owner_checkpoint_contracts=_contracts(("organism", "organism-identity")),
        owner_checkpoint_restorer=restored_owners.append,
    )

    assert restored.causal_tick == 1
    assert restored.records == initial.records
    assert len(restored_owners) == 1
    assert restored_owners[0][0].owner == "organism"

    with pytest.raises(ManifestError, match="not part of the stored World cut"):
        DurableWorldRuntime.restore(
            world_id,
            "different-presence",
            manifest_store,
            journal_store,
            owner_checkpoint_verifier=_verifier,
        owner_checkpoint_contracts=_contracts(("organism", "organism-identity")),
            owner_checkpoint_restorer=lambda _refs: None,
        )


def test_restore_refuses_runtime_if_owner_restore_changes_checkpoint_dependency(tmp_path):
    world_id = WorldId("world-restore-mutated-dependency")
    manifest_store = WorldCausalManifestStore(tmp_path / "manifest.json")
    journal_store = DurableJournalSnapshotStore(tmp_path / "journal", world_id)
    owner_state = tmp_path / "organism.json"
    owner_state.write_bytes(b"durable owner state")
    _initialize(
        world_id,
        "presence-a",
        manifest_store,
        journal_store,
        initial_owner_references=(_checkpoint(owner_state),),
        registered_owner_ids=(("organism", "organism-identity"),),
        owner_checkpoint_verifier=_verifier,
        owner_checkpoint_contracts=_contracts(("organism", "organism-identity")),
    )

    def mutating_restore(_references):
        owner_state.write_bytes(b"changed while restoring")

    with pytest.raises(ManifestError, match="owner restoration validation"):
        DurableWorldRuntime.restore(
            world_id,
            "presence-a",
            manifest_store,
            journal_store,
            owner_checkpoint_verifier=_verifier,
        owner_checkpoint_contracts=_contracts(("organism", "organism-identity")),
            owner_checkpoint_restorer=mutating_restore,
        )

    # Refusal is local to restore; this does not claim the manifest itself
    # provides cross-owner recovery or a crash-consistency protocol.
    assert manifest_store._require_active().causal_tick == 0


def test_unverifiable_owner_checkpoint_keeps_durable_manifest_pending(tmp_path):
    world_id = WorldId("world-incomplete")
    manifest_store = WorldCausalManifestStore(tmp_path / "manifest.json")
    journal_store = DurableJournalSnapshotStore(tmp_path / "journal", world_id)
    initial_owner = tmp_path / "organism-0.json"
    initial_owner.write_bytes(b"initial")
    runtime = _initialize(
        world_id,
        "presence-a",
        manifest_store,
        journal_store,
        initial_owner_references=(_checkpoint(initial_owner),),
        registered_owner_ids=(("organism", "organism-identity"),),
        owner_checkpoint_verifier=_verifier,
        owner_checkpoint_contracts=_contracts(("organism", "organism-identity")),
    )

    missing = CheckpointReference(
        owner="organism",
        owner_identity="organism-identity",
        checkpoint_id="missing",
        schema_version="v1",
        locator=str(tmp_path / "missing.json"),
        sha256="0" * 64,
    )
    with pytest.raises(ManifestError, match="owner checkpoint validation"):
        runtime.tick("transition-1", lambda _tick: (missing,))

    assert manifest_store.read().status is ManifestStatus.RECOVERY_REQUIRED
    with pytest.raises(RecoveryRequiredError):
        runtime.tick("transition-2", lambda _tick: ())


def test_explicit_recovery_restores_prior_cut_through_owner_and_records_receipt(tmp_path):
    world_id = WorldId("world-recovery")
    manifest_store = WorldCausalManifestStore(tmp_path / "manifest.json")
    journal_store = DurableJournalSnapshotStore(tmp_path / "journal", world_id)
    owner_state = tmp_path / "organism-0.json"
    owner_state.write_bytes(b"organism at committed tick zero")
    runtime = _initialize(
        world_id,
        "presence-a",
        manifest_store,
        journal_store,
        initial_owner_references=(_checkpoint(owner_state),),
        registered_owner_ids=(("organism", "organism-identity"),),
        owner_checkpoint_verifier=_verifier,
        owner_checkpoint_contracts=_contracts(("organism", "organism-identity")),
    )
    live_owner_state = tmp_path / "live-organism-state.json"
    live_owner_state.write_bytes(b"partially changed by failed tick")

    missing = CheckpointReference(
        owner="organism",
        owner_identity="organism-identity",
        checkpoint_id="unpublished",
        schema_version="v1",
        locator=str(tmp_path / "missing.json"),
        sha256="0" * 64,
    )
    with pytest.raises(ManifestError, match="owner checkpoint validation"):
        runtime.tick("transition-failed", lambda _tick: (missing,))

    restored = []

    def restore_owner(references):
        restored.append(references)
        live_owner_state.write_bytes(Path(references[0].locator).read_bytes())

    recovered = DurableWorldRuntime.recover_previous_cut(
        world_id,
        "presence-a",
        manifest_store,
        journal_store,
        owner_checkpoint_verifier=_verifier,
        owner_checkpoint_contracts=_contracts(("organism", "organism-identity")),
        owner_checkpoint_restorer=restore_owner,
    )

    assert live_owner_state.read_bytes() == b"organism at committed tick zero"
    assert restored and restored[0][0].owner == "organism"
    assert recovered.status is ManifestStatus.COMMITTED
    assert recovered.causal_tick == 0
    assert tuple(record.record_type for record in recovered.records) == ("world.causal_cut.committed",)
    assert recovered.manifest.presence_owner_links == (("presence-a", "organism", "organism-identity"),)
    assert manifest_store._require_active() == recovered.manifest

    receipt_files = tuple((tmp_path / "recoveries").glob("*.json"))
    assert len(receipt_files) == 1
    receipt = json.loads(receipt_files[0].read_text())
    assert receipt["world_id"] == world_id.value
    assert receipt["transition_id"] == "transition-failed"
    assert receipt["restored_causal_tick"] == 0


def test_failed_owner_recovery_keeps_recovery_required_manifest(tmp_path):
    world_id = WorldId("world-recovery-fails")
    manifest_store = WorldCausalManifestStore(tmp_path / "manifest.json")
    journal_store = DurableJournalSnapshotStore(tmp_path / "journal", world_id)
    owner_state = tmp_path / "organism.json"
    owner_state.write_bytes(b"durable")
    runtime = _initialize(
        world_id,
        "presence-a",
        manifest_store,
        journal_store,
        initial_owner_references=(_checkpoint(owner_state),),
        registered_owner_ids=(("organism", "organism-identity"),),
        owner_checkpoint_verifier=_verifier,
        owner_checkpoint_contracts=_contracts(("organism", "organism-identity")),
    )
    with pytest.raises(RuntimeError, match="tick callback failed"):
        runtime.tick(
            "transition-failed",
            lambda _tick: (_ for _ in ()).throw(RuntimeError("tick callback failed")),
        )
    pending = manifest_store.read()

    with pytest.raises(RuntimeError, match="owner restore failed"):
        DurableWorldRuntime.recover_previous_cut(
            world_id,
            "presence-a",
            manifest_store,
            journal_store,
            owner_checkpoint_verifier=_verifier,
        owner_checkpoint_contracts=_contracts(("organism", "organism-identity")),
            owner_checkpoint_restorer=lambda _refs: (_ for _ in ()).throw(RuntimeError("owner restore failed")),
        )

    assert manifest_store.read() == pending
    with pytest.raises(ManifestRecoveryRequiredError):
        manifest_store._require_active()


def test_retry_after_recovery_publication_interruption_fsyncs_existing_receipt_directory(
    tmp_path, monkeypatch
):
    world_id = WorldId("world-recovery-retry")
    manifest_store = WorldCausalManifestStore(tmp_path / "manifest.json")
    journal_store = DurableJournalSnapshotStore(tmp_path / "journal", world_id)
    owner_state = tmp_path / "organism.json"
    owner_state.write_bytes(b"durable")
    runtime = _initialize(
        world_id,
        "presence-a",
        manifest_store,
        journal_store,
        initial_owner_references=(_checkpoint(owner_state),),
        registered_owner_ids=(("organism", "organism-identity"),),
        owner_checkpoint_verifier=_verifier,
        owner_checkpoint_contracts=_contracts(("organism", "organism-identity")),
    )
    with pytest.raises(RuntimeError, match="tick callback failed"):
        runtime.tick(
            "transition-retry",
            lambda _tick: (_ for _ in ()).throw(RuntimeError("tick callback failed")),
        )

    original_write = manifest_store._write

    def interrupt_complete_publication(manifest):
        if manifest.status is ManifestStatus.COMMITTED:
            raise OSError("simulated interruption before manifest publication")
        original_write(manifest)

    original_fsync_directory = manifest_module._fsync_directory
    fsynced_directories = []

    def record_fsync_directory(path):
        fsynced_directories.append(Path(path))
        original_fsync_directory(Path(path))

    monkeypatch.setattr(manifest_module, "_fsync_directory", record_fsync_directory)
    monkeypatch.setattr(manifest_store, "_write", interrupt_complete_publication)
    with pytest.raises(OSError, match="simulated interruption"):
        DurableWorldRuntime.recover_previous_cut(
            world_id,
            "presence-a",
            manifest_store,
            journal_store,
            owner_checkpoint_verifier=_verifier,
        owner_checkpoint_contracts=_contracts(("organism", "organism-identity")),
            owner_checkpoint_restorer=lambda _refs: None,
        )

    receipt_dir = tmp_path / "recoveries"
    assert manifest_store.read().status is ManifestStatus.RECOVERY_REQUIRED
    assert receipt_dir.is_dir()
    assert tmp_path in fsynced_directories
    assert receipt_dir in fsynced_directories
    receipt_path = next(receipt_dir.glob("*.json"))
    first_receipt = receipt_path.read_bytes()

    with pytest.raises(OSError, match="simulated interruption"):
        DurableWorldRuntime.recover_previous_cut(
            world_id,
            "presence-a",
            manifest_store,
            journal_store,
            owner_checkpoint_verifier=_verifier,
            owner_checkpoint_contracts=_contracts(("organism", "organism-identity")),
            owner_checkpoint_restorer=lambda _refs: None,
        )
    assert receipt_path.read_bytes() == first_receipt

    # A retry may record a different diagnostic without changing which
    # transition or prior cut is being recovered.
    manifest_store._record_failure("transition-retry", "different failure detail")
    assert manifest_store.read().failure == "different failure detail"

    fsynced_directories.clear()
    monkeypatch.setattr(manifest_store, "_write", original_write)
    recovered = DurableWorldRuntime.recover_previous_cut(
        world_id,
        "presence-a",
        manifest_store,
        journal_store,
        owner_checkpoint_verifier=_verifier,
        owner_checkpoint_contracts=_contracts(("organism", "organism-identity")),
        owner_checkpoint_restorer=lambda _refs: None,
    )

    assert recovered.status is ManifestStatus.COMMITTED
    assert receipt_dir in fsynced_directories
    assert fsynced_directories.index(receipt_dir) < len(fsynced_directories) - 1
    assert fsynced_directories[-1] == tmp_path
    assert receipt_path.read_bytes() == first_receipt


def test_recovery_receipts_keep_distinct_transition_identities(tmp_path):
    world_id = WorldId("world-recovery-distinct-transitions")
    manifest_store = WorldCausalManifestStore(tmp_path / "manifest.json")
    journal_store = DurableJournalSnapshotStore(tmp_path / "journal", world_id)
    owner_state = tmp_path / "organism.json"
    owner_state.write_bytes(b"durable")
    runtime = _initialize(
        world_id,
        "presence-a",
        manifest_store,
        journal_store,
        initial_owner_references=(_checkpoint(owner_state),),
        registered_owner_ids=(("organism", "organism-identity"),),
        owner_checkpoint_verifier=_verifier,
        owner_checkpoint_contracts=_contracts(("organism", "organism-identity")),
    )

    for transition_id in ("transition-one", "transition-two"):
        with pytest.raises(ManifestError, match="owner checkpoint validation"):
            runtime.tick(
                transition_id,
                lambda _tick: (
                    CheckpointReference(
                        owner="organism",
                        owner_identity="organism-identity",
                        checkpoint_id="missing",
                        schema_version="v1",
                        locator=str(tmp_path / f"{transition_id}-missing.json"),
                        sha256="0" * 64,
                    ),
                ),
            )
        runtime = DurableWorldRuntime.recover_previous_cut(
            world_id,
            "presence-a",
            manifest_store,
            journal_store,
            owner_checkpoint_verifier=_verifier,
            owner_checkpoint_contracts=_contracts(("organism", "organism-identity")),
            owner_checkpoint_restorer=lambda _refs: None,
        )

    receipt_files = tuple((tmp_path / "recoveries").glob("*.json"))
    assert len(receipt_files) == 2
    assert {json.loads(path.read_bytes())["transition_id"] for path in receipt_files} == {
        "transition-one",
        "transition-two",
    }


def test_recovered_transition_identity_cannot_be_reused_for_different_proposal(tmp_path):
    world_id = WorldId("world-recovery-reused-transition")
    manifest_store = WorldCausalManifestStore(tmp_path / "manifest.json")
    journal_store = DurableJournalSnapshotStore(tmp_path / "journal", world_id)
    owner_state = tmp_path / "organism.json"
    owner_state.write_bytes(b"durable")
    runtime = _initialize(
        world_id,
        "presence-a",
        manifest_store,
        journal_store,
        initial_owner_references=(_checkpoint(owner_state),),
        registered_owner_ids=(("organism", "organism-identity"),),
        owner_checkpoint_verifier=_verifier,
        owner_checkpoint_contracts=_contracts(("organism", "organism-identity")),
    )
    with pytest.raises(RuntimeError, match="tick callback failed"):
        runtime.tick(
            "reused-transition",
            lambda _tick: (_ for _ in ()).throw(RuntimeError("tick callback failed")),
        )
    DurableWorldRuntime.recover_previous_cut(
        world_id,
        "presence-a",
        manifest_store,
        journal_store,
        owner_checkpoint_verifier=_verifier,
        owner_checkpoint_contracts=_contracts(("organism", "organism-identity")),
        owner_checkpoint_restorer=lambda _refs: None,
    )

    with pytest.raises(ManifestError, match="cannot be reused with different proposed state"):
        manifest_store._begin_transition(
            "reused-transition",
            step_participant_ids=(("modality", "modality-identity"),),
        )
    assert manifest_store.read().status is ManifestStatus.COMMITTED


def test_same_transition_id_on_later_tick_gets_a_distinct_recovery_receipt(tmp_path):
    world_id = WorldId("world-recovery-transition-across-ticks")
    manifest_store = WorldCausalManifestStore(tmp_path / "manifest.json")
    journal_store = DurableJournalSnapshotStore(tmp_path / "journal", world_id)
    owner_state = tmp_path / "organism.json"
    owner_state.write_bytes(b"durable")
    owner_reference = _checkpoint(owner_state)
    contracts = _contracts(("organism", "organism-identity"))
    runtime = _initialize(
        world_id,
        "presence-a",
        manifest_store,
        journal_store,
        initial_owner_references=(owner_reference,),
        registered_owner_ids=(("organism", "organism-identity"),),
        owner_checkpoint_verifier=_verifier,
        owner_checkpoint_contracts=contracts,
    )

    with pytest.raises(RuntimeError, match="first failure"):
        runtime.tick(
            "reused-transition",
            lambda _tick: (_ for _ in ()).throw(RuntimeError("first failure")),
        )
    runtime = DurableWorldRuntime.recover_previous_cut(
        world_id,
        "presence-a",
        manifest_store,
        journal_store,
        owner_checkpoint_verifier=_verifier,
        owner_checkpoint_contracts=contracts,
        owner_checkpoint_restorer=lambda _refs: None,
    )
    runtime.tick("successful-transition", lambda _tick: (owner_reference,))

    with pytest.raises(RuntimeError, match="second failure"):
        runtime.tick(
            "reused-transition",
            lambda _tick: (_ for _ in ()).throw(RuntimeError("second failure")),
        )
    DurableWorldRuntime.recover_previous_cut(
        world_id,
        "presence-a",
        manifest_store,
        journal_store,
        owner_checkpoint_verifier=_verifier,
        owner_checkpoint_contracts=contracts,
        owner_checkpoint_restorer=lambda _refs: None,
    )

    receipts = [json.loads(path.read_bytes()) for path in (tmp_path / "recoveries").glob("*.json")]
    assert len(receipts) == 2
    assert {receipt["transition_id"] for receipt in receipts} == {"reused-transition"}
    assert {receipt["pending_tick"] for receipt in receipts} == {1, 2}



def test_presence_lifecycle_detaches_and_deep_freezes_trigger_provenance():
    supplied = {"source": {"principal": "operator", "labels": ["manual"]}}
    transition = PresenceLifecycleTransition("REGISTER", PresenceId("presence-a"), supplied)
    supplied["source"]["principal"] = "changed"
    supplied["source"]["labels"].append("changed")

    assert transition.trigger_provenance["source"]["principal"] == "operator"
    assert transition.trigger_provenance["source"]["labels"] == ("manual",)
    with pytest.raises(TypeError):
        transition.trigger_provenance["source"]["principal"] = "mutated"


def test_recovery_required_error_hierarchy_is_unified():
    """Verify that ManifestRecoveryRequiredError is a subclass of runtime RecoveryRequiredError."""
    from world.manifest import RecoveryRequiredError as ManifestRecoveryError
    from world.runtime import RecoveryRequiredError as RuntimeRecoveryError

    err = ManifestRecoveryError("manifest failed")
    assert isinstance(err, RuntimeRecoveryError)
    assert isinstance(err, ManifestError)
    assert issubclass(ManifestRecoveryError, RuntimeRecoveryError)


def test_durable_runtime_status_consistent_with_external_store_recovery_required(tmp_path):
    """Ensure runtime.status cannot report COMMITTED when the authoritative store is RECOVERY_REQUIRED.

    Also tests:
    - Reading status does not mutate the store file.
    - Subsequent tick() fails with unified RecoveryRequiredError.
    - Explicit recover_previous_cut completes recovery and restores COMMITTED status.
    """
    world_id = WorldId("world-store-consistency")
    manifest_path = tmp_path / "manifest.json"
    manifest_store = WorldCausalManifestStore(manifest_path)
    journal_store = DurableJournalSnapshotStore(tmp_path / "journal", world_id)
    owner_state = tmp_path / "owner-0.json"
    owner_state.write_bytes(b"initial owner 0")
    contracts = _contracts(("organism", "organism-identity"))
    runtime = _initialize(
        world_id,
        "presence-a",
        manifest_store,
        journal_store,
        initial_owner_references=(_checkpoint(owner_state),),
        registered_owner_ids=(("organism", "organism-identity"),),
        owner_checkpoint_verifier=_verifier,
        owner_checkpoint_contracts=contracts,
    )

    assert runtime.status is ManifestStatus.COMMITTED

    # Simulate another actor or process marking the authoritative store as RECOVERY_REQUIRED
    # (e.g. after a crash during a transition or startup detecting interrupted state).
    manifest_store._begin_transition("external-transition")
    manifest_store._record_failure("external-transition", "external failure recorded")

    store_manifest = manifest_store.read()
    assert store_manifest.status is ManifestStatus.RECOVERY_REQUIRED
    store_file_mtime_before = manifest_path.stat().st_mtime_ns

    # Runtime status MUST NOT report COMMITTED when the authoritative store is RECOVERY_REQUIRED
    assert runtime.status is ManifestStatus.RECOVERY_REQUIRED

    # Reading status must not mutate the store file
    assert manifest_path.stat().st_mtime_ns == store_file_mtime_before

    # Attempting to tick on this runtime raises RecoveryRequiredError
    with pytest.raises(RecoveryRequiredError, match="outside this runtime|explicit recovery"):
        runtime.tick("transition-should-fail", lambda _t: ())

    # The runtime latched failure and remains RECOVERY_REQUIRED
    assert runtime.status is ManifestStatus.RECOVERY_REQUIRED

    # Explicit recovery restores the cut and returns a runtime with COMMITTED status
    recovered = DurableWorldRuntime.recover_previous_cut(
        world_id,
        "presence-a",
        manifest_store,
        journal_store,
        owner_checkpoint_verifier=_verifier,
        owner_checkpoint_contracts=contracts,
        owner_checkpoint_restorer=lambda _refs: None,
    )

    assert recovered.status is ManifestStatus.COMMITTED
    assert manifest_store.read().status is ManifestStatus.COMMITTED
