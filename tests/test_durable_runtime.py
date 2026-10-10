import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import world.manifest as manifest_module
from world import (
    CheckpointReference,
    DurableJournalSnapshotStore,
    DurableWorldRuntime,
    EventJournal,
    ManifestError,
    ManifestRecoveryRequiredError,
    ManifestStatus,
    OwnerCausalRecord,
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
    assert runtime.records == (record,)
    assert runtime.manifest.journal_last_sequence == 0
    assert len(runtime.manifest.checkpoint_references) == 2
    assert manifest_store._require_active() == runtime.manifest


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
    initial_journals = tuple(journal_path.glob("journal-*.json"))
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
    assert tuple(journal_path.glob("journal-*.json")) == initial_journals

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
    initial_journals = tuple(journal_path.glob("journal-*.json"))
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
    assert tuple(journal_path.glob("journal-*.json")) == initial_journals

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


def test_process_termination_during_partial_journal_write_requires_recovery(tmp_path):
    """A partial candidate journal temp is not a published causal cut.

    The child first publishes PREPARING and writes a candidate owner checkpoint file,
    then exits after writing/flushing only a prefix of the candidate journal
    snapshot temporary file. This exercises one process-termination boundary
    on the current local filesystem; it is not a power-loss test.
    """
    world_id = WorldId("world-partial-journal-write")
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
    initial_journals = tuple(journal_path.glob("journal-*.json"))
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
world_id = WorldId("world-partial-journal-write")
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
fdopen_calls = 0
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

def interrupt_journal_temp(fd, mode):
    global fdopen_calls
    fdopen_calls += 1
    if fdopen_calls == 2:
        # Call 1 publishes PREPARING's manifest temp; call 2 writes the
        # candidate journal temp. The assertions in the parent verify both
        # boundaries were reached in that order.
        return InterruptedWriter(fd, mode)
    return original_fdopen(fd, mode)

os.fdopen = interrupt_journal_temp

def write_candidate_owner(tick):
    assert tick == 1
    state = root / "owner-1.json"
    state.write_bytes(b"candidate owner state at tick one")
    from hashlib import sha256
    from world import CheckpointReference
    return (CheckpointReference(
        owner="organism",
        owner_identity="organism-identity",
        checkpoint_id="organism-checkpoint-1",
        schema_version="v1",
        locator=str(state),
        sha256=sha256(state.read_bytes()).hexdigest(),
    ),)

runtime.tick("transition-interrupted-mid-journal-write", write_candidate_owner)
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

    preparing = manifest_store.read()
    assert preparing.status is ManifestStatus.PREPARING
    assert preparing.causal_tick == committed_before.causal_tick
    assert preparing.commit_id == committed_before.commit_id
    assert preparing.checkpoint_references == committed_before.checkpoint_references
    assert (tmp_path / "owner-1.json").read_bytes() == b"candidate owner state at tick one"
    assert tuple(journal_path.glob("journal-*.json")) == initial_journals
    abandoned_temps = tuple(journal_path.glob(".journal-*"))
    assert len(abandoned_temps) == 1
    assert abandoned_temps[0].stat().st_size > 0
    with pytest.raises(json.JSONDecodeError):
        json.loads(abandoned_temps[0].read_bytes())

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
    assert not runtime.records

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
    assert tuple((tmp_path / "journal").glob("journal-1-*.json"))


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
    )
    assert runtime.manifest.presence_ids == ("presence-a", "presence-b")
    assert runtime.manifest.presence_owner_links == (
        ("presence-a", "organism", "organism-identity"),
        ("presence-b", "embodiment", "embodiment-identity"),
    )
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
        lambda _tick: (_checkpoint(next_owner),),
        unregister_presence=("presence-b",),
        unregister_owners=(("embodiment", "embodiment-identity"),),
    )
    assert runtime.manifest.presence_ids == ("presence-a",)
    assert runtime.manifest.presence_owner_links == (("presence-a", "organism", "organism-identity"),)


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
    journal_store = DurableJournalSnapshotStore(tmp_path / "journal", world_id)
    journal_reference = journal_store.write(EventJournal(), causal_tick=0)
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
    assert recovered.records == ()
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


def test_process_termination_before_journal_file_fsync_keeps_previous_cut(tmp_path):
    """A complete candidate journal temp is not committed before file fsync.

    The child exits when World calls fsync on the candidate journal temp,
    after the complete JSON payload has been written and flushed but before
    that fsync call can complete. This exercises process termination on the
    current local filesystem only; it is not a power-loss test.
    """
    world_id = WorldId("world-journal-fsync-interruption")
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
    initial_journals = tuple(journal_path.glob("journal-*.json"))
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
world_id = WorldId("world-journal-fsync-interruption")
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
original_fsync = os.fsync
fdopen_calls = 0
journal_fd = None
class TrackingWriter:
    def __init__(self, fd, mode):
        self.stream = original_fdopen(fd, mode)
    def __enter__(self):
        global journal_fd
        journal_fd = self.stream.fileno()
        return self
    def __exit__(self, exc_type, exc, traceback):
        return self.stream.__exit__(exc_type, exc, traceback)
    def write(self, payload):
        return self.stream.write(payload)
    def flush(self):
        return self.stream.flush()
    def fileno(self):
        return self.stream.fileno()

def track_journal_temp(fd, mode):
    global fdopen_calls
    fdopen_calls += 1
    if fdopen_calls == 2:
        # The first fdopen writes PREPARING's manifest temp. The second owns
        # the candidate journal temp whose fsync boundary is under test.
        return TrackingWriter(fd, mode)
    return original_fdopen(fd, mode)

def interrupt_journal_fsync(fd):
    if fd == journal_fd:
        os._exit(73)
    return original_fsync(fd)

os.fdopen = track_journal_temp
os.fsync = interrupt_journal_fsync

def write_candidate_owner(tick):
    assert tick == 1
    state = root / "owner-1.json"
    state.write_bytes(b"candidate owner state at tick one")
    from hashlib import sha256
    from world import CheckpointReference
    return (CheckpointReference(
        owner="organism",
        owner_identity="organism-identity",
        checkpoint_id="organism-checkpoint-1",
        schema_version="v1",
        locator=str(state),
        sha256=sha256(state.read_bytes()).hexdigest(),
    ),)

runtime.tick("transition-interrupted-before-journal-fsync", write_candidate_owner)
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

    preparing = manifest_store.read()
    assert preparing.status is ManifestStatus.PREPARING
    assert preparing.causal_tick == committed_before.causal_tick
    assert preparing.commit_id == committed_before.commit_id
    assert preparing.checkpoint_references == committed_before.checkpoint_references
    assert tuple(journal_path.glob("journal-*.json")) == initial_journals
    abandoned_temps = tuple(journal_path.glob(".journal-*"))
    assert len(abandoned_temps) == 1
    candidate_records = json.loads(abandoned_temps[0].read_bytes())
    assert isinstance(candidate_records, list)
    assert candidate_records[-1]["record_id"] == f"{world_id.value}:tick:1"
    assert candidate_records[-1]["causal_tick"] == 1

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
