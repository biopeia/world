import hashlib
import json
from dataclasses import replace
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
    OwnerRecoveryReceipt,
    RecoveryRequiredError,
    WorldCausalManifest,
    WorldCausalManifestStore,
    WorldId,
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


def _receipt(reference):
    return OwnerRecoveryReceipt(
        owner=reference.owner,
        owner_identity=reference.owner_identity,
        checkpoint_id=reference.checkpoint_id,
        schema_version=reference.schema_version,
        locator=reference.locator,
        sha256=reference.sha256,
        recoverable=True,
    )


class _OwnerContract:
    def __init__(self, receipt_factory=None):
        self.receipt_factory = receipt_factory or _receipt

    def validate_recoverability(self, reference):
        return self.receipt_factory(reference)

    def verify_restoration(self, reference):
        return _receipt(reference)


def _contracts(*keys):
    return {key: _OwnerContract() for key in keys}


def test_durable_tick_publishes_journal_and_owner_checkpoint_before_complete_manifest(tmp_path):
    world_id = WorldId("world-durable")
    manifest_store = WorldCausalManifestStore(tmp_path / "manifest.json")
    journal_store = DurableJournalSnapshotStore(tmp_path / "journal", world_id)
    initial_owner = tmp_path / "organism-0.json"
    initial_owner.write_bytes(b"organism at tick zero")
    runtime = DurableWorldRuntime.initialize(
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
    assert manifest_store.require_active() == runtime.manifest


def test_complete_cut_requires_nonempty_stable_owner_identity_closure(tmp_path):
    world_id = WorldId("world-owner-closure")
    manifest_store = WorldCausalManifestStore(tmp_path / "manifest.json")
    journal_store = DurableJournalSnapshotStore(tmp_path / "journal", world_id)

    with pytest.raises(ManifestError, match="at least one durable owner checkpoint"):
        DurableWorldRuntime.initialize(
            world_id,
            "presence-a",
            manifest_store,
            journal_store,
            initial_owner_references=(),
            registered_owner_ids=(),
            owner_checkpoint_verifier=_verifier,
            owner_checkpoint_contracts={},
        )

    initial_owner = tmp_path / "organism-0.json"
    initial_owner.write_bytes(b"owner at tick zero")
    runtime = DurableWorldRuntime.initialize(
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


def test_world_owner_registry_changes_only_through_explicit_lifecycle_tick(tmp_path):
    world_id = WorldId("world-owner-lifecycle")
    manifest_store = WorldCausalManifestStore(tmp_path / "manifest.json")
    journal_store = DurableJournalSnapshotStore(tmp_path / "journal", world_id)
    organism = tmp_path / "organism-0.json"
    organism.write_bytes(b"organism")
    runtime = DurableWorldRuntime.initialize(
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
    runtime = DurableWorldRuntime.initialize(
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
    assert store.require_active().causal_tick == 0


def test_lifecycle_registration_and_step_participants_are_explicit_and_exact(tmp_path):
    world_id = WorldId("world-participants")
    manifest_store = WorldCausalManifestStore(tmp_path / "manifest.json")
    journal_store = DurableJournalSnapshotStore(tmp_path / "journal", world_id)
    organism = tmp_path / "organism.json"
    organism.write_bytes(b"organism")
    runtime = DurableWorldRuntime.initialize(
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
    runtime = DurableWorldRuntime.initialize(
        world_id, "presence-a", store, journal,
        initial_owner_references=(_checkpoint(owner),),
        registered_owner_ids=(("organism", "organism-identity"),),
        owner_checkpoint_verifier=_verifier,
        owner_checkpoint_contracts=_contracts(("organism", "organism-identity")),
    )
    next_owner = tmp_path / "owner-next.json"
    next_owner.write_bytes(b"owner-next")
    runtime.tick(
        "presence-join",
        lambda _tick: (_checkpoint(next_owner),),
        register_presence=("presence-b",),
    )
    assert runtime.manifest.presence_ids == ("presence-a", "presence-b")
    runtime.tick(
        "presence-leave",
        lambda _tick: (_checkpoint(next_owner),),
        unregister_presence=("presence-b",),
    )
    assert runtime.manifest.presence_ids == ("presence-a",)


def test_failed_registration_keeps_prior_cut_and_pending_roster_for_recovery(tmp_path):
    world_id = WorldId("world-partial-adoption")
    store = WorldCausalManifestStore(tmp_path / "manifest.json")
    journal = DurableJournalSnapshotStore(tmp_path / "journal", world_id)
    owner = tmp_path / "owner.json"
    owner.write_bytes(b"owner")
    runtime = DurableWorldRuntime.initialize(
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


@pytest.mark.parametrize("receipt_error", ["missing", "owner", "checkpoint", "hash"])
def test_owner_recoverability_receipt_is_required_and_bound_to_exact_reference(tmp_path, receipt_error):
    world_id = WorldId(f"world-receipt-{receipt_error}")
    store = WorldCausalManifestStore(tmp_path / "manifest.json")
    journal = DurableJournalSnapshotStore(tmp_path / "journal", world_id)
    owner = tmp_path / "owner.json"
    owner.write_bytes(b"owner")
    contract = _OwnerContract()
    runtime = DurableWorldRuntime.initialize(
        world_id, "presence-a", store, journal,
        initial_owner_references=(_checkpoint(owner),),
        registered_owner_ids=(("organism", "organism-identity"),),
        owner_checkpoint_verifier=_verifier,
        owner_checkpoint_contracts={("organism", "organism-identity"): contract},
    )
    next_owner = tmp_path / "next-owner.json"
    next_owner.write_bytes(b"next")
    reference = _checkpoint(next_owner)
    if receipt_error == "missing":
        contract.receipt_factory = lambda _ref: None
    elif receipt_error == "owner":
        contract.receipt_factory = lambda ref: replace(_receipt(ref), owner="other")
    elif receipt_error == "checkpoint":
        original_receipt = _receipt(_checkpoint(owner))
        contract.receipt_factory = lambda _ref: original_receipt
    else:
        contract.receipt_factory = lambda ref: replace(_receipt(ref), sha256="0" * 64)

    with pytest.raises(ManifestError, match="receipts"):
        runtime.tick("receipt-transition", lambda _tick: (reference,))
    assert store.read().status is ManifestStatus.RECOVERY_REQUIRED
    assert store.read().causal_tick == 0


def test_restore_and_recovery_reject_publicly_published_cuts_without_owner_state(tmp_path):
    world_id = WorldId("world-journal-only")
    manifest_store = WorldCausalManifestStore(tmp_path / "manifest.json")
    journal_store = DurableJournalSnapshotStore(tmp_path / "journal", world_id)
    journal_reference = journal_store.write(EventJournal(), causal_tick=0)
    journal_only = WorldCausalManifest(
        world_id=world_id.value,
        causal_tick=0,
        presence_ids=("presence-a",),
        checkpoint_references=(journal_reference,),
        journal_last_sequence=None,
        journal_prefix_sha256="",
        commit_id="genesis:world-journal-only",
    )

    # The lower-level publication API rejects this cut. Also exercise restore
    # fail-closed against an invalid/legacy on-disk manifest that bypasses it.
    with pytest.raises(ManifestError, match="at least one durable owner checkpoint"):
        manifest_store.initialize(journal_only)
    manifest_store.path.parent.mkdir(parents=True, exist_ok=True)
    manifest_store.path.write_text(
        json.dumps(manifest_module._manifest_to_data(journal_only)), encoding="utf-8"
    )
    with pytest.raises(ManifestError, match="at least one durable owner checkpoint"):
        DurableWorldRuntime.restore(
            world_id,
            "presence-a",
            manifest_store,
            journal_store,
            owner_checkpoint_verifier=lambda references: not references,
            owner_checkpoint_contracts={},
            owner_checkpoint_restorer=lambda _references: None,
        )

    pending_journal_only = WorldCausalManifest(
        world_id=world_id.value,
        causal_tick=0,
        presence_ids=("presence-a",),
        checkpoint_references=(journal_reference,),
        journal_last_sequence=None,
        journal_prefix_sha256="",
        commit_id="genesis:world-journal-only",
        status=ManifestStatus.RECOVERY_REQUIRED,
        pending_transition_id="transition-pending",
        pending_tick=1,
    )
    manifest_store._write(pending_journal_only)
    with pytest.raises(ManifestError, match="at least one durable owner checkpoint"):
        DurableWorldRuntime.recover_previous_cut(
            world_id,
            "presence-a",
            manifest_store,
            journal_store,
            owner_checkpoint_verifier=lambda references: not references,
            owner_checkpoint_contracts={},
            owner_checkpoint_restorer=lambda _references: None,
        )


def test_restore_validates_world_cut_then_delegates_owner_restore(tmp_path):
    world_id = WorldId("world-restore")
    manifest_store = WorldCausalManifestStore(tmp_path / "manifest.json")
    journal_store = DurableJournalSnapshotStore(tmp_path / "journal", world_id)
    owner_state = tmp_path / "organism.json"
    owner_state.write_bytes(b"durable owner state")
    initial = DurableWorldRuntime.initialize(
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
    DurableWorldRuntime.initialize(
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

    with pytest.raises(ManifestError, match="dependency closure changed during restore"):
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
    assert manifest_store.require_active().causal_tick == 0


def test_unverifiable_owner_checkpoint_keeps_durable_manifest_pending(tmp_path):
    world_id = WorldId("world-incomplete")
    manifest_store = WorldCausalManifestStore(tmp_path / "manifest.json")
    journal_store = DurableJournalSnapshotStore(tmp_path / "journal", world_id)
    initial_owner = tmp_path / "organism-0.json"
    initial_owner.write_bytes(b"initial")
    runtime = DurableWorldRuntime.initialize(
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
    with pytest.raises(ManifestError, match="dependency closure"):
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
    runtime = DurableWorldRuntime.initialize(
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
    with pytest.raises(ManifestError, match="dependency closure"):
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
    assert manifest_store.require_active() == recovered.manifest

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
    runtime = DurableWorldRuntime.initialize(
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
        manifest_store.require_active()


def test_retry_after_recovery_publication_interruption_fsyncs_existing_receipt_directory(
    tmp_path, monkeypatch
):
    world_id = WorldId("world-recovery-retry")
    manifest_store = WorldCausalManifestStore(tmp_path / "manifest.json")
    journal_store = DurableJournalSnapshotStore(tmp_path / "journal", world_id)
    owner_state = tmp_path / "organism.json"
    owner_state.write_bytes(b"durable")
    runtime = DurableWorldRuntime.initialize(
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
