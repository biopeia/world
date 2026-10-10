"""Bounded Organism/Embodiment checkpoint and World-manifest integration fixture.

This test exercises normal publication and restore with synthetic local fixtures for Organism and Embodiment owners.
It does not prove process-crash recovery or validate complete Organism state.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from embodiment import EmbodimentCheckpoint, LivingBodyState, MetabolicLedger
from organism.cognition.graph import CognitiveGraph, PlasticEdge, PlasticNode
from organism.cognition.limits import KernelLimits
from organism.cognition.portable_checkpoint import CognitiveRuntimeCheckpoint
from organism.cognition.transition import CognitiveTransitionState, advance_cognition
from organism.cognition.types import EdgeKind, NodeKind
from organism.identity import OrganismIdentity
from world.durable_runtime import DurableJournalSnapshotStore, DurableWorldRuntime, PresenceOwnerLink, WorldGenesisSpec
from world.manifest import (
    CheckpointReference,
    ManifestError,
    ManifestStatus,
    WorldCausalManifestStore,
)
from world.runtime import PresenceId, WorldId


IDENTITY = OrganismIdentity.parse("12345678-1234-5678-1234-567812345678")
WORLD_ID = "checkpoint-integration-world"
BODY_ID = "fixture-body-1"
EPISODE_ID = "fixture-episode-1"


class _FixtureOwnerContract:
    """In-process test owner validating an exact local checkpoint reference."""

    def validate_checkpoint(self, encoded: bytes, owner_identity: str, schema_version: str) -> bool:
        return bool(encoded) and bool(owner_identity) and bool(schema_version)

    def verify_restoration(self, encoded: bytes, owner_identity: str, schema_version: str) -> bool:
        return self.validate_checkpoint(encoded, owner_identity, schema_version)


def _owner_contracts(references: tuple[CheckpointReference, ...]):
    return {
        (reference.owner, reference.owner_identity): _FixtureOwnerContract()
        for reference in references
        if reference.owner != "world-journal"
    }


def _owner_roster(references: tuple[CheckpointReference, ...]):
    return tuple(sorted((reference.owner, reference.owner_identity) for reference in references))


def _graph() -> CognitiveGraph:
    return CognitiveGraph(
        nodes=(
            PlasticNode("signal", NodeKind.SENSE),
            PlasticNode("readout", NodeKind.READOUT),
        ),
        edges=(
            PlasticEdge(
                source_id="signal",
                target_id="readout",
                kind=EdgeKind.EXCITATORY,
                weight=0.75,
                plasticity=0.0,
                delay_ticks=0,
            ),
        ),
        kernel_limits=KernelLimits(),
    )


def _reference(owner: str, identity: str, checkpoint_id: str, path: Path, schema: str) -> CheckpointReference:
    return CheckpointReference(
        owner=owner,
        owner_identity=identity,
        checkpoint_id=checkpoint_id,
        schema_version=schema,
        locator=str(path),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def test_owner_checkpoints_publish_and_restore_one_bounded_causal_cut(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    store = WorldCausalManifestStore(tmp_path / "world-manifest.json")
    body = LivingBodyState(
        energy_reserve=1.5,
        max_energy=2.0,
        metabolic_capacity={"observation": 1.0, "cognition": 1.0, "persistence": 1.0, "maintenance": 1.0},
        metabolic_replenishment={"observation": 0.0, "cognition": 0.0, "persistence": 0.0, "maintenance": 0.0},
        metabolic_reserve={"observation": 0.5, "cognition": 0.5, "persistence": 0.5, "maintenance": 0.5},
    )
    ledger = MetabolicLedger(body_state=body)
    cognitive = CognitiveRuntimeCheckpoint(CognitiveTransitionState.initial(IDENTITY), _graph())
    world_id = WorldId(WORLD_ID)
    journal_store = DurableJournalSnapshotStore(tmp_path / "world-journal", world_id)

    # Begin from a durable owner cut. The World journal is empty at tick zero,
    # but the semantic owners still need identifiable durable state.
    initial_owner_payloads = {
        "organism-initial.json": cognitive.to_organism_checkpoint().to_bytes(),
        "embodiment-initial.json": EmbodimentCheckpoint(BODY_ID, EPISODE_ID, body, ledger).to_bytes(),
    }
    initial_owner_refs = []
    for name, payload in initial_owner_payloads.items():
        path = artifact_dir / name
        path.write_bytes(payload)
        owner = name.split("-")[0]
        identity = str(IDENTITY) if owner == "organism" else BODY_ID if owner == "embodiment" else "fixture-environment"
        schema = {
            "organism": "organism-cognitive-v1",
            "embodiment": "embodiment-checkpoint-v1",
        }[owner]
        initial_owner_refs.append(_reference(owner, identity, f"{owner}-tick-0", path, schema))
    initial_refs = tuple(initial_owner_refs)

    def verify_owner_refs(references: tuple[CheckpointReference, ...]) -> bool:
        return all(
            Path(reference.locator).is_file()
            and hashlib.sha256(Path(reference.locator).read_bytes()).hexdigest() == reference.sha256
            for reference in references
        )

    runtime = DurableWorldRuntime.initialize(
        WorldGenesisSpec(
            world_id=world_id,
            initial_presence_ids=(str(IDENTITY),),
            registered_owner_ids=_owner_roster(initial_refs),
            initial_owner_references=initial_refs,
            owner_checkpoint_contracts=_owner_contracts(initial_refs),
            presence_owner_links=(PresenceOwnerLink(PresenceId(str(IDENTITY)), "organism", str(IDENTITY)),),
        ),
        str(IDENTITY),
        store,
        journal_store,
        owner_checkpoint_verifier=verify_owner_refs,
    )

    pending: dict[str, object] = {}

    def transition(causal_tick: int) -> None:
        assert causal_tick == 1
        signal = 0.5  # Explicit fixture input; this test exercises durable owner checkpoints, not Environment fields.
        next_state, _frame = advance_cognition(
            cognitive.state,
            cognitive.graph,
            {"signal": signal},
            causal_tick=causal_tick,
        )
        candidate_body = LivingBodyState.from_checkpoint(body.checkpoint())
        candidate_ledger = MetabolicLedger.from_checkpoint(ledger.checkpoint(), body_state=candidate_body)
        candidate_ledger.charge("cognition", 0.125)
        candidate_body.advance_age()
        pending["cognitive"] = CognitiveRuntimeCheckpoint(next_state, cognitive.graph)
        pending["body"] = candidate_body
        pending["ledger"] = candidate_ledger

    def persist_owner_checkpoints(causal_tick: int) -> tuple[CheckpointReference, ...]:
        transition(causal_tick)
        next_cognitive = pending["cognitive"]
        next_body = pending["body"]
        next_ledger = pending["ledger"]
        assert isinstance(next_cognitive, CognitiveRuntimeCheckpoint)
        assert isinstance(next_body, LivingBodyState)
        assert isinstance(next_ledger, MetabolicLedger)
        payloads = (
            (
                "organism",
                str(IDENTITY),
                f"organism-tick-{causal_tick}",
                "organism-cognitive-v1",
                next_cognitive.to_organism_checkpoint().to_bytes(),
            ),
            (
                "embodiment",
                BODY_ID,
                f"embodiment-tick-{causal_tick}",
                "embodiment-checkpoint-v1",
                EmbodimentCheckpoint(BODY_ID, EPISODE_ID, next_body, next_ledger).to_bytes(),
            ),
        )
        refs = []
        for owner, identity, checkpoint_id, schema, payload in payloads:
            path = artifact_dir / f"{owner}-tick-{causal_tick}.json"
            path.write_bytes(payload)
            refs.append(_reference(owner, identity, checkpoint_id, path, schema))
        return tuple(refs)

    committed_record = runtime.tick("transition-1", persist_owner_checkpoints)
    cognitive = pending["cognitive"]  # type: ignore[assignment]
    body = pending["body"]  # type: ignore[assignment]
    ledger = pending["ledger"]  # type: ignore[assignment]

    published = runtime.manifest
    assert published.status is ManifestStatus.COMMITTED
    assert published.causal_tick == 1
    assert published.journal_last_sequence == 0
    assert len(published.checkpoint_references) == 3

    organism_ref = next(ref for ref in published.checkpoint_references if ref.owner == "organism")
    embodiment_ref = next(ref for ref in published.checkpoint_references if ref.owner == "embodiment")
    restored_cognitive = CognitiveRuntimeCheckpoint.from_organism_checkpoint(
        type(cognitive.to_organism_checkpoint()).from_bytes(Path(organism_ref.locator).read_bytes()),
        kernel_limits=KernelLimits(),
    )
    restored_embodiment = EmbodimentCheckpoint.from_bytes(Path(embodiment_ref.locator).read_bytes())
    restored_body = restored_embodiment.body_state
    restored_ledger = restored_embodiment.metabolic_ledger
    journal_ref = next(ref for ref in published.checkpoint_references if ref.owner == "world-journal")
    restored_journal = journal_store.read(journal_ref)

    assert restored_cognitive.state.identity == IDENTITY
    assert restored_cognitive.state == cognitive.state
    assert restored_body.checkpoint() == body.checkpoint()
    assert restored_ledger.checkpoint() == ledger.checkpoint()
    assert restored_embodiment.body_id == BODY_ID
    assert restored_embodiment.episode_id == EPISODE_ID
    assert restored_ledger.body_state.energy_reserve == restored_body.energy_reserve
    assert restored_journal.prefix_digest() == published.journal_prefix_sha256
    assert restored_journal.replay()[0] == committed_record


def test_manifest_stays_recovery_required_when_dependency_is_unverified(tmp_path: Path) -> None:
    store = WorldCausalManifestStore(tmp_path / "manifest.json")
    checkpoint = tmp_path / "owner.json"
    checkpoint.write_bytes(b"initial owner state")
    initial_reference = _reference(
        "organism", str(IDENTITY), "organism-tick-0", checkpoint, "organism-cognitive-v1"
    )
    world_id = WorldId(WORLD_ID)
    journal_store = DurableJournalSnapshotStore(tmp_path / "journal", world_id)
    allow_verification = True

    def verify_refs(references: tuple[CheckpointReference, ...]) -> bool:
        return allow_verification and all(Path(reference.locator).is_file() for reference in references)

    runtime = DurableWorldRuntime.initialize(
        WorldGenesisSpec(
            world_id=world_id,
            initial_presence_ids=(str(IDENTITY),),
            registered_owner_ids=(("organism", str(IDENTITY)),),
            initial_owner_references=(initial_reference,),
            owner_checkpoint_contracts={
                ("organism", str(IDENTITY)): _FixtureOwnerContract()
            },
            presence_owner_links=(PresenceOwnerLink(PresenceId(str(IDENTITY)), "organism", str(IDENTITY)),),
        ),
        str(IDENTITY),
        store,
        journal_store,
        owner_checkpoint_verifier=verify_refs,
    )
    allow_verification = False

    with pytest.raises(ManifestError, match="dependency closure"):
        runtime.tick("transition-1", lambda _tick: (initial_reference,))

    assert runtime.manifest.status is ManifestStatus.RECOVERY_REQUIRED
    assert store.read().status is ManifestStatus.RECOVERY_REQUIRED


def test_durable_world_runtime_publishes_and_restores_owner_checkpoint_closure(
    tmp_path: Path,
) -> None:
    """Integrate owner APIs through World publication, not a hand-built manifest."""
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    world_id = WorldId("durable-checkpoint-integration-world")
    manifest_store = WorldCausalManifestStore(tmp_path / "world-manifest.json")
    journal_store = DurableJournalSnapshotStore(tmp_path / "world-journal", world_id)
    body = LivingBodyState(
        energy_reserve=1.5,
        max_energy=2.0,
        metabolic_capacity={"observation": 1.0, "cognition": 1.0, "persistence": 1.0, "maintenance": 1.0},
        metabolic_replenishment={"observation": 0.0, "cognition": 0.0, "persistence": 0.0, "maintenance": 0.0},
        metabolic_reserve={"observation": 0.5, "cognition": 0.5, "persistence": 0.5, "maintenance": 0.5},
    )
    ledger = MetabolicLedger(body_state=body)
    cognitive = CognitiveRuntimeCheckpoint(CognitiveTransitionState.initial(IDENTITY), _graph())

    def write_bytes(name: str, payload: bytes, tick: int) -> CheckpointReference:
        path = artifact_dir / f"{name}-{tick}.json"
        path.write_bytes(payload)
        return _reference(
            name,
            str(IDENTITY) if name == "organism" else BODY_ID if name == "embodiment" else "fixture-environment",
            f"{name}-tick-{tick}",
            path,
            {
                "organism": "organism-cognitive-v1",
                "embodiment": "embodiment-checkpoint-v1",
            }[name],
        )

    initial_refs = (
        write_bytes("organism", cognitive.to_organism_checkpoint().to_bytes(), 0),
        write_bytes(
            "embodiment", EmbodimentCheckpoint(BODY_ID, EPISODE_ID, body, ledger).to_bytes(), 0
        ),
    )

    def verify_owner_refs(references: tuple[CheckpointReference, ...]) -> bool:
        return all(
            Path(reference.locator).is_file()
            and hashlib.sha256(Path(reference.locator).read_bytes()).hexdigest() == reference.sha256
            for reference in references
        )

    genesis = WorldGenesisSpec(
        world_id=world_id,
        initial_presence_ids=(str(IDENTITY),),
        registered_owner_ids=_owner_roster(initial_refs),
        initial_owner_references=initial_refs,
        owner_checkpoint_contracts=_owner_contracts(initial_refs),
        presence_owner_links=(PresenceOwnerLink(PresenceId(str(IDENTITY)), "organism", str(IDENTITY)),),
    )
    runtime = DurableWorldRuntime.initialize(
        genesis,
        str(IDENTITY),
        manifest_store,
        journal_store,
        owner_checkpoint_verifier=verify_owner_refs,
    )
    pending: dict[str, object] = {}

    def persist_tick(causal_tick: int) -> tuple[CheckpointReference, ...]:
        candidate_body = LivingBodyState.from_checkpoint(body.checkpoint())
        candidate_ledger = MetabolicLedger.from_checkpoint(
            ledger.checkpoint(), body_state=candidate_body
        )
        signal = 0.5  # Explicit fixture input; Environment semantics are outside this recovery test.
        next_cognition, _ = advance_cognition(
            cognitive.state,
            cognitive.graph,
            {"signal": signal},
            causal_tick=causal_tick,
        )
        candidate_body.consume_energy(0.01)
        candidate_body.advance_age()
        candidate_ledger.charge("cognition", 0.125)
        next_cognitive = CognitiveRuntimeCheckpoint(next_cognition, cognitive.graph)
        pending.update(
            body=candidate_body,
            ledger=candidate_ledger,
            cognitive=next_cognitive,
        )
        return (
            write_bytes("organism", next_cognitive.to_organism_checkpoint().to_bytes(), causal_tick),
            write_bytes(
                "embodiment",
                EmbodimentCheckpoint(BODY_ID, EPISODE_ID, candidate_body, candidate_ledger).to_bytes(),
                causal_tick,
            ),
        )

    record = runtime.tick("checkpoint-transition-1", persist_tick)
    body = pending["body"]  # type: ignore[assignment]
    ledger = pending["ledger"]  # type: ignore[assignment]
    cognitive = pending["cognitive"]  # type: ignore[assignment]

    restored: dict[str, object] = {}

    def restore_owner_refs(references: tuple[CheckpointReference, ...]) -> None:
        for reference in references:
            payload = Path(reference.locator).read_bytes()
            if reference.owner == "organism":
                envelope_type = type(cognitive.to_organism_checkpoint())
                envelope = envelope_type.from_bytes(payload)
                restored["cognitive"] = CognitiveRuntimeCheckpoint.from_organism_checkpoint(
                    envelope, kernel_limits=KernelLimits()
                )
            elif reference.owner == "embodiment":
                restored["embodiment"] = EmbodimentCheckpoint.from_bytes(payload)
            else:
                raise AssertionError(f"unexpected owner reference {reference.owner}")

    restored_world = DurableWorldRuntime.restore(
        world_id,
        str(IDENTITY),
        manifest_store,
        journal_store,
        owner_checkpoint_verifier=verify_owner_refs,
        owner_checkpoint_contracts=_owner_contracts(initial_refs),
        owner_checkpoint_restorer=restore_owner_refs,
    )

    restored_embodiment = restored["embodiment"]
    assert isinstance(restored_embodiment, EmbodimentCheckpoint)
    restored_cognitive = restored["cognitive"]
    assert isinstance(restored_cognitive, CognitiveRuntimeCheckpoint)
    assert restored_world.causal_tick == 1
    assert restored_world.records == (record,)
    assert restored_world.manifest.journal_last_sequence == 0
    assert restored_cognitive.state == cognitive.state
    assert restored_embodiment.body_state.checkpoint() == body.checkpoint()
    assert restored_embodiment.metabolic_ledger.checkpoint() == ledger.checkpoint()

    # Continue exactly one more causal tick from the restored owner state.
    body = restored_embodiment.body_state
    ledger = restored_embodiment.metabolic_ledger
    cognitive = restored_cognitive
    second_record = restored_world.tick("checkpoint-transition-2", persist_tick)
    body = pending["body"]  # type: ignore[assignment]
    ledger = pending["ledger"]  # type: ignore[assignment]
    cognitive = pending["cognitive"]  # type: ignore[assignment]

    assert second_record.causal_tick == 2
    assert restored_world.causal_tick == 2
    assert restored_world.records == (record, second_record)
    assert restored_world.manifest.status is ManifestStatus.COMMITTED
    assert restored_world.manifest.journal_last_sequence == 1
    assert restored_world.manifest.presence_ids == (str(IDENTITY),)
    assert {reference.checkpoint_id for reference in restored_world.manifest.checkpoint_references} == {
        "organism-tick-2",
        "embodiment-tick-2",
        "journal-prefix-2",
    }

    second_restore: dict[str, object] = {}

    def restore_second_cut(references: tuple[CheckpointReference, ...]) -> None:
        for reference in references:
            payload = Path(reference.locator).read_bytes()
            if reference.owner == "organism":
                envelope_type = type(cognitive.to_organism_checkpoint())
                second_restore["cognitive"] = CognitiveRuntimeCheckpoint.from_organism_checkpoint(
                    envelope_type.from_bytes(payload), kernel_limits=KernelLimits()
                )
            elif reference.owner == "embodiment":
                second_restore["embodiment"] = EmbodimentCheckpoint.from_bytes(payload)
            else:
                raise AssertionError(f"unexpected owner reference {reference.owner}")

    second_world = DurableWorldRuntime.restore(
        world_id,
        str(IDENTITY),
        manifest_store,
        journal_store,
        owner_checkpoint_verifier=verify_owner_refs,
        owner_checkpoint_contracts=_owner_contracts(
            tuple(
                reference
                for reference in restored_world.manifest.checkpoint_references
                if reference.owner != "world-journal"
            )
        ),
        owner_checkpoint_restorer=restore_second_cut,
    )
    second_embodiment = second_restore["embodiment"]
    second_cognitive = second_restore["cognitive"]
    assert isinstance(second_embodiment, EmbodimentCheckpoint)
    assert isinstance(second_cognitive, CognitiveRuntimeCheckpoint)
    assert second_world.causal_tick == 2
    assert second_world.records == (record, second_record)
    assert second_world.manifest.journal_last_sequence == 1
    assert second_world.manifest.presence_ids == (str(IDENTITY),)
    assert second_cognitive.state.identity == IDENTITY
    assert second_cognitive.state == cognitive.state
    assert second_embodiment.body_state.checkpoint() == body.checkpoint()
    assert second_embodiment.metabolic_ledger.checkpoint() == ledger.checkpoint()
