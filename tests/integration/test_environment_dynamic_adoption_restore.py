"""World-owned integration for dynamic Environment adoption and recovery."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from environment import (
    DynamicEnvironmentCandidate,
    DynamicEnvironmentOwner,
    DynamicObjectState,
    FrameDescriptor,
    OwnerRestorationReference,
    PhysicalStepCorrelation,
)
from world.durable_runtime import (
    DurableJournalSnapshotStore,
    DurableWorldRuntime,
    PresenceOwnerLink,
    WorldGenesisSpec,
)
from world.manifest import CheckpointReference, ManifestStatus, WorldCausalManifestStore
from world.physical_step import PhysicalStepContext
from world.runtime import PresenceId, WorldId


WORLD_ID = "world.environment-dynamic-fixture"
PRESENCE_ID = "presence.environment-dynamic-fixture"
ENVIRONMENT_ID = "environment.dynamic-fixture"
OWNER_ID = "environment.dynamic-owner-fixture"
OBJECT_ID = "object.movable-fixture"
FRAME_ID = "environment-frame.fixture"
REVISION = "environment-revision.fixture-r1"


class _EnvironmentCheckpointContract:
    def __init__(self) -> None:
        self.restored_owner: DynamicEnvironmentOwner | None = None
        self.restoration_reference: OwnerRestorationReference | None = None

    def validate_checkpoint(self, encoded: bytes, owner_identity: str, schema_version: str) -> bool:
        return DynamicEnvironmentOwner.validate_checkpoint(encoded, owner_identity, schema_version)

    def verify_restoration(self, encoded: bytes, owner_identity: str, schema_version: str) -> bool:
        if (
            self.restored_owner is None
            or self.restoration_reference is None
            or self.restored_owner.owner_id != owner_identity
            or self.restoration_reference.schema_version != schema_version
        ):
            return False
        try:
            self.restored_owner.verify_restoration(self.restoration_reference)
        except ValueError:
            return False
        return True


def _reference(directory: Path, *, tick: int, encoded: bytes) -> CheckpointReference:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"environment-{tick}.checkpoint"
    path.write_bytes(encoded)
    return CheckpointReference(
        owner="environment",
        owner_identity=OWNER_ID,
        checkpoint_id=f"environment-cut-{tick}",
        schema_version=DynamicEnvironmentOwner.CHECKPOINT_SCHEMA,
        locator=str(path),
        sha256=hashlib.sha256(encoded).hexdigest(),
    )


def _runtime(root: Path, environment: DynamicEnvironmentOwner):
    contract = _EnvironmentCheckpointContract()
    contracts = {("environment", OWNER_ID): contract}
    reference = _reference(root / "owners", tick=0, encoded=environment.checkpoint_bytes())
    world_id = WorldId(WORLD_ID)
    runtime = DurableWorldRuntime.initialize(
        WorldGenesisSpec(
            world_id=world_id,
            initial_presence_ids=(PRESENCE_ID,),
            registered_owner_ids=(("environment", OWNER_ID),),
            initial_owner_references=(reference,),
            owner_checkpoint_contracts=contracts,
            presence_owner_links=(
                PresenceOwnerLink(PresenceId(PRESENCE_ID), "environment", OWNER_ID),
            ),
        ),
        PRESENCE_ID,
        WorldCausalManifestStore(root / "manifest.json"),
        DurableJournalSnapshotStore(root / "journal", world_id),
        owner_checkpoint_verifier=lambda refs: all(
            contract.validate_checkpoint(Path(ref.locator).read_bytes(), ref.owner_identity, ref.schema_version)
            for ref in refs
            if ref.owner != "world-journal"
        ),
    )
    return runtime, contracts, contract


def _initial_owner() -> DynamicEnvironmentOwner:
    return DynamicEnvironmentOwner(
        owner_id=OWNER_ID,
        world_id=WORLD_ID,
        environment_id=ENVIRONMENT_ID,
        environment_revision=REVISION,
        frame=FrameDescriptor(
            FRAME_ID,
            "world-frame.fixture",
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            "RIGHT_HANDED",
            "Z",
        ),
        object_state=DynamicObjectState(OBJECT_ID, FRAME_ID, (0.0, 0.0, 0.0)),
    )


def _step(runtime, environment, root: Path, *, target_tick: int, reject: bool = False):
    before = environment.checkpoint_bytes()
    # World-side expectations are captured before candidate production.  The
    # candidate cannot define the correlation against which it is validated.
    world_context = PhysicalStepContext(
        world_id=runtime.manifest.world_id,
        source_causal_tick=runtime.causal_tick,
        target_causal_tick=target_tick,
        physical_step_id=f"fixture-step-{target_tick}",
        body_id="body.environment-fixture",
        embodiment_episode_id="episode.environment-fixture",
        morphology_revision="morphology.environment-fixture",
        environment_revision=REVISION,
        solver_config_revision="solver-config.fixture-r1",
    )
    expected_correlation = PhysicalStepCorrelation(
        world_id=world_context.world_id,
        causal_tick=world_context.source_causal_tick,
        physical_step_id=world_context.physical_step_id,
        environment_id=ENVIRONMENT_ID,
        environment_revision=world_context.environment_revision,
    )

    def persist(proposed_world_tick: int):
        assert proposed_world_tick == target_tick
        assert proposed_world_tick == expected_correlation.causal_tick + 1
        candidate_correlation = expected_correlation
        if reject:
            candidate_correlation = PhysicalStepCorrelation(
                world_id=expected_correlation.world_id,
                causal_tick=expected_correlation.causal_tick + 1,
                physical_step_id=f"stale-{expected_correlation.physical_step_id}",
                environment_id=expected_correlation.environment_id,
                environment_revision=expected_correlation.environment_revision,
            )
        old_position = environment.object_state.position_m
        candidate = DynamicEnvironmentCandidate(
            candidate_correlation,
            DynamicObjectState(OBJECT_ID, FRAME_ID, (old_position[0] + 0.25, old_position[1], old_position[2])),
        )
        environment.validate_and_adopt(candidate, expected_correlation=expected_correlation)
        return (_reference(root / "owners", tick=proposed_world_tick, encoded=environment.checkpoint_bytes()),)

    if reject:
        with pytest.raises(ValueError, match="correlation"):
            runtime.tick(f"dynamic-cut-{target_tick}", persist)
        assert environment.checkpoint_bytes() == before
        assert runtime.status is ManifestStatus.RECOVERY_REQUIRED
        assert runtime.causal_tick == target_tick - 1
        return

    runtime.tick(f"dynamic-cut-{target_tick}", persist)
    assert runtime.status is ManifestStatus.COMMITTED
    assert runtime.causal_tick == target_tick


def _normalized_cut(runtime):
    # Journal hashes bind local owner-reference locators; compare the stable
    # committed owner identities/content, not storage-root-specific bytes.
    return (
        runtime.manifest.world_id,
        runtime.manifest.causal_tick,
        runtime.manifest.commit_id,
        tuple(
            (ref.owner, ref.owner_identity, ref.checkpoint_id, ref.schema_version, ref.sha256)
            for ref in runtime.manifest.checkpoint_references
            if ref.owner != "world-journal"
        ),
    )


def test_dynamic_environment_transition_is_adopted_committed_restored_and_continued(tmp_path: Path):
    environment = _initial_owner()
    runtime, contracts, contract = _runtime(tmp_path, environment)
    _step(runtime, environment, tmp_path, target_tick=1)
    assert environment.object_state.position_m == (0.25, 0.0, 0.0)

    restored: dict[str, DynamicEnvironmentOwner] = {}

    def restore_owner_checkpoints(references):
        reference = next(
            ref for ref in references
            if ref.owner == "environment" and ref.owner_identity == OWNER_ID
        )
        contract.restoration_reference = OwnerRestorationReference(
            owner_identity=reference.owner_identity,
            schema_version=reference.schema_version,
            checkpoint_sha256=reference.sha256,
            environment_id=ENVIRONMENT_ID,
            environment_revision=REVISION,
            world_id=world_id.value,
        )
        contract.restored_owner = DynamicEnvironmentOwner.from_checkpoint_bytes(
            Path(reference.locator).read_bytes()
        )
        restored["environment"] = contract.restored_owner

    world_id = WorldId(WORLD_ID)
    restored_runtime = DurableWorldRuntime.restore(
        world_id,
        PRESENCE_ID,
        WorldCausalManifestStore(tmp_path / "manifest.json"),
        DurableJournalSnapshotStore(tmp_path / "journal", world_id),
        owner_checkpoint_verifier=lambda refs: all(
            contracts[(ref.owner, ref.owner_identity)].validate_checkpoint(
                Path(ref.locator).read_bytes(), ref.owner_identity, ref.schema_version
            )
            for ref in refs if ref.owner != "world-journal"
        ),
        owner_checkpoint_contracts=contracts,
        owner_checkpoint_restorer=restore_owner_checkpoints,
    )
    assert restored_runtime.status is ManifestStatus.COMMITTED
    assert restored_runtime.causal_tick == 1
    assert restored["environment"].object_state.position_m == (0.25, 0.0, 0.0)

    _step(restored_runtime, restored["environment"], tmp_path, target_tick=2)
    assert restored["environment"].object_state.position_m == (0.5, 0.0, 0.0)
    second_reference = next(
        ref for ref in restored_runtime.manifest.checkpoint_references
        if ref.owner == "environment" and ref.owner_identity == OWNER_ID
    )
    assert DynamicEnvironmentOwner.from_checkpoint_bytes(
        Path(second_reference.locator).read_bytes()
    ).object_state.position_m == (0.5, 0.0, 0.0)


def test_identical_fixture_input_produces_same_owner_state_and_committed_cut(tmp_path: Path):
    outcomes = []
    for name in ("first", "second"):
        root = tmp_path / name
        environment = _initial_owner()
        runtime, _, _ = _runtime(root, environment)
        _step(runtime, environment, root, target_tick=1)
        outcomes.append((environment.checkpoint_bytes(), _normalized_cut(runtime)))
    assert outcomes[0] == outcomes[1]


def test_stale_environment_candidate_rejection_prevents_next_world_commit(tmp_path: Path):
    environment = _initial_owner()
    runtime, _, _ = _runtime(tmp_path, environment)
    previous_cut = runtime.manifest
    _step(runtime, environment, tmp_path, target_tick=1, reject=True)
    assert runtime.manifest.status is ManifestStatus.RECOVERY_REQUIRED
    assert runtime.manifest.causal_tick == previous_cut.causal_tick == 0
