"""Production World join for one owner-projected articulated MuJoCo step.

This verifies integration wiring and durable publication, not calibrated
physical behavior or crash/power-loss durability.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import mujoco
import pytest
from embodiment.observation import BodyPoseSample
from embodiment.physics3d.articulated_spec import ArticulatedBodySpec
from embodiment.physics3d.articulated_state import (
    ActiveArticulatedMorphology,
    ArticulatedEmbodimentPhysicalState,
    JointKinematicState,
)
from embodiment.physics3d.joint_torque import (
    EmbodimentJointTorqueBinding,
    JointTorqueChannelBinding,
    JointTorqueInputChannel,
    JointTorqueRequest,
    project_joint_torque_command,
)
from embodiment.physics3d.mechanics import JointSpec, JointTopology, SegmentSpec, SurfaceMaterial
from embodiment.physics3d.physical_transition import EmbodimentPhysicalState
from environment import FrameDescriptor, StaticEnvironmentOwner, StaticPhysicalScene
from world.adapters.articulated_mujoco import (
    ArticulatedMujocoConfiguration,
    advance_articulated_mujoco_step,
)
import world.adapters.articulated_mujoco as articulated_adapter
from world.durable_runtime import (
    DurableJournalSnapshotStore,
    DurableWorldRuntime,
    PresenceOwnerLink,
    WorldGenesisSpec,
)
from world.journal import CausalRecord
from world.manifest import CheckpointReference, ManifestStatus, WorldCausalManifestStore
from world.runtime import PresenceId, WorldId


WORLD_ID = "world.articulated-production-join"
PRESENCE_ID = "presence.articulated-production-join"
BODY_ID = "body.articulated-production-join"
EPISODE_ID = "episode.articulated-production-join"
MORPHOLOGY_REVISION = "morphology.articulated-production-r1"
ENVIRONMENT_ID = "environment.articulated-production-join"
ENVIRONMENT_OWNER_ID = "owner.environment.articulated-production-join"
SURFACE_FINGERPRINT = "sha256:fixture-joint-surface-r1"
SOLVER_REVISION = f"mujoco-{mujoco.__version__}-articulated-production-join-r1"
FRAME = FrameDescriptor(
    frame_id="physics3d.world",
    parent_frame_id="world",
    origin_m=(0.0, 0.0, 0.0),
    x_axis_in_parent=(1.0, 0.0, 0.0),
    y_axis_in_parent=(0.0, 1.0, 0.0),
    z_axis_in_parent=(0.0, 0.0, 1.0),
    handedness="RIGHT_HANDED",
    up_axis="Z",
)


@dataclass(frozen=True)
class _OwnerContract:
    owner_identity: str
    schema_version: str
    validator: Callable[[bytes], bool]

    def validate_checkpoint(self, encoded: bytes, owner_identity: str, schema_version: str) -> bool:
        return (
            owner_identity == self.owner_identity
            and schema_version == self.schema_version
            and self.validator(encoded)
        )

    def verify_restoration(self, encoded: bytes, owner_identity: str, schema_version: str) -> bool:
        return self.validate_checkpoint(encoded, owner_identity, schema_version)


def _body_spec() -> ArticulatedBodySpec:
    return ArticulatedBodySpec(
        body_kind="articulated-production-join-fixture",
        state_schema_version=1,
        spawn_height=0.0,
        base_link_name="body.root",
        joint_specs=(JointSpec("joint.hinge", (0.0, 1.0, 0.0), -1.0, 1.0, 2.0, 0.1, 2.0),),
        joint_topology=(JointTopology("joint.hinge", "body.root", "body.arm", (0.0, 0.0, -0.2)),),
        segments={
            "body.root": SegmentSpec(1.0, (0.2, 0.2, 0.2), (0.0, 0.0, 0.0), (0.5, 0.5, 0.5, 1.0)),
            "body.arm": SegmentSpec(0.5, (0.1, 0.4, 0.1), (0.0, 0.0, -0.2), (0.2, 0.3, 0.4, 1.0)),
        },
        contact_link_names=("body.root", "body.arm"),
        structural_neighbour_names=(("body.root", "body.arm"),),
        body_material=SurfaceMaterial(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        fixed_base=True,
    )


def _reference(directory: Path, owner: str, identity: str, schema: str, tick: int, encoded: bytes) -> CheckpointReference:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{owner}-{tick}.checkpoint"
    path.write_bytes(encoded)
    return CheckpointReference(
        owner=owner,
        owner_identity=identity,
        checkpoint_id=f"tick-{tick}",
        schema_version=schema,
        locator=str(path),
        sha256=hashlib.sha256(encoded).hexdigest(),
    )


def test_production_join_adopts_commits_and_rejects_solver_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = _body_spec()
    morphology = ActiveArticulatedMorphology.from_spec(
        body_id=BODY_ID,
        embodiment_episode_id=EPISODE_ID,
        morphology_revision=MORPHOLOGY_REVISION,
        spec=spec,
    )
    root = EmbodimentPhysicalState(
        world_id=WORLD_ID,
        causal_tick=0,
        body_id=BODY_ID,
        embodiment_episode_id=EPISODE_ID,
        morphology_revision=MORPHOLOGY_REVISION,
        pose=BodyPoseSample(
            body_id=BODY_ID,
            reference_frame=FRAME.frame_id,
            body_frame="body.root",
            position_m=(0.0, 0.0, 0.0),
            orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
            sample_kind="genesis",
            sample_value=0,
        ),
        linear_velocity_m_per_s=(0.0, 0.0, 0.0),
        angular_velocity_rad_per_s=(0.0, 0.0, 0.0),
    )
    body = ArticulatedEmbodimentPhysicalState(
        root=root,
        morphology=morphology,
        joints=(JointKinematicState("joint.hinge", 0.0, 0.0),),
    )
    environment = StaticEnvironmentOwner(
        StaticPhysicalScene(
            environment_id=ENVIRONMENT_ID,
            environment_revision="environment.articulated-production-r1",
            frame=FRAME,
            gravity_m_s2=(0.0, 0.0, 0.0),
            objects=(),
        ),
        owner_id=ENVIRONMENT_OWNER_ID,
        world_id=WORLD_ID,
    )
    binding = EmbodimentJointTorqueBinding(
        binding_id="binding.articulated-production-r1",
        body_id=BODY_ID,
        embodiment_episode_id=EPISODE_ID,
        morphology_revision=MORPHOLOGY_REVISION,
        source_surface_fingerprint=SURFACE_FINGERPRINT,
        channels=(JointTorqueChannelBinding("actuator.hinge", "joint.hinge", (0.0, 1.0, 0.0)),),
    )
    request = JointTorqueRequest(
        command_id="command.articulated-production-1",
        commitment_id="commitment.articulated-production-1",
        controller_id="controller.fixture",
        competence_id=None,
        surface_fingerprint=SURFACE_FINGERPRINT,
        channels=(JointTorqueInputChannel("actuator.hinge", 0.5, "joint_torque", "N*m"),),
        issued_at_tick=0,
        embodiment_id=EPISODE_ID,
    )
    torque = project_joint_torque_command(request, binding, morphology)
    configuration = ArticulatedMujocoConfiguration(
        revision=SOLVER_REVISION,
        timestep_s=0.01,
        iterations=50,
        tolerance=1e-10,
        solref=(0.02, 1.0),
        solimp=(0.9, 0.95, 0.001, 0.5, 2.0),
        contact_policy="max-sliding-body-torsion-rolling/no-self-collision-v1",
    )

    owner_directory = tmp_path / "owner-checkpoints"
    body_schema = "embodiment-articulated-physical-state-v2"
    contracts = {
        ("embodiment", BODY_ID): _OwnerContract(
            BODY_ID,
            body_schema,
            lambda encoded: _valid_articulated_checkpoint(encoded, morphology),
        ),
        ("environment", ENVIRONMENT_OWNER_ID): _OwnerContract(
            ENVIRONMENT_OWNER_ID,
            StaticEnvironmentOwner.CHECKPOINT_SCHEMA,
            lambda encoded: StaticEnvironmentOwner.validate_checkpoint(
                encoded,
                ENVIRONMENT_OWNER_ID,
                StaticEnvironmentOwner.CHECKPOINT_SCHEMA,
            ),
        ),
    }
    initial_refs = (
        _reference(owner_directory, "embodiment", BODY_ID, body_schema, 0, body.checkpoint_bytes()),
        _reference(
            owner_directory,
            "environment",
            ENVIRONMENT_OWNER_ID,
            StaticEnvironmentOwner.CHECKPOINT_SCHEMA,
            0,
            environment.checkpoint_bytes(),
        ),
    )
    world_id = WorldId(WORLD_ID)
    runtime = DurableWorldRuntime.initialize(
        WorldGenesisSpec(
            world_id=world_id,
            initial_presence_ids=(PRESENCE_ID,),
            registered_owner_ids=tuple(sorted(contracts)),
            initial_owner_references=initial_refs,
            owner_checkpoint_contracts=contracts,
            presence_owner_links=(PresenceOwnerLink(PresenceId(PRESENCE_ID), "embodiment", BODY_ID),),
        ),
        PRESENCE_ID,
        WorldCausalManifestStore(tmp_path / "world-manifest.json"),
        DurableJournalSnapshotStore(tmp_path / "world-journal", world_id),
        owner_checkpoint_verifier=lambda refs: all(
            contracts[(ref.owner, ref.owner_identity)].validate_checkpoint(
                Path(ref.locator).read_bytes(), ref.owner_identity, ref.schema_version
            )
            for ref in refs
            if ref.owner != "world-journal"
        ),
    )

    written_ticks: list[int] = []

    def write_checkpoints(context, adopted_body, adopted_environment):
        assert adopted_body.causal_tick == context.target_causal_tick == 1
        written_ticks.append(context.target_causal_tick)
        return (
            _reference(
                owner_directory,
                "embodiment",
                BODY_ID,
                body_schema,
                context.target_causal_tick,
                adopted_body.checkpoint_bytes(),
            ),
            _reference(
                owner_directory,
                "environment",
                ENVIRONMENT_OWNER_ID,
                StaticEnvironmentOwner.CHECKPOINT_SCHEMA,
                context.target_causal_tick,
                adopted_environment.checkpoint_bytes(),
            ),
        )

    outcome = advance_articulated_mujoco_step(
        runtime,
        commit_id="commit.articulated-production-1",
        physical_step_id="physical-step.articulated-production-1",
        body_state=body,
        torque=torque,
        environment_owner=environment,
        configuration=configuration,
        write_checkpoints=write_checkpoints,
    )

    assert isinstance(outcome.record, CausalRecord)
    assert runtime.causal_tick == outcome.record.causal_tick == 1
    assert outcome.body_state.causal_tick == 1
    assert outcome.body_state.joints[0].position_rad != 0.0
    assert environment.last_adopted_tick == 0

    def fail_solver(*args, **kwargs):
        raise RuntimeError("injected articulated solver failure")

    monkeypatch.setattr(articulated_adapter, "execute_articulated_mujoco_step", fail_solver)
    next_request = replace(
        request,
        command_id="command.articulated-production-failure",
        commitment_id="commitment.articulated-production-failure",
        issued_at_tick=1,
    )
    next_torque = project_joint_torque_command(next_request, binding, morphology)

    with pytest.raises(RuntimeError, match="injected articulated solver failure"):
        advance_articulated_mujoco_step(
            runtime,
            commit_id="commit.articulated-production-failure",
            physical_step_id="physical-step.articulated-production-failure",
            body_state=outcome.body_state,
            torque=next_torque,
            environment_owner=environment,
            configuration=configuration,
            write_checkpoints=write_checkpoints,
        )

    assert runtime.status is ManifestStatus.RECOVERY_REQUIRED
    assert runtime.causal_tick == runtime.manifest.causal_tick == 1
    assert runtime.manifest.commit_id == "commit.articulated-production-1"
    assert written_ticks == [1]


def _valid_articulated_checkpoint(
    encoded: bytes, morphology: ActiveArticulatedMorphology
) -> bool:
    try:
        ArticulatedEmbodimentPhysicalState.restore_checkpoint(
            encoded, active_morphology=morphology
        )
    except (TypeError, ValueError):
        return False
    return True
