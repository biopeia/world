"""Production World join for one owner-projected articulated MuJoCo step.

This verifies integration wiring and durable publication, not calibrated
physical behavior or crash/power-loss durability.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass, replace
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
from embodiment.physics3d.differential_motor_composition import (
    DifferentialMotorPair,
    EmbodimentDifferentialMotorBinding,
)
from embodiment.physics3d.joint_torque import (
    EmbodimentJointTorqueBinding,
    JointTorqueChannelBinding,
)
from embodiment.physics3d.mechanics import (
    JointSpec,
    JointTopology,
    SegmentSpec,
    SurfaceMaterial,
)
from embodiment.physics3d.physical_transition import EmbodimentPhysicalState
from environment import FrameDescriptor, StaticEnvironmentOwner, StaticPhysicalScene
from organism.actuation import MotorCommand

import world.adapters.articulated_mujoco as articulated_adapter
from world.adapters.articulated_mujoco import (
    ArticulatedMujocoConfiguration,
    advance_articulated_mujoco_step_from_motor_command,
)
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

    def verify_restoration(self, reference: CheckpointReference) -> bool:
        encoded = Path(reference.locator).read_bytes()
        return self.validate_checkpoint(encoded, reference.owner_identity, reference.schema_version)


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


@pytest.mark.parametrize(
    ("initial_position", "initial_velocity", "positive", "negative", "expected_torque"),
    [
        (0.9876290543817445, 1.9999654107650564, 0.75, 0.25, 0.5),
        (-0.9876290543817445, -1.9999654107650564, 0.25, 0.75, -0.5),
    ],
)
def test_production_join_adopts_commits_and_rejects_solver_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    initial_position: float, initial_velocity: float,
    positive: float, negative: float, expected_torque: float,
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
        # Exercise a near-limit source state with velocity directed outward;
        # the lower-bound case mirrors the recorded upper-bound edge state.
        joints=(JointKinematicState("joint.hinge", initial_position, initial_velocity),),
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
    command = MotorCommand(
        command_id="command.articulated-production-1",
        commitment_id="commitment.articulated-production-1",
        controller_id="controller.fixture",
        competence_id=None,
        surface_fingerprint=SURFACE_FINGERPRINT,
        channels=(("actuator.hinge.positive", positive), ("actuator.hinge.negative", negative)),
        issued_at_tick=0,
        embodiment_id=EPISODE_ID,
    )
    motor_binding = EmbodimentDifferentialMotorBinding(
        binding_id="binding.motor-differential-r1",
        body_id=BODY_ID,
        embodiment_episode_id=EPISODE_ID,
        morphology_revision=MORPHOLOGY_REVISION,
        source_surface_fingerprint=SURFACE_FINGERPRINT,
        pairs=(DifferentialMotorPair(
            joint_id="joint.hinge",
            positive_actuator_id="actuator.hinge.positive",
            negative_actuator_id="actuator.hinge.negative",
            scale=0.5,
        ),),
    )
    configuration = ArticulatedMujocoConfiguration(
        revision=SOLVER_REVISION,
        timestep_s=0.01,
        iterations=50,
        tolerance=1e-10,
        solref=(0.02, 1.0),
        solimp=(0.9, 0.95, 0.001, 0.5, 2.0),
        contact_policy="max-sliding-body-torsion-rolling/no-self-collision-v1",
        velocity_regulation_policy="m1-fixed-single-hinge-grid-v2",
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
        assert adopted_body.causal_tick == context.target_causal_tick
        assert context.target_causal_tick == runtime.causal_tick + 1
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

    outcome = advance_articulated_mujoco_step_from_motor_command(
        runtime,
        commit_id="commit.articulated-production-1",
        physical_step_id="physical-step.articulated-production-1",
        body_state=body,
        motor_command=command,
        active_morphology=morphology,
        differential_binding=motor_binding,
        torque_binding=binding,
        environment_owner=environment,
        configuration=configuration,
        write_checkpoints=write_checkpoints,
    )

    assert isinstance(outcome.record, CausalRecord)
    assert runtime.causal_tick == outcome.record.causal_tick == 1
    assert outcome.body_state.causal_tick == 1
    assert -1.0 <= outcome.body_state.joints[0].position_rad <= 1.0
    assert abs(outcome.body_state.joints[0].velocity_rad_per_s) <= 2.0
    assert body.causal_tick == 0
    assert body.joints[0] == JointKinematicState("joint.hinge", initial_position, initial_velocity)
    assert outcome.solver_evidence is not None
    assert outcome.solver_evidence["policy"] == "m1-fixed-single-hinge-grid-v2"
    assert outcome.solver_evidence["world_id"] == WORLD_ID
    assert outcome.solver_evidence["source_causal_tick"] == 0
    assert outcome.solver_evidence["target_causal_tick"] == 1
    assert outcome.solver_evidence["physical_step_id"] == "physical-step.articulated-production-1"
    assert outcome.solver_evidence["body_id"] == BODY_ID
    assert outcome.solver_evidence["morphology_revision"] == MORPHOLOGY_REVISION
    assert outcome.solver_evidence["source_position_rad"] == pytest.approx(initial_position)
    assert outcome.solver_evidence["source_velocity_rad_per_s"] == pytest.approx(initial_velocity)
    assert outcome.solver_evidence["requested_torque_nm"] == pytest.approx(expected_torque)
    assert outcome.solver_evidence["applied_torque_nm"] == pytest.approx(
        outcome.solver_evidence["requested_torque_nm"]
    )
    assert abs(outcome.solver_evidence["applied_torque_nm"]) <= outcome.solver_evidence["effort_limit_nm"]
    assert abs(outcome.solver_evidence["candidate_velocity_rad_per_s"]) <= outcome.solver_evidence["velocity_cap_rad_per_s"]
    assert -1.0 <= outcome.solver_evidence["candidate_position_rad"] <= 1.0
    assert outcome.solver_evidence["joint_limit_activation_margin_rad"] == pytest.approx(0.02)
    assert environment.last_adopted_tick == 0

    # Continue the real production join through repeated physical transitions
    # from the formerly failing edge state. Every candidate must satisfy the
    # same owner morphology bounds before its World cut is committed.
    for transition in range(25):
        command_at_cut = replace(
            command,
            command_id=f"command.articulated-production-repeat-{transition}",
            commitment_id=f"commitment.articulated-production-repeat-{transition}",
            issued_at_tick=outcome.body_state.causal_tick,
        )
        try:
            outcome = advance_articulated_mujoco_step_from_motor_command(
                runtime,
                commit_id=f"commit.articulated-production-repeat-{transition}",
                physical_step_id=f"physical-step.articulated-production-repeat-{transition}",
                body_state=outcome.body_state,
                motor_command=command_at_cut,
                active_morphology=morphology,
                differential_binding=motor_binding,
                torque_binding=binding,
                environment_owner=environment,
                configuration=configuration,
                write_checkpoints=write_checkpoints,
            )
        except ValueError as exc:
            pytest.fail(
                f"transition={transition}, source_tick={runtime.causal_tick}, "
                f"qpos={outcome.body_state.joints[0].position_rad}, "
                f"qvel={outcome.body_state.joints[0].velocity_rad_per_s}: {exc}"
            )
        joint = outcome.body_state.joints[0]
        assert -1.0 <= joint.position_rad <= 1.0
        assert abs(joint.velocity_rad_per_s) <= 2.0
    assert runtime.causal_tick == 26
    assert outcome.solver_evidence["source_causal_tick"] == 25
    assert outcome.solver_evidence["target_causal_tick"] == 26
    assert outcome.solver_evidence["physical_step_id"] == "physical-step.articulated-production-repeat-24"

    stale_command = replace(
        command,
        command_id="command.articulated-production-stale",
        commitment_id="commitment.articulated-production-stale",
    )
    with pytest.raises(ValueError, match="does not match the World source causal tick"):
        advance_articulated_mujoco_step_from_motor_command(
            runtime,
            commit_id="commit.articulated-production-stale",
            physical_step_id="physical-step.articulated-production-stale",
            body_state=outcome.body_state,
            motor_command=stale_command,
            active_morphology=morphology,
            differential_binding=motor_binding,
            torque_binding=binding,
            environment_owner=environment,
            configuration=configuration,
            write_checkpoints=write_checkpoints,
        )
    assert runtime.status is ManifestStatus.COMMITTED
    assert runtime.causal_tick == 26

    stale_morphology = ActiveArticulatedMorphology.from_spec(
        body_id=BODY_ID,
        embodiment_episode_id=EPISODE_ID,
        morphology_revision=MORPHOLOGY_REVISION,
        spec=replace(
            spec,
            joint_specs=(replace(spec.joint_specs[0], max_motor_torque=1.0),),
        ),
    )
    with pytest.raises(ValueError, match="active morphology content differs"):
        advance_articulated_mujoco_step_from_motor_command(
            runtime,
            commit_id="commit.articulated-production-stale-morphology",
            physical_step_id="physical-step.articulated-production-stale-morphology",
            body_state=outcome.body_state,
            motor_command=replace(command, issued_at_tick=runtime.causal_tick),
            active_morphology=stale_morphology,
            differential_binding=motor_binding,
            torque_binding=binding,
            environment_owner=environment,
            configuration=configuration,
            write_checkpoints=write_checkpoints,
        )
    assert runtime.status is ManifestStatus.COMMITTED
    assert runtime.causal_tick == 26

    original_trial = articulated_adapter._mujoco_trial

    def out_of_range_trial(*args, **kwargs):
        candidate = original_trial(*args, **kwargs)
        if candidate is not None:
            # Inject a physically returned but Embodiment-invalid candidate;
            # let World selector reject it before the causal commit boundary.
            candidate.qpos[0] = 1.01
        return candidate

    monkeypatch.setattr(articulated_adapter, "_mujoco_trial", out_of_range_trial)
    next_command = replace(
        command,
        command_id="command.articulated-production-failure",
        commitment_id="commitment.articulated-production-failure",
        issued_at_tick=runtime.causal_tick,
    )

    with pytest.raises(ValueError, match="no tested applied torque produces a morphology-compliant candidate"):
        advance_articulated_mujoco_step_from_motor_command(
            runtime,
            commit_id="commit.articulated-production-failure",
            physical_step_id="physical-step.articulated-production-failure",
            body_state=outcome.body_state,
            motor_command=next_command,
            active_morphology=morphology,
            differential_binding=motor_binding,
            torque_binding=binding,
            environment_owner=environment,
            configuration=configuration,
            write_checkpoints=write_checkpoints,
        )

    assert runtime.status is ManifestStatus.RECOVERY_REQUIRED
    assert runtime.causal_tick == runtime.manifest.causal_tick == 26
    assert runtime.manifest.commit_id == "commit.articulated-production-repeat-24"
    assert written_ticks == list(range(1, 27))


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
