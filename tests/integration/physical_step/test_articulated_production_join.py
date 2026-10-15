"""Production World join for one owner-projected articulated MuJoCo step.

This verifies integration wiring and durable publication, not calibrated
physical behavior or crash/power-loss durability.
"""

from __future__ import annotations

import hashlib
from threading import Thread
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
from embodiment.physics3d.spatial_render import RenderMorphologyProjection
from environment import (
    FrameDescriptor,
    StaticEnvironmentOwner,
    StaticPhysicalScene,
    StaticSphere,
)
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
from world.manifest import CheckpointReference, ManifestError, ManifestStatus, WorldCausalManifestStore
from world.runtime import PresenceId, TickInProgressError, WorldId
from world.spatial_projection import verify_committed_render_pose_projection

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
        (0.9876290543817445, 1.9999654107650564, 0.25, 0.75, -0.5),
        (-0.9876290543817445, -1.9999654107650564, 0.75, 0.25, 0.5),
    ],
)
@pytest.mark.parametrize("failure_kind", ("position", "velocity"))
def test_production_join_adopts_commits_and_rejects_solver_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    initial_position: float, initial_velocity: float,
    positive: float, negative: float, expected_torque: float, failure_kind: str,
) -> None:
    spec = _body_spec()
    morphology = ActiveArticulatedMorphology.from_spec(
        body_id=BODY_ID,
        embodiment_episode_id=EPISODE_ID,
        morphology_revision=MORPHOLOGY_REVISION,
        spec=spec,
    )
    render_projection = RenderMorphologyProjection.from_active_morphology(
        morphology, geometry_revision="geometry.articulated-production-r1"
    )
    wrong_render_projection = RenderMorphologyProjection.from_active_morphology(
        ActiveArticulatedMorphology.from_spec(
            body_id=BODY_ID,
            embodiment_episode_id=EPISODE_ID,
            morphology_revision="morphology.stale-r0",
            spec=spec,
        ),
        geometry_revision="geometry.articulated-production-r1",
    )
    joint_bounds = morphology.joint_ranges[0]
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
    # In the recorded positive near-limit pose the arm occupies this
    # Environment sphere's position; the negative case is the no-load
    # baseline for the mirrored joint boundary.
    contact_spheres = (
        StaticSphere(
            object_id="environment.sphere.contact-load",
            center_m=(-0.17, 0.0, -0.31),
            radius_m=0.08,
            frame_id=FRAME.frame_id,
            sliding_friction_coefficient=0.0,
        ),
    ) if initial_position > 0 else ()
    environment = StaticEnvironmentOwner(
        StaticPhysicalScene(
            environment_id=ENVIRONMENT_ID,
            environment_revision="environment.articulated-production-r1",
            frame=FRAME,
            gravity_m_s2=(0.0, 0.0, 0.0),
            objects=(),
            spheres=contact_spheres,
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

    arm_contact_count: list[int] = []
    production_trial = articulated_adapter._mujoco_trial

    def record_contact_trial(*args, **kwargs):
        candidate = production_trial(*args, **kwargs)
        if candidate is not None:
            model = args[1]
            arm_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "body.arm")
            environment_geom = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_GEOM, "__environment_sphere_0"
            )
            arm_contact_count.append(
                sum(
                    1
                    for contact in candidate.contact[: candidate.ncon]
                    if environment_geom in (contact.geom1, contact.geom2)
                    and model.geom_bodyid[
                        contact.geom2 if contact.geom1 == environment_geom else contact.geom1
                    ]
                    == arm_body
                )
            )
        return candidate

    monkeypatch.setattr(articulated_adapter, "_mujoco_trial", record_contact_trial)
    # A bad owner projection is rejected before World opens a causal
    # transition. The authoritative manifest, journal, and checkpoint set
    # remain exactly at the previous confirmed cut.
    before_invalid_projection = runtime.manifest
    before_invalid_records = runtime.records
    with pytest.raises(ValueError, match="owner render projection identity"):
        advance_articulated_mujoco_step_from_motor_command(
            runtime,
            commit_id="commit.articulated-production-invalid-render-identity",
            physical_step_id="physical-step.articulated-production-invalid-render-identity",
            body_state=body,
            motor_command=command,
            active_morphology=morphology,
            differential_binding=motor_binding,
            torque_binding=binding,
            environment_owner=environment,
            configuration=configuration,
            render_morphology_projection=wrong_render_projection,
            write_checkpoints=write_checkpoints,
        )
    assert runtime.manifest == before_invalid_projection
    assert runtime.causal_tick == before_invalid_projection.causal_tick == 0
    assert runtime.records == before_invalid_records

    original_cut_lookup = runtime.committed_cut_for
    competing_tick_errors = []

    def assert_cut_lookup_is_atomic(**kwargs):
        assert runtime._publication_guard_thread_id is not None
        if competing_tick_errors:
            return original_cut_lookup(**kwargs)
        finished = []

        def compete_for_next_cut():
            try:
                runtime.tick("commit.concurrent-during-render-enrichment", lambda _tick: ())
            except Exception as exc:
                finished.append(exc)

        thread = Thread(target=compete_for_next_cut)
        thread.start()
        thread.join(timeout=2)
        assert not thread.is_alive(), "competing tick must fail promptly while cut is guarded"
        assert len(finished) == 1
        competing_tick_errors.extend(finished)
        return original_cut_lookup(**kwargs)

    monkeypatch.setattr(runtime, "committed_cut_for", assert_cut_lookup_is_atomic)
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
        render_morphology_projection=render_projection,
        write_checkpoints=write_checkpoints,
    )
    monkeypatch.setattr(runtime, "committed_cut_for", original_cut_lookup)

    assert isinstance(outcome.record, CausalRecord)
    assert runtime.causal_tick == outcome.record.causal_tick == 1
    assert outcome.body_state.causal_tick == 1
    assert outcome.render_pose_projection is not None
    assert len(competing_tick_errors) == 1
    assert isinstance(competing_tick_errors[0], TickInProgressError)
    assert verify_committed_render_pose_projection(runtime, outcome.render_pose_projection)
    assert outcome.render_pose_projection.committed_cut.commit_id == "commit.articulated-production-1"
    assert outcome.render_pose_projection.owner_checkpoint_reference.owner == "embodiment"
    assert outcome.render_pose_projection.owner_checkpoint_reference.owner_identity == BODY_ID
    rendered = {pose.segment_id: pose for pose in outcome.render_pose_projection.candidate.poses}
    assert set(rendered) == {"body.arm", "body.root"}
    assert rendered["body.root"].translation_m == pytest.approx((0.0, 0.0, 0.0))
    q = outcome.body_state.joints[0].position_rad
    assert rendered["body.arm"].translation_m == pytest.approx(
        (0.0, 0.0, -0.2), abs=1e-7
    )
    assert rendered["body.arm"].orientation_xyzw == pytest.approx(
        (0.0, __import__("math").sin(q / 2), 0.0, __import__("math").cos(q / 2)), abs=1e-7
    )
    first_committed_pose_projection = outcome.render_pose_projection
    wrong_commit_receipt = replace(
        first_committed_pose_projection,
        committed_cut=replace(
            first_committed_pose_projection.committed_cut,
            commit_id="commit.same-tick-but-different-identity",
        ),
    )
    with pytest.raises(ManifestError, match="requested identity does not match"):
        verify_committed_render_pose_projection(runtime, wrong_commit_receipt)
    assert joint_bounds.lower_rad <= outcome.body_state.joints[0].position_rad <= joint_bounds.upper_rad
    assert abs(outcome.body_state.joints[0].velocity_rad_per_s) <= joint_bounds.max_velocity_rad_per_s
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
    assert abs(outcome.solver_evidence["candidate_velocity_rad_per_s"]) <= joint_bounds.max_velocity_rad_per_s
    assert joint_bounds.lower_rad <= outcome.solver_evidence["candidate_position_rad"] <= joint_bounds.upper_rad
    assert outcome.solver_evidence["joint_limit_activation_margin_rad"] == pytest.approx(0.02)
    assert environment.last_adopted_tick == 0
    if initial_position > 0:
        assert arm_contact_count and arm_contact_count[0] > 0, (
            "requested motor trial must contact the Environment sphere with body.arm"
        )

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
        assert joint_bounds.lower_rad <= joint.position_rad <= joint_bounds.upper_rad
        assert abs(joint.velocity_rad_per_s) <= joint_bounds.max_velocity_rad_per_s
    assert runtime.causal_tick == 26
    assert outcome.solver_evidence["source_causal_tick"] == 25
    assert outcome.solver_evidence["target_causal_tick"] == 26
    assert outcome.solver_evidence["physical_step_id"] == "physical-step.articulated-production-repeat-24"

    # Enrichment is optional and occurs after commit. Failure to construct the
    # transient view must not make the confirmed physical transition ambiguous.
    original_association = articulated_adapter.associate_render_pose_candidate
    monkeypatch.setattr(
        articulated_adapter,
        "associate_render_pose_candidate",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("injected projection failure")),
    )
    enrichment_failure = advance_articulated_mujoco_step_from_motor_command(
        runtime,
        commit_id="commit.articulated-production-render-enrichment-failure",
        physical_step_id="physical-step.articulated-production-render-enrichment-failure",
        body_state=outcome.body_state,
        motor_command=replace(
            command,
            command_id="command.articulated-production-render-enrichment-failure",
            commitment_id="commitment.articulated-production-render-enrichment-failure",
            issued_at_tick=runtime.causal_tick,
        ),
        active_morphology=morphology,
        differential_binding=motor_binding,
        torque_binding=binding,
        environment_owner=environment,
        configuration=configuration,
        render_morphology_projection=render_projection,
        write_checkpoints=write_checkpoints,
    )
    monkeypatch.setattr(articulated_adapter, "associate_render_pose_candidate", original_association)
    assert runtime.manifest.commit_id == "commit.articulated-production-render-enrichment-failure"
    assert enrichment_failure.render_pose_projection is None
    assert enrichment_failure.render_pose_error == "ValueError: injected projection failure"
    assert runtime.causal_tick == 27
    outcome = enrichment_failure
    # The old tick-1 snapshot must not be presented as a current-cut view once
    # World has committed later transitions. No arrival-order semantics are
    # inferred; stale association simply fails closed.
    with pytest.raises(ManifestError, match="requested identity does not match"):
        verify_committed_render_pose_projection(runtime, first_committed_pose_projection)

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
    assert runtime.causal_tick == 27

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
    assert runtime.causal_tick == 27

    durable_before_rejection = runtime.manifest
    journal_before_rejection = runtime.records
    references_before_rejection = durable_before_rejection.checkpoint_references
    journal_sequence_before_rejection = durable_before_rejection.journal_last_sequence
    journal_prefix_before_rejection = durable_before_rejection.journal_prefix_sha256
    original_trial = articulated_adapter._mujoco_trial

    def out_of_range_trial(*args, **kwargs):
        candidate = original_trial(*args, **kwargs)
        if candidate is not None:
            # Inject a physically returned but Embodiment-invalid candidate;
            # let World selector reject it before the causal commit boundary.
            if failure_kind == "position":
                candidate.qpos[0] = 1.01
            else:
                candidate.qpos[0] = 0.0
                candidate.qvel[0] = 2.01
        return candidate

    monkeypatch.setattr(articulated_adapter, "_mujoco_trial", out_of_range_trial)
    next_command = replace(
        command,
        command_id=f"command.articulated-production-failure-{failure_kind}",
        commitment_id=f"commitment.articulated-production-failure-{failure_kind}",
        issued_at_tick=runtime.causal_tick,
    )

    with pytest.raises(ValueError, match="no tested applied torque produces a morphology-compliant candidate"):
        advance_articulated_mujoco_step_from_motor_command(
            runtime,
            commit_id=f"commit.articulated-production-failure-{failure_kind}",
            physical_step_id=f"physical-step.articulated-production-failure-{failure_kind}",
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
    assert runtime.causal_tick == runtime.manifest.causal_tick == 27
    assert runtime.manifest.commit_id == "commit.articulated-production-render-enrichment-failure"
    assert runtime.manifest.checkpoint_references == references_before_rejection
    assert runtime.manifest.journal_last_sequence == journal_sequence_before_rejection
    assert runtime.manifest.journal_prefix_sha256 == journal_prefix_before_rejection
    assert runtime.records == journal_before_rejection
    assert written_ticks == list(range(1, 28))


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
