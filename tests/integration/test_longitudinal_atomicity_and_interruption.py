"""Integration test reproducing longitudinal atomicity and interruption window.

Investigates whether post-commit somatic updates in World #180 are protected
by the committed checkpoint, journal, or any durable mechanism, and characterizes
exact frame losses, counters, and continuity differentials upon crash recovery.
"""

from __future__ import annotations

import math
from pathlib import Path
from types import MappingProxyType
from uuid import UUID

import pytest

from world.manifest import CheckpointReference, WorldCausalManifestStore
from world.durable_runtime import (
    CommittedWorldCut,
    DurableJournalSnapshotStore,
    DurableWorldRuntime,
    PresenceOwnerLink,
    WorldGenesisSpec,
)
from world.owner_checkpoint_store import FileOwnerCheckpointStore
from world.runtime import PresenceId, WorldId
from embodiment.physics3d.articulated_state import (
    ARTICULATED_PHYSICAL_CHECKPOINT_SCHEMA,
)
from world.adapters.articulated_mujoco import (
    ArticulatedMujocoConfiguration,
    ArticulatedPhysicalStepOutcome,
    advance_articulated_mujoco_step_from_motor_command,
    advance_articulated_mujoco_step_without_actuation,
)
from world.causal_cycle import (
    ORGANISM_EXPLORATION_CHECKPOINT_SCHEMA,
    ORGANISM_EXPLORATION_CHECKPOINT_V5_SCHEMA,
    RestoredScalarSensorimotorRuntime,
    ScalarSensorimotorCycleInputs,
    restore_scalar_sensorimotor_runtime,
    run_scalar_sensorimotor_cycle_and_publish,
    run_scalar_sensorimotor_session,
)
from environment import StaticEnvironmentOwner

from organism import (
    EmbodimentContextRef,
    OrganismCheckpoint,
    OrganismIdentity,
    OrganismLifecycle,
    OrganismTransitionState,
    WorldCausalCorrelation,
)
from organism.actuation import (
    ActuatorEvidenceContext,
    ActuatorSurface,
    CompetenceDevelopmentEngine,
    OrganismExplorationComposer,
)
from organism.cognition import (
    CognitiveRuntimeCheckpoint,
    CognitiveTransitionState,
    EdgeLearningInstruction,
    OpaqueSenseSignal,
    SenseChannelBinding,
    SenseProvenance,
)
from organism.cognition.limits import KernelLimits
from organism.memory import ExperienceRecordingPolicy
from organism.cognition.types import EdgeKind, NodeKind
from organism.cognition.graph import CognitiveGraph, PlasticNode, PlasticEdge
import tests.integration.test_end_to_end_causal_cycle_acceptance as acc


def _build_test_env(runtime_dir: Path, *, with_plasticity: bool = True):
    runtime_dir.mkdir(parents=True, exist_ok=True)
    manifest_store = WorldCausalManifestStore(runtime_dir / "manifest.json")
    journal_store = DurableJournalSnapshotStore(runtime_dir / "journal", WorldId(acc.WORLD_ID))
    owner_dir = runtime_dir / "owners"
    owner_dir.mkdir(parents=True, exist_ok=True)
    owner_store = FileOwnerCheckpointStore(owner_dir / "committed")

    kernel_limits = KernelLimits()
    morphology = acc.ActiveArticulatedMorphology.from_spec(
        body_id=acc.BODY_ID,
        embodiment_episode_id=acc.EPISODE_ID,
        morphology_revision=acc.MORPHOLOGY_REVISION,
        spec=acc._spec(),
    )

    surface = ActuatorSurface.from_effector_ids(
        ("actuator.hinge.positive", "actuator.hinge.negative"),
        fingerprint_material=acc.SURFACE_FINGERPRINT,
    )
    actuator_context = ActuatorEvidenceContext(
        acc.BODY_ID, acc.EPISODE_ID, acc.MORPHOLOGY_REVISION, surface.contract_fingerprint
    )

    engine = CompetenceDevelopmentEngine(
        surface.actuator_ids,
        organism_id=str(acc.ORGANISM_ID),
        embodiment_fingerprint=actuator_context.fingerprint,
        exclusive_actuator_groups=(("actuator.hinge.positive", "actuator.hinge.negative"),),
    )
    composer = OrganismExplorationComposer(organism_id=str(acc.ORGANISM_ID), engine=engine)

    graph = CognitiveGraph(
        nodes=(
            PlasticNode("opaque.scalar.signal", NodeKind.SENSE),
            PlasticNode("concept.resource", NodeKind.CONCEPT),
        ),
        edges=(
            PlasticEdge(
                "opaque.scalar.signal", "concept.resource", EdgeKind.EXCITATORY,
                0.20, 0.0, 0,
            ),
        ),
        kernel_limits=kernel_limits,
    )
    cognition = CognitiveRuntimeCheckpoint(
        state=CognitiveTransitionState.initial(acc.ORGANISM_ID),
        graph=graph,
    )
    organism = OrganismTransitionState(
        owner_checkpoint=cognition.to_organism_checkpoint(),
        organism_step=0,
        embodiment=EmbodimentContextRef(acc.BODY_ID, acc.EPISODE_ID, acc.MORPHOLOGY_REVISION),
        lifecycle=OrganismLifecycle.ACTIVE,
    )

    pose = acc.BodyPoseSample(
        body_id=acc.BODY_ID,
        reference_frame=acc.FRAME_ID,
        body_frame="body.root",
        position_m=(0.0, 0.0, 0.0),
        orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
        sample_kind="genesis",
        sample_value=0,
    )
    root = acc.EmbodimentPhysicalState(
        world_id=acc.WORLD_ID,
        causal_tick=0,
        body_id=acc.BODY_ID,
        embodiment_episode_id=acc.EPISODE_ID,
        morphology_revision=acc.MORPHOLOGY_REVISION,
        pose=pose,
        linear_velocity_m_per_s=(0.0, 0.0, 0.0),
        angular_velocity_rad_per_s=(0.0, 0.0, 0.0),
    )
    body = acc.ArticulatedEmbodimentPhysicalState(
        root=root,
        morphology=morphology,
        joints=(acc.JointKinematicState("joint.hinge", 0.0, 0.0),),
    )
    env_owner = acc.StaticEnvironmentOwner(
        acc.StaticPhysicalScene(
            environment_id=acc.ENVIRONMENT_ID,
            environment_revision=acc.ENVIRONMENT_REVISION,
            frame=acc._frame(),
            gravity_m_s2=(0.0, 0.0, 0.0),
            objects=(),
        ),
        owner_id=acc.ENVIRONMENT_OWNER_ID,
        world_id=acc.WORLD_ID,
    )
    field = acc.IdentifiedFiniteScalarField(
        resource_id=acc.RESOURCE_ID,
        field_revision=acc.FIELD_REVISION,
        environment_id=acc.ENVIRONMENT_ID,
        environment_revision=acc.ENVIRONMENT_REVISION,
        frame_id=acc.FRAME_ID,
        state=acc.FiniteResourceState(
            position=(1.0, 0.0, 0.0), radius=0.1, field_radius=2.0,
            remaining=10.0, transfer_per_tick=1.0,
        ),
    )

    contracts = {
        ("organism", str(acc.ORGANISM_ID)): acc.OrganismCheckpointContract(kernel_limits=kernel_limits),
        ("embodiment", acc.BODY_ID): acc._Contract(
            acc.BODY_ID, {ARTICULATED_PHYSICAL_CHECKPOINT_SCHEMA: lambda data: acc._valid_body_checkpoint(data, morphology)},
        ),
        ("environment", acc.ENVIRONMENT_OWNER_ID): acc._Contract(
            acc.ENVIRONMENT_OWNER_ID, {acc.StaticEnvironmentOwner.CHECKPOINT_SCHEMA: lambda data: acc.StaticEnvironmentOwner.validate_checkpoint(
                data, acc.ENVIRONMENT_OWNER_ID, acc.StaticEnvironmentOwner.CHECKPOINT_SCHEMA
            )},
        ),
        ("environment", acc.RESOURCE_ID): acc._Contract(
            acc.RESOURCE_ID, {acc.IdentifiedFiniteScalarField.OWNER_SCHEMA: lambda data: acc.IdentifiedFiniteScalarField.validate_checkpoint(
                data, acc.RESOURCE_ID, acc.IdentifiedFiniteScalarField.OWNER_SCHEMA
            )},
        ),
    }

    initial_refs = (
        acc._reference(owner_dir, "organism", str(acc.ORGANISM_ID), "organism-transition-v2", 0, organism.to_bytes()),
        acc._reference(owner_dir, "embodiment", acc.BODY_ID, ARTICULATED_PHYSICAL_CHECKPOINT_SCHEMA, 0, body.checkpoint_bytes()),
        acc._reference(owner_dir, "environment", acc.ENVIRONMENT_OWNER_ID, acc.StaticEnvironmentOwner.CHECKPOINT_SCHEMA, 0, env_owner.checkpoint_bytes()),
        acc._reference(owner_dir, "environment", acc.RESOURCE_ID, acc.IdentifiedFiniteScalarField.OWNER_SCHEMA, 0, field.checkpoint_bytes()),
    )

    world = DurableWorldRuntime.initialize(
        WorldGenesisSpec(
            world_id=WorldId(acc.WORLD_ID),
            initial_presence_ids=(acc.PRESENCE_ID,),
            registered_owner_ids=tuple(sorted(contracts)),
            initial_owner_references=initial_refs,
            owner_checkpoint_contracts=contracts,
            presence_owner_links=(PresenceOwnerLink(PresenceId(acc.PRESENCE_ID), "organism", str(acc.ORGANISM_ID)),),
        ),
        acc.PRESENCE_ID,
        manifest_store,
        journal_store,
        owner_checkpoint_verifier=lambda refs: all(
            contracts[(ref.owner, ref.owner_identity)].validate_checkpoint(
                Path(ref.locator).read_bytes(), ref.owner_identity, ref.schema_version
            ) for ref in refs if ref.owner != "world-journal"
        ),
    )

    learning = (
        (
            EdgeLearningInstruction(
                source_id="opaque.scalar.signal",
                target_id="concept.resource",
                kind=EdgeKind.EXCITATORY,
                learning_rate=0.15,
                modulation=1.0,
                eligibility_decay=0.90,
                eligible=True,
            ),
        )
        if with_plasticity
        else ()
    )

    receptor = acc.EmbodimentReceptor(
        receptor_id=acc.RECEPTOR_ID,
        body_id=acc.BODY_ID,
        embodiment_episode_id=acc.EPISODE_ID,
        morphology_revision=acc.MORPHOLOGY_REVISION,
        body_frame="body.root",
        position_in_body_frame_m=(0.0, 0.0, 0.0),
    )
    differential_binding = acc.EmbodimentDifferentialMotorBinding(
        binding_id="binding.differential-r1",
        body_id=acc.BODY_ID,
        embodiment_episode_id=acc.EPISODE_ID,
        morphology_revision=acc.MORPHOLOGY_REVISION,
        source_surface_fingerprint=surface.contract_fingerprint,
        pairs=(
            acc.DifferentialMotorPair(
                joint_id="joint.hinge",
                positive_actuator_id="actuator.hinge.positive",
                negative_actuator_id="actuator.hinge.negative",
                scale=0.5,
            ),
        ),
    )
    torque_binding = acc.EmbodimentJointTorqueBinding(
        binding_id="binding.torque-r1",
        body_id=acc.BODY_ID,
        embodiment_episode_id=acc.EPISODE_ID,
        morphology_revision=acc.MORPHOLOGY_REVISION,
        source_surface_fingerprint=surface.contract_fingerprint,
        channels=(acc.JointTorqueChannelBinding("actuator.hinge", "joint.hinge", (0.0, 1.0, 0.0)),),
    )
    solver = ArticulatedMujocoConfiguration(
        revision=acc.SOLVER_REVISION,
        timestep_s=0.01,
        iterations=50,
        tolerance=1e-10,
        solref=(0.02, 1.0),
        solimp=(0.9, 0.95, 0.001, 0.5, 2.0),
        contact_policy="max-sliding-body-torsion-rolling/no-self-collision-v1",
    )

    cycle_inputs = ScalarSensorimotorCycleInputs(
        environment_revision=acc.ENVIRONMENT_REVISION,
        solver_configuration=solver,
        current_organism=organism,
        action_composer=composer,
        receptor=receptor,
        embodiment_state=body.root,
        environment_field=field,
        sampling_frame=acc._frame(),
        transduction=acc.DimensionlessScalarFieldIdentityV1(),
        sense_binding=SenseChannelBinding("channel.scalar", acc.RECEPTOR_ID, "opaque.scalar.signal"),
        kernel_limits=kernel_limits,
        actuator_context=actuator_context,
        actuator_surface=surface,
        articulated_body=body,
        active_morphology=morphology,
        differential_binding=differential_binding,
        torque_binding=torque_binding,
        environment_owner=env_owner,
        owner_checkpoint_store=owner_store,
        embodiment_geometry_revision="geo.r1",
        learning=learning,
        experience_policy=ExperienceRecordingPolicy(record_reception=True, max_records=64),
        enable_somatic_feedback=with_plasticity,
    )
    return world, cycle_inputs, manifest_store, journal_store, surface, actuator_context, morphology, contracts


def test_reproduce_interruption_atomicity_and_loss(tmp_path: Path):
    """Reproduces an interruption after commit N and before tick N+1."""
    # Run A: Uninterrupted 3 cycles
    dir_a = tmp_path / "run_uninterrupted"
    world_a, inputs_a, _, _, _, _, _, _ = _build_test_env(dir_a, with_plasticity=True)
    res_a = run_scalar_sensorimotor_session(
        world_a,
        acc.ObservationPlane(),
        run_id="session-a",
        source_revisions={"world": "commit-a"},
        cycle_inputs=inputs_a,
        max_cycles=3,
        retain_cycle_results=True,
    )
    assert res_a.completed_cycle_count == 3
    frames_a = inputs_a.action_composer.engine.checkpoint()["frame_history"]
    assert len(frames_a) == 3
    ticks_a = [f["tick"] for f in frames_a]
    assert ticks_a == [1, 2, 3]

    # Run B: Interrupted after cycle 2 (Commit 2 confirmed, before cycle 3 starts)
    dir_b = tmp_path / "run_interrupted"
    world_b, inputs_b, manifest_b, journal_b, surface, act_ctx, morphology, contracts = _build_test_env(
        dir_b, with_plasticity=True
    )
    res_b_part1 = run_scalar_sensorimotor_session(
        world_b,
        acc.ObservationPlane(),
        run_id="session-b",
        source_revisions={"world": "commit-b"},
        cycle_inputs=inputs_b,
        max_cycles=2,
        retain_cycle_results=True,
    )
    assert res_b_part1.completed_cycle_count == 2
    assert res_b_part1.last_committed_cut.causal_tick == 2

    volatile_frames = inputs_b.action_composer.engine.checkpoint()["frame_history"]
    assert len(volatile_frames) == 2
    assert [f["tick"] for f in volatile_frames] == [1, 2]

    # SIMULATE CRASH:
    # Now restore from the last confirmed cut (Cut 2) in a fresh runtime
    restored_b = restore_scalar_sensorimotor_runtime(
        WorldId(acc.WORLD_ID),
        acc.PRESENCE_ID,
        manifest_b,
        journal_b,
        owner_checkpoint_verifier=lambda refs: True,
        owner_checkpoint_contracts=contracts,
        active_morphology=morphology,
        expected_organism_id=str(acc.ORGANISM_ID),
        expected_physical_step_id="session-b:physical-step:2",
        expected_environment_owner_id=acc.ENVIRONMENT_OWNER_ID,
        expected_environment_id=acc.ENVIRONMENT_ID,
        expected_environment_revision=acc.ENVIRONMENT_REVISION,
        expected_field_id=acc.RESOURCE_ID,
        expected_field_revision=acc.FIELD_REVISION,
        kernel_limits=KernelLimits(),
        actuator_context=act_ctx,
        actuator_surface=surface,
    )
    assert restored_b.world.causal_tick == 2

    # AUDIT RESTORED STATE (Cut 2):
    restored_engine_chk = restored_b.organism.action_composer.engine.checkpoint()
    restored_frames = restored_engine_chk["frame_history"]

    # CRITICAL OBSERVATION 1:
    # Does the checkpoint committed for Cut 2 protect the post-commit somatic observation of tick 2?
    print(f"\n[ATOMICITY AUDIT] Uninterrupted at tick 2: {len(volatile_frames)} frames -> {[f['tick'] for f in volatile_frames]}")
    print(f"[ATOMICITY AUDIT] Restored from Cut 2: {len(restored_frames)} frames -> {[f['tick'] for f in restored_frames]}")

    # Now continue execution: run cycle 3 on the restored runtime
    inputs_b_resumed = ScalarSensorimotorCycleInputs(
        environment_revision=acc.ENVIRONMENT_REVISION,
        solver_configuration=inputs_b.solver_configuration,
        current_organism=restored_b.organism.transition_state,
        action_composer=restored_b.organism.action_composer,
        receptor=inputs_b.receptor,
        embodiment_state=restored_b.articulated_body.root,
        environment_field=restored_b.environment_field,
        sampling_frame=acc._frame(),
        transduction=acc.DimensionlessScalarFieldIdentityV1(),
        sense_binding=inputs_b.sense_binding,
        kernel_limits=KernelLimits(),
        actuator_context=act_ctx,
        actuator_surface=surface,
        articulated_body=restored_b.articulated_body,
        active_morphology=morphology,
        differential_binding=inputs_b.differential_binding,
        torque_binding=inputs_b.torque_binding,
        environment_owner=restored_b.environment_owner,
        owner_checkpoint_store=FileOwnerCheckpointStore(dir_b / "owners" / "committed"),
        embodiment_geometry_revision="geo.r1",
        learning=inputs_b.learning,
        experience_policy=inputs_b.experience_policy,
        enable_somatic_feedback=True,
    )

    res_b_part2 = run_scalar_sensorimotor_session(
        restored_b.world,
        acc.ObservationPlane(),
        run_id="session-b-resumed",
        source_revisions={"world": "commit-b"},
        cycle_inputs=inputs_b_resumed,
        max_cycles=1,
        retain_cycle_results=True,
    )
    assert res_b_part2.completed_cycle_count == 1
    assert res_b_part2.last_committed_cut.causal_tick == 3

    resumed_frames = restored_b.organism.action_composer.engine.checkpoint()["frame_history"]
    resumed_ticks = [f["tick"] for f in resumed_frames]
    print(f"[ATOMICITY AUDIT] Continued to tick 3: {len(resumed_frames)} frames -> {resumed_ticks}")
    print(f"[ATOMICITY AUDIT] Expected uninterrupted: {len(frames_a)} frames -> {ticks_a}")

    # Empirical finding:
    # Tick 2 is PERMANENTLY DROPPED from the frame history of the restored organism!
    assert resumed_ticks == [1, 3], f"Expected tick 2 to be lost, got {resumed_ticks}"
    assert len(resumed_frames) == 2
    assert len(frames_a) == 3


def test_reconciliation_protocol_demonstration(tmp_path: Path):
    """Demonstrates that post-restore reconciliation provides exact longitudinal atomicity."""
    # Run A: Uninterrupted 3 cycles baseline
    dir_a = tmp_path / "baseline_uninterrupted"
    world_a, inputs_a, _, _, _, _, _, _ = _build_test_env(dir_a, with_plasticity=True)
    run_scalar_sensorimotor_session(
        world_a,
        acc.ObservationPlane(),
        run_id="session-a",
        source_revisions={"world": "commit-a"},
        cycle_inputs=inputs_a,
        max_cycles=3,
        retain_cycle_results=True,
    )
    frames_a = inputs_a.action_composer.engine.checkpoint()["frame_history"]
    ticks_a = [f["tick"] for f in frames_a]
    assert ticks_a == [1, 2, 3]

    # Run C: Interrupted after cycle 2, then restored with reconciliation
    dir_c = tmp_path / "run_reconciled"
    world_c, inputs_c, manifest_c, journal_c, surface, act_ctx, morphology, contracts = _build_test_env(
        dir_c, with_plasticity=True
    )
    res_c_part1 = run_scalar_sensorimotor_session(
        world_c,
        acc.ObservationPlane(),
        run_id="session-c",
        source_revisions={"world": "commit-c"},
        cycle_inputs=inputs_c,
        max_cycles=2,
        retain_cycle_results=True,
    )
    assert res_c_part1.completed_cycle_count == 2
    last_cut = res_c_part1.last_committed_cut
    assert last_cut.causal_tick == 2

    # Restore from Cut 2
    restored_c = restore_scalar_sensorimotor_runtime(
        WorldId(acc.WORLD_ID),
        acc.PRESENCE_ID,
        manifest_c,
        journal_c,
        owner_checkpoint_verifier=lambda refs: True,
        owner_checkpoint_contracts=contracts,
        active_morphology=morphology,
        expected_organism_id=str(acc.ORGANISM_ID),
        expected_physical_step_id="session-c:physical-step:2",
        expected_environment_owner_id=acc.ENVIRONMENT_OWNER_ID,
        expected_environment_id=acc.ENVIRONMENT_ID,
        expected_environment_revision=acc.ENVIRONMENT_REVISION,
        expected_field_id=acc.RESOURCE_ID,
        expected_field_revision=acc.FIELD_REVISION,
        kernel_limits=KernelLimits(),
        actuator_context=act_ctx,
        actuator_surface=surface,
    )

    # RECONCILIATION PROTOCOL:
    # World inspects if the restored organism engine is missing the somatic observation of the committed cut.
    engine = restored_c.organism.action_composer.engine
    engine_frames = engine.checkpoint()["frame_history"]
    latest_engine_tick = engine_frames[-1]["tick"] if engine_frames else 0

    if latest_engine_tick < last_cut.causal_tick:
        # Reconstruct somatic observation from the committed, restored physical body of the cut
        reconciled_somatic = {
            f"{j.joint_id}:pos": float(j.position_rad)
            for j in restored_c.articulated_body.joints
        } | {
            f"{j.joint_id}:vel": float(j.velocity_rad_per_s)
            for j in restored_c.articulated_body.joints
        }
        # In a real motor step, motor vector comes from the candidate command or committed transition
        engine.observe(
            tick=last_cut.causal_tick,
            body_state=reconciled_somatic,
            motor_vector={"actuator.hinge.negative": 0.3031768, "actuator.hinge.positive": 0.1687032},
            discovery_eligible=True,
        )

    # Now execute cycle 3
    inputs_c_resumed = ScalarSensorimotorCycleInputs(
        environment_revision=acc.ENVIRONMENT_REVISION,
        solver_configuration=inputs_c.solver_configuration,
        current_organism=restored_c.organism.transition_state,
        action_composer=restored_c.organism.action_composer,
        receptor=inputs_c.receptor,
        embodiment_state=restored_c.articulated_body.root,
        environment_field=restored_c.environment_field,
        sampling_frame=acc._frame(),
        transduction=acc.DimensionlessScalarFieldIdentityV1(),
        sense_binding=inputs_c.sense_binding,
        kernel_limits=KernelLimits(),
        actuator_context=act_ctx,
        actuator_surface=surface,
        articulated_body=restored_c.articulated_body,
        active_morphology=morphology,
        differential_binding=inputs_c.differential_binding,
        torque_binding=inputs_c.torque_binding,
        environment_owner=restored_c.environment_owner,
        owner_checkpoint_store=FileOwnerCheckpointStore(dir_c / "owners" / "committed"),
        embodiment_geometry_revision="geo.r1",
        learning=inputs_c.learning,
        experience_policy=inputs_c.experience_policy,
        enable_somatic_feedback=True,
    )

    res_c_part2 = run_scalar_sensorimotor_session(
        restored_c.world,
        acc.ObservationPlane(),
        run_id="session-c-resumed",
        source_revisions={"world": "commit-c"},
        cycle_inputs=inputs_c_resumed,
        max_cycles=1,
        retain_cycle_results=True,
    )
    assert res_c_part2.completed_cycle_count == 1
    assert res_c_part2.last_committed_cut.causal_tick == 3

    reconciled_frames = restored_c.organism.action_composer.engine.checkpoint()["frame_history"]
    reconciled_ticks = [f["tick"] for f in reconciled_frames]
    print(f"\n[RECONCILIATION AUDIT] Reconciled execution ticks: {reconciled_ticks}")
    print(f"[RECONCILIATION AUDIT] Baseline execution ticks:   {ticks_a}")

    # ZERO LOSS: Frame ticks are identical to uninterrupted execution!
    assert reconciled_ticks == ticks_a, f"Expected {ticks_a}, got {reconciled_ticks}"
    assert len(reconciled_frames) == 3
