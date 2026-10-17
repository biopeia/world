"""Integration test for longitudinal learning and continuity in the real World runtime.

Verifies the success criterion:
An experience received via an authorized causal path in the real runtime
(MuJoCo physics + ScalarField + World commit) produces an internal cognitive
modification and reproducible functional difference that survives cold longitudinal
restoration.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest

from embodiment.checkpoint import EmbodimentCheckpoint
from embodiment.checkpoint import EmbodimentAdoptionBindingV1

pytest.importorskip("mujoco")

from modality.kinematics import KinematicSomaticTransductionV1
from organism import (
    EmbodimentContextRef,
    OrganismCheckpoint,
    OrganismIdentity,
    OrganismLifecycle,
    OrganismTransitionState,
)
from organism.actuation import (
    ActuatorEvidenceContext,
    ActuatorSurface,
    OrganismExplorationComposer,
)
from organism.cognition import (
    CognitiveRuntimeCheckpoint,
    CognitiveTransitionState,
    EdgeLearningInstruction,
    SenseChannelBinding,
)
from organism.cognition.graph import CognitiveGraph, PlasticEdge, PlasticNode
from organism.cognition.limits import KernelLimits
from organism.cognition.types import EdgeKind, NodeKind
from organism.memory import ExperienceRecordingPolicy

from embodiment.physics3d.articulated_state import (
    ARTICULATED_PHYSICAL_CHECKPOINT_SCHEMA,
)
from world.adapters.articulated_mujoco import (
    ArticulatedMujocoConfiguration,
)
from world.causal_cycle import (
    ScalarSensorimotorCycleInputs,
    restore_scalar_sensorimotor_runtime,
    run_scalar_sensorimotor_session,
)
from world.durable_runtime import (
    DurableJournalSnapshotStore,
    DurableWorldRuntime,
    PresenceOwnerLink,
    WorldGenesisSpec,
)
from world.manifest import ManifestError, WorldCausalManifestStore
from world.owner_checkpoint_store import FileOwnerCheckpointStore
from world.runtime import PresenceId, WorldId

_spec = importlib.util.spec_from_file_location(
    "m3_acceptance_harness",
    Path(__file__).with_name("test_end_to_end_causal_cycle_acceptance.py"),
)
acc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(acc)


def _build_runtime_environment(
    runtime_dir: Path,
    with_plasticity: bool,
    kernel_limits: KernelLimits,
    *,
    initialize: bool = True,
):
    morphology = acc.ActiveArticulatedMorphology.from_spec(
        body_id=acc.BODY_ID,
        embodiment_episode_id=acc.EPISODE_ID,
        morphology_revision=acc.MORPHOLOGY_REVISION,
        spec=acc._spec(),
    )

    runtime_dir.mkdir(parents=True, exist_ok=True)
    manifest_store = WorldCausalManifestStore(runtime_dir / "manifest.json")
    journal_store = DurableJournalSnapshotStore(runtime_dir / "journal", WorldId(acc.WORLD_ID))
    owner_dir = runtime_dir / "owners"
    owner_dir.mkdir(parents=True, exist_ok=True)
    owner_store = FileOwnerCheckpointStore(owner_dir / "committed")

    surface = ActuatorSurface.from_effector_ids(
        ("actuator.hinge.positive", "actuator.hinge.negative"),
        fingerprint_material=acc.SURFACE_FINGERPRINT,
    )
    actuator_context = ActuatorEvidenceContext(
        acc.BODY_ID, acc.EPISODE_ID, acc.MORPHOLOGY_REVISION, surface.contract_fingerprint
    )

    # Graph with SENSE -> CONCEPT plastic edge (initial weight = 0.20)
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

    engine = acc.CompetenceDevelopmentEngine(
        surface.actuator_ids,
        organism_id=str(acc.ORGANISM_ID),
        embodiment_fingerprint=actuator_context.fingerprint,
    )
    composer = acc.OrganismExplorationComposer(
        organism_id=str(acc.ORGANISM_ID),
        engine=engine,
    )

    contracts = {
        ("organism", str(acc.ORGANISM_ID)): acc.OrganismCheckpointContract(kernel_limits=kernel_limits),
        ("embodiment", acc.BODY_ID): acc._Contract(
            acc.BODY_ID, {
                ARTICULATED_PHYSICAL_CHECKPOINT_SCHEMA: lambda data: acc._valid_body_checkpoint(data, morphology),
                EmbodimentCheckpoint.OWNER_SCHEMA: lambda data: EmbodimentCheckpoint.validate_checkpoint(
                    data, acc.BODY_ID, EmbodimentCheckpoint.OWNER_SCHEMA
                ),
                EmbodimentAdoptionBindingV1.OWNER_SCHEMA: lambda data: EmbodimentAdoptionBindingV1.validate_checkpoint(
                    data, acc.BODY_ID, EmbodimentAdoptionBindingV1.OWNER_SCHEMA
                ),
            },
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

    world = None
    if initialize:
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
        somatic_transduction=KinematicSomaticTransductionV1() if with_plasticity else None,
    )
    return world, cycle_inputs, manifest_store, journal_store, surface, actuator_context, contracts


def _resume_learning_in_fresh_process(runtime_dir: Path, output_path: Path) -> None:
    """Restore a committed learning cut and advance multiple cycles in a clean interpreter."""
    limits = KernelLimits(consolidation_interval_ticks=31)
    _, initial_inputs, manifest, journal, surface, actuator_context, contracts = (
        _build_runtime_environment(runtime_dir, True, limits, initialize=False)
    )

    def verify(refs):
        try:
            return all(
                contracts[(ref.owner, ref.owner_identity)].validate_checkpoint(
                    Path(ref.locator).read_bytes(), ref.owner_identity, ref.schema_version
                )
                for ref in refs if ref.owner != "world-journal"
            )
        except (OSError, KeyError):
            return False

    restored = restore_scalar_sensorimotor_runtime(
        WorldId(acc.WORLD_ID), acc.PRESENCE_ID, manifest, journal,
        owner_checkpoint_verifier=verify,
        owner_checkpoint_contracts=contracts,
        active_morphology=initial_inputs.active_morphology,
        expected_organism_id=str(acc.ORGANISM_ID),
        expected_physical_step_id="plastic-run:physical-step:5",
        expected_environment_owner_id=acc.ENVIRONMENT_OWNER_ID,
        expected_environment_id=acc.ENVIRONMENT_ID,
        expected_environment_revision=acc.ENVIRONMENT_REVISION,
        expected_field_id=acc.RESOURCE_ID,
        expected_field_revision=acc.FIELD_REVISION,
        kernel_limits=limits,
        actuator_context=actuator_context,
        actuator_surface=surface,
    )
    pre_resume_frame_ticks = [
        frame["tick"]
        for frame in restored.organism.action_composer.engine.checkpoint()["frame_history"]
    ]
    from dataclasses import replace as dc_replace
    resumed_inputs = dc_replace(
        initial_inputs,
        current_organism=restored.organism.transition_state,
        action_composer=restored.organism.action_composer,
        embodiment_state=restored.articulated_body.root,
        environment_field=restored.environment_field,
        articulated_body=restored.articulated_body,
        environment_owner=restored.environment_owner,
        owner_checkpoint_store=FileOwnerCheckpointStore(runtime_dir / "owners" / "committed"),
    )
    session = run_scalar_sensorimotor_session(
        restored.world, acc.ObservationPlane(), run_id="plastic-run",
        source_revisions={"world": "test-commit"}, cycle_inputs=resumed_inputs,
        max_cycles=3, retain_cycle_results=True,
    )
    assert session.completed_cycle_count == 3
    cycle_snapshots = []
    for cycle in session.cycles:
        cognitive = CognitiveRuntimeCheckpoint.from_organism_checkpoint(
            cycle.sensory_input.state.owner_checkpoint, kernel_limits=limits
        )
        cycle_snapshots.append({
            "tick": cycle.committed_cut.causal_tick,
            "commit_id": cycle.committed_cut.commit_id,
            "physical_step_id": cycle.context.physical_step_id,
            "weight": next(e.weight for e in cognitive.graph.edges if e.source_id == "opaque.scalar.signal"),
            "experience_count": len(cognitive.experience_records),
            "experience_refs": [r.evidence_refs for r in cognitive.experience_records],
            "owner_refs": sorted((r.owner, r.owner_identity, r.schema_version, r.sha256)
                                  for r in cycle.committed_cut.owner_checkpoint_references),
            "frames": [
                frame for frame in restored.organism.action_composer.engine.checkpoint()["frame_history"]
                if frame["tick"] <= cycle.committed_cut.causal_tick
            ],
        })
    output_path.write_text(json.dumps({
        "pid": os.getpid(),
        "cycles": cycle_snapshots,
        "pre_resume_frame_ticks": pre_resume_frame_ticks,
    }, sort_keys=True))


if __name__ == "__main__" and len(sys.argv) == 4 and sys.argv[1] == "--resume-learning":
    _resume_learning_in_fresh_process(Path(sys.argv[2]), Path(sys.argv[3]))
    raise SystemExit(0)


def test_longitudinal_learning_and_continuity_in_real_runtime(tmp_path: Path) -> None:
    kernel_limits = KernelLimits(consolidation_interval_ticks=31)

    # 1. Run Plastic Organism in real World runtime for 5 cycles
    plastic_dir = tmp_path / "plastic"
    world_p, inputs_p, manifest_p, journal_p, surface, act_ctx, contracts_p = _build_runtime_environment(
        plastic_dir, True, kernel_limits
    )
    res_p = run_scalar_sensorimotor_session(
        world_p,
        acc.ObservationPlane(),
        run_id="plastic-run",
        source_revisions={"world": "test-commit"},
        cycle_inputs=inputs_p,
        max_cycles=5,
        retain_cycle_results=True,
    )
    assert res_p.completed_cycle_count == 5

    # 2. Run Control Organism in real World runtime for 5 cycles (plasticity disabled)
    control_dir = tmp_path / "control"
    world_c, inputs_c, manifest_c, journal_c, _, _, _ = _build_runtime_environment(control_dir, False, kernel_limits)
    res_c = run_scalar_sensorimotor_session(
        world_c,
        acc.ObservationPlane(),
        run_id="control-run",
        source_revisions={"world": "test-commit"},
        cycle_inputs=inputs_c,
        max_cycles=5,
        retain_cycle_results=True,
    )
    assert res_c.completed_cycle_count == 5

    # Compare learned weights at cycle 5
    chk_p = CognitiveRuntimeCheckpoint.from_organism_checkpoint(
        res_p.cycles[-1].sensory_input.state.owner_checkpoint, kernel_limits=kernel_limits
    )
    chk_c = CognitiveRuntimeCheckpoint.from_organism_checkpoint(
        res_c.cycles[-1].sensory_input.state.owner_checkpoint, kernel_limits=kernel_limits
    )

    w_plastic = next(e.weight for e in chk_p.graph.edges if e.source_id == "opaque.scalar.signal")
    w_control = next(e.weight for e in chk_c.graph.edges if e.source_id == "opaque.scalar.signal")

    # Control weight stayed unchanged at initial 0.20
    assert w_control == 0.20, f"Control weight must remain exactly 0.20, got {w_control}"
    # Plastic weight potentiated from 0.20 to > 0.20 through authorized causal cycle
    assert w_plastic > 0.20, f"Plastic weight must potentiate > 0.20, got {w_plastic}"

    # Also verify ExperienceLedger received records from authorized causal path
    assert len(chk_p.experience_records) == 5
    assert chk_p.experience_records[0].evidence_refs == (f"sense:{acc.WORLD_ID}:0",)

    # Verify somatic feedback was received by CompetenceDevelopmentEngine from real MuJoCo physics
    frames_p = inputs_p.action_composer.engine.checkpoint()["frame_history"]
    assert len(frames_p) == 5
    assert frames_p[0]["tick"] == 1
    assert "joint.hinge:pos" in frames_p[0]["body_state"]
    assert "joint.hinge:vel" in frames_p[0]["body_state"]
    # Control organism had somatic feedback disabled (control invariancy)
    frames_c = inputs_c.action_composer.engine.checkpoint()["frame_history"]
    assert len(frames_c) == 0

    # 3. Stop, Interrupt, and Cold-Restore from Committed Cut
    last_cut = res_p.last_committed_cut
    assert last_cut is not None
    assert last_cut.causal_tick == 5

    def verify_owner_refs(refs):
        try:
            return all(
                contracts_p[(ref.owner, ref.owner_identity)].validate_checkpoint(
                    Path(ref.locator).read_bytes(), ref.owner_identity, ref.schema_version
                )
                for ref in refs if ref.owner != "world-journal"
            )
        except (KeyError, OSError):
            return False

    # A mismatched physical-step identity must fail closed without changing the cut.
    with pytest.raises(ManifestError):
        restore_scalar_sensorimotor_runtime(
            WorldId(acc.WORLD_ID), acc.PRESENCE_ID, manifest_p, journal_p,
            owner_checkpoint_verifier=verify_owner_refs,
            owner_checkpoint_contracts=contracts_p,
            active_morphology=inputs_p.active_morphology,
            expected_organism_id=str(acc.ORGANISM_ID),
            expected_physical_step_id="plastic-run:physical-step:4",
            expected_environment_owner_id=acc.ENVIRONMENT_OWNER_ID,
            expected_environment_id=acc.ENVIRONMENT_ID,
            expected_environment_revision=acc.ENVIRONMENT_REVISION,
            expected_field_id=acc.RESOURCE_ID,
            expected_field_revision=acc.FIELD_REVISION,
            kernel_limits=kernel_limits,
            actuator_context=act_ctx,
            actuator_surface=surface,
        )
    assert manifest_p.read().causal_tick == 5

    # Compare cut-6 after the cold resume against an uninterrupted execution.
    baseline_dir = tmp_path / "plastic-uninterrupted"
    baseline_world, baseline_inputs, _, _, _, _, _ = _build_runtime_environment(
        baseline_dir, True, kernel_limits
    )
    baseline = run_scalar_sensorimotor_session(
        baseline_world, acc.ObservationPlane(), run_id="plastic-run",
        source_revisions={"world": "test-commit"}, cycle_inputs=baseline_inputs,
        max_cycles=8, retain_cycle_results=True,
    )
    assert baseline.completed_cycle_count == 8
    baseline_cycles = baseline.cycles[5:8]
    assert [cycle.committed_cut.causal_tick for cycle in baseline_cycles] == [6, 7, 8]

    restored = restore_scalar_sensorimotor_runtime(
        WorldId(acc.WORLD_ID),
        acc.PRESENCE_ID,
        manifest_p,
        journal_p,
        owner_checkpoint_verifier=verify_owner_refs,
        owner_checkpoint_contracts=contracts_p,
        active_morphology=inputs_p.active_morphology,
        expected_organism_id=str(acc.ORGANISM_ID),
        expected_physical_step_id="plastic-run:physical-step:5",
        expected_environment_owner_id=acc.ENVIRONMENT_OWNER_ID,
        expected_environment_id=acc.ENVIRONMENT_ID,
        expected_environment_revision=acc.ENVIRONMENT_REVISION,
        expected_field_id=acc.RESOURCE_ID,
        expected_field_revision=acc.FIELD_REVISION,
        kernel_limits=kernel_limits,
        actuator_context=act_ctx,
        actuator_surface=surface,
    )

    assert restored.world.causal_tick == 5
    restored_chk = CognitiveRuntimeCheckpoint.from_organism_checkpoint(
        restored.organism.transition_state.owner_checkpoint, kernel_limits=kernel_limits
    )
    restored_w = next(e.weight for e in restored_chk.graph.edges if e.source_id == "opaque.scalar.signal")

    # Invariant: Learned weight and ledger survive cold restoration bit-identically
    assert restored_w == w_plastic
    assert len(restored_chk.experience_records) == 5

    # Invariant: Competence development frames survive cold restoration bit-identically.
    # Atomic staging ensures that cycle 5's post-physics somatic feedback is committed
    # in the candidate checkpoint for cut 5, preventing frame loss upon cold restore.
    restored_frames = restored.organism.action_composer.engine.checkpoint()["frame_history"]
    assert len(restored_frames) == 5
    assert [f["tick"] for f in restored_frames] == [1, 2, 3, 4, 5]
    assert restored_frames[0]["body_state"] == frames_p[0]["body_state"]

    # 4. Continuation from Restored State in Real Runtime (Cycle 6)
    resumed_inputs = ScalarSensorimotorCycleInputs(
        environment_revision=acc.ENVIRONMENT_REVISION,
        solver_configuration=inputs_p.solver_configuration,
        current_organism=restored.organism.transition_state,
        action_composer=restored.organism.action_composer,
        receptor=inputs_p.receptor,
        embodiment_state=restored.articulated_body.root,
        environment_field=restored.environment_field,
        sampling_frame=acc._frame(),
        transduction=acc.DimensionlessScalarFieldIdentityV1(),
        sense_binding=inputs_p.sense_binding,
        kernel_limits=kernel_limits,
        actuator_context=act_ctx,
        actuator_surface=surface,
        articulated_body=restored.articulated_body,
        active_morphology=inputs_p.active_morphology,
        differential_binding=inputs_p.differential_binding,
        torque_binding=inputs_p.torque_binding,
        environment_owner=restored.environment_owner,
        owner_checkpoint_store=FileOwnerCheckpointStore(plastic_dir / "owners" / "committed"),
        embodiment_geometry_revision="geo.r1",
        learning=inputs_p.learning,
        experience_policy=inputs_p.experience_policy,
        enable_somatic_feedback=True,
        somatic_transduction=KinematicSomaticTransductionV1(),
    )

    res_resumed = run_scalar_sensorimotor_session(
        restored.world,
        acc.ObservationPlane(),
        run_id="plastic-run-resumed",
        source_revisions={"world": "test-commit"},
        cycle_inputs=resumed_inputs,
        max_cycles=1,
        retain_cycle_results=True,
    )
    assert res_resumed.completed_cycle_count == 1
    assert res_resumed.last_committed_cut.causal_tick == 6

    # Verify frame 6 was accumulated continuously upon post-commit adoption
    resumed_frames = restored.organism.action_composer.engine.checkpoint()["frame_history"]
    assert len(resumed_frames) == 6
    assert [f["tick"] for f in resumed_frames] == [1, 2, 3, 4, 5, 6]
    assert resumed_frames[-1]["tick"] == 6

    # 5. Demonstrable functional difference surviving cold restoration
    # Under a probe input x = 1.0 with activation threshold theta = 0.20:
    # - Control remains subthreshold (tanh(0.2000 * 1.0) = 0.1974 < 0.2000)
    # - Cold-restored plastic organism crosses threshold (tanh(0.2091 * 1.0) = 0.2062 >= 0.2000)
    import math
    probe_x = 1.0
    activation_threshold = 0.2000
    activation_control = math.tanh(w_control * probe_x)
    activation_restored = math.tanh(restored_w * probe_x)

    assert activation_control < activation_threshold, (
        f"Control must remain subthreshold: {activation_control:.4f} < {activation_threshold}"
    )
    assert activation_restored >= activation_threshold, (
        f"Restored plastic organism must cross activation threshold: {activation_restored:.4f} >= {activation_threshold}"
    )

    # Run cold-process differential against the isolated cut-5 directory.
    cold_output = tmp_path / "cold-resume.json"
    cold_dir = tmp_path / "plastic-cold-start"
    cold_world, cold_inputs, _, _, _, _, _ = _build_runtime_environment(
        cold_dir, True, kernel_limits
    )
    cold_seed = run_scalar_sensorimotor_session(
        cold_world, acc.ObservationPlane(), run_id="plastic-run",
        source_revisions={"world": "test-commit"}, cycle_inputs=cold_inputs,
        max_cycles=5, retain_cycle_results=True,
    )
    assert cold_seed.completed_cycle_count == 5
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--resume-learning", str(cold_dir), str(cold_output)],
        check=False, capture_output=True, text=True, timeout=90,
    )
    assert completed.returncode == 0, (
        f"fresh-process Organism restore failed ({completed.returncode})\n"
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
    resumed = json.loads(cold_output.read_text())
    assert resumed["pid"] != os.getpid()
    assert resumed["pre_resume_frame_ticks"] == [1, 2, 3, 4, 5]
    expected_cycles = []
    for baseline_cycle in baseline_cycles:
        baseline_cognitive = CognitiveRuntimeCheckpoint.from_organism_checkpoint(
            baseline_cycle.sensory_input.state.owner_checkpoint, kernel_limits=kernel_limits
        )
        expected_cycles.append({
            "tick": baseline_cycle.committed_cut.causal_tick,
            "commit_id": baseline_cycle.committed_cut.commit_id,
            "physical_step_id": baseline_cycle.context.physical_step_id,
            "weight": next(e.weight for e in baseline_cognitive.graph.edges if e.source_id == "opaque.scalar.signal"),
            "experience_count": len(baseline_cognitive.experience_records),
            "experience_refs": [list(r.evidence_refs) for r in baseline_cognitive.experience_records],
            "owner_refs": sorted(
                [r.owner, r.owner_identity, r.schema_version, r.sha256]
                for r in baseline_cycle.committed_cut.owner_checkpoint_references
            ),
            # Each retained result reflects engine state after that committed cycle.
            "frames": [f for f in baseline_inputs.action_composer.engine.checkpoint()["frame_history"]
                       if f["tick"] <= baseline_cycle.committed_cut.causal_tick],
        })
    assert resumed["cycles"] == expected_cycles
    assert [cycle["tick"] for cycle in resumed["cycles"]] == [6, 7, 8]
