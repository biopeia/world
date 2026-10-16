"""Longitudinal continuity with the compact Organism checkpoint (organism#183, Phase 2).

An interrupted v5 session — stopped at a committed cut, every owner restored
from disk through World's production restore, then resumed with the same run
identity — must produce exactly the same committed cuts, byte for byte, as an
uninterrupted session: same commit ids, same Organism/Embodiment/Environment
checkpoint digests and the same action-history dependency heads. The
Organism's cognitive and competence state evolves across the run, so equality
is not trivial.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("mujoco")

_spec = importlib.util.spec_from_file_location(
    "m3_acceptance_harness", Path(__file__).with_name("test_end_to_end_causal_cycle_acceptance.py"))
acc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(acc)

RUN_ID = "acceptance-run"


def _cut_digest(result) -> dict:
    refs = result.committed_cut.owner_checkpoint_references
    return {
        "tick": result.committed_cut.causal_tick,
        "commit_id": result.committed_cut.commit_id,
        "owners": sorted((r.owner, r.owner_identity, r.schema_version, r.sha256, r.dependency_sha256)
                         for r in refs if r.owner != "world-journal"),
    }


def _restore(directory: Path, physical_step_id: str):
    original = acc.restore_scalar_sensorimotor_runtime

    def restore(*args, **kwargs):
        return original(*args, **{**kwargs, "expected_physical_step_id": physical_step_id})

    acc.restore_scalar_sensorimotor_runtime = restore
    try:
        return acc._restore_runtime_from_directory(directory, organism_v5=True)
    finally:
        acc.restore_scalar_sensorimotor_runtime = original


def _resume(directory: Path, restored, morphology, cycles: int, collect) -> None:
    surface = acc.ActuatorSurface.from_effector_ids(
        ("actuator.hinge.positive", "actuator.hinge.negative"), fingerprint_material=acc.SURFACE_FINGERPRINT)
    context = acc.ActuatorEvidenceContext(acc.BODY_ID, acc.EPISODE_ID, acc.MORPHOLOGY_REVISION,
                                          surface.contract_fingerprint)
    receptor = acc.EmbodimentReceptor(
        receptor_id=acc.RECEPTOR_ID, body_id=acc.BODY_ID, embodiment_episode_id=acc.EPISODE_ID,
        morphology_revision=acc.MORPHOLOGY_REVISION, body_frame="body.root", position_in_body_frame_m=(0.0, 0.0, 0.0))
    motor = acc.EmbodimentDifferentialMotorBinding(
        binding_id="binding.acceptance.differential-r1", body_id=acc.BODY_ID, embodiment_episode_id=acc.EPISODE_ID,
        morphology_revision=acc.MORPHOLOGY_REVISION, source_surface_fingerprint=surface.contract_fingerprint,
        pairs=(acc.DifferentialMotorPair(joint_id="joint.hinge", positive_actuator_id="actuator.hinge.positive",
                                         negative_actuator_id="actuator.hinge.negative", scale=0.5),))
    torque = acc.EmbodimentJointTorqueBinding(
        binding_id="binding.acceptance.torque-r1", body_id=acc.BODY_ID, embodiment_episode_id=acc.EPISODE_ID,
        morphology_revision=acc.MORPHOLOGY_REVISION, source_surface_fingerprint=surface.contract_fingerprint,
        channels=(acc.JointTorqueChannelBinding("actuator.hinge", "joint.hinge", (0.0, 1.0, 0.0)),))
    solver = acc.ArticulatedMujocoConfiguration(
        revision=acc.SOLVER_REVISION, timestep_s=0.01, iterations=50, tolerance=1e-10, solref=(0.02, 1.0),
        solimp=(0.9, 0.95, 0.001, 0.5, 2.0), contact_policy="max-sliding-body-torsion-rolling/no-self-collision-v1")
    acc.run_scalar_sensorimotor_session(
        restored.world, acc.ObservationPlane(), run_id=RUN_ID, source_revisions={"world": "test-commit"},
        cycle_inputs=acc.ScalarSensorimotorCycleInputs(
            environment_revision=acc.ENVIRONMENT_REVISION, solver_configuration=solver,
            current_organism=restored.organism.transition_state, action_composer=restored.organism.action_composer,
            receptor=receptor, embodiment_state=restored.articulated_body.root,
            environment_field=restored.environment_field, sampling_frame=acc._frame(),
            transduction=acc.DimensionlessScalarFieldIdentityV1(),
            sense_binding=acc.SenseChannelBinding(acc.CHANNEL_ID, acc.RECEPTOR_ID, "opaque.scalar.signal"),
            kernel_limits=acc.KernelLimits(), actuator_context=context, actuator_surface=surface,
            articulated_body=restored.articulated_body, active_morphology=morphology, differential_binding=motor,
            torque_binding=torque, environment_owner=restored.environment_owner,
            owner_checkpoint_store=acc.FileOwnerCheckpointStore(directory / "owners" / "committed"),
            embodiment_geometry_revision="geometry.acceptance.sensorimotor-r1"),
        max_cycles=cycles, on_cycle_result=collect)


def test_interrupted_v5_session_continues_byte_identically(tmp_path: Path) -> None:
    total, stop_at = 24, 10
    uninterrupted: list = []
    acc._run(tmp_path / "uninterrupted", use_session=True, session_cycles=total, organism_v5=True,
             on_cycle_result=uninterrupted.append)
    interrupted: list = []
    acc._run(tmp_path / "interrupted", use_session=True, session_cycles=stop_at, organism_v5=True,
             on_cycle_result=interrupted.append)
    restored, morphology = _restore(tmp_path / "interrupted", f"{RUN_ID}:physical-step:{stop_at}")
    assert restored.world.causal_tick == stop_at
    _resume(tmp_path / "interrupted", restored, morphology, total - stop_at, interrupted.append)

    assert len(uninterrupted) == len(interrupted) == total
    assert [_cut_digest(r) for r in interrupted] == [_cut_digest(r) for r in uninterrupted]

    # Non-trivial: Organism state actually evolves, and the history grows.
    organism = [next(r for r in res.committed_cut.owner_checkpoint_references if r.owner == "organism")
                for res in uninterrupted]
    assert len({r.sha256 for r in organism}) == total
    transition = [json.loads(Path(r.locator).read_bytes())["transition_state_base64"] for r in organism]
    assert len(set(transition)) > 1
    heads = [r.dependency_sha256 for r in organism if r.dependency_sha256]
    assert len(set(heads)) > 1
