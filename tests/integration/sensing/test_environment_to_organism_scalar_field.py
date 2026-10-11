"""World-coordinated Environment-to-Organism scalar sensing fixture.

This exercises a narrow target contract, not scientific learning or a
production sensor/solver adapter. Environment owns the field, Embodiment owns
the receiver, Modality owns the declared numerical mapping, Organism receives
only an opaque signal, and World supplies the causal step and commit boundary.
The canonical Organism-owned transition is invoked through World's runtime
boundary and its accepted state is included in the committed owner checkpoint.
The fixture still does not prove a production sensor adapter, a physical
solver adapter, learned behavior, or end-to-end migration acceptance.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from embodiment.observation import BodyPoseSample
from embodiment.physics3d.physical_transition import (
    EmbodimentPhysicalState,
    PhysicalStepCorrelation as EmbodimentStepCorrelation,
    SolverBodyCandidate,
)
from embodiment.physics3d.receptor import EmbodimentReceptor, project_receptor_sample_point
from environment import (
    FiniteResourceState,
    FrameDescriptor,
    IdentifiedFiniteScalarField,
    PhysicalStepCorrelation as EnvironmentStepCorrelation,
)
from modality import DimensionlessScalarFieldIdentityV1
from organism.cognition.graph import CognitiveGraph, PlasticNode
from organism.cognition.limits import KernelLimits
from organism.cognition.sense_input import (
    OpaqueSenseSignal,
    SenseChannelBinding,
    SenseProvenance,
)
from organism.cognition import CognitiveRuntimeCheckpoint
from organism.cognition.transition import CognitiveTransitionState
from organism.transition import (
    EmbodimentContextRef,
    OrganismLifecycle,
    OrganismTransitionState,
)
from organism.cognition.types import NodeKind
from organism.identity import OrganismIdentity

from world.durable_runtime import (
    DurableJournalSnapshotStore,
    DurableWorldRuntime,
    PresenceOwnerLink,
    WorldGenesisSpec,
)
from world.manifest import (
    CheckpointReference,
    ManifestStatus,
    WorldCausalManifestStore,
)
from world.physical_step import (
    PhysicalStepCandidate,
    PhysicalStepContext,
    PhysicalStepPorts,
    run_physical_step,
)
from world.organism_transition import advance_organism_for_physical_step
from world.runtime import PresenceId, WorldId


WORLD_ID = "scalar-sensing-world"
WORLD_PRESENCE_ID = "scalar-sensing-presence"
ORGANISM_ID = OrganismIdentity.parse("12345678-1234-5678-1234-567812345678")
BODY_ID = "body.scalar-sensing"
EPISODE_ID = "episode.scalar-sensing"
MORPHOLOGY_REVISION = "morphology.scalar-sensing.v1"
ENVIRONMENT_ID = "environment.scalar-sensing"
ENVIRONMENT_REVISION = "environment-revision-1"
RESOURCE_ID = "resource.scalar-field-1"
FIELD_REVISION = "field-revision-1"
FRAME_ID = "environment-local"
RECEPTOR_ID = "receptor.scalar-1"
CHANNEL_ID = "channel.opaque-1"


class _FixtureOwnerContract:
    """In-process owner contract for this integration fixture only."""

    def validate_checkpoint(self, encoded: bytes, owner_identity: str, schema_version: str) -> bool:
        return bool(encoded and owner_identity and schema_version)

    def verify_restoration(self, encoded: bytes, owner_identity: str, schema_version: str) -> bool:
        return self.validate_checkpoint(encoded, owner_identity, schema_version)


def _graph() -> CognitiveGraph:
    return CognitiveGraph(
        nodes=(PlasticNode("opaque-input", NodeKind.SENSE),),
        edges=(),
        kernel_limits=KernelLimits(),
    )


def _body_state(causal_tick: int, *, physical_step: EmbodimentStepCorrelation | None = None):
    pose = BodyPoseSample(
        body_id=BODY_ID,
        reference_frame=FRAME_ID,
        body_frame="body-root",
        position_m=(0.0, 0.0, 0.0),
        orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
        sample_kind="physics_step" if physical_step else "declared_initial",
        sample_value=physical_step.physical_step_id if physical_step else "initial",
    )
    return EmbodimentPhysicalState(
        world_id=WORLD_ID,
        causal_tick=causal_tick,
        body_id=BODY_ID,
        embodiment_episode_id=EPISODE_ID,
        morphology_revision=MORPHOLOGY_REVISION,
        pose=pose,
        linear_velocity_m_per_s=(0.0, 0.0, 0.0),
        angular_velocity_rad_per_s=(0.0, 0.0, 0.0),
        last_physical_step=physical_step,
    )


def _frame() -> FrameDescriptor:
    return FrameDescriptor(
        frame_id=FRAME_ID,
        parent_frame_id="world-frame",
        origin_m=(0.0, 0.0, 0.0),
        x_axis_in_parent=(1.0, 0.0, 0.0),
        y_axis_in_parent=(0.0, 1.0, 0.0),
        z_axis_in_parent=(0.0, 0.0, 1.0),
        handedness="RIGHT_HANDED",
        up_axis="Z",
    )


def _field() -> IdentifiedFiniteScalarField:
    return IdentifiedFiniteScalarField(
        resource_id=RESOURCE_ID,
        field_revision=FIELD_REVISION,
        environment_id=ENVIRONMENT_ID,
        environment_revision=ENVIRONMENT_REVISION,
        frame_id=FRAME_ID,
        state=FiniteResourceState(
            position=(1.0, 0.0, 0.0),
            radius=0.1,
            field_radius=2.0,
            remaining=10.0,
            transfer_per_tick=1.0,
        ),
    )


def _reference(
    owner: str,
    owner_identity: str,
    tick: int,
    payload: bytes,
    artifact_dir: Path,
) -> CheckpointReference:
    path = artifact_dir / f"{owner}-{tick}.checkpoint"
    path.write_bytes(payload)
    return CheckpointReference(
        owner=owner,
        owner_identity=owner_identity,
        checkpoint_id=f"{owner}-tick-{tick}",
        schema_version=f"{owner}-fixture-v1",
        locator=str(path),
        sha256=hashlib.sha256(payload).hexdigest(),
    )


def _runtime(tmp_path: Path, *, stale_receptor: bool = False):
    artifacts = tmp_path / "owner-checkpoints"
    artifacts.mkdir()
    field = _field()
    body = _body_state(0)
    cognition = CognitiveTransitionState.initial(ORGANISM_ID)
    initial_organism_state = OrganismTransitionState(
        owner_checkpoint=CognitiveRuntimeCheckpoint(
            state=cognition, graph=_graph()
        ).to_organism_checkpoint(),
        organism_step=0,
        embodiment=EmbodimentContextRef(
            BODY_ID, EPISODE_ID, MORPHOLOGY_REVISION
        ),
        lifecycle=OrganismLifecycle.ACTIVE,
    )
    initial_refs = (
        _reference("organism", str(ORGANISM_ID), 0, initial_organism_state.to_bytes(), artifacts),
        _reference("embodiment", BODY_ID, 0, body.checkpoint_bytes(), artifacts),
        _reference("environment", RESOURCE_ID, 0, field.checkpoint_bytes(), artifacts),
    )

    def verify_refs(references: tuple[CheckpointReference, ...]) -> bool:
        return all(
            Path(reference.locator).is_file()
            and hashlib.sha256(Path(reference.locator).read_bytes()).hexdigest() == reference.sha256
            for reference in references
        )

    owners = tuple(sorted((reference.owner, reference.owner_identity) for reference in initial_refs))
    runtime = DurableWorldRuntime.initialize(
        WorldGenesisSpec(
            world_id=WorldId(WORLD_ID),
            initial_presence_ids=(WORLD_PRESENCE_ID,),
            registered_owner_ids=owners,
            initial_owner_references=initial_refs,
            owner_checkpoint_contracts={
                (reference.owner, reference.owner_identity): _FixtureOwnerContract()
                for reference in initial_refs
            },
            presence_owner_links=(
                PresenceOwnerLink(PresenceId(WORLD_PRESENCE_ID), "organism", str(ORGANISM_ID)),
            ),
        ),
        WORLD_PRESENCE_ID,
        WorldCausalManifestStore(tmp_path / "world-manifest.json"),
        DurableJournalSnapshotStore(tmp_path / "world-journal", WorldId(WORLD_ID)),
        owner_checkpoint_verifier=verify_refs,
    )
    receptor = EmbodimentReceptor(
        receptor_id=RECEPTOR_ID,
        body_id=BODY_ID,
        embodiment_episode_id=EPISODE_ID,
        morphology_revision="stale-morphology" if stale_receptor else MORPHOLOGY_REVISION,
        body_frame="body-root",
        position_in_body_frame_m=(0.0, 0.0, 0.0),
    )
    return runtime, artifacts, field, body, initial_organism_state, receptor


def _run_fixture(tmp_path: Path, *, stale_receptor: bool = False):
    runtime, artifacts, field, body, organism_state, receptor = _runtime(
        tmp_path, stale_receptor=stale_receptor
    )
    graph = _graph()
    pending: dict[str, object] = {"initial_organism_state": organism_state}

    def project_body(context: PhysicalStepContext):
        point = project_receptor_sample_point(receptor, body, context)
        pending["receptor_point"] = point
        return point

    def project_environment(context: PhysicalStepContext):
        # Correlation is derived from the independently issued World context,
        # before any Environment sample or solver candidate exists.
        correlation = EnvironmentStepCorrelation(
            world_id=context.world_id,
            causal_tick=context.source_causal_tick,
            physical_step_id=context.physical_step_id,
            environment_id=ENVIRONMENT_ID,
            environment_revision=ENVIRONMENT_REVISION,
        )
        point = pending["receptor_point"]
        sample = field.sample_field(
            point.position_m, frame=_frame(), correlation=correlation
        )
        assert sample.resource_id == RESOURCE_ID
        assert sample.field_revision == FIELD_REVISION
        assert sample.world_id == context.world_id
        assert sample.causal_tick == context.source_causal_tick
        assert sample.physical_step_id == context.physical_step_id
        assert sample.environment_id == ENVIRONMENT_ID
        assert sample.environment_revision == ENVIRONMENT_REVISION
        assert sample.frame_id == FRAME_ID
        assert 0.0 <= sample.value <= 1.0
        expected_provenance = SenseProvenance(
            world_id=context.world_id,
            causal_tick=context.source_causal_tick,
            physical_step_id=context.physical_step_id,
            environment_id=ENVIRONMENT_ID,
            environment_revision=ENVIRONMENT_REVISION,
            sample_frame_id=point.frame_id,
            sample_position_m=point.position_m,
            body_id=context.body_id,
            embodiment_episode_id=context.embodiment_episode_id,
            morphology_revision=context.morphology_revision,
        )
        provenance = SenseProvenance(
            world_id=sample.world_id,
            causal_tick=sample.causal_tick,
            physical_step_id=sample.physical_step_id,
            environment_id=sample.environment_id,
            environment_revision=sample.environment_revision,
            sample_frame_id=sample.frame_id,
            sample_position_m=sample.point_m,
            body_id=context.body_id,
            embodiment_episode_id=context.embodiment_episode_id,
            morphology_revision=context.morphology_revision,
        )
        assert provenance == expected_provenance
        signal = OpaqueSenseSignal(
            channel_id=CHANNEL_ID,
            receptor_id=RECEPTOR_ID,
            value=DimensionlessScalarFieldIdentityV1().transduce(sample.value),
            provenance=provenance,
            contract_id=DimensionlessScalarFieldIdentityV1.CONTRACT_ID,
            contract_revision=DimensionlessScalarFieldIdentityV1.REVISION,
            quantity_kind=DimensionlessScalarFieldIdentityV1.QUANTITY_KIND,
            unit=DimensionlessScalarFieldIdentityV1.UNIT,
        )
        accepted = advance_organism_for_physical_step(
            context,
            organism_state,
            (SenseChannelBinding(CHANNEL_ID, RECEPTOR_ID, "opaque-input"),),
            (signal,),
            expected_provenance=expected_provenance,
            kernel_limits=KernelLimits(),
        )
        pending["signal"] = signal
        pending["organism_state"] = accepted
        pending["sample"] = sample
        return sample

    def execute_solver(context: PhysicalStepContext, _body_projection, sample):
        # The deterministic solver fixture is deliberately numerical-only.
        assert sample.world_id == context.world_id
        candidate_correlation = EmbodimentStepCorrelation(
            world_id=context.world_id,
            causal_tick=context.source_causal_tick,
            physical_step_id=context.physical_step_id,
            body_id=context.body_id,
            embodiment_episode_id=context.embodiment_episode_id,
            morphology_revision=context.morphology_revision,
            environment_revision=context.environment_revision,
            solver_config_revision=context.solver_config_revision,
        )
        body_candidate = SolverBodyCandidate(
            correlation=candidate_correlation,
            reference_frame=FRAME_ID,
            body_frame="body-root",
            position_m=body.pose.position_m,
            orientation_xyzw=body.pose.orientation_xyzw,
            linear_velocity_m_per_s=body.linear_velocity_m_per_s,
            angular_velocity_rad_per_s=body.angular_velocity_rad_per_s,
        )
        pending["body_candidate"] = body_candidate
        return PhysicalStepCandidate(body_candidate, sample)

    def adopt_body(context: PhysicalStepContext, candidate: SolverBodyCandidate):
        pending["body_next"] = body.adopt_candidate(candidate, world_context=context)

    def adopt_environment(context: PhysicalStepContext, candidate):
        if (
            candidate.resource_id != RESOURCE_ID
            or candidate.field_revision != FIELD_REVISION
            or candidate.world_id != context.world_id
            or candidate.causal_tick != context.source_causal_tick
            or candidate.physical_step_id != context.physical_step_id
            or candidate.environment_id != ENVIRONMENT_ID
            or candidate.environment_revision != context.environment_revision
        ):
            raise ValueError("Environment sample identity or correlation is stale")
        pending["environment_adopted"] = True

    def write_checkpoints(context: PhysicalStepContext):
        next_body = pending["body_next"]
        assert isinstance(next_body, EmbodimentPhysicalState)
        return (
            _reference("organism", str(ORGANISM_ID), context.target_causal_tick,
                       pending["organism_state"].to_bytes(), artifacts),
            _reference("embodiment", BODY_ID, context.target_causal_tick,
                       next_body.checkpoint_bytes(), artifacts),
            _reference("environment", RESOURCE_ID, context.target_causal_tick,
                       field.checkpoint_bytes(), artifacts),
        )

    ports = PhysicalStepPorts(
        project_body=project_body,
        project_environment=project_environment,
        execute_solver=execute_solver,
        adopt_body=adopt_body,
        adopt_environment=adopt_environment,
        write_checkpoints=write_checkpoints,
    )
    record = run_physical_step(
        runtime,
        commit_id="commit-scalar-sensing-1",
        physical_step_id="physical-step-scalar-sensing-1",
        body_id=BODY_ID,
        embodiment_episode_id=EPISODE_ID,
        morphology_revision=MORPHOLOGY_REVISION,
        environment_revision=ENVIRONMENT_REVISION,
        solver_config_revision="deterministic-sensing-solver-v1",
        ports=ports,
    )
    return runtime, pending, record


def test_environment_scalar_field_reaches_organism_as_opaque_signal_and_commits(
    tmp_path: Path,
) -> None:
    runtime, pending, record = _run_fixture(tmp_path)

    assert runtime.status is ManifestStatus.COMMITTED
    assert runtime.causal_tick == 1
    assert record.causal_tick == 1
    sample = pending["sample"]
    assert sample.resource_id == RESOURCE_ID
    assert sample.field_revision == FIELD_REVISION
    signal = pending["signal"]
    assert signal.channel_id == CHANNEL_ID
    assert signal.receptor_id == RECEPTOR_ID
    assert signal.value == pytest.approx(0.25)
    accepted = pending["organism_state"]
    assert isinstance(accepted, OrganismTransitionState)
    assert accepted.organism_step == 1
    assert accepted.world_correlation.causal_tick == 0
    assert accepted.owner_checkpoint != pending["initial_organism_state"].owner_checkpoint
    organism_ref = next(
        reference
        for reference in runtime.manifest.checkpoint_references
        if reference.owner == "organism"
    )
    persisted = OrganismTransitionState.from_bytes(Path(organism_ref.locator).read_bytes())
    assert persisted == accepted
    cognition = CognitiveRuntimeCheckpoint.from_organism_checkpoint(
        persisted.owner_checkpoint, kernel_limits=KernelLimits()
    )
    assert cognition.state.last_causal_tick == 0
    assert cognition.graph.node_by_id("opaque-input") is not None
    assert len(runtime.manifest.checkpoint_references) == 4  # three owners + World journal


def test_stale_receptor_morphology_is_rejected_without_next_committed_cut(
    tmp_path: Path,
) -> None:
    runtime, *_ = _runtime(tmp_path, stale_receptor=True)
    body = _body_state(0)

    def project_body(context: PhysicalStepContext):
        return project_receptor_sample_point(
            EmbodimentReceptor(
                receptor_id=RECEPTOR_ID,
                body_id=BODY_ID,
                embodiment_episode_id=EPISODE_ID,
                morphology_revision="stale-morphology",
                body_frame="body-root",
                position_in_body_frame_m=(0.0, 0.0, 0.0),
            ),
            body,
            context,
        )

    def unused(*_args):
        raise AssertionError("stale receptor must fail before sampling/solver/adoption")

    with pytest.raises(ValueError, match="receptor identity or morphology"):
        run_physical_step(
            runtime,
            commit_id="commit-scalar-sensing-stale",
            physical_step_id="physical-step-scalar-sensing-stale",
            body_id=BODY_ID,
            embodiment_episode_id=EPISODE_ID,
            morphology_revision=MORPHOLOGY_REVISION,
            environment_revision=ENVIRONMENT_REVISION,
            solver_config_revision="deterministic-sensing-solver-v1",
            ports=PhysicalStepPorts(
                project_body=project_body,
                project_environment=unused,
                execute_solver=unused,
                adopt_body=unused,
                adopt_environment=unused,
                write_checkpoints=unused,
            ),
        )

    assert runtime.status is ManifestStatus.RECOVERY_REQUIRED
    assert runtime.causal_tick == 0
    assert runtime.manifest.causal_tick == 0
    assert runtime.manifest.pending_tick == 1


def test_organism_transition_rejects_provenance_for_another_physical_step(
    tmp_path: Path,
) -> None:
    _, _, _, _, organism_state, _ = _runtime(tmp_path)
    context = PhysicalStepContext(
        world_id=WORLD_ID,
        source_causal_tick=0,
        target_causal_tick=1,
        physical_step_id="physical-step-expected",
        body_id=BODY_ID,
        embodiment_episode_id=EPISODE_ID,
        morphology_revision=MORPHOLOGY_REVISION,
        environment_revision=ENVIRONMENT_REVISION,
        solver_config_revision="deterministic-sensing-solver-v1",
    )
    mismatched_provenance = SenseProvenance(
        world_id=WORLD_ID,
        causal_tick=0,
        physical_step_id="physical-step-stale",
        environment_id=ENVIRONMENT_ID,
        environment_revision=ENVIRONMENT_REVISION,
        sample_frame_id=FRAME_ID,
        sample_position_m=(0.0, 0.0, 0.0),
        body_id=BODY_ID,
        embodiment_episode_id=EPISODE_ID,
        morphology_revision=MORPHOLOGY_REVISION,
    )

    with pytest.raises(ValueError, match="does not match the World physical step"):
        advance_organism_for_physical_step(
            context,
            organism_state,
            (),
            (),
            expected_provenance=mismatched_provenance,
            kernel_limits=KernelLimits(),
        )

    assert organism_state.organism_step == 0
