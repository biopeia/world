"""World-owned composition of a bounded sensorimotor causal cycle.

This integration boundary composes public owner APIs but does not own
cognition, action policy, body/environment state, numerical physics, or
observation meaning. Owner checkpoint bytes are durably published by an
opaque checkpoint store before World can commit their references. The
post-commit owner-to-Observation bindings are part of this TARGET runtime
composition, not delegated to a test or launcher.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Protocol, runtime_checkable

from embodiment.physics3d.articulated_state import (
    ARTICULATED_PHYSICAL_CHECKPOINT_SCHEMA,
    ActiveArticulatedMorphology,
    ArticulatedEmbodimentPhysicalState,
)
from embodiment.physics3d.differential_motor_composition import (
    EmbodimentDifferentialMotorBinding,
)
from embodiment.physics3d.joint_torque import EmbodimentJointTorqueBinding
from embodiment.physics3d.physical_transition import EmbodimentPhysicalState
from embodiment.physics3d.receptor import EmbodimentReceptor
from environment import (
    EnvironmentObservationContext,
    FrameDescriptor,
    IdentifiedFiniteScalarField,
    StaticEnvironmentOwner,
)
from modality import DimensionlessScalarFieldIdentityV1
from organism.actuation import (
    ORGANISM_EXPLORATION_CHECKPOINT_SCHEMA,
    ActuatorEvidenceContext,
    ActuatorSurface,
    OrganismExplorationComposer,
    RestoredOrganismExplorationCheckpoint,
)
from organism.cognition import SenseChannelBinding
from organism.cognition.limits import KernelLimits
from organism.transition import OrganismTransitionState

from .adapters.articulated_mujoco import (
    ArticulatedMujocoConfiguration,
    ArticulatedPhysicalStepOutcome,
    advance_articulated_mujoco_step_from_motor_command,
)
from .durable_runtime import (
    CommittedWorldCut,
    DurableJournalSnapshotStore,
    DurableWorldRuntime,
    OwnerCheckpointContract,
    OwnerCheckpointVerifier,
    WorldCausalManifestStore,
)
from .manifest import CheckpointReference, ManifestError
from .observation_publication import (
    ObservationPlaneLike,
    ObservationPublication,
    OwnerObservation,
    publish_committed_owner_observations,
)
from .organism_transition import (
    ScalarFieldOrganismInput,
    advance_organism_from_scalar_field_for_physical_step,
)
from .owner_checkpoint_store import (
    OwnerCheckpointPayload,
    OwnerCheckpointStore,
)
from .physical_step import PhysicalStepContext
from .runtime import PresenceId, WorldId


class _ActionCandidate(Protocol):
    """Structural view of an Organism-owned staged action result."""

    motor_command: object | None
    proposal: object | None


@runtime_checkable
class _OrganismActionComposer(Protocol):
    """Public Organism port; World never inspects its private state."""

    def prepare(
        self,
        transition: OrganismTransitionState,
        *,
        actuator_context: object,
        surface: object,
        target_causal_tick: int,
        physical_step_id: str,
        commit_id: str,
    ) -> _ActionCandidate: ...

    def checkpoint_for_candidate(
        self,
        candidate: _ActionCandidate,
        *,
        transition_state: OrganismTransitionState,
    ) -> bytes: ...

    def adopt_after_world_commit(
        self,
        candidate: _ActionCandidate,
        *,
        committed_cut: CommittedWorldCut,
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class CausalCycleResult:
    """Evidence-bearing result for one successfully committed cycle."""

    context: PhysicalStepContext
    sensory_input: ScalarFieldOrganismInput
    action_candidate: _ActionCandidate
    physical_outcome: ArticulatedPhysicalStepOutcome
    committed_cut: CommittedWorldCut
    observation_publication: ObservationPublication


@dataclass(frozen=True, slots=True)
class RestoredScalarSensorimotorRuntime:
    """A fully restored, receipt-bound composition ready for another cycle.

    This object is returned only after World has validated the durable cut,
    issued its committed-cut receipt, and each semantic owner's public restore
    API has reconstructed the referenced state. It is a bounded fixture for
    the current production scalar sensorimotor composition, not a general
    migration-complete Organism checkpoint.
    """

    world: DurableWorldRuntime
    committed_cut: CommittedWorldCut
    organism: RestoredOrganismExplorationCheckpoint
    articulated_body: ArticulatedEmbodimentPhysicalState
    environment_owner: StaticEnvironmentOwner
    environment_field: IdentifiedFiniteScalarField


def restore_scalar_sensorimotor_runtime(
    world_id: WorldId,
    presence_id: str,
    manifest_store: WorldCausalManifestStore,
    journal_store: DurableJournalSnapshotStore,
    *,
    owner_checkpoint_verifier: OwnerCheckpointVerifier,
    owner_checkpoint_contracts: Mapping[tuple[str, str], OwnerCheckpointContract],
    active_morphology: ActiveArticulatedMorphology,
    expected_organism_id: str,
    expected_physical_step_id: str,
    expected_environment_owner_id: str,
    expected_environment_id: str,
    expected_environment_revision: str,
    expected_field_id: str,
    expected_field_revision: str,
    actuator_context: ActuatorEvidenceContext,
    actuator_surface: ActuatorSurface,
    kernel_limits: KernelLimits | None = None,
) -> RestoredScalarSensorimotorRuntime:
    """Restore the production causal-cycle owners from one committed World cut.

    The World restore path validates the manifest, journal, exact owner roster,
    checkpoint contracts and recoverability before returning. This composition
    then obtains the receipt from that restored World runtime, uses it at the
    Organism public restore boundary, restores Embodiment and Environment via
    their public APIs, and exposes the composition only after all identities
    and revisions have been checked. The owner bytes remain opaque to World
    except for the existing public checkpoint-contract validation.
    """
    if not isinstance(world_id, WorldId):
        raise TypeError("world_id must be a WorldId")
    if not isinstance(active_morphology, ActiveArticulatedMorphology):
        raise TypeError("active_morphology must be an ActiveArticulatedMorphology")
    if not isinstance(actuator_context, ActuatorEvidenceContext):
        raise TypeError("actuator_context must be an ActuatorEvidenceContext")
    if not isinstance(actuator_surface, ActuatorSurface):
        raise TypeError("actuator_surface must be an ActuatorSurface")
    if (
        actuator_context.body_id != active_morphology.body_id
        or actuator_context.embodiment_episode_id != active_morphology.embodiment_episode_id
        or actuator_context.morphology_revision != active_morphology.morphology_revision
        or actuator_context.surface_fingerprint != actuator_surface.contract_fingerprint
    ):
        raise ValueError("actuator applicability must match the active Embodiment morphology and surface")
    expected_owners = {
        ("organism", expected_organism_id),
        ("embodiment", active_morphology.body_id),
        ("environment", expected_environment_owner_id),
        ("environment", expected_field_id),
    }
    staged: dict[tuple[str, str], tuple[CheckpointReference, bytes]] = {}

    def stage_owner_checkpoints(references: tuple[CheckpointReference, ...]) -> None:
        if len(references) != len(expected_owners):
            raise ManifestError("committed cut does not contain the required scalar-cycle owner set")
        for reference in references:
            key = (reference.owner, reference.owner_identity)
            if key not in expected_owners or key in staged:
                raise ManifestError("committed cut has an unexpected or duplicate scalar-cycle owner")
            try:
                encoded = Path(reference.locator).read_bytes()
            except OSError as error:
                raise ManifestError("committed owner checkpoint is unavailable") from error
            if sha256(encoded).hexdigest() != reference.sha256:
                raise ManifestError("committed owner checkpoint digest does not match its World reference")
            staged[key] = (reference, encoded)
        if set(staged) != expected_owners:
            raise ManifestError("committed cut is missing a required scalar-cycle owner")

    restored_world = DurableWorldRuntime.restore(
        world_id,
        presence_id,
        manifest_store,
        journal_store,
        owner_checkpoint_verifier=owner_checkpoint_verifier,
        owner_checkpoint_contracts=owner_checkpoint_contracts,
        owner_checkpoint_restorer=stage_owner_checkpoints,
    )
    manifest = restored_world.manifest
    cut = restored_world.committed_cut_for(
        world_id=world_id.value,
        causal_tick=manifest.causal_tick,
        commit_id=manifest.commit_id,
        expected_owner_references=tuple(value[0] for value in staged.values()),
    )

    organism_reference, organism_bytes = staged[("organism", expected_organism_id)]
    if organism_reference.schema_version != ORGANISM_EXPLORATION_CHECKPOINT_SCHEMA:
        raise ManifestError("committed Organism checkpoint schema is not supported by this composition")
    try:
        organism = OrganismExplorationComposer.restore_from_committed_checkpoint(
            organism_bytes,
            committed_cut=cut,
            expected_organism_id=expected_organism_id,
            expected_physical_step_id=expected_physical_step_id,
            expected_actuator_context=actuator_context,
            expected_surface=actuator_surface,
            kernel_limits=kernel_limits,
        )
    except (TypeError, ValueError) as error:
        raise ManifestError("Organism checkpoint does not restore against the committed World receipt") from error

    body_reference, body_bytes = staged[("embodiment", active_morphology.body_id)]
    if body_reference.schema_version != ARTICULATED_PHYSICAL_CHECKPOINT_SCHEMA:
        raise ManifestError("committed Embodiment checkpoint schema is not supported by this composition")
    try:
        articulated_body = ArticulatedEmbodimentPhysicalState.restore_checkpoint(
            body_bytes, active_morphology=active_morphology
        )
    except (TypeError, ValueError) as error:
        raise ManifestError("Embodiment checkpoint is incompatible with the active morphology") from error

    environment_reference, environment_bytes = staged[("environment", expected_environment_owner_id)]
    if environment_reference.schema_version != StaticEnvironmentOwner.CHECKPOINT_SCHEMA:
        raise ManifestError("committed Environment checkpoint schema is not supported by this composition")
    try:
        environment_owner = StaticEnvironmentOwner.from_checkpoint_bytes(environment_bytes)
    except (TypeError, ValueError) as error:
        raise ManifestError("Environment owner checkpoint cannot be restored") from error

    field_reference, field_bytes = staged[("environment", expected_field_id)]
    if field_reference.schema_version != IdentifiedFiniteScalarField.OWNER_SCHEMA:
        raise ManifestError("committed scalar-field checkpoint schema is not supported by this composition")
    try:
        environment_field = IdentifiedFiniteScalarField.from_checkpoint_bytes(field_bytes)
    except (TypeError, ValueError) as error:
        raise ManifestError("Environment scalar-field checkpoint cannot be restored") from error

    if (
        str(organism.transition_state.identity) != expected_organism_id
        or organism.transition_state.world_correlation is None
        or organism.transition_state.world_correlation.world_id != world_id.value
        or organism.transition_state.world_correlation.causal_tick != cut.causal_tick - 1
    ):
        raise ManifestError("restored Organism transition does not match the committed cut lineage")
    if (
        articulated_body.root.world_id != world_id.value
        or articulated_body.root.causal_tick != cut.causal_tick
        or articulated_body.root.body_id != active_morphology.body_id
        or articulated_body.root.embodiment_episode_id != active_morphology.embodiment_episode_id
        or articulated_body.root.morphology_revision != active_morphology.morphology_revision
    ):
        raise ManifestError("restored Embodiment state does not match the committed World cut")
    if (
        environment_owner.owner_id != expected_environment_owner_id
        or environment_owner.world_id != world_id.value
        or environment_owner.scene.environment_id != expected_environment_id
        or environment_owner.scene.environment_revision != expected_environment_revision
    ):
        raise ManifestError("restored Environment owner identity or revision is incompatible")
    if (
        environment_field.resource_id != expected_field_id
        or environment_field.environment_id != expected_environment_id
        or environment_field.environment_revision != expected_environment_revision
        or environment_field.field_revision != expected_field_revision
    ):
        raise ManifestError("restored scalar field identity or revision is incompatible")

    return RestoredScalarSensorimotorRuntime(
        world=restored_world,
        committed_cut=cut,
        organism=organism,
        articulated_body=articulated_body,
        environment_owner=environment_owner,
        environment_field=environment_field,
    )


def run_scalar_sensorimotor_cycle_and_publish(
    world: DurableWorldRuntime,
    observation_plane: ObservationPlaneLike,
    *,
    commit_id: str,
    physical_step_id: str,
    environment_revision: str,
    solver_configuration: ArticulatedMujocoConfiguration,
    current_organism: OrganismTransitionState,
    action_composer: _OrganismActionComposer,
    receptor: EmbodimentReceptor,
    embodiment_state: EmbodimentPhysicalState,
    environment_field: IdentifiedFiniteScalarField,
    sampling_frame: FrameDescriptor,
    transduction: DimensionlessScalarFieldIdentityV1,
    sense_binding: SenseChannelBinding,
    kernel_limits: KernelLimits,
    actuator_context: object,
    actuator_surface: object,
    articulated_body: ArticulatedEmbodimentPhysicalState,
    active_morphology: ActiveArticulatedMorphology,
    differential_binding: EmbodimentDifferentialMotorBinding,
    torque_binding: EmbodimentJointTorqueBinding,
    environment_owner: StaticEnvironmentOwner,
    owner_checkpoint_store: OwnerCheckpointStore,
) -> CausalCycleResult:
    """Sense, transition, issue an Organism command, solve, adopt, commit, publish.

    The World publication guard makes the pre-step context and post-commit
    publication one single-writer critical section. A missing action command
    is a failed cycle: it is never replaced with a fabricated or neutral one.
    """
    if not isinstance(world, DurableWorldRuntime):
        raise TypeError("world must be a DurableWorldRuntime")
    if not isinstance(current_organism, OrganismTransitionState):
        raise TypeError("current_organism must be an OrganismTransitionState")
    if not isinstance(action_composer, _OrganismActionComposer):
        raise TypeError("action_composer must implement the Organism action port")
    if not isinstance(owner_checkpoint_store, OwnerCheckpointStore):
        raise TypeError("owner_checkpoint_store must implement the opaque owner checkpoint store")
    source_tick = world.causal_tick
    if embodiment_state.causal_tick != source_tick:
        raise ValueError("Embodiment source state does not match the current World cut")
    if articulated_body.root.causal_tick != source_tick:
        raise ValueError("articulated body does not match the current World cut")
    if environment_owner.scene.environment_revision != environment_revision:
        raise ValueError("Environment owner revision differs from the requested cycle")
    if environment_field.environment_revision != environment_revision:
        raise ValueError("scalar field revision context differs from the requested cycle")
    if solver_configuration.revision == "":
        raise ValueError("solver configuration revision is required")

    with world.committed_cut_publication_guard():
        source_tick = world.causal_tick
        context = PhysicalStepContext(
            world_id=world.world_id.value,
            source_causal_tick=source_tick,
            target_causal_tick=source_tick + 1,
            physical_step_id=physical_step_id,
            body_id=articulated_body.root.body_id,
            embodiment_episode_id=articulated_body.root.embodiment_episode_id,
            morphology_revision=articulated_body.morphology_revision,
            environment_revision=environment_revision,
            solver_config_revision=solver_configuration.revision,
        )
        sensed = advance_organism_from_scalar_field_for_physical_step(
            context,
            current_organism,
            receptor=receptor,
            embodiment_state=embodiment_state,
            environment_field=environment_field,
            sampling_frame=sampling_frame,
            transduction=transduction,
            binding=sense_binding,
            kernel_limits=kernel_limits,
        )
        candidate = action_composer.prepare(
            sensed.state,
            actuator_context=actuator_context,
            surface=actuator_surface,
            target_causal_tick=context.target_causal_tick,
            physical_step_id=context.physical_step_id,
            commit_id=commit_id,
        )
        command = candidate.motor_command
        if command is None:
            raise ValueError("Organism produced no motor command for this causal cycle")

        written_references: tuple[CheckpointReference, ...] = ()

        def persist(
            step_context: PhysicalStepContext,
            adopted_body: ArticulatedEmbodimentPhysicalState,
            adopted_environment: StaticEnvironmentOwner,
        ) -> tuple[CheckpointReference, ...]:
            nonlocal written_references
            if step_context != context:
                raise ManifestError("World-issued physical context changed before commit")
            organism_checkpoint = action_composer.checkpoint_for_candidate(
                candidate,
                transition_state=sensed.state,
            )
            written_references = tuple(
                owner_checkpoint_store.write(
                    step_context,
                    (
                        OwnerCheckpointPayload(
                            owner="organism",
                            owner_identity=str(sensed.state.identity),
                            schema_version=ORGANISM_EXPLORATION_CHECKPOINT_SCHEMA,
                            encoded=organism_checkpoint,
                        ),
                        OwnerCheckpointPayload(
                            owner="embodiment",
                            owner_identity=adopted_body.root.body_id,
                            schema_version=ARTICULATED_PHYSICAL_CHECKPOINT_SCHEMA,
                            encoded=adopted_body.checkpoint_bytes(),
                        ),
                        OwnerCheckpointPayload(
                            owner="environment",
                            owner_identity=adopted_environment.owner_id,
                            schema_version=StaticEnvironmentOwner.CHECKPOINT_SCHEMA,
                            encoded=adopted_environment.checkpoint_bytes(),
                        ),
                        OwnerCheckpointPayload(
                            owner="environment",
                            owner_identity=environment_field.resource_id,
                            schema_version=environment_field.OWNER_SCHEMA,
                            encoded=environment_field.checkpoint_bytes(),
                        ),
                    ),
                )
            )
            _require_checkpoint(
                written_references,
                owner="organism",
                owner_identity=str(sensed.state.identity),
                encoded=organism_checkpoint,
            )
            _require_checkpoint(
                written_references,
                owner="embodiment",
                owner_identity=adopted_body.root.body_id,
                encoded=adopted_body.checkpoint_bytes(),
            )
            _require_checkpoint(
                written_references,
                owner="environment",
                owner_identity=adopted_environment.owner_id,
                encoded=adopted_environment.checkpoint_bytes(),
            )
            _require_checkpoint(
                written_references,
                owner="environment",
                owner_identity=environment_field.resource_id,
                encoded=environment_field.checkpoint_bytes(),
            )
            return written_references

        physical = advance_articulated_mujoco_step_from_motor_command(
            world,
            commit_id=commit_id,
            physical_step_id=physical_step_id,
            body_state=articulated_body,
            motor_command=command,
            active_morphology=active_morphology,
            differential_binding=differential_binding,
            torque_binding=torque_binding,
            environment_owner=environment_owner,
            configuration=solver_configuration,
            write_checkpoints=lambda step_context, adopted_body, adopted_environment: persist(
                step_context, adopted_body, adopted_environment
            ),
        )
        if physical.record.causal_tick != context.target_causal_tick:
            raise ManifestError("physical commit does not match the prepared causal cycle")
        cut = world.committed_cut_for(
            world_id=context.world_id,
            causal_tick=context.target_causal_tick,
            commit_id=commit_id,
            expected_owner_references=written_references,
        )
        action_composer.adopt_after_world_commit(candidate, committed_cut=cut)
        # This is the production domain-to-Observation binding. The test
        # supplies neither the owner checkpoint set nor claim construction.
        from observation.embodiment import embodiment_body_pose_observation
        from observation.environment import environment_finite_scalar_field_observation

        body_reference = _reference_for_owner(
            cut, "embodiment", physical.body_state.root.body_id
        )
        field_reference = _reference_for_owner(
            cut, "environment", environment_field.resource_id
        )
        body_claim = embodiment_body_pose_observation(
            physical.body_state.body_pose_observation_projection()
        )
        field_projection = project_scalar_field_after_commit(
            environment_field,
            context=context,
            point_m=sensed.receptor_point.position_m,
            frame=sampling_frame,
        )
        field_claim = environment_finite_scalar_field_observation(field_projection)
        observations = (
            OwnerObservation(body_reference, body_claim),
            OwnerObservation(field_reference, field_claim),
        )
        publication = publish_committed_owner_observations(
            world,
            observation_plane,
            committed_cut=cut,
            owner_observations=observations,
        )
        return CausalCycleResult(
            context,
            sensed,
            candidate,
            physical,
            cut,
            publication,
        )


def project_scalar_field_after_commit(
    environment_field: IdentifiedFiniteScalarField,
    *,
    context: PhysicalStepContext,
    point_m: tuple[float, float, float],
    frame: FrameDescriptor,
):
    """Invoke Environment's owner projection for a committed target cut."""
    observation_context = EnvironmentObservationContext(
        world_id=context.world_id,
        source_causal_tick=context.source_causal_tick,
        target_causal_tick=context.target_causal_tick,
        physical_step_id=context.physical_step_id,
        environment_id=environment_field.environment_id,
        environment_revision=environment_field.environment_revision,
        resource_id=environment_field.resource_id,
        field_revision=environment_field.field_revision,
    )
    return environment_field.project_observation(
        point_m, frame=frame, context=observation_context
    )


def _require_checkpoint(
    references: tuple[CheckpointReference, ...],
    *,
    owner: str,
    owner_identity: str,
    encoded: bytes,
) -> None:
    digest = sha256(encoded).hexdigest()
    if not any(
        reference.owner == owner
        and reference.owner_identity == owner_identity
        and reference.sha256 == digest
        for reference in references
    ):
        raise ManifestError(
            f"durable cut lacks the exact {owner} owner checkpoint for this cycle"
        )


def _reference_for_owner(
    cut: CommittedWorldCut, owner: str, owner_identity: str
) -> CheckpointReference:
    matches = tuple(
        reference
        for reference in cut.owner_checkpoint_references
        if reference.owner == owner and reference.owner_identity == owner_identity
    )
    if len(matches) != 1:
        raise ManifestError(
            f"committed cut must contain exactly one {owner} checkpoint for {owner_identity}"
        )
    return matches[0]


__all__ = [
    "CausalCycleResult",
    "RestoredScalarSensorimotorRuntime",
    "project_scalar_field_after_commit",
    "restore_scalar_sensorimotor_runtime",
    "run_scalar_sensorimotor_cycle_and_publish",
]
