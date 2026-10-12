"""World-owned composition of a bounded sensorimotor causal cycle.

This integration boundary composes public owner APIs but does not own
cognition, action policy, body/environment state, numerical physics, or
observation meaning. Owner checkpoint bytes are durably published by an
opaque checkpoint store before World can commit their references. The
post-commit owner-to-Observation bindings are part of this TARGET runtime
composition, not delegated to a test or launcher.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Protocol, runtime_checkable

from embodiment.physics3d.articulated_state import (
    ActiveArticulatedMorphology,
    ArticulatedEmbodimentPhysicalState,
    ARTICULATED_PHYSICAL_CHECKPOINT_SCHEMA,
)
from embodiment.physics3d.differential_motor_composition import (
    EmbodimentDifferentialMotorBinding,
)
from embodiment.physics3d.joint_torque import EmbodimentJointTorqueBinding
from embodiment.physics3d.receptor import EmbodimentReceptor
from embodiment.physics3d.physical_transition import EmbodimentPhysicalState
from environment import (
    EnvironmentObservationContext,
    FrameDescriptor,
    IdentifiedFiniteScalarField,
    StaticEnvironmentOwner,
)
from modality import DimensionlessScalarFieldIdentityV1
from organism.cognition import SenseChannelBinding
from organism.cognition.limits import KernelLimits
from organism.transition import OrganismTransitionState

from .adapters.articulated_mujoco import (
    ArticulatedMujocoConfiguration,
    ArticulatedPhysicalStepOutcome,
    advance_articulated_mujoco_step_from_motor_command,
)
from .durable_runtime import CommittedWorldCut, DurableWorldRuntime
from .manifest import CheckpointReference, ManifestError
from .observation_publication import (
    ObservationPlaneLike,
    ObservationPublication,
    OwnerObservation,
    publish_committed_owner_observations,
)
from .owner_checkpoint_store import (
    OwnerCheckpointPayload,
    OwnerCheckpointStore,
)
from .organism_transition import (
    ScalarFieldOrganismInput,
    advance_organism_from_scalar_field_for_physical_step,
)
from .physical_step import PhysicalStepContext

ORGANISM_ACTION_CHECKPOINT_SCHEMA = "biopeia-organism-exploration-checkpoint-v3"


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
                            schema_version=ORGANISM_ACTION_CHECKPOINT_SCHEMA,
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
    "ORGANISM_ACTION_CHECKPOINT_SCHEMA",
    "project_scalar_field_after_commit",
    "run_scalar_sensorimotor_cycle_and_publish",
]
