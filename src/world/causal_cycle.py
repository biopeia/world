"""World-owned composition of a bounded sensorimotor causal cycle.

This integration boundary composes public owner APIs but does not own
cognition, action policy, body/environment state, numerical physics, or
observation meaning. Owner checkpoint bytes are durably published by an
opaque checkpoint store before World can commit their references. The
post-commit owner-to-Observation bindings are part of this TARGET runtime
composition, not delegated to a test or launcher.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from enum import StrEnum
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
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
    ObservationPublicationFailure,
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
class ScalarSensorimotorCycleInputs:
    """Initial state and fixed owner capabilities for the production cycle."""

    environment_revision: str
    solver_configuration: ArticulatedMujocoConfiguration
    current_organism: OrganismTransitionState
    action_composer: _OrganismActionComposer
    receptor: EmbodimentReceptor
    embodiment_state: EmbodimentPhysicalState
    environment_field: IdentifiedFiniteScalarField
    sampling_frame: FrameDescriptor
    transduction: DimensionlessScalarFieldIdentityV1
    sense_binding: SenseChannelBinding
    kernel_limits: KernelLimits
    actuator_context: object
    actuator_surface: object
    articulated_body: ArticulatedEmbodimentPhysicalState
    active_morphology: ActiveArticulatedMorphology
    differential_binding: EmbodimentDifferentialMotorBinding
    torque_binding: EmbodimentJointTorqueBinding
    environment_owner: StaticEnvironmentOwner
    owner_checkpoint_store: OwnerCheckpointStore

    def as_cycle_arguments(self) -> dict[str, object]:
        return {
            "environment_revision": self.environment_revision,
            "solver_configuration": self.solver_configuration,
            "current_organism": self.current_organism,
            "action_composer": self.action_composer,
            "receptor": self.receptor,
            "embodiment_state": self.embodiment_state,
            "environment_field": self.environment_field,
            "sampling_frame": self.sampling_frame,
            "transduction": self.transduction,
            "sense_binding": self.sense_binding,
            "kernel_limits": self.kernel_limits,
            "actuator_context": self.actuator_context,
            "actuator_surface": self.actuator_surface,
            "articulated_body": self.articulated_body,
            "active_morphology": self.active_morphology,
            "differential_binding": self.differential_binding,
            "torque_binding": self.torque_binding,
            "environment_owner": self.environment_owner,
            "owner_checkpoint_store": self.owner_checkpoint_store,
        }


class ScalarSensorimotorSessionStatus(StrEnum):
    COMPLETED = "completed"
    STOPPED = "stopped"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class ScalarSensorimotorSessionResult:
    """Operational outcome; only returned cycle results denote full publications."""

    world_id: str
    run_id: str
    source_revisions: Mapping[str, str]
    status: ScalarSensorimotorSessionStatus
    cycles: tuple[CausalCycleResult, ...]
    last_committed_cut: CommittedWorldCut | None
    error: str | None = None
    failure_kind: str | None = None
    publication_delivery_status_uncertain: bool = False


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


def run_scalar_sensorimotor_session(
    world: DurableWorldRuntime,
    observation_plane: ObservationPlaneLike,
    *,
    run_id: str,
    cycle_inputs: ScalarSensorimotorCycleInputs,
    source_revisions: Mapping[str, str],
    max_cycles: int | None = 1,
    stop_requested: Callable[[], bool] | None = None,
    on_cycle_result: Callable[[CausalCycleResult], None] | None = None,
    commit_id_prefix: str | None = None,
    physical_step_id_prefix: str | None = None,
) -> ScalarSensorimotorSessionResult:
    """Run production causal cycles serially until bounded completion or stop.

    A cycle is synchronous and is never interrupted here. Stop is sampled only
    between cycles, so a requested stop allows the in-flight production cycle
    (including its publication) to return or fail before the session exits.
    ``max_cycles=None`` is continuous mode and therefore requires a stop
    predicate. Source revisions are caller-supplied provenance, not inferred.
    """
    if not isinstance(world, DurableWorldRuntime):
        raise TypeError("world must be a DurableWorldRuntime")
    if not isinstance(cycle_inputs, ScalarSensorimotorCycleInputs):
        raise TypeError("cycle_inputs must be ScalarSensorimotorCycleInputs")
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("run_id must be a non-empty string")
    if max_cycles is not None and (type(max_cycles) is not int or max_cycles < 1):
        raise ValueError("max_cycles must be a positive integer or None")
    if max_cycles is None and stop_requested is None:
        raise ValueError("continuous mode requires stop_requested")
    if stop_requested is not None and not callable(stop_requested):
        raise TypeError("stop_requested must be callable")
    if on_cycle_result is not None and not callable(on_cycle_result):
        raise TypeError("on_cycle_result must be callable")
    revisions = dict(source_revisions)
    if not revisions or any(
        not isinstance(key, str)
        or not key.strip()
        or not isinstance(value, str)
        or not value.strip()
        for key, value in revisions.items()
    ):
        raise ValueError("source_revisions must map non-empty names to non-empty revisions")
    commit_prefix = commit_id_prefix or f"{run_id}:commit:"
    physical_step_prefix = physical_step_id_prefix or f"{run_id}:physical-step:"
    if not commit_prefix.strip() or not physical_step_prefix.strip():
        raise ValueError("cycle identifier prefixes must be non-empty")

    def current_cut() -> CommittedWorldCut | None:
        manifest = world.manifest
        if manifest.status.value != "COMMITTED":
            return None
        owner_references = tuple(
            reference
            for reference in manifest.checkpoint_references
            if reference.owner != "world-journal"
        )
        return world.committed_cut_for(
            world_id=manifest.world_id,
            causal_tick=manifest.causal_tick,
            commit_id=manifest.commit_id,
            expected_owner_references=owner_references,
        )

    completed: list[CausalCycleResult] = []
    last_cut: CommittedWorldCut | None
    error: str | None = None
    try:
        last_cut = current_cut()
    except Exception as cut_exc:
        last_cut = None
        error = (
            "initial committed-cut verification failed: "
            f"{type(cut_exc).__name__}: {cut_exc}"
        )

    status = ScalarSensorimotorSessionStatus.COMPLETED
    failure_kind: str | None = None
    publication_delivery_status_uncertain = False
    inputs = cycle_inputs
    try:
        while max_cycles is None or len(completed) < max_cycles:
            if stop_requested is not None and stop_requested():
                status = ScalarSensorimotorSessionStatus.STOPPED
                break
            target_tick = world.causal_tick + 1
            result = run_scalar_sensorimotor_cycle_and_publish(
                world,
                observation_plane,
                commit_id=f"{commit_prefix}{target_tick}",
                physical_step_id=f"{physical_step_prefix}{target_tick}",
                **inputs.as_cycle_arguments(),
            )
            completed.append(result)
            last_cut = result.committed_cut
            if on_cycle_result is not None:
                on_cycle_result(result)
            inputs = replace(
                inputs,
                current_organism=result.sensory_input.state,
                embodiment_state=result.physical_outcome.body_state.root,
                articulated_body=result.physical_outcome.body_state,
            )
    except Exception as exc:
        status = ScalarSensorimotorSessionStatus.FAILED
        error = f"{type(exc).__name__}: {exc}"
        failure_kind = type(exc).__name__
        if isinstance(exc, ObservationPublicationFailure):
            publication_delivery_status_uncertain = exc.delivery_status_uncertain
        # A production cycle can commit before an Observation delivery error.
        # Report that cut only if World can still verify it as the current cut;
        # the failed cycle remains absent from ``cycles``.
        try:
            last_cut = current_cut()
        except Exception as cut_exc:
            # Keep the last previously verified cut, but make clear that it
            # may not reflect a commit that became current before the failure.
            error = (
                f"{error}; current cut verification failed: "
                f"{type(cut_exc).__name__}: {cut_exc}"
            )

    return ScalarSensorimotorSessionResult(
        world_id=world.world_id.value,
        run_id=run_id,
        source_revisions=MappingProxyType(revisions),
        status=status,
        cycles=tuple(completed),
        last_committed_cut=last_cut,
        error=error,
        failure_kind=failure_kind,
        publication_delivery_status_uncertain=publication_delivery_status_uncertain,
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
    "ScalarSensorimotorCycleInputs",
    "ScalarSensorimotorSessionResult",
    "ScalarSensorimotorSessionStatus",
    "project_scalar_field_after_commit",
    "restore_scalar_sensorimotor_runtime",
    "run_scalar_sensorimotor_cycle_and_publish",
    "run_scalar_sensorimotor_session",
]
