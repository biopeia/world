"""Production composition for physical commit followed by passive publication."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from enum import Enum

from .durable_runtime import (
    CommittedWorldCut,
    DurableWorldRuntime,
    TransitionCommitState,
)
from .journal import CausalRecord
from .manifest import CheckpointReference
from .observation_publication import (
    ObservationPlaneLike,
    ObservationPublicationFailure,
    ObservationPublication,
    OwnerObservation,
    publish_committed_owner_observations,
)
from .physical_step import (
    PhysicalStepContext,
    PhysicalStepPorts,
    run_physical_step,
)


@dataclass(frozen=True, slots=True)
class PhysicalStepPublicationResult:
    """Caller-visible outcome for one attempted physical step.

    Outcomes report evidence, not permission to retry. In particular,
    ``NOT_COMMITTED`` may still require explicit recovery if owner-side work
    or World preparation began; callers must honor ``recovery_required`` and
    must never repeat a step merely because publication was incomplete.
    """

    outcome: "PhysicalStepOutcome"
    causal_record: CausalRecord | None = None
    committed_cut: CommittedWorldCut | None = None
    publication: ObservationPublication | None = None
    failure: str | None = None
    recovery_required: bool = False


class PhysicalStepOutcome(str, Enum):
    """Four caller-visible outcomes for one attempted causal transition."""

    NOT_COMMITTED = "NOT_COMMITTED"
    COMMITTED_AND_PUBLISHED = "COMMITTED_AND_PUBLISHED"
    COMMITTED_PUBLICATION_INCOMPLETE = "COMMITTED_PUBLICATION_INCOMPLETE"
    COMMIT_UNKNOWN = "COMMIT_UNKNOWN"


def run_physical_step_and_publish[
    BodyProjection,
    EnvironmentProjection,
    BodyCandidate,
    EnvironmentCandidate,
](
    world: DurableWorldRuntime,
    plane: ObservationPlaneLike,
    *,
    commit_id: str,
    physical_step_id: str,
    body_id: str,
    embodiment_episode_id: str,
    morphology_revision: str,
    environment_revision: str,
    solver_config_revision: str,
    ports: PhysicalStepPorts[
        BodyProjection,
        EnvironmentProjection,
        BodyCandidate,
        EnvironmentCandidate,
    ],
    project_owner_observations: Callable[
        [PhysicalStepContext, CommittedWorldCut], Iterable[OwnerObservation]
    ],
) -> PhysicalStepPublicationResult:
    """Run owner adoption, commit the cut, then publish owner projections.

    The projection callback is invoked only after ``run_physical_step`` returns
    successfully and World verifies the exact committed owner references. It
    should call domain-owned projection and Observation mapping APIs; this
    composition layer does not construct or modify semantic claims.

    This function never retries or recovers a transition. Its result is a
    classification of the available in-process evidence; callers must stop
    on ``COMMIT_UNKNOWN`` or when ``recovery_required`` is true. A committed
    cut with incomplete publication remains authoritative and is not stepped
    again here.
    """

    with world.committed_cut_publication_guard():
        expected_source = world.manifest
        issued_contexts: list[PhysicalStepContext] = []
        written_references: list[CheckpointReference] = []

        def write_and_capture(step_context: PhysicalStepContext):
            issued_contexts.append(step_context)
            references = tuple(ports.write_checkpoints(step_context))
            written_references[:] = references
            return references

        execution_ports = replace(ports, write_checkpoints=write_and_capture)
        try:
            record = run_physical_step(
                world,
                commit_id=commit_id,
                physical_step_id=physical_step_id,
                body_id=body_id,
                embodiment_episode_id=embodiment_episode_id,
                morphology_revision=morphology_revision,
                environment_revision=environment_revision,
                solver_config_revision=solver_config_revision,
                ports=execution_ports,
            )
        except Exception as exc:
            inspection = world.inspect_transition_commit(
                transition_id=commit_id,
                expected_source=expected_source,
            )
            failure = f"{type(exc).__name__}: {exc}"
            if inspection.state is TransitionCommitState.NOT_COMMITTED:
                return PhysicalStepPublicationResult(
                    PhysicalStepOutcome.NOT_COMMITTED,
                    failure=failure,
                    recovery_required=world.status.value == "RECOVERY_REQUIRED",
                )
            if inspection.state is TransitionCommitState.COMMITTED:
                return PhysicalStepPublicationResult(
                    PhysicalStepOutcome.COMMITTED_PUBLICATION_INCOMPLETE,
                    committed_cut=inspection.committed_cut,
                    failure=failure,
                )
            return PhysicalStepPublicationResult(
                PhysicalStepOutcome.COMMIT_UNKNOWN,
                failure=failure,
                recovery_required=True,
            )
        if len(issued_contexts) != 1:
            inspection = world.inspect_transition_commit(
                transition_id=commit_id,
                expected_source=expected_source,
            )
            if inspection.state is not TransitionCommitState.COMMITTED:
                return PhysicalStepPublicationResult(
                    PhysicalStepOutcome.COMMIT_UNKNOWN,
                    causal_record=record,
                    failure="successful physical step issued an unexpected number of contexts",
                    recovery_required=True,
                )
            return PhysicalStepPublicationResult(
                PhysicalStepOutcome.COMMITTED_PUBLICATION_INCOMPLETE,
                causal_record=record,
                committed_cut=inspection.committed_cut,
                failure="successful physical step issued an unexpected number of contexts",
            )
        context = issued_contexts[0]
        if (
            context.world_id != record.world_id
            or context.target_causal_tick != record.causal_tick
            or context.physical_step_id != physical_step_id
        ):
            inspection = world.inspect_transition_commit(
                transition_id=commit_id,
                expected_source=expected_source,
            )
            if inspection.state is not TransitionCommitState.COMMITTED:
                return PhysicalStepPublicationResult(
                    PhysicalStepOutcome.COMMIT_UNKNOWN,
                    causal_record=record,
                    failure="committed physical record differs from its issued context",
                    recovery_required=True,
                )
            return PhysicalStepPublicationResult(
                PhysicalStepOutcome.COMMITTED_PUBLICATION_INCOMPLETE,
                causal_record=record,
                committed_cut=inspection.committed_cut,
                failure="committed physical record differs from its issued context",
            )
        try:
            cut = world.committed_cut_for(
                world_id=record.world_id,
                causal_tick=record.causal_tick,
                commit_id=commit_id,
                expected_owner_references=written_references,
            )
        except Exception as exc:
            inspection = world.inspect_transition_commit(
                transition_id=commit_id,
                expected_source=expected_source,
            )
            if inspection.state is TransitionCommitState.COMMITTED:
                return PhysicalStepPublicationResult(
                    PhysicalStepOutcome.COMMITTED_PUBLICATION_INCOMPLETE,
                    causal_record=record,
                    committed_cut=inspection.committed_cut,
                    failure=f"{type(exc).__name__}: {exc}",
                )
            return PhysicalStepPublicationResult(
                PhysicalStepOutcome.COMMIT_UNKNOWN,
                causal_record=record,
                failure=f"{type(exc).__name__}: {exc}",
                recovery_required=True,
            )

        try:
            owner_observations = tuple(project_owner_observations(context, cut))
        except Exception as exc:
            publication = ObservationPublication(
                committed_cut=cut,
                delivery_sequences=(),
                complete=False,
                failure=f"{type(exc).__name__}: {exc}",
            )
            return PhysicalStepPublicationResult(
                PhysicalStepOutcome.COMMITTED_PUBLICATION_INCOMPLETE,
                record,
                cut,
                publication,
                publication.failure,
            )
        try:
            publication = publish_committed_owner_observations(
                world,
                plane,
                committed_cut=cut,
                owner_observations=owner_observations,
            )
        except ObservationPublicationFailure as exc:
            publication = ObservationPublication(
                committed_cut=exc.committed_cut,
                delivery_sequences=exc.delivery_sequences,
                complete=False,
                failed_index=exc.failed_index,
                delivery_status_uncertain=exc.delivery_status_uncertain,
                failure=exc.failure,
            )
            return PhysicalStepPublicationResult(
                PhysicalStepOutcome.COMMITTED_PUBLICATION_INCOMPLETE,
                record,
                cut,
                publication,
                publication.failure,
            )
        except Exception as exc:
            publication = ObservationPublication(
                committed_cut=cut,
                delivery_sequences=(),
                complete=False,
                failure=f"{type(exc).__name__}: {exc}",
            )
            return PhysicalStepPublicationResult(
                PhysicalStepOutcome.COMMITTED_PUBLICATION_INCOMPLETE,
                record,
                cut,
                publication,
                publication.failure,
            )
        return PhysicalStepPublicationResult(
            PhysicalStepOutcome.COMMITTED_AND_PUBLISHED,
            record,
            cut,
            publication,
        )


__all__ = [
    "PhysicalStepOutcome",
    "PhysicalStepPublicationResult",
    "run_physical_step_and_publish",
]
