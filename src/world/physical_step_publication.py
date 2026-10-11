"""Production composition for physical commit followed by passive publication."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace

from .durable_runtime import CommittedWorldCut, DurableWorldRuntime
from .journal import CausalRecord
from .manifest import CheckpointReference
from .observation_publication import (
    ObservationPlaneLike,
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
    """Committed physical transition and its verified Observation delivery."""

    causal_record: CausalRecord
    committed_cut: CommittedWorldCut
    publication: ObservationPublication


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
    """

    with world.committed_cut_publication_guard():
        issued_contexts: list[PhysicalStepContext] = []
        written_references: list[CheckpointReference] = []

        def write_and_capture(step_context: PhysicalStepContext):
            issued_contexts.append(step_context)
            references = tuple(ports.write_checkpoints(step_context))
            written_references[:] = references
            return references

        execution_ports = replace(ports, write_checkpoints=write_and_capture)
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
        if len(issued_contexts) != 1:
            raise RuntimeError(
                "a successful physical step must issue exactly one context"
            )
        context = issued_contexts[0]
        if (
            context.world_id != record.world_id
            or context.target_causal_tick != record.causal_tick
            or context.physical_step_id != physical_step_id
        ):
            raise RuntimeError(
                "committed physical record differs from its issued context"
            )
        cut = world.committed_cut_for(
            world_id=record.world_id,
            causal_tick=record.causal_tick,
            commit_id=commit_id,
            expected_owner_references=written_references,
        )
        owner_observations = tuple(project_owner_observations(context, cut))
        publication = publish_committed_owner_observations(
            world,
            plane,
            committed_cut=cut,
            owner_observations=owner_observations,
        )
        return PhysicalStepPublicationResult(record, cut, publication)


__all__ = ["PhysicalStepPublicationResult", "run_physical_step_and_publish"]
