"""Protocol-driven coordination for an owner-adopted physical World step.

World supplies causal identity and publishes the cut; owner adapters retain
projection, candidate-validation, adoption, and checkpoint semantics. The
numerical solver remains an injected capability with no semantic authority.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from .durable_runtime import DurableWorldRuntime
from .journal import CausalRecord
from .manifest import CheckpointReference


@dataclass(frozen=True, slots=True)
class PhysicalStepContext:
    """World-issued identity for one physical transition.

    ``source_causal_tick`` names the state projected into the solver;
    ``target_causal_tick`` names the World cut that may be committed after
    owner adoption. They are deliberately explicit and differ by one.
    """

    world_id: str
    source_causal_tick: int
    target_causal_tick: int
    physical_step_id: str
    body_id: str
    embodiment_episode_id: str
    morphology_revision: str
    environment_revision: str
    solver_config_revision: str

    def __post_init__(self) -> None:
        identities = (
                self.world_id,
                self.physical_step_id,
                self.body_id,
                self.embodiment_episode_id,
                self.morphology_revision,
                self.environment_revision,
                self.solver_config_revision,
        )
        if any(type(value) is not str or not value.strip() for value in identities):
            raise ValueError("physical-step identity and revisions must be non-empty")
        if (
            type(self.source_causal_tick) is not int
            or type(self.target_causal_tick) is not int
        ):
            raise ValueError("physical-step ticks must be integers")
        if (
            self.source_causal_tick < 0
            or self.target_causal_tick != self.source_causal_tick + 1
        ):
            raise ValueError(
                "physical-step target tick must immediately follow its source tick"
            )


@dataclass(frozen=True, slots=True)
class PhysicalStepCandidate[BodyCandidate, EnvironmentCandidate]:
    """Opaque numerical result split by the semantic owner that may adopt it."""

    body: BodyCandidate
    environment: EnvironmentCandidate
    # Optional World-owned solver evidence; semantic owner state remains
    # exclusively in body/environment candidates.
    evidence: dict[str, object] | None = None


@dataclass(frozen=True, slots=True)
class PhysicalStepPorts[
    BodyProjection,
    EnvironmentProjection,
    BodyCandidate,
    EnvironmentCandidate,
]:
    """Domain adapters and solver injected into the World coordinator."""

    project_body: Callable[[PhysicalStepContext], BodyProjection]
    project_environment: Callable[[PhysicalStepContext], EnvironmentProjection]
    execute_solver: Callable[
        [PhysicalStepContext, BodyProjection, EnvironmentProjection],
        PhysicalStepCandidate[BodyCandidate, EnvironmentCandidate],
    ]
    adopt_body: Callable[[PhysicalStepContext, BodyCandidate], None]
    adopt_environment: Callable[[PhysicalStepContext, EnvironmentCandidate], None]
    write_checkpoints: Callable[[PhysicalStepContext], Iterable[CheckpointReference]]


def run_physical_step[
    BodyProjection,
    EnvironmentProjection,
    BodyCandidate,
    EnvironmentCandidate,
](
    world: DurableWorldRuntime,
    *,
    commit_id: str,
    physical_step_id: str,
    body_id: str,
    embodiment_episode_id: str,
    morphology_revision: str,
    environment_revision: str,
    solver_config_revision: str,
    ports: PhysicalStepPorts[
        BodyProjection, EnvironmentProjection, BodyCandidate, EnvironmentCandidate
    ],
) -> CausalRecord:
    """Project, solve, adopt and publish one durable World physical cut.

    Every owner operation and solver execution is correlated to the explicit
    source/target ticks and identity bundle in ``PhysicalStepContext``. The
    owner adapters must reject stale or incompatible candidates. Exceptions
    propagate through ``DurableWorldRuntime.tick``, which prevents publication
    of the cut and latches recovery-required status. This coordinator does not
    roll back in-memory owner mutations; recovery restores owners from the last
    committed cut using their checkpoint contracts.
    """

    def transition(target_tick: int) -> Iterable[CheckpointReference]:
        source_tick = world.causal_tick
        if target_tick != source_tick + 1:
            raise RuntimeError(
                "World tick changed before physical-step context was issued"
            )
        context = PhysicalStepContext(
            world_id=world.world_id.value,
            source_causal_tick=source_tick,
            target_causal_tick=target_tick,
            physical_step_id=physical_step_id,
            body_id=body_id,
            embodiment_episode_id=embodiment_episode_id,
            morphology_revision=morphology_revision,
            environment_revision=environment_revision,
            solver_config_revision=solver_config_revision,
        )
        body_projection = ports.project_body(context)
        environment_projection = ports.project_environment(context)
        candidate = ports.execute_solver(
            context, body_projection, environment_projection
        )
        if not isinstance(candidate, PhysicalStepCandidate):
            raise TypeError("physical solver must return a PhysicalStepCandidate")
        ports.adopt_body(context, candidate.body)
        ports.adopt_environment(context, candidate.environment)
        return tuple(ports.write_checkpoints(context))

    return world.tick(commit_id, transition)
