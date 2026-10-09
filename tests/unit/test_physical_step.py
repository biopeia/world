from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pytest

from world.physical_step import (
    PhysicalStepCandidate,
    PhysicalStepContext,
    PhysicalStepPorts,
    run_physical_step,
)


@dataclass
class _World:
    world_id: object
    causal_tick: int = 4
    committed: bool = False
    status: str = "ACTIVE"

    def tick(self, commit_id: str, transition: Callable):
        assert commit_id == "commit-5"
        try:
            refs = tuple(transition(self.causal_tick + 1))
        except Exception:
            self.status = "RECOVERY_REQUIRED"
            raise
        self.committed = True
        self.causal_tick += 1
        return refs


@dataclass(frozen=True)
class _WorldId:
    value: str = "world-1"


def _ports(events: list[tuple[str, PhysicalStepContext]]) -> PhysicalStepPorts:
    def event(name: str):
        return lambda context: events.append((name, context))

    return PhysicalStepPorts(
        project_body=lambda context: (
            events.append(("body_projection", context)) or "body"
        ),
        project_environment=lambda context: (
            events.append(("environment_projection", context)) or "environment"
        ),
        execute_solver=lambda context, body, environment: (
            events.append(("solver", context, body, environment))
            or PhysicalStepCandidate("body-result", "environment-result")
        ),
        adopt_body=lambda context, candidate: events.append(
            ("body_adoption", context, candidate)
        ),
        adopt_environment=lambda context, candidate: events.append(
            ("environment_adoption", context, candidate)
        ),
        write_checkpoints=lambda context: events.append(("checkpoints", context)) or (),
    )


def test_coordinator_issues_source_and_target_ticks_and_runs_owner_adoption_before_checkpoint():
    world = _World(_WorldId())
    events: list[tuple] = []

    result = run_physical_step(
        world,  # type: ignore[arg-type]
        commit_id="commit-5",
        physical_step_id="physics-5",
        body_id="body-1",
        embodiment_episode_id="episode-1",
        morphology_revision="morphology-r2",
        environment_revision="environment-r3",
        solver_config_revision="solver-r4",
        ports=_ports(events),
    )

    assert result == ()
    assert world.committed is True
    assert world.causal_tick == 5
    assert [event[0] for event in events] == [
        "body_projection",
        "environment_projection",
        "solver",
        "body_adoption",
        "environment_adoption",
        "checkpoints",
    ]
    contexts = [event[1] for event in events]
    assert all(context is contexts[0] for context in contexts)
    context = contexts[0]
    assert isinstance(context, PhysicalStepContext)
    assert (context.source_causal_tick, context.target_causal_tick) == (4, 5)
    assert context.world_id == "world-1"
    assert context.physical_step_id == "physics-5"


@pytest.mark.parametrize(
    "source,target",
    [(-1, 0), (2, 2), (2, 4)],
)
def test_context_rejects_invalid_tick_relationship(source: int, target: int) -> None:
    with pytest.raises(ValueError, match="target tick"):
        PhysicalStepContext(
            "world-1",
            source,
            target,
            "step-1",
            "body-1",
            "episode-1",
            "morph-r1",
            "env-r1",
            "solver-r1",
        )


def test_context_rejects_empty_correlation_identity() -> None:
    with pytest.raises(ValueError, match="must be non-empty"):
        PhysicalStepContext(
            "world-1",
            0,
            1,
            "step-1",
            "",
            "episode-1",
            "morph-r1",
            "env-r1",
            "solver-r1",
        )


@pytest.mark.parametrize(
    "field,value",
    [("world_id", 1), ("physical_step_id", None), ("body_id", object())],
)
def test_context_rejects_non_string_identity(field: str, value: object) -> None:
    values = {
        "world_id": "world-1",
        "source_causal_tick": 0,
        "target_causal_tick": 1,
        "physical_step_id": "step-1",
        "body_id": "body-1",
        "embodiment_episode_id": "episode-1",
        "morphology_revision": "morph-r1",
        "environment_revision": "env-r1",
        "solver_config_revision": "solver-r1",
    }
    values[field] = value
    with pytest.raises(ValueError, match="must be non-empty"):
        PhysicalStepContext(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize("field", ["source_causal_tick", "target_causal_tick"])
@pytest.mark.parametrize("value", [True, 1.0, "1"])
def test_context_rejects_non_integer_ticks(field: str, value: object) -> None:
    values: dict[str, object] = {
        "world_id": "world-1",
        "source_causal_tick": 0,
        "target_causal_tick": 1,
        "physical_step_id": "step-1",
        "body_id": "body-1",
        "embodiment_episode_id": "episode-1",
        "morphology_revision": "morph-r1",
        "environment_revision": "env-r1",
        "solver_config_revision": "solver-r1",
    }
    values[field] = value
    with pytest.raises(ValueError, match="ticks must be integers"):
        PhysicalStepContext(**values)  # type: ignore[arg-type]


def test_solver_failure_prevents_commit_and_propagates_recovery_requirement() -> None:
    world = _World(_WorldId())
    events: list[tuple] = []
    ports = _ports(events)

    def fail_solver(context, body, environment):
        events.append(("solver_failed", context))
        raise RuntimeError("solver failed")

    ports = PhysicalStepPorts(
        project_body=ports.project_body,
        project_environment=ports.project_environment,
        execute_solver=fail_solver,
        adopt_body=ports.adopt_body,
        adopt_environment=ports.adopt_environment,
        write_checkpoints=ports.write_checkpoints,
    )
    with pytest.raises(RuntimeError, match="solver failed"):
        run_physical_step(
            world,  # type: ignore[arg-type]
            commit_id="commit-5",
            physical_step_id="physics-5",
            body_id="body-1",
            embodiment_episode_id="episode-1",
            morphology_revision="morphology-r2",
            environment_revision="environment-r3",
            solver_config_revision="solver-r4",
            ports=ports,
        )

    assert world.causal_tick == 4
    assert world.committed is False
    assert world.status == "RECOVERY_REQUIRED"
    assert [event[0] for event in events] == [
        "body_projection",
        "environment_projection",
        "solver_failed",
    ]
