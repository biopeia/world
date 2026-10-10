"""Contract tests for the optional, low-level MuJoCo executor."""

from __future__ import annotations

import math

import mujoco
import pytest

from world.adapters.mujoco import MujocoExecutionError, execute_mujoco_step
from world.physical_step import PhysicalStepContext


def _context() -> PhysicalStepContext:
    return PhysicalStepContext(
        world_id="adapter-test-world",
        source_causal_tick=7,
        target_causal_tick=8,
        physical_step_id="adapter-test-step-8",
        body_id="body.adapter-test",
        embodiment_episode_id="episode.adapter-test",
        morphology_revision="morphology-r1",
        environment_revision="environment-r1",
        solver_config_revision=f"mujoco-{mujoco.__version__}-euler-dt0.01-r1",
    )


def _model() -> mujoco.MjModel:
    # This is an explicit test projection, not a Biopeia Environment format.
    return mujoco.MjModel.from_xml_string(
        '<mujoco><option timestep="0.01" gravity="0 0 -9.81"/>'
        '<worldbody><body pos="0 0 1"><freejoint/>'
        '<geom type="sphere" size="0.1" mass="1"/>'
        '</body></worldbody></mujoco>'
    )


def _run(model: mujoco.MjModel):
    return execute_mujoco_step(
        _context(),
        model=model,
        qpos=(0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0),
        qvel=(0.0,) * 6,
        ctrl=(),
    )


def test_executor_returns_deterministic_candidate_with_supplied_step_context() -> None:
    first = _run(_model())
    second = _run(_model())

    assert first.context == _context()
    assert first.context.target_causal_tick == 8
    assert first.qpos == second.qpos
    assert first.qvel == second.qvel
    assert first.qvel[2] < 0
    assert all(math.isfinite(value) for value in (*first.qpos, *first.qvel))


@pytest.mark.parametrize(
    ("qpos", "qvel", "ctrl", "message"),
    [
        ((0.0,), (0.0,) * 6, (), "qpos"),
        ((0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0), (0.0,), (), "qvel"),
        ((0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0), (0.0,) * 6, (0.0,), "ctrl"),
        ((0.0, 0.0, 1.0, 1.0, 0.0, 0.0, float("nan")), (0.0,) * 6, (), "finite"),
        ((0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0), (False,) + (0.0,) * 5, (), "numeric"),
    ],
)
def test_executor_rejects_invalid_explicit_vectors(qpos, qvel, ctrl, message) -> None:
    with pytest.raises(MujocoExecutionError, match=message):
        execute_mujoco_step(
            _context(), model=_model(), qpos=qpos, qvel=qvel, ctrl=ctrl
        )


def test_executor_requires_explicit_control_vector() -> None:
    with pytest.raises(TypeError, match="ctrl"):
        execute_mujoco_step(
            _context(),
            model=_model(),
            qpos=(0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0),
            qvel=(0.0,) * 6,
        )


def test_executor_rejects_non_mujoco_model() -> None:
    with pytest.raises(MujocoExecutionError, match="compiled mujoco.MjModel"):
        execute_mujoco_step(
            _context(), model=object(), qpos=(), qvel=(), ctrl=()
        )


def test_executor_rejects_untyped_correlation_context() -> None:
    with pytest.raises(MujocoExecutionError, match="PhysicalStepContext"):
        execute_mujoco_step(
            object(), model=_model(), qpos=(0.0,) * 7, qvel=(0.0,) * 6, ctrl=()
        )


def test_executor_reports_missing_optional_dependency(monkeypatch) -> None:
    import sys

    monkeypatch.setitem(sys.modules, "mujoco", None)
    with pytest.raises(RuntimeError, match="optional 'mujoco' dependency"):
        execute_mujoco_step(
            _context(), model=object(), qpos=(), qvel=(), ctrl=()
        )

