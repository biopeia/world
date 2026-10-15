from __future__ import annotations

import pytest
import mujoco

from world.adapters.articulated_mujoco import _mujoco_trial, _select_compliant_torque


def test_outward_request_is_regulated_when_position_would_cross_limit() -> None:
    applied, state = _select_compliant_torque(
        1.0, 2.0, -1.0, 1.0, 2.0,
        lambda torque: (0.9 + torque * 0.2, 0.5 + torque * 0.2),
    )
    assert applied <= 0.5 + 1e-9
    assert -1.0 <= state[0] <= 1.0
    assert abs(state[1]) <= 2.0


def test_infeasible_velocity_bound_fails_closed() -> None:
    with pytest.raises(ValueError, match="no tested applied torque"):
        _select_compliant_torque(
            0.0, 2.0, -1.0, 1.0, 2.0, lambda _torque: (0.5, 2.01)
        )


def test_infeasible_position_bound_fails_closed_independently() -> None:
    with pytest.raises(ValueError, match="no tested applied torque"):
        _select_compliant_torque(
            0.0, 2.0, -1.0, 1.0, 2.0, lambda _torque: (1.01, 0.5)
        )


def test_search_is_deterministic() -> None:
    trial = lambda torque: (1.1 - torque * 0.2, 1.1 - torque * 0.2)
    first = _select_compliant_torque(1.0, 2.0, -1.0, 1.0, 1.0, trial)
    second = _select_compliant_torque(1.0, 2.0, -1.0, 1.0, 1.0, trial)
    assert first == second


@pytest.mark.parametrize(("position", "velocity", "torque"), [
    (0.98, 1.9, 1.0),
    (-0.98, -1.9, -1.0),
])
def test_repeated_boundary_transitions_remain_position_and_velocity_compliant(
    position: float, velocity: float, torque: float,
) -> None:
    # The trial reads the current state on every iteration: unlike a fixed
    # lambda response, this exercises a sequence of state-dependent selections.
    # This remains a selector regression, not physical-efficacy evidence.
    states = []
    for _ in range(50):
        source_position, source_velocity = position, velocity

        def trial(request: float) -> tuple[float, float]:
            return (
                source_position + source_velocity * 0.01 + request * 0.01,
                source_velocity + request * 0.5,
            )

        applied, candidate = _select_compliant_torque(
            torque, 2.0, -1.0, 1.0, 2.0,
            trial,
        )
        assert -2.0 <= applied <= 2.0
        assert -1.0 <= candidate[0] <= 1.0
        assert abs(candidate[1]) <= 2.0
        position, velocity = candidate
        states.append(candidate)

    assert len(set(states)) > 1


def test_trial_search_does_not_mutate_source_state() -> None:
    model = mujoco.MjModel.from_xml_string("""
        <mujoco><worldbody><body name="arm">
        <joint name="hinge" type="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0 0 1" size="0.1" mass="1"/>
        </body></worldbody></mujoco>
    """)
    source = mujoco.MjData(model)
    source.qpos[:] = (0.1,)
    source.qvel[:] = (0.2,)
    source.qfrc_applied[:] = (0.3,)
    before = (source.qpos.copy(), source.qvel.copy(), source.qfrc_applied.copy())
    trial = _mujoco_trial(
        mujoco, model, source.qpos.copy(), source.qvel.copy(),
        source.qfrc_applied.copy(), 0, -0.5,
    )
    assert trial is not None
    assert trial is not source
    assert source.qpos.tolist() == before[0].tolist()
    assert source.qvel.tolist() == before[1].tolist()
    assert source.qfrc_applied.tolist() == before[2].tolist()
