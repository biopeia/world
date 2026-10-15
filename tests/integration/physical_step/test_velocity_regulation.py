from __future__ import annotations

import pytest
import mujoco

from world.adapters.articulated_mujoco import _mujoco_trial, _select_compliant_torque


def test_outward_request_at_velocity_limit_is_regulated() -> None:
    applied, velocity = _select_compliant_torque(
        1.0, 2.0, 1.0, lambda torque: 0.9 + torque * 0.2
    )
    assert applied <= 0.5
    assert abs(velocity) <= 1.0


def test_infeasible_velocity_bound_fails_closed() -> None:
    with pytest.raises(ValueError, match="no tested applied torque"):
        _select_compliant_torque(0.0, 2.0, 1.0, lambda _torque: 1.01)


def test_search_is_deterministic() -> None:
    trial = lambda torque: 1.1 - torque * 0.2
    first = _select_compliant_torque(1.0, 2.0, 1.0, trial)
    second = _select_compliant_torque(1.0, 2.0, 1.0, trial)
    assert first == second


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
