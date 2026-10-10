"""The World core and adapter import surface do not eagerly require MuJoCo."""

from __future__ import annotations

import importlib
import sys


def test_adapter_module_import_does_not_import_optional_engine(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "mujoco", None)

    module = importlib.import_module("world.adapters.mujoco")

    assert callable(module.execute_mujoco_step)
