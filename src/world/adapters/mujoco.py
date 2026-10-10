"""An optional, engine-specific MuJoCo single-step executor.

The caller supplies an already compiled model and explicit generalized state
and control vectors. This module deliberately does not define a Biopeia scene
schema, assemble owner projections, derive units or identity from native
vector positions, or mutate owner state. Callers must resolve domain units,
stable body/actuator correspondence, and omitted-command semantics before this
boundary. The returned values are only a solver candidate; owner
validation/adoption and World commit remain separate operations.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Sequence

from ..physical_step import PhysicalStepContext


class MujocoExecutionError(ValueError):
    """Raised when explicit MuJoCo inputs are incompatible or invalid."""


@dataclass(frozen=True, slots=True)
class MujocoStepCandidate:
    """Immutable numerical result carrying the supplied step context."""

    context: PhysicalStepContext
    qpos: tuple[float, ...]
    qvel: tuple[float, ...]


def _finite_vector(name: str, values: Sequence[float]) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise MujocoExecutionError(f"{name} must be a numeric sequence")
    try:
        result = tuple(values)
    except TypeError as exc:
        raise MujocoExecutionError(f"{name} must be a numeric sequence") from exc
    for index, value in enumerate(result):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise MujocoExecutionError(f"{name}[{index}] must be numeric")
        if not math.isfinite(value):
            raise MujocoExecutionError(f"{name}[{index}] must be finite")
    return tuple(float(value) for value in result)


def execute_mujoco_step(
    context: PhysicalStepContext,
    *,
    model: object,
    qpos: Sequence[float],
    qvel: Sequence[float],
    ctrl: Sequence[float],
) -> MujocoStepCandidate:
    """Run exactly one MuJoCo step from explicit state and control vectors.

    ``model`` must already have been compiled from caller-supplied projections
    and declared solver configuration. This executor can verify only the
    compiled model's effective timestep, not whether its author explicitly
    declared it or whether the configuration revision is truthful. This
    executor does not infer an absent control vector: callers must provide
    ``ctrl`` explicitly, including an empty vector for a model with no
    actuators. Unsupported Biopeia projection semantics are the caller's
    responsibility to reject before this boundary. The context is carried
    for correlation only: this function checks its type but does not prove
    that World issued it or authorize the transition.

    MuJoCo is an optional dependency. Importing :mod:`world` or this module
    does not require it; invoking this function does.
    """

    if not isinstance(context, PhysicalStepContext):
        raise MujocoExecutionError("context must be a PhysicalStepContext")

    try:
        import mujoco
    except ImportError as exc:  # pragma: no cover - exercised without the extra
        raise RuntimeError(
            "MuJoCo execution requires the optional 'mujoco' dependency"
        ) from exc

    if not isinstance(model, mujoco.MjModel):
        raise MujocoExecutionError("model must be a compiled mujoco.MjModel")
    if not math.isfinite(float(model.opt.timestep)) or model.opt.timestep <= 0:
        raise MujocoExecutionError("compiled model must declare a positive timestep")

    positions = _finite_vector("qpos", qpos)
    velocities = _finite_vector("qvel", qvel)
    controls = _finite_vector("ctrl", ctrl)
    if len(positions) != model.nq:
        raise MujocoExecutionError(f"qpos must contain exactly {model.nq} values")
    if len(velocities) != model.nv:
        raise MujocoExecutionError(f"qvel must contain exactly {model.nv} values")
    if len(controls) != model.nu:
        raise MujocoExecutionError(f"ctrl must contain exactly {model.nu} values")

    data = mujoco.MjData(model)
    data.qpos[:] = positions
    data.qvel[:] = velocities
    data.ctrl[:] = controls
    mujoco.mj_step(model, data)

    return MujocoStepCandidate(
        context=context,
        qpos=tuple(float(value) for value in data.qpos),
        qvel=tuple(float(value) for value in data.qvel),
    )

