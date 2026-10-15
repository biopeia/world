"""Optional numerical join for owner-projected articulated bodies/static scenes.

This adapter has no authoritative state. Every call rebuilds the numerical
scene from owner projections; direct joint forces have no actuator activation
state and warm-starting is disabled. Owner adoption and durable publication
remain the responsibility of ``run_physical_step`` and its owner ports.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from embodiment.physics3d.articulated_state import (
    ActiveArticulatedMorphology,
    ArticulatedEmbodimentPhysicalState,
    JointKinematicState,
    SolverArticulatedCandidate,
)
from embodiment.physics3d.differential_motor_composition import (
    EmbodimentDifferentialMotorBinding,
    compose_differential_motor_command,
)
from embodiment.physics3d.joint_torque import (
    EmbodimentJointTorqueBinding,
    JointTorqueCandidate,
    project_joint_torque_command,
)
from embodiment.physics3d.physical_projection import (
    BodyPhysicalInput,
    project_body_physical_input,
)
from embodiment.physics3d.physical_transition import (
    PhysicalStepCorrelation,
    SolverBodyCandidate,
)
from environment.static_scene import (
    EnvironmentPhysicalProjection,
    StaticEnvironmentOwner,
    StaticSceneCandidate,
)
from environment.static_scene import (
    PhysicalStepCorrelation as EnvironmentStepCorrelation,
)
from organism.actuation import MotorCommand

from world.durable_runtime import DurableWorldRuntime
from world.journal import CausalRecord
from world.manifest import CheckpointReference
from world.physical_step import (
    PhysicalStepCandidate,
    PhysicalStepContext,
    PhysicalStepPorts,
    run_physical_step,
)


def _select_compliant_torque(requested: float, effort: float, cap: float, trial):
    """Choose nearest compliant value from a deterministic bounded grid."""
    requested_velocity = trial(requested)
    if requested_velocity is not None and abs(requested_velocity) <= cap:
        return requested, requested_velocity
    candidates = [requested]
    candidates.extend(-effort + (2 * effort * i / 1000) for i in range(1001))
    feasible = []
    for applied in candidates:
        velocity = trial(applied)
        if velocity is not None and abs(velocity) <= cap:
            feasible.append((abs(applied - requested), applied, velocity))
    if not feasible:
        raise ValueError("no tested applied torque produces a velocity-compliant candidate")
    _, applied, velocity = min(feasible, key=lambda item: (item[0], item[1]))
    return applied, velocity


def _mujoco_trial(mujoco, model, qpos, qvel, qfrc, dof: int, applied: float):
    """Run one trial on a fresh data object, isolated from the source state."""
    import numpy as np

    trial_data = mujoco.MjData(model)
    trial_data.qpos[:] = qpos
    trial_data.qvel[:] = qvel
    trial_data.qfrc_applied[:] = qfrc
    trial_data.qfrc_applied[dof] = applied
    mujoco.mj_step(model, trial_data)
    if not np.all(np.isfinite(trial_data.qpos)) or not np.all(np.isfinite(trial_data.qvel)) or any(w.number for w in trial_data.warning):
        return None
    return trial_data


def advance_articulated_mujoco_step_from_motor_command(
    world: DurableWorldRuntime,
    *,
    commit_id: str,
    physical_step_id: str,
    body_state: ArticulatedEmbodimentPhysicalState,
    motor_command: MotorCommand,
    active_morphology: ActiveArticulatedMorphology,
    differential_binding: EmbodimentDifferentialMotorBinding,
    torque_binding: EmbodimentJointTorqueBinding,
    environment_owner: StaticEnvironmentOwner,
    configuration: ArticulatedMujocoConfiguration,
    write_checkpoints: Callable[
        [PhysicalStepContext, ArticulatedEmbodimentPhysicalState, StaticEnvironmentOwner],
        Iterable[CheckpointReference],
    ],
) -> ArticulatedPhysicalStepOutcome:
    """Compose a TARGET MotorCommand through Embodiment and commit one step.

    This production join preserves stable actuator identity and the explicit
    Embodiment-owned differential mapping. It does not infer units, issue a
    command for omitted channels, or make the solver authoritative.
    """
    context = (
        body_state.root.body_id,
        body_state.root.embodiment_episode_id,
        body_state.morphology_revision,
    )
    morphology_context = (
        active_morphology.body_id,
        active_morphology.embodiment_episode_id,
        active_morphology.morphology_revision,
    )
    if context != morphology_context:
        raise ValueError("active morphology does not match the Embodiment body state")
    if body_state.morphology.morphology_sha256 != active_morphology.morphology_sha256:
        raise ValueError("active morphology content differs from the Embodiment body state")
    if motor_command.issued_at_tick != body_state.causal_tick:
        raise ValueError("motor command does not match the World source causal tick")
    request = compose_differential_motor_command(
        motor_command,
        differential_binding,
        torque_binding,
        active_morphology,
    )
    torque = project_joint_torque_command(request, torque_binding, active_morphology)
    return advance_articulated_mujoco_step(
        world,
        commit_id=commit_id,
        physical_step_id=physical_step_id,
        body_state=body_state,
        torque=torque,
        environment_owner=environment_owner,
        configuration=configuration,
        write_checkpoints=write_checkpoints,
    )


@dataclass(frozen=True, slots=True)
class ArticulatedMujocoConfiguration:
    """Explicit numerical policy, not a calibrated physical/material model.

    Euler integration, Newton constraints, elliptic friction cones, no body
    self-collision, and stateless direct torque are this revision's supported
    policies. Pair sliding friction is the maximum of the two owner values;
    torsional/rolling friction comes from the body descriptor. All contact and
    joint-limit compliance parameters are explicit MuJoCo numerical inputs.
    """

    revision: str
    timestep_s: float
    iterations: int
    tolerance: float
    solref: tuple[float, float]
    solimp: tuple[float, float, float, float, float]
    contact_policy: str
    velocity_regulation_policy: str = "none"

    def __post_init__(self) -> None:
        if not isinstance(self.revision, str) or not self.revision.strip():
            raise ValueError("solver configuration revision is required")
        if type(self.iterations) is not int or self.iterations < 1:
            raise ValueError("iterations must be a positive integer")
        for value in (self.timestep_s, self.tolerance, *self.solref, *self.solimp):
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
                raise ValueError("solver parameters must be finite numbers")
        if self.timestep_s <= 0 or self.tolerance <= 0:
            raise ValueError("timestep and tolerance must be positive")
        if len(self.solref) != 2 or self.solref[0] < 2 * self.timestep_s or self.solref[1] <= 0:
            raise ValueError("solref requires time constant >= 2 timesteps and positive damping")
        if len(self.solimp) != 5:
            raise ValueError("solimp requires five parameters")
        dmin, dmax, width, midpoint, power = self.solimp
        if not 0 < dmin <= dmax < 1 or width <= 0 or not 0 < midpoint < 1 or power < 1:
            raise ValueError("invalid explicit constraint impedance")
        if self.contact_policy != "max-sliding-body-torsion-rolling/no-self-collision-v1":
            raise ValueError("unsupported explicit contact policy")
        if self.velocity_regulation_policy not in {"none", "m1-fixed-single-hinge-grid-v1"}:
            raise ValueError("unsupported velocity regulation policy")


@dataclass(frozen=True, slots=True)
class ArticulatedPhysicalInput:
    body: BodyPhysicalInput
    torque: JointTorqueCandidate


def execute_articulated_mujoco_step(
    context: PhysicalStepContext,
    body_input: ArticulatedPhysicalInput,
    environment: EnvironmentPhysicalProjection,
    *,
    configuration: ArticulatedMujocoConfiguration,
) -> PhysicalStepCandidate[SolverArticulatedCandidate, StaticSceneCandidate]:
    """Solve one step; return unaccepted candidates, never mutate owner state.

    Supported morphology: owner-generated URDF with revolute joints and a
    single fixed or free root. Supported Environment: static planes/spheres in
    the body's named reference frame. Restitution and root drag are rejected
    rather than silently dropped. Missing torque channels remain absent.
    """
    import mujoco
    import numpy as np

    body = body_input.body
    torque = body_input.torque
    expected = PhysicalStepCorrelation(
        context.world_id, context.source_causal_tick, context.physical_step_id,
        context.body_id, context.embodiment_episode_id, context.morphology_revision,
        context.environment_revision, context.solver_config_revision,
    )
    if body.correlation != expected or configuration.revision != context.solver_config_revision:
        raise ValueError("body/configuration differs from World-issued context")
    ec = environment.correlation
    if (ec.world_id, ec.causal_tick, ec.physical_step_id, ec.environment_revision) != (
        context.world_id, context.source_causal_tick, context.physical_step_id, context.environment_revision,
    ):
        raise ValueError("Environment projection differs from World-issued context")
    scene = environment.scene
    if body.reference_frame != scene.frame.frame_id:
        raise ValueError("body and Environment must share an explicitly named reference frame")
    material = body.body_material
    if any(value < 0 for value in (material.lateral_friction, material.spinning_friction, material.rolling_friction)):
        raise ValueError("friction must be non-negative")
    if (material.restitution, material.linear_damping, material.angular_damping) != (0, 0, 0):
        raise ValueError("restitution/root drag require a separately supported physical model")
    if torque.issued_at_tick != context.source_causal_tick:
        raise ValueError("torque is stale for the World source cut")

    description = ET.fromstring(body.articulated_description)
    links = {item.attrib["name"] for item in description.findall("link")}
    topology = {item.attrib["name"]: item for item in description.findall("joint")}
    children = {item.find("child").attrib["link"] for item in topology.values()}
    roots = links - children
    if roots != {body.body_frame}:
        raise ValueError("owner body frame must identify the single URDF root")
    joint_ids = {item.joint_id for item in body.joints}
    revolute_ids = {name for name, joint in topology.items() if joint.attrib["type"] == "revolute"}
    if joint_ids != revolute_ids or any(j.attrib["type"] not in {"revolute", "fixed"} for j in topology.values()):
        raise ValueError("only exact owner-described revolute/fixed joint topology is supported")
    seen = set()
    for channel in torque.channels:
        if channel.joint_id in seen or channel.joint_id not in joint_ids:
            raise ValueError("duplicate or unknown stable joint command")
        seen.add(channel.joint_id)
        joint = topology[channel.joint_id]
        if (channel.body_id, channel.embodiment_episode_id, channel.morphology_revision) != (
            context.body_id, context.embodiment_episode_id, context.morphology_revision,
        ):
            raise ValueError("joint command targets a different body/episode/morphology")
        axis = tuple(float(x) for x in joint.find("axis").attrib["xyz"].split())
        effort = float(joint.find("limit").attrib["effort"])
        if (channel.joint_parent_link_id, channel.joint_child_link_id, channel.axis) != (
            joint.find("parent").attrib["link"], joint.find("child").attrib["link"], axis,
        ):
            raise ValueError("joint command binding differs from owner topology/axis")
        if (channel.quantity, channel.unit, channel.axis_frame_id, channel.positive_direction) != (
            "joint_torque", "N*m", f"joint-frame:{channel.joint_id}@{context.morphology_revision}",
            "right_hand_about_declared_axis",
        ):
            raise ValueError("joint torque quantity/unit/frame/direction is unsupported")
        if (channel.minimum_nm, channel.neutral_nm, channel.maximum_nm) != (-effort, 0, effort):
            raise ValueError("joint command range differs from owner effort descriptor")
        if isinstance(channel.torque_nm, bool) or not math.isfinite(channel.torque_nm) or not -effort <= channel.torque_nm <= effort:
            raise ValueError("joint torque is non-finite or out of range")

    spec = mujoco.MjSpec.from_string(body.articulated_description)
    spec.compiler.fusestatic = False
    spec.compiler.discardvisual = True
    spec.option.timestep = configuration.timestep_s
    spec.option.gravity = scene.gravity_m_s2
    spec.option.iterations = configuration.iterations
    spec.option.tolerance = configuration.tolerance
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_EULER
    spec.option.solver = mujoco.mjtSolver.mjSOL_NEWTON
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.disableflags = int(mujoco.mjtDisableBit.mjDSBL_WARMSTART)
    root = spec.body(body.body_frame)
    root.pos = body.position_m
    x, y, z, w = body.orientation_xyzw
    root.quat = (w, x, y, z)
    if body.fixed_base:
        if any(v != 0 for v in (*body.linear_velocity_m_per_s, *body.angular_velocity_rad_per_s)):
            raise ValueError("fixed root cannot have nonzero root velocity")
    else:
        root.add_freejoint(name="__world_numerical_free_root__")
    for limits in body.joint_ranges:
        joint = spec.joint(limits.joint_id)
        joint.range = (limits.lower_rad, limits.upper_rad)
        joint.solref_limit = configuration.solref
        joint.solimp_limit = configuration.solimp

    body_geoms = tuple(spec.geoms)
    for index, geom in enumerate(body_geoms):
        geom.name = f"__body_geom_{index}"
        geom.contype = geom.conaffinity = 0
    environment_geoms = []
    for index, plane in enumerate(scene.objects):
        geom = spec.worldbody.add_geom(
            name=f"__environment_plane_{index}", type=mujoco.mjtGeom.mjGEOM_PLANE,
            pos=tuple(plane.offset_m * component for component in plane.normal),
            zaxis=plane.normal, size=(0, 0, 0), contype=0, conaffinity=0,
        )
        environment_geoms.append((geom, plane.sliding_friction_coefficient))
    for index, sphere in enumerate(scene.spheres):
        geom = spec.worldbody.add_geom(
            name=f"__environment_sphere_{index}", type=mujoco.mjtGeom.mjGEOM_SPHERE,
            pos=sphere.center_m, size=(sphere.radius_m, 0, 0), contype=0, conaffinity=0,
        )
        environment_geoms.append((geom, sphere.sliding_friction_coefficient))
    for geom in body_geoms:
        if geom.parent.name not in body.contact_link_names:
            continue
        for external, sliding in environment_geoms:
            if sliding is None:
                raise ValueError("Environment contact friction must be explicit")
            friction = max(material.lateral_friction, sliding)
            spec.add_pair(
                geomname1=geom.name, geomname2=external.name, condim=6,
                friction=(friction, friction, material.spinning_friction, material.rolling_friction, material.rolling_friction),
                solref=configuration.solref, solreffriction=configuration.solref,
                solimp=configuration.solimp, margin=0, gap=0,
            )
    model = spec.compile()
    if model.nu or model.na:
        raise ValueError("actuator state is unsupported by this stateless direct-torque adapter")
    data = mujoco.MjData(model)
    root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body.body_frame)
    if not body.fixed_base:
        free_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "__world_numerical_free_root__")
        address = model.jnt_dofadr[free_id]
        local_angular = np.empty(3)
        inverse = np.array((w, -x, -y, -z))
        mujoco.mju_rotVecQuat(local_angular, np.array(body.angular_velocity_rad_per_s), inverse)
        data.qvel[address:address + 3] = body.linear_velocity_m_per_s
        data.qvel[address + 3:address + 6] = local_angular
    for joint in body.joints:
        native = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint.joint_id)
        data.qpos[model.jnt_qposadr[native]] = joint.position_rad
        data.qvel[model.jnt_dofadr[native]] = joint.velocity_rad_per_s
    for channel in torque.channels:
        native = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, channel.joint_id)
        data.qfrc_applied[model.jnt_dofadr[native]] = channel.torque_nm
    regulation = None
    if configuration.velocity_regulation_policy == "none":
        mujoco.mj_step(model, data)
    else:
        # M1 regulation is deliberately limited to one fixed-base hinge. MuJoCo
        # joint ranges are positional; they do not enforce Embodiment speed caps.
        # Search candidate applied torques by re-simulating from an identical
        # source state. Never repair/clamp the resulting qvel.
        if len(body.joints) != 1 or not body.fixed_base:
            raise ValueError("M1 velocity regulation requires fixed-base single hinge")
        joint_id = body.joints[0].joint_id
        joint_native = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        dof = int(model.jnt_dofadr[joint_native])
        velocity_cap = next(
            item.max_velocity_rad_per_s for item in body.joint_ranges if item.joint_id == joint_id
        )
        effort_cap = float(topology[joint_id].find("limit").attrib["effort"])
        requested_torque = float(data.qfrc_applied[dof])
        if abs(float(data.qvel[dof])) > velocity_cap:
            raise ValueError("source joint velocity already exceeds Embodiment maximum")
        source_qpos = np.array(data.qpos, copy=True)
        source_qvel = np.array(data.qvel, copy=True)
        source_qfrc = np.array(data.qfrc_applied, copy=True)

        def trial(applied: float):
            return _mujoco_trial(mujoco, model, source_qpos, source_qvel, source_qfrc, dof, applied)

        # Deterministic bounded grid is a tested-candidate search, not a proof
        # that no feasible torque exists between samples. Preserve the requested
        # value exactly when it is already compliant.
        applied_torque, candidate_velocity = _select_compliant_torque(
            requested_torque, effort_cap, velocity_cap,
            lambda applied: (None if (result := trial(applied)) is None else float(result.qvel[dof])),
        )
        data = trial(applied_torque)
        # Keep regulation evidence World-owned and inspectable without rewriting
        # the Organism command or Embodiment's state/limits.
        regulation = {
            "policy": "m1-fixed-single-hinge-grid-v1",
            "joint_id": joint_id,
            "requested_torque_nm": requested_torque,
            "applied_torque_nm": float(applied_torque),
            "effort_limit_nm": effort_cap,
            "velocity_cap_rad_per_s": velocity_cap,
            "candidate_velocity_rad_per_s": candidate_velocity,
        }
    if not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.qvel)) or any(warning.number for warning in data.warning):
        raise ValueError("MuJoCo produced invalid/warning-reset numerical state")
    mujoco.mj_forward(model, data)
    velocity = np.empty(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, root_id, velocity, 0)
    # mj_objectVelocity is centered on the body inertial frame, not its origin.
    velocity[3:] += np.cross(velocity[:3], data.xpos[root_id] - data.xipos[root_id])
    qw, qx, qy, qz = data.xquat[root_id]
    root_candidate = SolverBodyCandidate(
        expected, body.reference_frame, body.body_frame, tuple(data.xpos[root_id]),
        (float(qx), float(qy), float(qz), float(qw)), tuple(velocity[3:]), tuple(velocity[:3]),
    )
    joints = []
    for item in body.joints:
        native = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, item.joint_id)
        joints.append(JointKinematicState(item.joint_id, float(data.qpos[model.jnt_qposadr[native]]), float(data.qvel[model.jnt_dofadr[native]])))
    return PhysicalStepCandidate(
        SolverArticulatedCandidate(root_candidate, tuple(joints)),
        StaticSceneCandidate(environment.correlation, scene),
        evidence=regulation,
    )


@dataclass(frozen=True, slots=True)
class ArticulatedPhysicalStepOutcome:
    """Committed record and accepted body plus transient solver evidence.

    ``solver_evidence`` is returned to the caller only; it is not part of the
    committed CausalRecord or durable journal. Do not treat it as persisted.
    """

    record: CausalRecord
    body_state: ArticulatedEmbodimentPhysicalState
    solver_evidence: dict[str, object] | None = None


def advance_articulated_mujoco_step(
    world: DurableWorldRuntime,
    *,
    commit_id: str,
    physical_step_id: str,
    body_state: ArticulatedEmbodimentPhysicalState,
    torque: JointTorqueCandidate,
    environment_owner: StaticEnvironmentOwner,
    configuration: ArticulatedMujocoConfiguration,
    write_checkpoints: Callable[
        [PhysicalStepContext, ArticulatedEmbodimentPhysicalState, StaticEnvironmentOwner],
        Iterable[CheckpointReference],
    ],
) -> ArticulatedPhysicalStepOutcome:
    """Run the owner-projected articulated physical step through World commit.

    This is the production integration join for the currently supported
    articulated Embodiment and static Environment capabilities. It assembles
    no domain state: Embodiment and Environment produce/validate their own
    projections and adoption results; the caller's checkpoint writer persists
    the accepted owner states and returns their World references.

    Environment's static owner correlation tracks the source causal state
    being transitioned, matching its established adoption/checkpoint contract.
    Embodiment's body projection and action torque are likewise source-scoped;
    World commits the resulting transition at the target causal tick.
    """
    accepted: dict[str, ArticulatedEmbodimentPhysicalState] = {}
    solver_evidence: dict[str, object] = {}

    def project_body(context: PhysicalStepContext) -> ArticulatedPhysicalInput:
        if body_state.causal_tick != context.source_causal_tick:
            raise ValueError("Embodiment state is not at the World source causal tick")
        correlation = PhysicalStepCorrelation(
            context.world_id,
            context.source_causal_tick,
            context.physical_step_id,
            context.body_id,
            context.embodiment_episode_id,
            context.morphology_revision,
            context.environment_revision,
            context.solver_config_revision,
        )
        projection = project_body_physical_input(body_state, correlation=correlation)
        return ArticulatedPhysicalInput(projection, torque)

    def project_environment(context: PhysicalStepContext) -> EnvironmentPhysicalProjection:
        correlation = EnvironmentStepCorrelation(
            context.world_id,
            context.source_causal_tick,
            context.physical_step_id,
            environment_owner.scene.environment_id,
            context.environment_revision,
        )
        return environment_owner.project_physical_step(correlation)

    def execute_solver(context, body_projection, environment_projection):
        candidate = execute_articulated_mujoco_step(
            context,
            body_projection,
            environment_projection,
            configuration=configuration,
        )
        solver_evidence.update(candidate.evidence or {})
        return candidate

    def adopt_body(context, candidate: SolverArticulatedCandidate) -> None:
        accepted["body"] = body_state.adopt_candidate(candidate, world_context=context)

    def adopt_environment(context, candidate: StaticSceneCandidate) -> None:
        expected = EnvironmentStepCorrelation(
            context.world_id,
            context.source_causal_tick,
            context.physical_step_id,
            environment_owner.scene.environment_id,
            context.environment_revision,
        )
        environment_owner.validate_and_adopt(candidate, expected_correlation=expected)

    def persist(context):
        adopted = accepted.get("body")
        if adopted is None:
            raise RuntimeError("Embodiment did not publish an adopted state")
        return tuple(write_checkpoints(context, adopted, environment_owner))

    record = run_physical_step(
        world,
        commit_id=commit_id,
        physical_step_id=physical_step_id,
        body_id=body_state.root.body_id,
        embodiment_episode_id=body_state.root.embodiment_episode_id,
        morphology_revision=body_state.morphology_revision,
        environment_revision=environment_owner.scene.environment_revision,
        solver_config_revision=configuration.revision,
        ports=PhysicalStepPorts(
            project_body=project_body,
            project_environment=project_environment,
            execute_solver=execute_solver,
            adopt_body=adopt_body,
            adopt_environment=adopt_environment,
            write_checkpoints=persist,
        ),
    )
    return ArticulatedPhysicalStepOutcome(record, accepted["body"], dict(solver_evidence))
