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
    NoActuationProjection,
    project_no_actuation,
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
from world.spatial_projection import CommittedRenderPoseProjection, associate_render_pose_candidate


def _select_compliant_torque(
    requested: float,
    effort: float,
    lower: float,
    upper: float,
    velocity_cap: float,
    trial,
):
    """Choose the nearest tested torque satisfying owner morphology bounds."""
    requested_state = trial(requested)
    if (
        requested_state is not None
        and lower <= requested_state[0] <= upper
        and abs(requested_state[1]) <= velocity_cap
    ):
        return requested, requested_state
    candidates = [requested]
    candidates.extend(-effort + (2 * effort * i / 1000) for i in range(1001))
    feasible = []
    for applied in candidates:
        state = trial(applied)
        if (
            state is not None
            and lower <= state[0] <= upper
            and abs(state[1]) <= velocity_cap
        ):
            feasible.append((abs(applied - requested), applied, state))
    if not feasible:
        raise ValueError("no tested applied torque produces a morphology-compliant candidate")
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
    render_morphology_projection: object | None = None,
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
        active_morphology=active_morphology,
        torque_binding=torque_binding,
        environment_owner=environment_owner,
        configuration=configuration,
        render_morphology_projection=render_morphology_projection,
        write_checkpoints=write_checkpoints,
    )


def advance_articulated_mujoco_step_without_actuation(
    world: DurableWorldRuntime,
    *,
    commit_id: str,
    physical_step_id: str,
    body_state: ArticulatedEmbodimentPhysicalState,
    no_actuation: NoActuationProjection,
    active_morphology: ActiveArticulatedMorphology,
    torque_binding: EmbodimentJointTorqueBinding,
    environment_owner: StaticEnvironmentOwner,
    configuration: ArticulatedMujocoConfiguration,
    render_morphology_projection: object | None = None,
    write_checkpoints: Callable[
        [PhysicalStepContext, ArticulatedEmbodimentPhysicalState, StaticEnvironmentOwner],
        Iterable[CheckpointReference],
    ],
) -> ArticulatedPhysicalStepOutcome:
    """Advance one authorized physical step with no motor command requested."""
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
    expected = project_no_actuation(
        torque_binding, active_morphology, issued_at_tick=body_state.causal_tick
    )
    if no_actuation != expected:
        raise ValueError("no-actuation projection is stale or differs from active Embodiment binding")
    return advance_articulated_mujoco_step(
        world,
        commit_id=commit_id,
        physical_step_id=physical_step_id,
        body_state=body_state,
        torque=no_actuation,
        active_morphology=active_morphology,
        torque_binding=torque_binding,
        environment_owner=environment_owner,
        configuration=configuration,
        render_morphology_projection=render_morphology_projection,
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
        if self.velocity_regulation_policy not in {
            "none",
            "m1-fixed-single-hinge-grid-v1",
            "m1-fixed-single-hinge-grid-v2",
        }:
            raise ValueError("unsupported velocity regulation policy")


@dataclass(frozen=True, slots=True)
class ArticulatedPhysicalInput:
    body: BodyPhysicalInput
    torque: JointTorqueCandidate | NoActuationProjection
    active_morphology: ActiveArticulatedMorphology
    torque_binding: EmbodimentJointTorqueBinding


def execute_articulated_mujoco_step(
    context: PhysicalStepContext,
    body_input: ArticulatedPhysicalInput,
    environment: EnvironmentPhysicalProjection,
    *,
    configuration: ArticulatedMujocoConfiguration,
    render_morphology_projection: object | None = None,
    render_pose_sink: Callable[[object], None] | None = None,
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
    if isinstance(torque, NoActuationProjection):
        active = body_input.active_morphology
        binding = body_input.torque_binding
        if (
            (active.body_id, active.embodiment_episode_id, active.morphology_revision)
            != (context.body_id, context.embodiment_episode_id, context.morphology_revision)
            or body.morphology_sha256 != active.morphology_sha256
        ):
            raise ValueError("active Embodiment morphology differs from World source context/body")
        expected_no_actuation = project_no_actuation(
            binding, active, issued_at_tick=context.source_causal_tick
        )
        if torque != expected_no_actuation:
            raise ValueError("no-actuation projection is stale or differs from World source context")
        torque_channels = ()
    elif isinstance(torque, JointTorqueCandidate):
        if torque.issued_at_tick != context.source_causal_tick:
            raise ValueError("torque is stale for the World source cut")
        torque_channels = torque.channels
    else:
        raise TypeError("physical step requires an explicit Embodiment torque or no-actuation projection")

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
    for channel in torque_channels:
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
        if configuration.velocity_regulation_policy == "m1-fixed-single-hinge-grid-v2":
            # MuJoCo activates joint-limit constraints at the boundary. With a
            # finite causal timestep, a near-limit state at the allowed peak
            # velocity can cross the Embodiment range before the next state is
            # observable. Activate the unchanged range's constraint early by
            # at most one timestep of the owner velocity cap. This is a solver
            # margin, not a changed morphology range or a qpos clamp.
            joint.margin = min(
                (limits.upper_rad - limits.lower_rad) / 2,
                limits.max_velocity_rad_per_s * configuration.timestep_s,
            )

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
    for channel in torque_channels:
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
        joint_range = next(item for item in body.joint_ranges if item.joint_id == joint_id)
        velocity_cap = joint_range.max_velocity_rad_per_s
        effort_cap = float(topology[joint_id].find("limit").attrib["effort"])
        requested_torque = float(data.qfrc_applied[dof])
        if abs(float(data.qvel[dof])) > velocity_cap:
            raise ValueError("source joint velocity already exceeds Embodiment maximum")
        source_qpos = np.array(data.qpos, copy=True)
        source_qvel = np.array(data.qvel, copy=True)
        source_qfrc = np.array(data.qfrc_applied, copy=True)

        qpos_address = int(model.jnt_qposadr[joint_native])

        def trial(applied: float):
            result = _mujoco_trial(
                mujoco, model, source_qpos, source_qvel, source_qfrc, dof, applied
            )
            if result is None:
                return None
            return float(result.qpos[qpos_address]), float(result.qvel[dof])

        # Deterministic bounded grid is a tested-candidate search, not a proof
        # that no feasible torque exists between samples. Preserve the requested
        # value exactly when it is already compliant.
        if isinstance(torque, NoActuationProjection):
            # No-actuation is not permission for the solver to invent a force
            # that makes the candidate compliant. Integrate exactly the
            # zero-torque request and reject it below if owner limits fail.
            applied_torque = 0.0
            candidate_state = trial(applied_torque)
            if candidate_state is None:
                raise ValueError("no-actuation zero-torque trial produced invalid MuJoCo state")
            if not (
                joint_range.lower_rad <= candidate_state[0] <= joint_range.upper_rad
                and abs(candidate_state[1]) <= velocity_cap
            ):
                raise ValueError(
                    "no-actuation zero-torque candidate violates Embodiment joint limits"
                )
        else:
            applied_torque, candidate_state = _select_compliant_torque(
                requested_torque,
                effort_cap,
                joint_range.lower_rad,
                joint_range.upper_rad,
                velocity_cap,
                trial,
            )
        data = _mujoco_trial(
            mujoco, model, source_qpos, source_qvel, source_qfrc, dof, applied_torque
        )
        if data is None:
            raise ValueError("selected torque no longer produces a valid MuJoCo candidate")
        # Keep regulation evidence World-owned and inspectable without rewriting
        # the Organism command or Embodiment's state/limits.
        regulation = {
            "policy": configuration.velocity_regulation_policy,
            "world_id": context.world_id,
            "source_causal_tick": context.source_causal_tick,
            "target_causal_tick": context.target_causal_tick,
            "physical_step_id": context.physical_step_id,
            "body_id": context.body_id,
            "embodiment_episode_id": context.embodiment_episode_id,
            "morphology_revision": context.morphology_revision,
            "solver_config_revision": context.solver_config_revision,
            "joint_id": joint_id,
            "source_position_rad": float(source_qpos[qpos_address]),
            "source_velocity_rad_per_s": float(source_qvel[dof]),
            "requested_torque_nm": requested_torque,
            "applied_torque_nm": float(applied_torque),
            "effort_limit_nm": effort_cap,
            "position_range_rad": [joint_range.lower_rad, joint_range.upper_rad],
            "joint_limit_activation_margin_rad": float(model.jnt_margin[joint_native]),
            "candidate_position_rad": candidate_state[0],
            "velocity_cap_rad_per_s": velocity_cap,
            "candidate_velocity_rad_per_s": candidate_state[1],
        }
    if not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.qvel)) or any(warning.number for warning in data.warning):
        raise ValueError("MuJoCo produced invalid/warning-reset numerical state")
    mujoco.mj_forward(model, data)
    if render_morphology_projection is not None:
        if render_pose_sink is None:
            raise ValueError("render pose extraction requires an internal result sink")
        render_pose_sink(
            _render_pose_candidate_from_mujoco(
                mujoco, model, data, context, render_morphology_projection
            )
        )
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
    render_pose_projection: CommittedRenderPoseProjection | None = None
    # A committed physical transition remains successful if optional render
    # enrichment cannot be completed. This field makes that limitation explicit.
    render_pose_error: str | None = None


def advance_articulated_mujoco_step(
    world: DurableWorldRuntime,
    *,
    commit_id: str,
    physical_step_id: str,
    body_state: ArticulatedEmbodimentPhysicalState,
    torque: JointTorqueCandidate | NoActuationProjection,
    active_morphology: ActiveArticulatedMorphology,
    torque_binding: EmbodimentJointTorqueBinding,
    environment_owner: StaticEnvironmentOwner,
    configuration: ArticulatedMujocoConfiguration,
    render_morphology_projection: object | None = None,
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
    if render_morphology_projection is not None:
        _validate_render_projection_identity(
            render_morphology_projection,
            body_id=body_state.root.body_id,
            embodiment_episode_id=body_state.root.embodiment_episode_id,
            morphology_revision=body_state.morphology_revision,
        )
    accepted: dict[str, ArticulatedEmbodimentPhysicalState] = {}
    solver_evidence: dict[str, object] = {}
    render_pose_candidates: list[object] = []

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
        return ArticulatedPhysicalInput(
            projection, torque, active_morphology, torque_binding
        )

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
            render_morphology_projection=render_morphology_projection,
            render_pose_sink=render_pose_candidates.append if render_morphology_projection is not None else None,
        )
        if render_morphology_projection is not None and len(render_pose_candidates) != 1:
            raise RuntimeError("physical solver must produce exactly one transient render pose candidate")
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

    # The committed-cut lookup and transient projection must refer to the cut
    # this call just produced, even if another caller tries to advance World.
    # The guard's single-tick allowance is deliberately scoped over commit and
    # enrichment; it is not a durable publication or history mechanism.
    committed_projection = None
    render_pose_error = None
    with world.committed_cut_publication_guard():
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
        if render_morphology_projection is not None:
            try:
                cut = world.committed_cut_for(
                    world_id=record.world_id,
                    causal_tick=record.causal_tick,
                    commit_id=commit_id,
                )
                references = tuple(
                    reference
                    for reference in cut.owner_checkpoint_references
                    if reference.owner == "embodiment" and reference.owner_identity == body_state.root.body_id
                )
                if len(references) != 1:
                    raise ValueError("committed cut must contain exactly one Embodiment checkpoint for this body")
                committed_projection = associate_render_pose_candidate(
                    world, cut, references[0], render_pose_candidates[0]
                )
            except Exception as exc:
                # Do not turn a confirmed physical commit into an ambiguous
                # failed call merely because the optional transient view failed.
                render_pose_error = f"{type(exc).__name__}: {exc}"
    return ArticulatedPhysicalStepOutcome(
        record, accepted["body"], dict(solver_evidence), committed_projection, render_pose_error
    )


def _render_pose_candidate_from_mujoco(
    mujoco, model, data, context: PhysicalStepContext, projection: object
) -> object:
    """Create Embodiment's candidate from actual MuJoCo body transforms.

    The projection is owner supplied and carries the geometry revision. World
    maps only its authored renderable segment IDs to exact compiled body names;
    it never synthesizes geometry or transforms.
    """
    segments = getattr(projection, "segments", None)
    validate = getattr(projection, "validate_dynamic_pose_candidate", None)
    if not isinstance(segments, tuple) or not callable(validate):
        raise TypeError("render morphology projection must be an owner-authored validated projection")
    identities = (
        (getattr(projection, "body_id", None), context.body_id),
        (getattr(projection, "embodiment_episode_id", None), context.embodiment_episode_id),
        (getattr(projection, "morphology_revision", None), context.morphology_revision),
    )
    if any(actual != expected for actual, expected in identities):
        raise ValueError("owner render projection identity does not match World physical-step context")
    poses = []
    segment_ids = []
    for segment in segments:
        segment_id = getattr(segment, "segment_id", None)
        if type(segment_id) is not str or not segment_id.strip() or segment_id in segment_ids:
            raise ValueError("renderable segment IDs must be unique non-empty strings")
        segment_ids.append(segment_id)
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, segment_id)
        if body_id <= 0:
            raise ValueError(f"renderable segment {segment_id!r} is not an exact MuJoCo body")
        qw, qx, qy, qz = (float(value) for value in data.xquat[body_id])
        poses.append(
            _make_render_link_pose(
                segment_id,
                tuple(float(value) for value in data.xpos[body_id]),
                (qx, qy, qz, qw),
            )
        )
    candidate = validate(world_context=context, poses=tuple(poses))
    return candidate


def _validate_render_projection_identity(
    projection: object, *, body_id: str, embodiment_episode_id: str, morphology_revision: str
) -> None:
    """Reject owner projection identity mismatch before opening a World transition."""
    validate = getattr(projection, "validate_dynamic_pose_candidate", None)
    segments = getattr(projection, "segments", None)
    if not isinstance(segments, tuple) or not callable(validate):
        raise TypeError("render morphology projection must be an owner-authored validated projection")
    expected = (body_id, embodiment_episode_id, morphology_revision)
    actual = (
        getattr(projection, "body_id", None),
        getattr(projection, "embodiment_episode_id", None),
        getattr(projection, "morphology_revision", None),
    )
    if actual != expected:
        raise ValueError("owner render projection identity does not match World physical-step context")


def _make_render_link_pose(segment_id: str, translation_m: tuple[float, ...], orientation_xyzw: tuple[float, ...]):
    """Instantiate the owner-owned candidate value without pinning its package."""
    # Import only when this optional capability is requested. Embodiment owns
    # RenderLinkPose validation and the dependency is not required by base World.
    from embodiment.physics3d.spatial_render import RenderLinkPose

    return RenderLinkPose(segment_id, translation_m, orientation_xyzw)
