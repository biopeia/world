"""Load and construct the explicitly configured scalar M1 operator profile.

This module is deliberately opt-in and imports cross-domain runtime packages
only when a World M1 process is invoked. The profile is operator input, never
an observation or scientific claim.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


PROFILE_SCHEMA = "biopeia.world.m1.scalar-sensorimotor-profile"
PROFILE_SCHEMA_VERSION = 1
DOCS_REVISION = "586dcbb8376c2800fc7ea8e04ba0b486db3e87db"


class ProfileError(ValueError):
    """Invalid or unsupported operator profile input."""


def _reject_constant(value: str) -> None:
    raise ProfileError(f"non-finite JSON number is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ProfileError(f"duplicate profile key: {key}")
        result[key] = value
    return result


def _object(value: Any, expected: set[str], name: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise ProfileError(f"{name} must be an object")
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise ProfileError(f"{name} fields mismatch; missing={missing}, unknown={unknown}")
    return value


def _text(value: Any, name: str) -> str:
    if type(value) is not str or not value.strip():
        raise ProfileError(f"{name} must be a non-empty string")
    return value


def _identifier(value: Any, name: str) -> str:
    text = _text(value, name)
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}", text) is None:
        raise ProfileError(f"{name} is not a supported identifier")
    return text


def _number(value: Any, name: str, *, minimum: float | None = None) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ProfileError(f"{name} must be a finite number")
    if minimum is not None and value < minimum:
        raise ProfileError(f"{name} must be >= {minimum}")
    return float(value)


def _vector(value: Any, name: str, size: int, *, minimum: float | None = None) -> tuple[float, ...]:
    if type(value) is not list or len(value) != size:
        raise ProfileError(f"{name} must be a {size}-element array")
    return tuple(_number(item, f"{name}[{index}]", minimum=minimum) for index, item in enumerate(value))


def _finite_tree(value: Any, path: str = "profile") -> None:
    if type(value) is float and not math.isfinite(value):
        raise ProfileError(f"{path} contains a non-finite number")
    if type(value) is dict:
        for key, child in value.items():
            _finite_tree(child, f"{path}.{key}")
    elif type(value) is list:
        for index, child in enumerate(value):
            _finite_tree(child, f"{path}[{index}]")


def load_operator_profile(path: str | Path) -> tuple[dict[str, Any], str]:
    """Load a bounded, non-symlink JSON profile and return its SHA-256."""
    supplied = Path(path).expanduser()
    absolute = supplied.absolute()
    try:
        resolved = absolute.resolve(strict=True)
    except OSError as exc:
        raise ProfileError(f"profile file is unavailable: {exc}") from exc
    if resolved != absolute or absolute.is_symlink() or not absolute.is_file():
        raise ProfileError("profile path must be a regular non-symlink file")
    raw = absolute.read_bytes()
    if len(raw) > 1024 * 1024:
        raise ProfileError("profile file exceeds 1 MiB")
    try:
        profile = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProfileError(f"profile must be valid UTF-8 JSON: {exc}") from exc
    _validate_profile(profile)
    return profile, hashlib.sha256(raw).hexdigest()


def _validate_profile(value: Any) -> None:
    profile = _object(value, {
        "schema", "schema_version", "profile_id", "profile_kind", "provenance",
        "world", "organism", "embodiment", "environment", "modality", "solver",
        "observation_bridge",
    }, "profile")
    _finite_tree(profile)
    if profile["schema"] != PROFILE_SCHEMA or type(profile["schema_version"]) is not int or profile["schema_version"] != PROFILE_SCHEMA_VERSION:
        raise ProfileError("unsupported profile schema/version")
    if profile["profile_kind"] != "operator_supplied_scenario":
        raise ProfileError("profile_kind must identify explicit operator-supplied scenario data")
    _identifier(profile["profile_id"], "profile_id")
    provenance = _object(profile["provenance"], {
        "repository", "revision", "document", "scenario", "authorization", "classification",
    }, "provenance")
    if provenance["repository"] != "biopeia/docs" or provenance["revision"] != DOCS_REVISION:
        raise ProfileError("profile provenance does not match its reviewed Docs source revision")
    for key in provenance:
        _text(provenance[key], f"provenance.{key}")
    if re.fullmatch(r"[0-9a-f]{40}", provenance["revision"]) is None:
        raise ProfileError("provenance.revision must be a full lowercase Git SHA")

    world = _object(profile["world"], {"id", "presence_id", "initial_causal_tick"}, "world")
    _identifier(world["id"], "world.id")
    _identifier(world["presence_id"], "world.presence_id")
    if type(world["initial_causal_tick"]) is not int or world["initial_causal_tick"] != 0:
        raise ProfileError("fresh M1 profile must start at causal tick 0")

    organism = _object(profile["organism"], {"id", "step", "lifecycle", "sense_node_id", "kernel_limits"}, "organism")
    _identifier(organism["id"], "organism.id")
    try:
        uuid.UUID(organism["id"])
    except ValueError as exc:
        raise ProfileError("organism.id must be a UUID") from exc
    _identifier(organism["sense_node_id"], "organism.sense_node_id")
    if type(organism["step"]) is not int or organism["step"] != 0 or organism["lifecycle"] != "ACTIVE":
        raise ProfileError("unsupported initial Organism transition")
    limits = _object(organism["kernel_limits"], {
        "max_nodes", "max_concepts", "max_edges", "max_tentative_edges",
        "max_structural_mutations_per_consolidation", "consolidation_interval_ticks",
        "max_plastic_checkpoint_bytes", "max_owner_checkpoint_bytes",
        "max_consolidation_candidates", "max_salient_event_traces", "consolidation_epoch_ticks",
        "slow_support_epochs", "fast_consolidation_threshold", "fast_min_reliability",
        "max_incoming_consolidated_weight_norm", "reacclimation_ticks", "max_episodic_episodes",
        "max_episodic_episode_records", "max_episodic_retrieval_candidates", "max_episodic_replay_items",
        "max_episodic_interpretations_per_episode", "max_episodic_checkpoint_bytes",
        "episodic_epoch_ticks", "episodic_min_consolidation_epochs",
    }, "organism.kernel_limits")
    for name, item in limits.items():
        _number(item, f"organism.kernel_limits.{name}", minimum=0)
    if any(type(item) is not int for name, item in limits.items() if name not in {
        "fast_consolidation_threshold", "fast_min_reliability", "max_incoming_consolidated_weight_norm"
    }):
        raise ProfileError("count and interval kernel limits must be integers")
    if limits["fast_consolidation_threshold"] > 1 or limits["fast_min_reliability"] > 1:
        raise ProfileError("kernel reliability thresholds must be <= 1")

    embodiment = _object(profile["embodiment"], {
        "body_id", "episode_id", "morphology_revision", "geometry_revision", "frame_id", "root_frame", "joints",
        "segments", "body_material", "fixed_base", "receptor", "actuator_surface",
        "differential_binding", "torque_binding",
    }, "embodiment")
    for key in ("body_id", "episode_id", "morphology_revision", "geometry_revision", "frame_id", "root_frame"):
        _identifier(embodiment[key], f"embodiment.{key}")
    if embodiment["fixed_base"] is not True or type(embodiment["joints"]) is not list or len(embodiment["joints"]) != 1:
        raise ProfileError("M1 profile requires its declared fixed-base single-hinge body")
    joint = _object(embodiment["joints"][0], {
        "id", "axis", "lower", "upper", "max_motor_torque", "passive_damping", "max_velocity",
        "parent_link", "child_link", "origin", "initial_position", "initial_velocity",
    }, "embodiment.joints[0]")
    for key in ("id", "parent_link", "child_link"):
        _identifier(joint[key], f"embodiment.joints[0].{key}")
    _vector(joint["axis"], "embodiment.joints[0].axis", 3)
    _vector(joint["origin"], "embodiment.joints[0].origin", 3)
    axis_norm = math.sqrt(sum(float(x) ** 2 for x in joint["axis"]))
    if not math.isclose(axis_norm, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ProfileError("joint axis must be unit length")
    lower = _number(joint["lower"], "joint.lower")
    upper = _number(joint["upper"], "joint.upper")
    if lower >= upper:
        raise ProfileError("joint lower limit must be below upper limit")
    for key in ("max_motor_torque", "max_velocity"):
        _number(joint[key], f"joint.{key}", minimum=0)
    _number(joint["passive_damping"], "joint.passive_damping", minimum=0)
    initial_position = _number(joint["initial_position"], "joint.initial_position")
    initial_velocity = _number(joint["initial_velocity"], "joint.initial_velocity")
    if not lower <= initial_position <= upper:
        raise ProfileError("initial joint position must be within declared limits")
    if abs(initial_velocity) > joint["max_velocity"]:
        raise ProfileError("initial joint velocity exceeds the declared maximum")
    _identifier(joint["parent_link"], "joint.parent_link")
    _identifier(joint["child_link"], "joint.child_link")
    material = _object(embodiment["body_material"], {
        "lateral_friction", "spinning_friction", "rolling_friction", "restitution",
        "linear_damping", "angular_damping",
    }, "embodiment.body_material")
    for key, value in material.items():
        _number(value, f"embodiment.body_material.{key}", minimum=0)
    segments = _object(embodiment["segments"], {"body.root", "body.arm"}, "embodiment.segments")
    for name, segment in segments.items():
        segment = _object(segment, {"mass", "size", "origin", "color"}, f"embodiment.segments.{name}")
        _number(segment["mass"], f"segments.{name}.mass", minimum=0.000001)
        _vector(segment["size"], f"segments.{name}.size", 3, minimum=0.000001)
        _vector(segment["origin"], f"segments.{name}.origin", 3)
        color = _vector(segment["color"], f"segments.{name}.color", 4, minimum=0)
        if any(channel > 1 for channel in color):
            raise ProfileError(f"segments.{name}.color channels must be <= 1")
    receptor = _object(embodiment["receptor"], {"id", "body_frame", "position_in_body_frame_m"}, "embodiment.receptor")
    for key in ("id", "body_frame"):
        _identifier(receptor[key], f"embodiment.receptor.{key}")
    _vector(receptor["position_in_body_frame_m"], "receptor.position_in_body_frame_m", 3)
    surface = _object(embodiment["actuator_surface"], {"ids", "fingerprint_material"}, "embodiment.actuator_surface")
    if surface["ids"] != ["actuator.hinge.positive", "actuator.hinge.negative"]:
        raise ProfileError("unsupported actuator surface")
    _text(surface["fingerprint_material"], "embodiment.actuator_surface.fingerprint_material")
    for actuator_id in surface["ids"]:
        _identifier(actuator_id, "actuator_surface.ids[]")
    differential = _object(embodiment["differential_binding"], {
        "id", "positive_actuator_id", "negative_actuator_id", "scale",
    }, "embodiment.differential_binding")
    torque = _object(embodiment["torque_binding"], {"id", "channel_id", "joint_id", "axis"}, "embodiment.torque_binding")
    for section, data in (("differential_binding", differential), ("torque_binding", torque)):
        for key in data:
            if key != "scale" and key != "axis":
                _text(data[key], f"embodiment.{section}.{key}")
    if (differential["positive_actuator_id"], differential["negative_actuator_id"]) != tuple(surface["ids"]):
        raise ProfileError("differential binding must use the declared actuator surface")
    _number(differential["scale"], "differential_binding.scale", minimum=0)
    _vector(torque["axis"], "torque_binding.axis", 3)
    if torque["joint_id"] != joint["id"]:
        raise ProfileError("torque binding must target the declared joint")
    if torque["axis"] != joint["axis"]:
        raise ProfileError("torque binding axis must match the declared joint axis")
    if embodiment["root_frame"] not in segments:
        raise ProfileError("root_frame must reference a declared body segment")

    environment = _object(profile["environment"], {"id", "revision", "owner_id", "gravity_m_s2", "field"}, "environment")
    for key in ("id", "revision", "owner_id"):
        _identifier(environment[key], f"environment.{key}")
    field = _object(environment["field"], {
        "id", "revision", "position_m", "resource_radius_m", "field_radius_m", "remaining", "transfer_per_tick",
    }, "environment.field")
    for key in ("id", "revision"):
        _identifier(field[key], f"environment.field.{key}")
    _vector(environment["gravity_m_s2"], "environment.gravity_m_s2", 3)
    _vector(field["position_m"], "environment.field.position_m", 3)
    for key in ("resource_radius_m", "field_radius_m", "remaining", "transfer_per_tick"):
        _number(field[key], f"environment.field.{key}", minimum=0)
    if field["resource_radius_m"] <= 0 or field["field_radius_m"] <= 0:
        raise ProfileError("field radii must be positive")
    modality = _object(profile["modality"], {"channel_id", "sense_node_id", "identity_transduction_revision"}, "modality")
    for key in modality:
        _text(modality[key], f"modality.{key}")
    if modality["identity_transduction_revision"] != "dimensionless-scalar-field-identity-v1":
        raise ProfileError("unsupported Modality transduction")
    if modality["sense_node_id"] != organism["sense_node_id"]:
        raise ProfileError("Modality sense node must match the Organism profile node")
    solver = _object(profile["solver"], {
        "engine", "version", "revision", "timestep_s", "iterations", "tolerance", "solref", "solimp", "contact_policy",
    }, "solver")
    if solver["engine"] != "mujoco" or solver["version"] != "3.15.0":
        raise ProfileError("profile solver version must match the M1 operator profile revision")
    for key in ("engine", "version", "revision"):
        _identifier(solver[key], f"solver.{key}")
    _text(solver["contact_policy"], "solver.contact_policy")
    _number(solver["timestep_s"], "solver.timestep_s", minimum=0.000000001)
    _number(solver["tolerance"], "solver.tolerance", minimum=0)
    if type(solver["iterations"]) is not int or solver["iterations"] < 1:
        raise ProfileError("solver.iterations must be a positive integer")
    _vector(solver["solref"], "solver.solref", 2, minimum=0)
    if any(value <= 0 for value in solver["solref"]):
        raise ProfileError("solver.solref values must be positive")
    solimp = _vector(solver["solimp"], "solver.solimp", 5, minimum=0)
    if any(value > 1 for value in solimp[:3]):
        raise ProfileError("first three solver.solimp values must be <= 1")
    if receptor["body_frame"] != embodiment["root_frame"]:
        raise ProfileError("receptor body frame must reference the declared root frame")
    if joint["parent_link"] not in segments or joint["child_link"] not in segments:
        raise ProfileError("joint topology must reference declared body segments")
    bridge = _object(profile["observation_bridge"], {"host", "allowed_origin", "max_response_bytes"}, "observation_bridge")
    if bridge["host"] != "127.0.0.1" or type(bridge["max_response_bytes"]) is not int or bridge["max_response_bytes"] < 1:
        raise ProfileError("Observation bridge must use the configured loopback interface and positive response cap")
    _text(bridge["allowed_origin"], "observation_bridge.allowed_origin")
    origin = urlsplit(bridge["allowed_origin"])
    if origin.scheme != "http" or origin.hostname not in {"127.0.0.1", "localhost"} or origin.path or origin.query or origin.fragment:
        raise ProfileError("Observation bridge allowed origin must be an HTTP loopback origin")


@dataclass(frozen=True, slots=True)
class BuiltM1Runtime:
    world: Any
    observation_plane: Any
    cycle_inputs: Any
    source_revisions: dict[str, str]
    profile_id: str
    profile_sha256: str
    observation_bridge: Any


def construct_fresh_runtime(
    profile: dict[str, Any],
    profile_sha256: str,
    state_root: Path,
    source_revisions: dict[str, str],
    *,
    observation_port: int,
    organism_checkpoint: str = "v4",
) -> BuiltM1Runtime:
    """Construct World and owner state solely from validated profile inputs.

    ``organism_checkpoint="v5"`` opts into the compact Organism exploration
    checkpoint whose action history is a World-verified dependency
    (organism#183, world#90). ``"v4"`` keeps the existing format.
    """
    if organism_checkpoint not in ("v4", "v5"):
        raise ProfileError("organism_checkpoint must be 'v4' or 'v5'")
    # Keep this boundary safe even when called directly rather than through the
    # CLI profile loader. The CLI's digest is provenance for the original JSON;
    # revalidation ensures its decoded content remains admissible.
    _validate_profile(profile)
    if type(profile_sha256) is not str or re.fullmatch(r"[0-9a-f]{64}", profile_sha256) is None:
        raise ProfileError("profile_sha256 must be a lowercase SHA-256 digest")
    if type(source_revisions) is not dict or not source_revisions or any(
        not isinstance(key, str) or not key.strip() or not isinstance(value, str) or not value.strip()
        for key, value in source_revisions.items()
    ):
        raise ProfileError("source revisions must be a non-empty mapping")
    try:
        import mujoco
        from embodiment.observation import BodyPoseSample
        from embodiment.physics3d.articulated_spec import ArticulatedBodySpec
        from embodiment.physics3d.articulated_state import (
            ARTICULATED_PHYSICAL_CHECKPOINT_SCHEMA,
            ActiveArticulatedMorphology,
            ArticulatedEmbodimentPhysicalState,
            JointKinematicState,
        )
        from embodiment.physics3d.differential_motor_composition import (
            DifferentialMotorPair,
            EmbodimentDifferentialMotorBinding,
        )
        from embodiment.physics3d.joint_torque import (
            EmbodimentJointTorqueBinding,
            JointTorqueChannelBinding,
        )
        from embodiment.physics3d.mechanics import (
            JointSpec,
            JointTopology,
            SegmentSpec,
            SurfaceMaterial,
        )
        from embodiment.physics3d.physical_transition import EmbodimentPhysicalState
        from embodiment.physics3d.receptor import EmbodimentReceptor
        from environment import (
            FiniteResourceState,
            FrameDescriptor,
            IdentifiedFiniteScalarField,
            StaticEnvironmentOwner,
            StaticPhysicalScene,
        )
        from modality import DimensionlessScalarFieldIdentityV1
        from observation.http_bridge import LocalObservationReadBridge
        from observation.plane import ObservationPlane
    except ImportError as exc:
        raise ProfileError(f"M1 runtime dependencies are unavailable: {exc}") from exc

    # The package imports below are intentionally local: World's base package
    # remains usable without the optional cross-domain runtime extra.
    from organism.actuation import (
        ActuatorEvidenceContext,
        ActuatorSurface,
        CompetenceDevelopmentEngine,
        OrganismCheckpointContract,
        OrganismExplorationComposer,
    )
    from organism.actuation.action_history import HistoryIndex
    from organism.cognition import CognitiveRuntimeCheckpoint, SenseChannelBinding

    from .checkpoint_dependency import resolve_dependency
    from organism.cognition.graph import CognitiveGraph, PlasticNode
    from organism.cognition.limits import KernelLimits
    from organism.cognition.transition import CognitiveTransitionState
    from organism.cognition.types import NodeKind
    from organism.identity import OrganismIdentity
    from organism.transition import (
        EmbodimentContextRef,
        OrganismLifecycle,
        OrganismTransitionState,
    )
    from world.adapters.articulated_mujoco import ArticulatedMujocoConfiguration
    from world.causal_cycle import ScalarSensorimotorCycleInputs
    from world.durable_runtime import (
        DurableJournalSnapshotStore,
        DurableWorldRuntime,
        PresenceOwnerLink,
        WorldGenesisSpec,
        WorldCausalManifestStore,
    )
    from world.owner_checkpoint_store import (
        FileOwnerCheckpointStore,
        OwnerCheckpointPayload,
    )
    from world.runtime import PresenceId, WorldId

    solver_profile = profile["solver"]
    if str(getattr(mujoco, "__version__", "")) != solver_profile["version"]:
        raise ProfileError("installed MuJoCo version does not match profile solver version")

    world_cfg = profile["world"]
    org_cfg = profile["organism"]
    body_cfg = profile["embodiment"]
    env_cfg = profile["environment"]
    field_cfg = env_cfg["field"]
    world_id = world_cfg["id"]
    organism_id = OrganismIdentity.parse(org_cfg["id"])
    frame = FrameDescriptor(
        frame_id=body_cfg["frame_id"],
        parent_frame_id="world",
        origin_m=(0.0, 0.0, 0.0),
        x_axis_in_parent=(1.0, 0.0, 0.0),
        y_axis_in_parent=(0.0, 1.0, 0.0),
        z_axis_in_parent=(0.0, 0.0, 1.0),
        handedness="RIGHT_HANDED",
        up_axis="Z",
    )

    joint_cfg, = body_cfg["joints"]
    joint_spec = JointSpec(
        joint_cfg["id"], tuple(joint_cfg["axis"]), joint_cfg["lower"], joint_cfg["upper"],
        joint_cfg["max_motor_torque"], joint_cfg["passive_damping"], joint_cfg["max_velocity"],
    )
    joint_topology = JointTopology(
        joint_cfg["id"], joint_cfg["parent_link"], joint_cfg["child_link"], tuple(joint_cfg["origin"])
    )
    segments = {
        name: SegmentSpec(data["mass"], tuple(data["size"]), tuple(data["origin"]), tuple(data["color"]))
        for name, data in body_cfg["segments"].items()
    }
    material = SurfaceMaterial(**body_cfg["body_material"])
    body_spec = ArticulatedBodySpec(
        body_kind="acceptance-fixed-base-hinge",
        state_schema_version=1,
        spawn_height=0.0,
        base_link_name=body_cfg["root_frame"],
        joint_specs=(joint_spec,),
        joint_topology=(joint_topology,),
        segments=segments,
        contact_link_names=tuple(sorted(segments)),
        structural_neighbour_names=((joint_cfg["parent_link"], joint_cfg["child_link"]),),
        body_material=material,
        fixed_base=body_cfg["fixed_base"],
    )
    morphology = ActiveArticulatedMorphology.from_spec(
        body_id=body_cfg["body_id"],
        embodiment_episode_id=body_cfg["episode_id"],
        morphology_revision=body_cfg["morphology_revision"],
        spec=body_spec,
    )
    pose = BodyPoseSample(
        body_id=body_cfg["body_id"], reference_frame=body_cfg["frame_id"],
        body_frame=body_cfg["root_frame"], position_m=(0.0, 0.0, 0.0),
        orientation_xyzw=(0.0, 0.0, 0.0, 1.0), sample_kind="genesis", sample_value=0,
    )
    root = EmbodimentPhysicalState(
        world_id=world_id, causal_tick=0, body_id=body_cfg["body_id"],
        embodiment_episode_id=body_cfg["episode_id"], morphology_revision=body_cfg["morphology_revision"],
        pose=pose, linear_velocity_m_per_s=(0.0, 0.0, 0.0), angular_velocity_rad_per_s=(0.0, 0.0, 0.0),
    )
    articulated_body = ArticulatedEmbodimentPhysicalState(
        root=root, morphology=morphology,
        joints=(JointKinematicState(joint_cfg["id"], joint_cfg["initial_position"], joint_cfg["initial_velocity"]),),
    )
    env_owner = StaticEnvironmentOwner(
        StaticPhysicalScene(
            environment_id=env_cfg["id"], environment_revision=env_cfg["revision"], frame=frame,
            gravity_m_s2=tuple(env_cfg["gravity_m_s2"]), objects=(),
        ), owner_id=env_cfg["owner_id"], world_id=world_id,
    )
    environment_field = IdentifiedFiniteScalarField(
        resource_id=field_cfg["id"], field_revision=field_cfg["revision"],
        environment_id=env_cfg["id"], environment_revision=env_cfg["revision"],
        frame_id=body_cfg["frame_id"],
        state=FiniteResourceState(
            position=tuple(field_cfg["position_m"]), radius=field_cfg["resource_radius_m"],
            field_radius=field_cfg["field_radius_m"], remaining=field_cfg["remaining"],
            transfer_per_tick=field_cfg["transfer_per_tick"],
        ),
    )
    limits = KernelLimits(**org_cfg["kernel_limits"])
    cognition = CognitiveRuntimeCheckpoint(
        state=CognitiveTransitionState.initial(organism_id),
        graph=CognitiveGraph(
            nodes=(PlasticNode(org_cfg["sense_node_id"], NodeKind.SENSE),),
            edges=(), kernel_limits=limits,
        ),
    )
    organism_state = OrganismTransitionState(
        owner_checkpoint=cognition.to_organism_checkpoint(),
        organism_step=org_cfg["step"],
        embodiment=EmbodimentContextRef(
            body_cfg["body_id"], body_cfg["episode_id"], body_cfg["morphology_revision"]
        ),
        lifecycle=OrganismLifecycle[org_cfg["lifecycle"]],
    )
    surface_cfg = body_cfg["actuator_surface"]
    surface = ActuatorSurface.from_effector_ids(
        tuple(surface_cfg["ids"]), fingerprint_material=surface_cfg["fingerprint_material"]
    )
    actuator_context = ActuatorEvidenceContext(
        body_cfg["body_id"], body_cfg["episode_id"], body_cfg["morphology_revision"],
        surface.contract_fingerprint,
    )
    differential_cfg = body_cfg["differential_binding"]
    engine = CompetenceDevelopmentEngine(
        surface.actuator_ids,
        organism_id=str(organism_id),
        embodiment_fingerprint=actuator_context.fingerprint,
    )
    composer = OrganismExplorationComposer(
        organism_id=str(organism_id), engine=engine,
        history=HistoryIndex() if organism_checkpoint == "v5" else None,
    )
    receptor_cfg = body_cfg["receptor"]
    receptor = EmbodimentReceptor(
        receptor_id=receptor_cfg["id"], body_id=body_cfg["body_id"],
        embodiment_episode_id=body_cfg["episode_id"], morphology_revision=body_cfg["morphology_revision"],
        body_frame=receptor_cfg["body_frame"], position_in_body_frame_m=tuple(receptor_cfg["position_in_body_frame_m"]),
    )
    differential_binding = EmbodimentDifferentialMotorBinding(
        binding_id=differential_cfg["id"], body_id=body_cfg["body_id"],
        embodiment_episode_id=body_cfg["episode_id"], morphology_revision=body_cfg["morphology_revision"],
        source_surface_fingerprint=surface.contract_fingerprint,
        pairs=(DifferentialMotorPair(
            joint_id=joint_cfg["id"],
            positive_actuator_id=differential_cfg["positive_actuator_id"],
            negative_actuator_id=differential_cfg["negative_actuator_id"], scale=differential_cfg["scale"],
        ),),
    )
    torque_cfg = body_cfg["torque_binding"]
    torque_binding = EmbodimentJointTorqueBinding(
        binding_id=torque_cfg["id"], body_id=body_cfg["body_id"],
        embodiment_episode_id=body_cfg["episode_id"], morphology_revision=body_cfg["morphology_revision"],
        source_surface_fingerprint=surface.contract_fingerprint,
        channels=(JointTorqueChannelBinding(torque_cfg["channel_id"], torque_cfg["joint_id"], tuple(torque_cfg["axis"])),),
    )
    solver = ArticulatedMujocoConfiguration(
        revision=solver_profile["revision"], timestep_s=solver_profile["timestep_s"],
        iterations=solver_profile["iterations"], tolerance=solver_profile["tolerance"],
        solref=tuple(solver_profile["solref"]), solimp=tuple(solver_profile["solimp"]),
        contact_policy=solver_profile["contact_policy"],
        velocity_regulation_policy="m1-fixed-single-hinge-grid-v2",
    )

    contracts: dict[tuple[str, str], Any] = {
        ("organism", str(organism_id)): OrganismCheckpointContract(
            kernel_limits=limits,
            read_dependency=(lambda digest: resolve_dependency(state_root / "owners", digest))
            if organism_checkpoint == "v5" else None,
        ),
        ("embodiment", body_cfg["body_id"]): _CheckpointContract(
            body_cfg["body_id"],
            {ARTICULATED_PHYSICAL_CHECKPOINT_SCHEMA: lambda encoded: _valid_body_checkpoint(encoded, morphology)},
        ),
        ("environment", env_cfg["owner_id"]): _CheckpointContract(
            env_cfg["owner_id"],
            {StaticEnvironmentOwner.CHECKPOINT_SCHEMA: lambda encoded: StaticEnvironmentOwner.validate_checkpoint(
                encoded, env_cfg["owner_id"], StaticEnvironmentOwner.CHECKPOINT_SCHEMA
            )},
        ),
        ("environment", field_cfg["id"]): _CheckpointContract(
            field_cfg["id"],
            {IdentifiedFiniteScalarField.OWNER_SCHEMA: lambda encoded: IdentifiedFiniteScalarField.validate_checkpoint(
                encoded, field_cfg["id"], IdentifiedFiniteScalarField.OWNER_SCHEMA
            )},
        ),
    }
    store = FileOwnerCheckpointStore(state_root / "owners")
    genesis_payloads = (
        OwnerCheckpointPayload("organism", str(organism_id), "organism-transition-v2", organism_state.to_bytes()),
        OwnerCheckpointPayload("embodiment", body_cfg["body_id"], ARTICULATED_PHYSICAL_CHECKPOINT_SCHEMA, articulated_body.checkpoint_bytes()),
        OwnerCheckpointPayload("environment", env_cfg["owner_id"], StaticEnvironmentOwner.CHECKPOINT_SCHEMA, env_owner.checkpoint_bytes()),
        OwnerCheckpointPayload("environment", field_cfg["id"], IdentifiedFiniteScalarField.OWNER_SCHEMA, environment_field.checkpoint_bytes()),
    )
    initial_refs = store.write_genesis(genesis_payloads)
    manifest_store = WorldCausalManifestStore(state_root / "manifest.json")
    journal_store = DurableJournalSnapshotStore(state_root / "journal", WorldId(world_id))

    def verify_owner_checkpoints(references) -> bool:
        try:
            for reference in references:
                if reference.owner == "world-journal":
                    continue
                contract = contracts[(reference.owner, reference.owner_identity)]
                encoded = Path(reference.locator).read_bytes()
                if not contract.validate_checkpoint(encoded, reference.owner_identity, reference.schema_version):
                    return False
            return True
        except (KeyError, OSError, TypeError, ValueError):
            return False

    presence_id = world_cfg["presence_id"]
    world = DurableWorldRuntime.initialize(
        WorldGenesisSpec(
            world_id=WorldId(world_id),
            initial_presence_ids=(presence_id,),
            registered_owner_ids=tuple(sorted(contracts)),
            initial_owner_references=initial_refs,
            owner_checkpoint_contracts=contracts,
            presence_owner_links=(PresenceOwnerLink(PresenceId(presence_id), "organism", str(organism_id)),),
        ),
        presence_id,
        manifest_store,
        journal_store,
        owner_checkpoint_verifier=verify_owner_checkpoints,
    )

    cycle_inputs = ScalarSensorimotorCycleInputs(
        environment_revision=env_cfg["revision"], solver_configuration=solver,
        current_organism=organism_state, action_composer=composer, receptor=receptor,
        embodiment_state=root, environment_field=environment_field, sampling_frame=frame,
        transduction=DimensionlessScalarFieldIdentityV1(),
        sense_binding=SenseChannelBinding(
            profile["modality"]["channel_id"], receptor_cfg["id"], profile["modality"]["sense_node_id"]
        ),
        kernel_limits=limits, actuator_context=actuator_context, actuator_surface=surface,
        articulated_body=articulated_body, active_morphology=morphology,
        embodiment_geometry_revision=body_cfg["geometry_revision"],
        differential_binding=differential_binding, torque_binding=torque_binding,
        environment_owner=env_owner, owner_checkpoint_store=store,
    )
    bridge = LocalObservationReadBridge.create(
        ObservationPlane(), host=profile["observation_bridge"]["host"], port=observation_port,
        allowed_origin=profile["observation_bridge"]["allowed_origin"],
        max_response_bytes=profile["observation_bridge"]["max_response_bytes"],
    )
    return BuiltM1Runtime(
        world=world, observation_plane=bridge.server.plane, cycle_inputs=cycle_inputs,
        source_revisions=source_revisions, profile_id=profile["profile_id"],
        profile_sha256=profile_sha256, observation_bridge=bridge,
    )


class _CheckpointContract:
    def __init__(self, identity: str, validators: dict[str, Any]) -> None:
        self.identity = identity
        self.validators = validators

    def validate_checkpoint(self, encoded: bytes, owner_identity: str, schema_version: str) -> bool:
        validator = self.validators.get(schema_version)
        return owner_identity == self.identity and validator is not None and validator(encoded)

    def verify_restoration(self, reference: Any) -> bool:
        try:
            return self.validate_checkpoint(
                Path(reference.locator).read_bytes(), reference.owner_identity, reference.schema_version
            )
        except OSError:
            return False


def _valid_body_checkpoint(encoded: bytes, morphology: Any) -> bool:
    from embodiment.physics3d.articulated_state import ArticulatedEmbodimentPhysicalState

    try:
        ArticulatedEmbodimentPhysicalState.restore_checkpoint(encoded, active_morphology=morphology)
    except (TypeError, ValueError):
        return False
    return True


__all__ = ["BuiltM1Runtime", "ProfileError", "construct_fresh_runtime", "load_operator_profile"]
