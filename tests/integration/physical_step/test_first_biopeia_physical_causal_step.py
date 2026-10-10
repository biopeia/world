"""Minimal end-to-end physical causal-step integration fixture.

This bounded integration executes one real MuJoCo step over an explicit
Environment-owned static scene, then exercises owner adoption, the World
durable manifest protocol, and passive semantic Observation. It is not a
scientific validation or a production solver adapter. Coverage is split across
the free-root scenario (command, adoption, commit, Observation and repeated
fresh-run determinism) and the articulated scenario (checkpoint restore and a
subsequent step). Local checkpoint writes do not test power-loss durability,
and restart continuation is not compared with uninterrupted replay.
"""

from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import mujoco
import pytest
from embodiment.checkpoint import EmbodimentCheckpoint
from embodiment.metabolism import MetabolicLedger
from embodiment.observation import BodyPoseSample
from embodiment.physics3d.actuation_projection import project_motor_command
from embodiment.physics3d.articulated_spec import ArticulatedBodySpec
from embodiment.physics3d.articulated_state import (
    ActiveArticulatedMorphology,
    ArticulatedEmbodimentPhysicalState,
    JointKinematicState,
    SolverArticulatedCandidate,
)
from embodiment.physics3d.mechanics import JointSpec, JointTopology, SegmentSpec
from embodiment.physics3d.physical_projection import project_body_physical_input
from embodiment.physics3d.physical_transition import (
    EmbodimentPhysicalState,
    PhysicalStepCorrelation,
    SolverBodyCandidate,
)
from embodiment.physiology import LivingBodyState
from environment import (
    FrameDescriptor,
    StaticEnvironmentOwner,
    StaticGroundPlane,
    StaticPhysicalScene,
    StaticSceneCandidate,
    project_static_scene,
)
from environment import (
    PhysicalStepCorrelation as EnvironmentStepCorrelation,
)
from observation.semantic import (
    AcquisitionBasis,
    ActorRef,
    AuthorityDomain,
    DomainTimeRef,
    IdentityAssertion,
    IdentityRef,
    SemanticClaim,
    SemanticContractRef,
    SemanticFamily,
    SemanticObservation,
    SubjectScope,
)
from organism.actuation import (
    ActionArbitrator,
    ActionCommitment,
    ActionEvaluation,
    ActionJustification,
    ActionProposal,
    ActionSource,
    ActuatorSurface,
    MotorIntent,
    SequenceController,
    motor_command_from_frame,
)
from organism.agency import (
    ActionAffordance,
    ActionIntent,
    ExecutiveAdmissionPolicy,
    IntentStatus,
    admit_afforded_action,
)
from organism.identity import OrganismIdentity

from world.durable_runtime import (
    DurableJournalSnapshotStore,
    DurableWorldRuntime,
    PresenceOwnerLink,
    WorldGenesisSpec,
)
from world.manifest import (
    CheckpointReference,
    ManifestStatus,
    WorldCausalManifestStore,
)
from world.physical_step import (
    PhysicalStepCandidate,
    PhysicalStepContext,
    PhysicalStepPorts,
    run_physical_step,
)
from world.runtime import PresenceId, WorldId

WORLD_ID = "e2e-physical-world"
ORGANISM_ID = OrganismIdentity.parse("12345678-1234-5678-1234-567812345678")
BODY_ID = "body.integration-1"
EPISODE_ID = "episode.integration-1"
MORPHOLOGY_REVISION = "morphology.integration-r1"
ENVIRONMENT_REVISION = "environment.integration-r1"
ENVIRONMENT_ID = "environment.integration-1"
SOLVER_CONFIGURATION_REVISION = f"mujoco-{mujoco.__version__}-free-root-box-euler-dt0.01-ground-half-extent-10m-r2"


@dataclass(frozen=True)
class _Effector:
    effector_id: str
    quantity: str = "applied_force"
    unit: str = "N"
    minimum: float = 0.0
    neutral: float = 0.0
    maximum: float = 1.0
    saturation: str = "reject"


@dataclass(frozen=True)
class _PhysicalChannel:
    actuator_id: str
    effector_id: str
    command_kind: str = "force"
    quantity: str = "applied_force"
    unit: str = "N"
    minimum: float = 0.0
    neutral: float = 0.0
    maximum: float = 1.0
    saturation: str = "reject"
    available: bool = True


@dataclass(frozen=True)
class _PhysicalSurface:
    contract_fingerprint: str
    morphology_revision: str
    channels: tuple[_PhysicalChannel, ...]


@dataclass(frozen=True)
class _Morphology:
    body_id: str
    embodiment_episode_id: str
    morphology_revision: str
    effectors: tuple[_Effector, ...]


class _FilesystemOwnerContract:
    """Test verifier/restorer contract for immutable checkpoint files."""

    def __init__(
        self,
        *,
        owner_identity: str,
        schema_version: str,
        validate: Callable[[bytes], bool],
    ) -> None:
        self._owner_identity = owner_identity
        self._schema_version = schema_version
        self._validate = validate

    def validate_checkpoint(self, encoded: bytes, owner_identity: str, schema_version: str) -> bool:
        if owner_identity != self._owner_identity or schema_version != self._schema_version:
            return False
        try:
            return self._validate(encoded) is True
        except Exception:
            return False

    def verify_restoration(self, encoded: bytes, owner_identity: str, schema_version: str) -> bool:
        return self.validate_checkpoint(encoded, owner_identity, schema_version)


def _write_checkpoint(
    directory: Path,
    *,
    owner: str,
    owner_identity: str,
    checkpoint_id: str,
    schema: str,
    encoded: bytes,
) -> CheckpointReference:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{owner}-{checkpoint_id}.bin"
    path.write_bytes(encoded)
    return CheckpointReference(
        owner=owner,
        owner_identity=owner_identity,
        checkpoint_id=checkpoint_id,
        schema_version=schema,
        locator=str(path),
        sha256=hashlib.sha256(encoded).hexdigest(),
    )


def _free_root_box_spec() -> ArticulatedBodySpec:
    return ArticulatedBodySpec(
        body_kind="free-root-box-fixture",
        state_schema_version=1,
        spawn_height=0.0,
        base_link_name="body.root",
        joint_specs=(),
        joint_topology=(),
        segments={"body.root": SegmentSpec(1.0, (0.2, 0.2, 0.2), (0.0, 0.0, 0.0), (0.5, 0.5, 0.5, 1.0))},
        contact_link_names=("body.root",),
        structural_neighbour_names=(),
    )


def _environment_owner() -> StaticEnvironmentOwner:
    scene = StaticPhysicalScene(
        environment_id=ENVIRONMENT_ID,
        environment_revision=ENVIRONMENT_REVISION,
        frame=FrameDescriptor(
            frame_id="physics3d.world",
            parent_frame_id="world",
            origin_m=(0.0, 0.0, 0.0),
            x_axis_in_parent=(1.0, 0.0, 0.0),
            y_axis_in_parent=(0.0, 1.0, 0.0),
            z_axis_in_parent=(0.0, 0.0, 1.0),
            handedness="RIGHT_HANDED",
            up_axis="Z",
        ),
        gravity_m_s2=(0.0, 0.0, -9.81),
        objects=(StaticGroundPlane("ground.z0", (0.0, 0.0, 1.0), 0.0),),
    )
    return StaticEnvironmentOwner(
        scene, owner_id=ENVIRONMENT_ID, world_id=WORLD_ID
    )


def _execute_mujoco_step(body_input, scene, actuation):
    """Execute one step for the explicitly projected free-root box fixture."""
    frame = scene.frame
    if (
        body_input.body_kind != "free-root-box-fixture"
        or body_input.reference_frame != "physics3d.world"
        or frame.frame_id != body_input.reference_frame
        or frame.parent_frame_id != "world"
        or frame.origin_m != (0.0, 0.0, 0.0)
        or frame.x_axis_in_parent != (1.0, 0.0, 0.0)
        or frame.y_axis_in_parent != (0.0, 1.0, 0.0)
        or frame.z_axis_in_parent != (0.0, 0.0, 1.0)
        or frame.handedness != "RIGHT_HANDED"
        or frame.up_axis != "Z"
    ):
        raise ValueError("fixture solver supports only the declared identity physics3d.world frame")
    if len(scene.objects) != 1 or scene.objects[0].normal != (0.0, 0.0, 1.0):
        raise ValueError("fixture solver supports only one +Z ground plane")
    # Consume the actual Embodiment-projected morphology rather than silently
    # substituting an unrelated solver geometry. This fixture deliberately
    # supports exactly one root link with one box collision and explicit mass.
    projected = ET.fromstring(body_input.articulated_description)
    links = projected.findall("link")
    if len(links) != 1 or links[0].attrib.get("name") != "body.root" or projected.findall("joint"):
        raise ValueError("fixture solver requires the projected single-link free-root morphology")
    mass_node = links[0].find("inertial/mass")
    box_node = links[0].find("collision/geometry/box")
    if mass_node is None or box_node is None:
        raise ValueError("fixture solver requires projected root mass and box collision geometry")
    mass = float(mass_node.attrib["value"])
    size = tuple(float(value) for value in box_node.attrib["size"].split())
    if mass != 1.0 or size != (0.2, 0.2, 0.2):
        raise ValueError("fixture solver supports only a 1 kg, 0.2 m root box")
    plane = scene.objects[0]
    gravity = " ".join(str(value) for value in scene.gravity_m_s2)
    xml = f"""
    <mujoco model="biopeia-physical-step-fixture">
      <compiler angle="radian" coordinate="local"/>
      <option timestep="0.01" integrator="Euler" gravity="{gravity}"/>
      <worldbody>
        <geom name="ground" type="plane" pos="0 0 {plane.offset_m}" size="10 10 0.1"/>
        <body name="fixture-body" pos="{' '.join(str(value) for value in body_input.position_m)}">
          <freejoint name="root.free"/>
          <geom name="body.root.collider" type="box" size="0.1 0.1 0.1" mass="1"/>
        </body>
      </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "fixture-body")
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root.free")
    qpos_address = int(model.jnt_qposadr[joint_id])
    qvel_address = int(model.jnt_dofadr[joint_id])
    data.qpos[qpos_address : qpos_address + 7] = (
        *body_input.position_m,
        body_input.orientation_xyzw[3],
        *body_input.orientation_xyzw[:3],
    )
    data.qvel[qvel_address : qvel_address + 6] = (
        *body_input.linear_velocity_m_per_s,
        *body_input.angular_velocity_rad_per_s,
    )
    mujoco.mj_forward(model, data)
    for channel in actuation.channels:
        if channel.effector_id != "root.force.x" or channel.quantity != "applied_force" or channel.unit != "N":
            raise ValueError("fixture solver requires the declared root.force.x channel in newtons")
        # MuJoCo's xfrc_applied is a world-frame force/torque at the body's COM.
        data.xfrc_applied[body_id, 0] = channel.value
    mujoco.mj_step(model, data)
    qpos = tuple(float(value) for value in data.qpos[qpos_address : qpos_address + 7])
    position = qpos[:3]
    quaternion_wxyz = qpos[3:]
    linear_velocity = tuple(float(value) for value in data.qvel[qvel_address : qvel_address + 3])
    angular_velocity = tuple(float(value) for value in data.qvel[qvel_address + 3 : qvel_address + 6])
    return position, (quaternion_wxyz[1], quaternion_wxyz[2], quaternion_wxyz[3], quaternion_wxyz[0]), linear_velocity, angular_velocity


def _store_initial_state(root: Path, environment: StaticEnvironmentOwner, physical: EmbodimentPhysicalState):
    organism_payload = json.dumps(
        {"organism_id": str(ORGANISM_ID), "controller_id": "controller.integration", "initial_step": 0},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    references = (
        _write_checkpoint(
            root,
            owner="organism",
            owner_identity=str(ORGANISM_ID),
            checkpoint_id="organism-tick-0",
            schema="organism-action-fixture-v1",
            encoded=organism_payload,
        ),
        _write_checkpoint(
            root,
            owner="embodiment",
            owner_identity=BODY_ID,
            checkpoint_id="embodiment-tick-0",
            schema="embodiment-physical-state-v2",
            encoded=physical.checkpoint_bytes(),
        ),
        _write_checkpoint(
            root,
            owner="environment",
            owner_identity=ENVIRONMENT_ID,
            checkpoint_id="environment-tick-0",
            schema=StaticEnvironmentOwner.CHECKPOINT_SCHEMA,
            encoded=environment.checkpoint_bytes(),
        ),
    )
    return references, organism_payload


def _run_valid_step(
    root: Path, *, reverse_surface: bool = True, wrong_environment_world: bool = False
) -> tuple[dict[str, object], tuple[SemanticObservation[object], ...]]:
    world_id = WorldId(WORLD_ID)
    environment = _environment_owner()
    initial_pose = BodyPoseSample(
        body_id=BODY_ID,
        reference_frame="physics3d.world",
        body_frame="body.root",
        position_m=(0.0, 0.0, 0.5),
        orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
        sample_kind="genesis",
        sample_value=0,
    )
    physical_state = EmbodimentPhysicalState(
        world_id=WORLD_ID,
        causal_tick=0,
        body_id=BODY_ID,
        embodiment_episode_id=EPISODE_ID,
        morphology_revision=MORPHOLOGY_REVISION,
        pose=initial_pose,
        linear_velocity_m_per_s=(0.0, 0.0, 0.0),
        angular_velocity_rad_per_s=(0.0, 0.0, 0.0),
    )
    initial_references, organism_initial_payload = _store_initial_state(root / "owners", environment, physical_state)

    def validate_embodiment(encoded: bytes) -> bool:
        restored = EmbodimentPhysicalState.restore_checkpoint(encoded)
        return (
            restored.body_id == BODY_ID
            and restored.embodiment_episode_id == EPISODE_ID
            and restored.world_id == WORLD_ID
        )

    def validate_environment(encoded: bytes) -> bool:
        return StaticEnvironmentOwner.validate_checkpoint(
            encoded, ENVIRONMENT_ID, StaticEnvironmentOwner.CHECKPOINT_SCHEMA
        )

    def validate_organism(encoded: bytes) -> bool:
        payload = json.loads(encoded)
        return payload.get("organism_id") == str(ORGANISM_ID)

    contracts = {
        ("organism", str(ORGANISM_ID)): _FilesystemOwnerContract(
            owner_identity=str(ORGANISM_ID),
            schema_version="organism-action-fixture-v1",
            validate=validate_organism,
        ),
        ("embodiment", BODY_ID): _FilesystemOwnerContract(
            owner_identity=BODY_ID,
            schema_version="embodiment-physical-state-v2",
            validate=validate_embodiment,
        ),
        ("environment", ENVIRONMENT_ID): _FilesystemOwnerContract(
            owner_identity=ENVIRONMENT_ID,
            schema_version=StaticEnvironmentOwner.CHECKPOINT_SCHEMA,
            validate=validate_environment,
        ),
    }
    presence_id = PresenceId("presence.physical-e2e")
    genesis = WorldGenesisSpec(
        world_id=world_id,
        initial_presence_ids=(presence_id.value,),
        registered_owner_ids=tuple(sorted(contracts)),
        initial_owner_references=initial_references,
        owner_checkpoint_contracts=contracts,
        presence_owner_links=(
            PresenceOwnerLink(presence_id, "organism", str(ORGANISM_ID)),
        ),
    )
    manifest_store = WorldCausalManifestStore(root / "world-manifest.json")
    journal_store = DurableJournalSnapshotStore(root / "journal", world_id)
    durable_world = DurableWorldRuntime.initialize(
        genesis,
        presence_id.value,
        manifest_store,
        journal_store,
        owner_checkpoint_verifier=lambda refs: all(
            contracts[(ref.owner, ref.owner_identity)].validate_checkpoint(Path(ref.locator).read_bytes(), ref.owner_identity, ref.schema_version)
            for ref in refs
        ),
    )

    actuator_surface = ActuatorSurface.from_effector_ids(
        ("joint.elbow", "joint.wrist"),
        fingerprint_material="physical-e2e-surface-v1",
    )
    morphology = _Morphology(
        BODY_ID,
        EPISODE_ID,
        MORPHOLOGY_REVISION,
        (_Effector("root.force.x"), _Effector("root.force.y")),
    )
    entries = (
        _PhysicalChannel(actuator_surface.actuator_ids[0], "root.force.x"),
        _PhysicalChannel(actuator_surface.actuator_ids[1], "root.force.y"),
    )
    if reverse_surface:
        entries = tuple(reversed(entries))
    surface = _PhysicalSurface(actuator_surface.contract_fingerprint, MORPHOLOGY_REVISION, entries)
    # This exercises the target-native admitted action path from an explicit,
    # deterministic primitive readout. It is not a claim of autonomous
    # cognition or learning. SequenceController below is only the low-level
    # controller that materializes a command for the selected commitment.
    affordance = ActionAffordance(
        affordance_id="affordance.physical-e2e",
        competence_id="competence.fixture",
        anticipated_effect_id="effect.forward",
        context_ref="context.physical-e2e",
        embodiment_id=EPISODE_ID,
        prediction_confidence=0.4,
        controllability=0.8,
        executability_confidence=0.9,
        prediction_ref="prediction.physical-e2e",
        evidence_refs=("causal-evidence.physical-e2e",),
    )
    decision = admit_afforded_action(
        affordances=(affordance,),
        primitive_readouts={"competence.fixture": 0.9},
        epistemic_value=lambda _competence_id: None,
        homeostatic_relevance=lambda _competence_id: 0.0,
        policy=ExecutiveAdmissionPolicy(readout_threshold=0.1, minimum_relevance=0.2),
    )
    assert decision is not None
    assert decision.competence_id == "competence.fixture"
    assert decision.supporting_affordance_id == affordance.affordance_id
    intent = ActionIntent(
        intent_id="intent.physical-e2e",
        competence_id=decision.competence_id,
        anticipated_effect_id=decision.anticipated_effect_id,
        context_ref=affordance.context_ref,
        embodiment_id=affordance.embodiment_id,
        origin_refs=decision.origin_refs,
        prediction_ref=decision.prediction_ref,
        confidence=decision.confidence,
        epistemic_relevance=decision.epistemic_relevance,
        homeostatic_relevance=decision.homeostatic_relevance,
        supporting_affordance_id=decision.supporting_affordance_id,
        created_tick=0,
        activated_tick=None,
        last_progress_tick=0,
        status=IntentStatus.PENDING,
        termination_reason=None,
        admission=decision.admission,
    )
    intent.transition(IntentStatus.ACTIVE, tick=1)
    assert intent.status is IntentStatus.ACTIVE
    proposal = ActionProposal(
        proposal_id="proposal.physical-e2e",
        source=ActionSource.PROSPECTION,
        effect_target_id=decision.anticipated_effect_id,
        competence_id=decision.competence_id,
        justification=ActionJustification(
            effect_target_id=decision.anticipated_effect_id,
            competence_id=decision.competence_id,
            prediction_id=decision.prediction_ref,
            evidence_refs=decision.origin_refs,
        ),
        evaluation=ActionEvaluation(
            epistemic_relevance=decision.epistemic_relevance,
            homeostatic_relevance=decision.homeostatic_relevance,
            effect_confidence=decision.confidence,
            controllability=affordance.controllability,
            uncertainty=1.0 - decision.confidence,
        ),
        intent_id=intent.intent_id,
    )
    arbitration = ActionArbitrator().choose(proposals=(proposal,), tick=1)
    assert arbitration.proposal is proposal
    commitment = ActionCommitment(
        commitment_id="commitment.physical-e2e",
        proposal_id=proposal.proposal_id,
        effect_target_id=proposal.effect_target_id,
        competence_id=proposal.competence_id,
        started_tick=1,
        controller_id="controller.integration",
        surface_fingerprint=actuator_surface.contract_fingerprint,
        embodiment_id=EPISODE_ID,
        intent_id=intent.intent_id,
    )
    assert commitment.intent_id == intent.intent_id
    # One command channel only: the missing right channel remains NO COMMAND.
    controller = SequenceController(
        "controller.integration",
        ((MotorIntent(actuator_surface.actuator_ids[0], 0.5),),),
    )
    # Modality is not in this outbound path: its current target contract is
    # sensory transduction only, not motor-command adaptation.
    staged: dict[str, object] = {}

    def project_body(context: PhysicalStepContext):
        assert manifest_store.read().status is ManifestStatus.PREPARING
        correlation = PhysicalStepCorrelation(
            world_id=context.world_id,
            causal_tick=context.source_causal_tick,
            physical_step_id=context.physical_step_id,
            body_id=context.body_id,
            embodiment_episode_id=context.embodiment_episode_id,
            morphology_revision=context.morphology_revision,
            environment_revision=context.environment_revision,
            solver_config_revision=context.solver_config_revision,
        )
        assert correlation.causal_tick == physical_state.causal_tick
        projected_body = project_body_physical_input(
            _free_root_box_spec(),
            physical_state.pose,
            correlation=correlation,
            linear_velocity_m_per_s=physical_state.linear_velocity_m_per_s,
            angular_velocity_rad_per_s=physical_state.angular_velocity_rad_per_s,
        )
        frame = controller.frame(commitment, tick=context.target_causal_tick)
        command = motor_command_from_frame(
            commitment,
            frame,
            command_id=f"motor-command-{context.target_causal_tick}",
            issued_at_tick=context.target_causal_tick,
        )
        assert command.commitment_id == commitment.commitment_id
        assert command.competence_id == decision.competence_id
        physiology = LivingBodyState(energy_reserve=1.0, max_energy=2.0)
        embodiment_checkpoint = EmbodimentCheckpoint(
            BODY_ID, EPISODE_ID, physiology, MetabolicLedger(body_state=physiology)
        )
        actuation = project_motor_command(
            command,
            surface,
            morphology,
            body_pose=physical_state.pose,
            embodiment_checkpoint=embodiment_checkpoint,
            command_kind="force",
        )
        assert len(actuation.channels) == 1
        assert actuation.channels[0].effector_id == "root.force.x"
        assert actuation.channels[0].value == 0.5
        staged.update(command=command, actuation=actuation, correlation=correlation)
        return projected_body, actuation

    def project_environment(context: PhysicalStepContext):
        assert context.environment_revision == environment.scene.environment_revision
        # This expected owner correlation is derived from World authority before
        # solver execution; adoption must never trust the candidate to define it.
        staged["environment_expected_correlation"] = EnvironmentStepCorrelation(
            world_id=context.world_id,
            causal_tick=context.source_causal_tick,
            physical_step_id=context.physical_step_id,
            environment_id=environment.scene.environment_id,
            environment_revision=context.environment_revision,
        )
        return environment.scene

    def execute_solver(context, body_projection, environment_projection):
        projected_body, actuation = body_projection
        position_m, orientation_xyzw, linear_velocity_m_per_s, angular_velocity_rad_per_s = (
            _execute_mujoco_step(projected_body, environment_projection, actuation)
        )
        correlation = PhysicalStepCorrelation(
            world_id=context.world_id,
            causal_tick=context.source_causal_tick,
            physical_step_id=context.physical_step_id,
            body_id=context.body_id,
            embodiment_episode_id=context.embodiment_episode_id,
            morphology_revision=context.morphology_revision,
            environment_revision=context.environment_revision,
            solver_config_revision=context.solver_config_revision,
        )
        body_candidate = SolverBodyCandidate(
            correlation=correlation,
            reference_frame=projected_body.reference_frame,
            body_frame=projected_body.body_frame,
            position_m=position_m,
            orientation_xyzw=orientation_xyzw,
            linear_velocity_m_per_s=linear_velocity_m_per_s,
            angular_velocity_rad_per_s=angular_velocity_rad_per_s,
        )
        environment_correlation = EnvironmentStepCorrelation(
            world_id=("unrelated-world" if wrong_environment_world else context.world_id),
            causal_tick=context.source_causal_tick,
            physical_step_id=context.physical_step_id,
            environment_id=environment_projection.environment_id,
            environment_revision=environment_projection.environment_revision,
        )
        environment_candidate = StaticSceneCandidate(
            environment_correlation, project_static_scene(environment_projection)
        )
        return PhysicalStepCandidate(body_candidate, environment_candidate)

    def adopt_body(context: PhysicalStepContext, candidate: SolverBodyCandidate) -> None:
        correlation = staged["correlation"]
        adopted_state = physical_state.adopt_candidate(candidate, expected_step=correlation)
        assert adopted_state.causal_tick == context.target_causal_tick
        staged.update(candidate=candidate, adopted=adopted_state)

    def adopt_environment(context: PhysicalStepContext, candidate: StaticSceneCandidate) -> None:
        expected_correlation = staged["environment_expected_correlation"]
        assert isinstance(expected_correlation, EnvironmentStepCorrelation)
        assert expected_correlation.world_id == context.world_id
        assert expected_correlation.causal_tick == context.source_causal_tick
        environment.validate_and_adopt(
            candidate,
            expected_correlation=expected_correlation,
        )
        environment_projection = {
            "environment_revision": environment.scene.environment_revision,
            "frame_id": environment.scene.frame.frame_id,
            "gravity_m_s2": environment.scene.gravity_m_s2,
            "ground_planes": tuple(
                (item.object_id, item.normal, item.offset_m)
                for item in environment.scene.objects
            ),
            "accepted_tick": environment.last_adopted_tick,
        }
        assert environment.last_adopted_tick == context.source_causal_tick
        staged["environment"] = environment_projection

    def write_owner_checkpoints(context: PhysicalStepContext) -> tuple[CheckpointReference, ...]:
        staged["checkpoint_writes_started"] = True
        causal_tick = context.target_causal_tick
        adopted_state = staged["adopted"]
        command = staged["command"]
        assert isinstance(adopted_state, EmbodimentPhysicalState)
        correlation = PhysicalStepCorrelation(
            world_id=context.world_id,
            causal_tick=context.source_causal_tick,
            physical_step_id=context.physical_step_id,
            body_id=context.body_id,
            embodiment_episode_id=context.embodiment_episode_id,
            morphology_revision=context.morphology_revision,
            environment_revision=context.environment_revision,
            solver_config_revision=context.solver_config_revision,
        )
        assert adopted_state.causal_tick == correlation.causal_tick + 1 == causal_tick
        previous_organism_state = json.loads(organism_initial_payload)
        next_organism_state = {
            **previous_organism_state,
            "last_command_id": command.command_id,
            "last_command_tick": causal_tick,
        }
        organism_payload = json.dumps(next_organism_state, sort_keys=True, separators=(",", ":")).encode()
        return (
            _write_checkpoint(
                root / "owners",
                owner="organism",
                owner_identity=str(ORGANISM_ID),
                checkpoint_id=f"organism-tick-{causal_tick}",
                schema="organism-action-fixture-v1",
                encoded=organism_payload,
            ),
            _write_checkpoint(
                root / "owners",
                owner="embodiment",
                owner_identity=BODY_ID,
                checkpoint_id=f"embodiment-tick-{causal_tick}",
                schema="embodiment-physical-state-v2",
                encoded=adopted_state.checkpoint_bytes(),
            ),
            _write_checkpoint(
                root / "owners",
                owner="environment",
                owner_identity=ENVIRONMENT_ID,
                checkpoint_id=f"environment-tick-{causal_tick}",
                schema=StaticEnvironmentOwner.CHECKPOINT_SCHEMA,
                encoded=environment.checkpoint_bytes(),
            ),
        )

    ports = PhysicalStepPorts(
        project_body=project_body,
        project_environment=project_environment,
        execute_solver=execute_solver,
        adopt_body=adopt_body,
        adopt_environment=adopt_environment,
        write_checkpoints=write_owner_checkpoints,
    )
    if wrong_environment_world:
        with pytest.raises(ValueError, match="candidate correlation does not match expected"):
            run_physical_step(
                durable_world,
                commit_id="physical-step-commit-1",
                physical_step_id="physical-step-1",
                body_id=BODY_ID,
                embodiment_episode_id=EPISODE_ID,
                morphology_revision=MORPHOLOGY_REVISION,
                environment_revision=ENVIRONMENT_REVISION,
                solver_config_revision=SOLVER_CONFIGURATION_REVISION,
                ports=ports,
            )
        assert durable_world.status is ManifestStatus.RECOVERY_REQUIRED
        assert durable_world.causal_tick == 0
        assert durable_world.manifest.pending_tick == 1
        assert environment.last_adopted_tick == -1
        assert staged.get("checkpoint_writes_started") is not True
        assert staged["environment_expected_correlation"].world_id == WORLD_ID
        assert staged["environment_expected_correlation"].physical_step_id == "physical-step-1"
        return {"status": durable_world.status.value, "tick": durable_world.causal_tick}, ()

    record = run_physical_step(
        durable_world,
        commit_id="physical-step-commit-1",
        physical_step_id="physical-step-1",
        body_id=BODY_ID,
        embodiment_episode_id=EPISODE_ID,
        morphology_revision=MORPHOLOGY_REVISION,
        environment_revision=ENVIRONMENT_REVISION,
        solver_config_revision=SOLVER_CONFIGURATION_REVISION,
        ports=ports,
    )
    assert durable_world.status is ManifestStatus.COMMITTED
    assert durable_world.causal_tick == 1
    adopted = staged["adopted"]
    assert isinstance(adopted, EmbodimentPhysicalState)
    body_reference = next(ref for ref in durable_world.manifest.checkpoint_references if ref.owner == "embodiment")
    restored = EmbodimentPhysicalState.restore_checkpoint(Path(body_reference.locator).read_bytes())
    assert restored == adopted
    assert restored.causal_tick == staged["correlation"].causal_tick + 1 == record.causal_tick
    assert restored.last_physical_step == staged["correlation"]

    body_identity = IdentityAssertion(
        IdentityRef("body_id", BODY_ID, ActorRef("domain", "embodiment"), scope=EPISODE_ID),
        ActorRef("boundary", "embodiment-owner"),
    )
    world_identity = IdentityAssertion(
        IdentityRef("world_id", WORLD_ID, ActorRef("domain", "world")),
        ActorRef("boundary", "world-owner"),
    )
    environment_identity = IdentityAssertion(
        IdentityRef("environment_id", ENVIRONMENT_ID, ActorRef("domain", "environment")),
        ActorRef("boundary", "environment-owner"),
    )
    body_observation = SemanticObservation(
        claim=SemanticClaim(
            family=SemanticFamily(AuthorityDomain.EMBODIMENT, ("body", "pose"), SemanticContractRef("body-pose", "v1")),
            subject=SubjectScope("body", identities=(body_identity,)),
            epistemic_kind="authoritative_state",
            value={
                "position_m": restored.pose.position_m,
                "orientation_xyzw": restored.pose.orientation_xyzw,
                "position_unit": "m",
                "linear_velocity_m_per_s": restored.linear_velocity_m_per_s,
                "angular_velocity_rad_per_s": restored.angular_velocity_rad_per_s,
                "reference_frame": restored.pose.reference_frame,
                "body_frame": restored.pose.body_frame,
            },
        ),
        acquisition=AcquisitionBasis("authoritative_state"),
        asserter=ActorRef("boundary", "embodiment-owner"),
        producer=ActorRef("component", "observation-adapter"),
        domain_time=(
            DomainTimeRef(AuthorityDomain.EMBODIMENT, "world_causal_tick", restored.causal_tick),
            DomainTimeRef(AuthorityDomain.EMBODIMENT, "morphology_revision", restored.morphology_revision),
        ),
    )
    environment_projection = staged["environment"]
    assert isinstance(environment_projection, dict)
    environment_observation = SemanticObservation(
        claim=SemanticClaim(
            family=SemanticFamily(AuthorityDomain.ENVIRONMENT, ("static_scene",), SemanticContractRef("environment-static-scene", "v1")),
            subject=SubjectScope("environment", identities=(environment_identity,)),
            epistemic_kind="authoritative_state",
            value=environment_projection,
        ),
        acquisition=AcquisitionBasis("authoritative_state"),
        asserter=ActorRef("boundary", "environment-owner"),
        producer=ActorRef("component", "observation-adapter"),
        domain_time=(DomainTimeRef(AuthorityDomain.ENVIRONMENT, "environment_revision", ENVIRONMENT_REVISION),),
        context_time=(DomainTimeRef(AuthorityDomain.WORLD, "causal_tick", record.causal_tick),),
    )
    world_observation = SemanticObservation(
        claim=SemanticClaim(
            family=SemanticFamily(AuthorityDomain.WORLD, ("causal_cut",), SemanticContractRef("world-causal-cut", "v1")),
            subject=SubjectScope("world", identities=(world_identity,)),
            epistemic_kind="authoritative_state",
            value={"commit_id": durable_world.manifest.commit_id, "status": durable_world.manifest.status.value},
        ),
        acquisition=AcquisitionBasis("authoritative_state"),
        asserter=ActorRef("boundary", "world-owner"),
        producer=ActorRef("component", "observation-adapter"),
        domain_time=(DomainTimeRef(AuthorityDomain.WORLD, "causal_tick", record.causal_tick),),
    )
    # Observation is created only after the World manifest is COMMITTED.
    assert durable_world.manifest.status is ManifestStatus.COMMITTED
    return (
        {
            "record_id": record.record_id,
            "tick": record.causal_tick,
            "command_channels": staged["command"].channels,
            "commanded_actuator_id": actuator_surface.actuator_ids[0],
            "omitted_actuator_id": actuator_surface.actuator_ids[1],
            "effector_ids": tuple(channel.effector_id for channel in staged["actuation"].channels),
            "adopted_pose": restored.pose.position_m,
            "checkpoint_digest": body_reference.sha256,
            "environment": environment_projection,
        },
        (world_observation, body_observation, environment_observation),
    )


def _run_stale_morphology(root: Path) -> tuple[ManifestStatus, int, int]:
    """Reuse the setup via a valid helper's construction, but fail before cut publication."""
    # The runtime setup is deliberately minimal and separate so the failure
    # assertion observes the real DurableWorldRuntime failure protocol.
    world_id = WorldId(WORLD_ID)
    environment = _environment_owner()
    initial_pose = BodyPoseSample(BODY_ID, "physics3d.world", "body.root", (0.0, 0.0, 0.5), (0.0, 0.0, 0.0, 1.0), "genesis", 0)
    physical = EmbodimentPhysicalState(
        WORLD_ID,
        0,
        BODY_ID,
        EPISODE_ID,
        MORPHOLOGY_REVISION,
        initial_pose,
        linear_velocity_m_per_s=(0.0, 0.0, 0.0),
        angular_velocity_rad_per_s=(0.0, 0.0, 0.0),
    )
    refs, organism_payload = _store_initial_state(root / "owners", environment, physical)
    validators: dict[tuple[str, str], tuple[str, Callable[[bytes], bool]]] = {
        ("organism", str(ORGANISM_ID)): (
            "organism-action-fixture-v1",
            lambda b: json.loads(b).get("organism_id") == str(ORGANISM_ID),
        ),
        ("embodiment", BODY_ID): (
            "embodiment-physical-state-v2",
            lambda b: (
                EmbodimentPhysicalState.restore_checkpoint(b).body_id == BODY_ID
                and EmbodimentPhysicalState.restore_checkpoint(b).embodiment_episode_id == EPISODE_ID
            ),
        ),
        ("environment", ENVIRONMENT_ID): (
            StaticEnvironmentOwner.CHECKPOINT_SCHEMA,
            lambda b: StaticEnvironmentOwner.validate_checkpoint(
                b, ENVIRONMENT_ID, StaticEnvironmentOwner.CHECKPOINT_SCHEMA
            ),
        ),
    }
    contracts = {
        key: _FilesystemOwnerContract(
            owner_identity=key[1], schema_version=schema, validate=validate
        )
        for key, (schema, validate) in validators.items()
    }
    presence_id = PresenceId("presence.stale-physical-e2e")
    manifest_store = WorldCausalManifestStore(root / "world-manifest.json")
    runtime = DurableWorldRuntime.initialize(
        WorldGenesisSpec(
            world_id,
            (presence_id.value,),
            tuple(sorted(contracts)),
            refs,
            contracts,
            (PresenceOwnerLink(presence_id, "organism", str(ORGANISM_ID)),),
        ),
        presence_id.value,
        manifest_store,
        DurableJournalSnapshotStore(root / "journal", world_id),
        owner_checkpoint_verifier=lambda owner_refs: all(
            contracts[(ref.owner, ref.owner_identity)].validate_checkpoint(Path(ref.locator).read_bytes(), ref.owner_identity, ref.schema_version)
            for ref in owner_refs
        ),
    )
    expected = PhysicalStepCorrelation(WORLD_ID, 0, "step-stale", BODY_ID, EPISODE_ID, MORPHOLOGY_REVISION, ENVIRONMENT_REVISION, SOLVER_CONFIGURATION_REVISION)
    stale = replace(expected, morphology_revision="morphology.previous")
    adopted_candidates: list[EmbodimentPhysicalState] = []

    def execute_stale_solver(context, _body, _environment):
        candidate = SolverBodyCandidate(
            stale,
            "physics3d.world",
            "body.root",
            (0.1, 0.0, 0.5),
            (0.0, 0.0, 0.0, 1.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
        )
        return PhysicalStepCandidate(candidate, None)

    def adopt_stale_body(_context, candidate):
        adopted_candidates.append(physical.adopt_candidate(candidate, expected_step=expected))

    ports = PhysicalStepPorts(
        project_body=lambda _context: None,
        project_environment=lambda _context: None,
        execute_solver=execute_stale_solver,
        adopt_body=adopt_stale_body,
        adopt_environment=lambda _context, _candidate: None,
        write_checkpoints=lambda _context: pytest.fail("stale candidate reached checkpoint writing"),
    )
    with pytest.raises(ValueError, match="authorized physical step"):
        run_physical_step(
            runtime,
            commit_id="stale-physical-step",
            physical_step_id="step-stale",
            body_id=BODY_ID,
            embodiment_episode_id=EPISODE_ID,
            morphology_revision=MORPHOLOGY_REVISION,
            environment_revision=ENVIRONMENT_REVISION,
            solver_config_revision=SOLVER_CONFIGURATION_REVISION,
            ports=ports,
        )
    manifest = manifest_store.read()
    assert physical.causal_tick == 0
    assert physical.pose.position_m == initial_pose.position_m
    assert adopted_candidates == []
    return manifest.status, manifest.causal_tick, manifest.pending_tick or 0


def test_first_biopeia_end_to_end_physical_causal_step_is_deterministic(tmp_path: Path) -> None:
    baseline, observations = _run_valid_step(tmp_path / "run-a")
    repeated, repeated_observations = _run_valid_step(tmp_path / "run-b")
    permuted, permuted_observations = _run_valid_step(tmp_path / "run-c", reverse_surface=False)

    assert baseline == repeated == permuted
    assert observations == repeated_observations == permuted_observations
    assert baseline["tick"] == 1
    assert baseline["record_id"] == f"{WORLD_ID}:tick:1"
    assert len(baseline["command_channels"]) == 1
    assert baseline["command_channels"] == ((baseline["commanded_actuator_id"], 0.5),)
    assert baseline["omitted_actuator_id"] not in dict(baseline["command_channels"])
    assert baseline["effector_ids"] == ("root.force.x",)
    assert baseline["adopted_pose"][0] > 0.0
    assert baseline["adopted_pose"][2] < 0.5  # Environment-owned gravity affected the solver result.
    assert [claim.authority for claim in observations] == [
        AuthorityDomain.WORLD,
        AuthorityDomain.EMBODIMENT,
        AuthorityDomain.ENVIRONMENT,
    ]
    assert observations[0].claim.value["status"] == ManifestStatus.COMMITTED.value


def test_wrong_world_environment_candidate_cannot_publish_the_next_world_cut(tmp_path: Path) -> None:
    result, observations = _run_valid_step(
        tmp_path / "wrong-environment-world", wrong_environment_world=True
    )
    assert result == {"status": ManifestStatus.RECOVERY_REQUIRED.value, "tick": 0}
    assert observations == ()


def test_stale_morphology_candidate_cannot_publish_the_next_world_cut(tmp_path: Path) -> None:
    status, committed_tick, pending_tick = _run_stale_morphology(tmp_path / "stale")
    assert status is ManifestStatus.RECOVERY_REQUIRED
    assert committed_tick == 0
    assert pending_tick == 1


def test_fixture_owner_codecs_bind_checkpoint_identity_and_schema() -> None:
    body = EmbodimentPhysicalState(
        WORLD_ID,
        0,
        BODY_ID,
        EPISODE_ID,
        MORPHOLOGY_REVISION,
        BodyPoseSample(BODY_ID, "physics3d.world", "body.root", (0.0, 0.0, 0.5), (0.0, 0.0, 0.0, 1.0), "genesis", 0),
        linear_velocity_m_per_s=(0.0, 0.0, 0.0),
        angular_velocity_rad_per_s=(0.0, 0.0, 0.0),
    )
    contract = _FilesystemOwnerContract(
        owner_identity=BODY_ID,
        schema_version="embodiment-physical-state-v2",
        validate=lambda encoded: (
            EmbodimentPhysicalState.restore_checkpoint(encoded).body_id == BODY_ID
        ),
    )
    encoded = body.checkpoint_bytes()

    assert contract.validate_checkpoint(encoded, BODY_ID, "embodiment-physical-state-v2")
    assert not contract.validate_checkpoint(encoded, "body.other", "embodiment-physical-state-v2")
    assert not contract.validate_checkpoint(encoded, BODY_ID, "other-schema")


def test_articulated_mujoco_candidate_is_adopted_and_world_committed(tmp_path: Path) -> None:
    """Exercise an articulated step, then restore and execute a next step.

    This remains a workspace-local integration fixture, not a production
    solver adapter. Checkpoint writes are local fixture writes, not proof of
    power-loss durability; restart continuation is not compared with
    uninterrupted deterministic replay.
    """
    articulated_schema = "embodiment-articulated-physical-state-v1"
    articulated_body_id = "body.articulated-integration"
    articulated_episode = "episode.articulated-integration"
    articulated_revision = "morphology.articulated-r1"
    articulated_spec = ArticulatedBodySpec(
        body_kind="articulated-integration-fixture",
        state_schema_version=1,
        spawn_height=0.0,
        base_link_name="body.root",
        joint_specs=(JointSpec("joint.hinge", (0.0, 1.0, 0.0), -1.0, 1.0, 2.0, 0.1, 2.0),),
        joint_topology=(JointTopology("joint.hinge", "body.root", "body.arm", (0.0, 0.0, -0.2)),),
        segments={
            "body.root": SegmentSpec(1.0, (0.2, 0.2, 0.2), (0.0, 0.0, 0.0), (0.5, 0.5, 0.5, 1.0)),
            "body.arm": SegmentSpec(0.5, (0.1, 0.4, 0.1), (0.0, 0.0, -0.2), (0.2, 0.3, 0.4, 1.0)),
        },
        contact_link_names=("body.root", "body.arm"),
        structural_neighbour_names=(("body.root", "body.arm"),),
    )
    active_morphology = ActiveArticulatedMorphology.from_spec(
        body_id=articulated_body_id,
        embodiment_episode_id=articulated_episode,
        morphology_revision=articulated_revision,
        spec=articulated_spec,
    )
    environment = _environment_owner()
    initial_pose = BodyPoseSample(
        articulated_body_id, "physics3d.world", "body.root", (0.0, 0.0, 0.8),
        (0.0, 0.0, 0.0, 1.0), "genesis", 0,
    )
    root_state = EmbodimentPhysicalState(
        world_id=WORLD_ID,
        causal_tick=0,
        body_id=articulated_body_id,
        embodiment_episode_id=articulated_episode,
        morphology_revision=articulated_revision,
        pose=initial_pose,
        linear_velocity_m_per_s=(0.0, 0.0, 0.0),
        angular_velocity_rad_per_s=(0.0, 0.0, 0.0),
    )
    owner_state = ArticulatedEmbodimentPhysicalState(
        root=root_state,
        morphology=active_morphology,
        joints=(JointKinematicState("joint.hinge", 0.0, 0.0),),
    )
    organism_payload = json.dumps(
        {"organism_id": str(ORGANISM_ID), "fixture": "articulated-step"},
        sort_keys=True, separators=(",", ":"),
    ).encode()
    initial_refs = (
        _write_checkpoint(
            tmp_path / "owners", owner="organism", owner_identity=str(ORGANISM_ID),
            checkpoint_id="organism-tick-0", schema="organism-action-fixture-v1", encoded=organism_payload,
        ),
        _write_checkpoint(
            tmp_path / "owners", owner="embodiment", owner_identity=articulated_body_id,
            checkpoint_id="embodiment-tick-0", schema=articulated_schema, encoded=owner_state.checkpoint_bytes(),
        ),
        _write_checkpoint(
            tmp_path / "owners", owner="environment", owner_identity=ENVIRONMENT_ID,
            checkpoint_id="environment-tick-0", schema=StaticEnvironmentOwner.CHECKPOINT_SCHEMA,
            encoded=environment.checkpoint_bytes(),
        ),
    )

    def valid_body_checkpoint(encoded: bytes) -> bool:
        return ArticulatedEmbodimentPhysicalState.restore_checkpoint(
            encoded, active_morphology=active_morphology
        ).root.body_id == articulated_body_id

    contracts = {
        ("organism", str(ORGANISM_ID)): _FilesystemOwnerContract(
            owner_identity=str(ORGANISM_ID), schema_version="organism-action-fixture-v1",
            validate=lambda encoded: json.loads(encoded).get("organism_id") == str(ORGANISM_ID),
        ),
        ("embodiment", articulated_body_id): _FilesystemOwnerContract(
            owner_identity=articulated_body_id, schema_version=articulated_schema,
            validate=valid_body_checkpoint,
        ),
        ("environment", ENVIRONMENT_ID): _FilesystemOwnerContract(
            owner_identity=ENVIRONMENT_ID, schema_version=StaticEnvironmentOwner.CHECKPOINT_SCHEMA,
            validate=lambda encoded: StaticEnvironmentOwner.validate_checkpoint(
                encoded, ENVIRONMENT_ID, StaticEnvironmentOwner.CHECKPOINT_SCHEMA
            ),
        ),
    }
    presence_id = PresenceId("presence.articulated-physical-step")
    manifest_store = WorldCausalManifestStore(tmp_path / "world-manifest.json")
    world = DurableWorldRuntime.initialize(
        WorldGenesisSpec(
            WorldId(WORLD_ID), (presence_id.value,), tuple(sorted(contracts)), initial_refs, contracts,
            (PresenceOwnerLink(presence_id, "organism", str(ORGANISM_ID)),),
        ),
        presence_id.value,
        manifest_store,
        DurableJournalSnapshotStore(tmp_path / "journal", WorldId(WORLD_ID)),
        owner_checkpoint_verifier=lambda refs: all(
            contracts[(ref.owner, ref.owner_identity)].validate_checkpoint(
                Path(ref.locator).read_bytes(), ref.owner_identity, ref.schema_version
            ) for ref in refs if ref.owner != "world"
        ),
    )
    correlation = PhysicalStepCorrelation(
        WORLD_ID, 0, "physical-step-articulated-1", articulated_body_id,
        articulated_episode, articulated_revision, ENVIRONMENT_REVISION,
        f"mujoco-{mujoco.__version__}-articulated-hinge-euler-dt0.01-r1",
    )
    # Correlation tick identifies the source owner state (tick 0); the
    # candidate transition advances Embodiment and World to committed tick 1.
    assert correlation.causal_tick == owner_state.root.causal_tick == 0
    projected = project_body_physical_input(
        articulated_spec, initial_pose, correlation=correlation,
        linear_velocity_m_per_s=root_state.linear_velocity_m_per_s,
        angular_velocity_rad_per_s=root_state.angular_velocity_rad_per_s,
    )
    # The numerical fixture is deliberately limited to this exact projected
    # one-hinge morphology; it does not substitute a different body model.
    urdf = ET.fromstring(projected.articulated_description)
    assert [node.attrib["name"] for node in urdf.findall("joint") if node.attrib["type"] == "revolute"] == ["joint.hinge"]
    assert {node.attrib["name"] for node in urdf.findall("link")} == {"body.root", "body.arm"}
    projected_links = {node.attrib["name"]: node for node in urdf.findall("link")}
    root_link = projected_links["body.root"]
    arm_link = projected_links["body.arm"]
    root_mass = float(root_link.find("inertial/mass").attrib["value"])
    root_size = tuple(float(value) / 2 for value in root_link.find("collision/geometry/box").attrib["size"].split())
    arm_mass = float(arm_link.find("inertial/mass").attrib["value"])
    arm_size = tuple(float(value) / 2 for value in arm_link.find("collision/geometry/box").attrib["size"].split())
    arm_collision_origin = tuple(float(value) for value in arm_link.find("collision/origin").attrib["xyz"].split())
    hinge = next(node for node in urdf.findall("joint") if node.attrib.get("name") == "joint.hinge")
    hinge_origin = tuple(float(value) for value in hinge.find("origin").attrib["xyz"].split())
    hinge_axis = tuple(float(value) for value in hinge.find("axis").attrib["xyz"].split())
    hinge_limit = hinge.find("limit")
    hinge_dynamics = hinge.find("dynamics")
    assert hinge_limit is not None and hinge_dynamics is not None
    xml = f"""
    <mujoco model="biopeia-articulated-step-fixture">
      <compiler angle="radian" coordinate="local"/>
      <option timestep="0.01" integrator="Euler" gravity="0 0 -9.81"/>
      <worldbody>
        <geom name="ground" type="plane" pos="0 0 0" size="10 10 0.1"/>
        <body name="body.root" pos="0 0 0.8">
          <freejoint name="root.free"/>
          <geom name="root.geom" type="box" size="{' '.join(map(str, root_size))}" mass="{root_mass}"/>
          <body name="body.arm" pos="{' '.join(map(str, hinge_origin))}">
            <joint name="joint.hinge" type="hinge" axis="{' '.join(map(str, hinge_axis))}" range="{hinge_limit.attrib['lower']} {hinge_limit.attrib['upper']}" damping="{hinge_dynamics.attrib['damping']}"/>
            <geom name="arm.geom" type="box" pos="{' '.join(map(str, arm_collision_origin))}" size="{' '.join(map(str, arm_size))}" mass="{arm_mass}"/>
          </body>
        </body>
      </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    root_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root.free")
    hinge_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint.hinge")
    root_qpos = int(model.jnt_qposadr[root_joint])
    root_qvel = int(model.jnt_dofadr[root_joint])
    hinge_qpos = int(model.jnt_qposadr[hinge_joint])
    hinge_qvel = int(model.jnt_dofadr[hinge_joint])
    data.qpos[root_qpos : root_qpos + 7] = (0.0, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0)
    # Apply a declared positive hinge torque in N*m for this deterministic fixture.
    data.qfrc_applied[hinge_qvel] = 0.5
    pending_step_result: dict[str, object] = {}

    def persist_owners(causal_tick: int) -> tuple[CheckpointReference, ...]:
        assert causal_tick == 1
        assert correlation.causal_tick + 1 == causal_tick
        assert manifest_store.read().status is ManifestStatus.PREPARING
        mujoco.mj_step(model, data)
        qpos = tuple(float(value) for value in data.qpos[root_qpos : root_qpos + 7])
        qvel = tuple(float(value) for value in data.qvel[root_qvel : root_qvel + 6])
        candidate = SolverArticulatedCandidate(
            root=SolverBodyCandidate(
                correlation, "physics3d.world", "body.root", qpos[:3],
                (qpos[4], qpos[5], qpos[6], qpos[3]), qvel[:3], qvel[3:6],
            ),
            joints=(JointKinematicState("joint.hinge", float(data.qpos[hinge_qpos]), float(data.qvel[hinge_qvel])),),
        )
        # Embodiment returns a candidate state without mutating its prior
        # value; Environment adoption follows before any references are emitted.
        adopted = owner_state.adopt_candidate(candidate, expected_step=correlation)
        assert adopted.root.causal_tick == causal_tick
        env_correlation = EnvironmentStepCorrelation(
            WORLD_ID, 0, correlation.physical_step_id, environment.scene.environment_id, ENVIRONMENT_REVISION
        )
        environment.validate_and_adopt(
            StaticSceneCandidate(env_correlation, project_static_scene(environment.scene)),
            expected_correlation=env_correlation,
        )
        pending_step_result.update(adopted=adopted, candidate=candidate, correlation=correlation)
        return (
            _write_checkpoint(
                tmp_path / "owners", owner="organism", owner_identity=str(ORGANISM_ID),
                checkpoint_id="organism-tick-1", schema="organism-action-fixture-v1", encoded=organism_payload,
            ),
            _write_checkpoint(
                tmp_path / "owners", owner="embodiment", owner_identity=articulated_body_id,
                checkpoint_id="embodiment-tick-1", schema=articulated_schema, encoded=adopted.checkpoint_bytes(),
            ),
            _write_checkpoint(
                tmp_path / "owners", owner="environment", owner_identity=ENVIRONMENT_ID,
                checkpoint_id="environment-tick-1", schema=StaticEnvironmentOwner.CHECKPOINT_SCHEMA,
                encoded=environment.checkpoint_bytes(),
            ),
        )

    world.tick("articulated-physical-step-1", persist_owners)
    assert world.status is ManifestStatus.COMMITTED
    assert world.causal_tick == 1
    adopted = pending_step_result["adopted"]
    correlation = pending_step_result["correlation"]
    assert isinstance(adopted, ArticulatedEmbodimentPhysicalState)
    assert isinstance(correlation, PhysicalStepCorrelation)
    body_ref = next(ref for ref in world.manifest.checkpoint_references if ref.owner == "embodiment")
    restored = ArticulatedEmbodimentPhysicalState.restore_checkpoint(
        Path(body_ref.locator).read_bytes(), active_morphology=active_morphology
    )
    assert restored == adopted
    assert restored.root.causal_tick == correlation.causal_tick + 1 == world.causal_tick
    assert restored.joints[0].joint_id == "joint.hinge"
    assert restored.joints[0].position_rad != 0.0
    assert restored.root.last_physical_step == correlation

    # Exercise runtime reconstruction using only the committed World manifest
    # and owner checkpoint references. This does not launch a separate OS
    # process or prove crash/power-loss behavior.
    restored_owners: dict[str, object] = {}

    def restore_owner_checkpoints(references: tuple[CheckpointReference, ...]) -> None:
        for reference in references:
            encoded = Path(reference.locator).read_bytes()
            if reference.owner == "embodiment":
                restored_owners[reference.owner] = ArticulatedEmbodimentPhysicalState.restore_checkpoint(
                    encoded, active_morphology=active_morphology
                )
            elif reference.owner == "environment":
                restored_owners[reference.owner] = StaticEnvironmentOwner.from_checkpoint_bytes(encoded)
            elif reference.owner == "organism":
                restored_owners[reference.owner] = json.loads(encoded)

    restarted_world = DurableWorldRuntime.restore(
        WorldId(WORLD_ID),
        presence_id.value,
        manifest_store,
        DurableJournalSnapshotStore(tmp_path / "journal", WorldId(WORLD_ID)),
        owner_checkpoint_verifier=lambda references: all(
            contracts[(reference.owner, reference.owner_identity)].validate_checkpoint(
                Path(reference.locator).read_bytes(), reference.owner_identity, reference.schema_version
            )
            for reference in references
            if reference.owner != "world-journal"
        ),
        owner_checkpoint_contracts=contracts,
        owner_checkpoint_restorer=restore_owner_checkpoints,
    )
    assert restarted_world.status is ManifestStatus.COMMITTED
    assert restarted_world.causal_tick == 1
    assert restarted_world.manifest.commit_id == "articulated-physical-step-1"
    assert restored_owners["embodiment"] == restored
    assert isinstance(restored_owners["environment"], StaticEnvironmentOwner)
    assert restored_owners["environment"].last_accepted_correlation == EnvironmentStepCorrelation(
        WORLD_ID, 0, correlation.physical_step_id, environment.scene.environment_id, ENVIRONMENT_REVISION
    )
    assert restored_owners["organism"] == json.loads(organism_payload)

    # Restore the committed owner values, execute a subsequent step using a
    # newly compiled numerical model, and publish the next cut. This checks
    # continuation from restored state, not equivalence to uninterrupted replay.
    restored_body = restored_owners["embodiment"]
    restored_environment = restored_owners["environment"]
    assert isinstance(restored_body, ArticulatedEmbodimentPhysicalState)
    assert isinstance(restored_environment, StaticEnvironmentOwner)
    restored_organism = restored_owners["organism"]
    assert isinstance(restored_organism, dict)
    next_correlation = replace(
        correlation,
        causal_tick=1,
        physical_step_id="physical-step-articulated-2",
    )
    assert next_correlation.causal_tick == restored_body.root.causal_tick == restarted_world.causal_tick
    next_projection = project_body_physical_input(
        articulated_spec,
        restored_body.root.pose,
        correlation=next_correlation,
        linear_velocity_m_per_s=restored_body.root.linear_velocity_m_per_s,
        angular_velocity_rad_per_s=restored_body.root.angular_velocity_rad_per_s,
    )
    assert next_projection.articulated_description == projected.articulated_description
    resumed_model = mujoco.MjModel.from_xml_string(xml)
    resumed_data = mujoco.MjData(resumed_model)
    resumed_root_joint = mujoco.mj_name2id(resumed_model, mujoco.mjtObj.mjOBJ_JOINT, "root.free")
    resumed_hinge_joint = mujoco.mj_name2id(resumed_model, mujoco.mjtObj.mjOBJ_JOINT, "joint.hinge")
    resumed_root_qpos = int(resumed_model.jnt_qposadr[resumed_root_joint])
    resumed_root_qvel = int(resumed_model.jnt_dofadr[resumed_root_joint])
    resumed_hinge_qpos = int(resumed_model.jnt_qposadr[resumed_hinge_joint])
    resumed_hinge_qvel = int(resumed_model.jnt_dofadr[resumed_hinge_joint])
    resumed_data.qpos[resumed_root_qpos : resumed_root_qpos + 7] = (
        *restored_body.root.pose.position_m,
        restored_body.root.pose.orientation_xyzw[3],
        *restored_body.root.pose.orientation_xyzw[:3],
    )
    resumed_data.qvel[resumed_root_qvel : resumed_root_qvel + 6] = (
        *restored_body.root.linear_velocity_m_per_s,
        *restored_body.root.angular_velocity_rad_per_s,
    )
    resumed_data.qpos[resumed_hinge_qpos] = restored_body.joints[0].position_rad
    resumed_data.qvel[resumed_hinge_qvel] = restored_body.joints[0].velocity_rad_per_s
    next_environment_correlation = EnvironmentStepCorrelation(
        WORLD_ID,
        1,
        next_correlation.physical_step_id,
        environment.scene.environment_id,
        ENVIRONMENT_REVISION,
    )
    pending_continuation: dict[str, object] = {}

    def persist_continued_owners(causal_tick: int) -> tuple[CheckpointReference, ...]:
        assert causal_tick == 2
        assert next_correlation.causal_tick + 1 == causal_tick
        assert manifest_store.read().status is ManifestStatus.PREPARING
        # The post-restore input is intentionally lower to stay within the
        # morphology's declared joint-velocity limit.
        resumed_data.qfrc_applied[resumed_hinge_qvel] = 0.05
        mujoco.mj_step(resumed_model, resumed_data)
        next_qpos = tuple(float(value) for value in resumed_data.qpos[resumed_root_qpos : resumed_root_qpos + 7])
        next_qvel = tuple(float(value) for value in resumed_data.qvel[resumed_root_qvel : resumed_root_qvel + 6])
        next_candidate = SolverArticulatedCandidate(
            root=SolverBodyCandidate(
                next_correlation,
                "physics3d.world",
                "body.root",
                next_qpos[:3],
                (next_qpos[4], next_qpos[5], next_qpos[6], next_qpos[3]),
                next_qvel[:3],
                next_qvel[3:6],
            ),
            joints=(
                JointKinematicState(
                    "joint.hinge",
                    float(resumed_data.qpos[resumed_hinge_qpos]),
                    float(resumed_data.qvel[resumed_hinge_qvel]),
                ),
            ),
        )
        continued_body = restored_body.adopt_candidate(
            next_candidate, expected_step=next_correlation
        )
        restored_environment.validate_and_adopt(
            StaticSceneCandidate(
                next_environment_correlation,
                project_static_scene(restored_environment.scene),
            ),
            expected_correlation=next_environment_correlation,
        )
        organism_tick_2 = json.dumps(
            {**restored_organism, "last_physical_tick": causal_tick},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        pending_continuation.update(body=continued_body, candidate=next_candidate)
        return (
            _write_checkpoint(
                tmp_path / "owners", owner="organism", owner_identity=str(ORGANISM_ID),
                checkpoint_id="organism-tick-2", schema="organism-action-fixture-v1", encoded=organism_tick_2,
            ),
            _write_checkpoint(
                tmp_path / "owners", owner="embodiment", owner_identity=articulated_body_id,
                checkpoint_id="embodiment-tick-2", schema=articulated_schema,
                encoded=continued_body.checkpoint_bytes(),
            ),
            _write_checkpoint(
                tmp_path / "owners", owner="environment", owner_identity=ENVIRONMENT_ID,
                checkpoint_id="environment-tick-2", schema=StaticEnvironmentOwner.CHECKPOINT_SCHEMA,
                encoded=restored_environment.checkpoint_bytes(),
            ),
        )

    continued_record = restarted_world.tick(
        "articulated-physical-step-2", persist_continued_owners
    )
    assert restarted_world.status is ManifestStatus.COMMITTED
    assert continued_record.causal_tick == 2
    assert restarted_world.manifest.commit_id == "articulated-physical-step-2"
    continued_body = pending_continuation["body"]
    assert isinstance(continued_body, ArticulatedEmbodimentPhysicalState)
    assert continued_body.root.causal_tick == next_correlation.causal_tick + 1 == continued_record.causal_tick
    committed_body_ref = next(
        reference for reference in restarted_world.manifest.checkpoint_references
        if reference.owner == "embodiment"
    )
    committed_body = ArticulatedEmbodimentPhysicalState.restore_checkpoint(
        Path(committed_body_ref.locator).read_bytes(), active_morphology=active_morphology
    )
    assert committed_body == continued_body
    assert committed_body.root.causal_tick == restarted_world.causal_tick
    assert committed_body.root.last_physical_step == next_correlation
    committed_environment_ref = next(
        reference for reference in restarted_world.manifest.checkpoint_references
        if reference.owner == "environment"
    )
    committed_environment = StaticEnvironmentOwner.from_checkpoint_bytes(
        Path(committed_environment_ref.locator).read_bytes()
    )
    assert committed_environment.last_accepted_correlation == next_environment_correlation
    committed_organism_ref = next(
        reference for reference in restarted_world.manifest.checkpoint_references
        if reference.owner == "organism"
    )
    assert json.loads(Path(committed_organism_ref.locator).read_bytes()) == {
        **restored_organism,
        "last_physical_tick": 2,
    }
