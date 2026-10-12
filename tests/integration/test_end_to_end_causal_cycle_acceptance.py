"""Two-run acceptance of the production TARGET sensorimotor composition.

This is bounded integration evidence, not a scientific efficacy, calibration,
or production-scale durability claim. Every semantic join is supplied by the
domain APIs and World production composition, never by a test-only causal
shortcut.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path

import mujoco
import pytest
from embodiment.observation import BodyPoseSample
from embodiment.physics3d.articulated_spec import ArticulatedBodySpec
from embodiment.physics3d.articulated_state import (
    ActiveArticulatedMorphology,
    ArticulatedEmbodimentPhysicalState,
    ARTICULATED_PHYSICAL_CHECKPOINT_SCHEMA,
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
from observation.plane import ObservationPlane
from organism.actuation import (
    ActuatorEvidenceContext,
    ActuatorSurface,
    CompetenceDevelopmentEngine,
    OrganismExplorationComposer,
)
from organism.cognition import CognitiveRuntimeCheckpoint, SenseChannelBinding
from organism.cognition.graph import CognitiveGraph, PlasticNode
from organism.cognition.limits import KernelLimits
from organism.cognition.sense_input import SenseProvenance
from organism.cognition.transition import CognitiveTransitionState
from organism.cognition.types import NodeKind
from organism.identity import OrganismIdentity
from organism.transition import (
    EmbodimentContextRef,
    OrganismLifecycle,
    OrganismTransitionState,
    WorldCausalCorrelation,
)
from world.durable_runtime import (
    DurableJournalSnapshotStore,
    DurableWorldRuntime,
    PresenceOwnerLink,
    WorldGenesisSpec,
)
from world.manifest import CheckpointReference, WorldCausalManifestStore
from world.runtime import PresenceId, WorldId
from world.causal_cycle import (
    ORGANISM_ACTION_CHECKPOINT_SCHEMA,
    run_scalar_sensorimotor_cycle_and_publish,
)
from world.adapters.articulated_mujoco import ArticulatedMujocoConfiguration
from world.owner_checkpoint_store import FileOwnerCheckpointStore

WORLD_ID = "world.acceptance.sensorimotor-v1"
PRESENCE_ID = "presence.acceptance.sensorimotor-v1"
ORGANISM_ID = OrganismIdentity.parse("12345678-1234-5678-1234-567812345678")
BODY_ID = "body.acceptance.sensorimotor-v1"
EPISODE_ID = "episode.acceptance.sensorimotor-v1"
MORPHOLOGY_REVISION = "morphology.acceptance.sensorimotor-r1"
ENVIRONMENT_ID = "environment.acceptance.sensorimotor-v1"
ENVIRONMENT_OWNER_ID = "owner.environment.acceptance.sensorimotor-v1"
RESOURCE_ID = "resource.acceptance.scalar-field-v1"
ENVIRONMENT_REVISION = "environment.acceptance.r1"
FIELD_REVISION = "field.acceptance.r1"
FRAME_ID = "acceptance.world-frame"
SURFACE_FINGERPRINT = "sha256:acceptance-joint-surface-r1"
RECEPTOR_ID = "receptor.acceptance.scalar-v1"
CHANNEL_ID = "channel.opaque.scalar-v1"
SOLVER_REVISION = f"mujoco-{mujoco.__version__}-acceptance-r1"


class _Contract:
    def __init__(self, identity: str, validators: dict[str, Callable[[bytes], bool]]):
        self.identity = identity
        self.validators = validators

    def validate_checkpoint(self, encoded: bytes, owner_identity: str, schema_version: str) -> bool:
        validator = self.validators.get(schema_version)
        return owner_identity == self.identity and validator is not None and validator(encoded)

    def verify_restoration(self, reference: CheckpointReference) -> bool:
        path = Path(reference.locator)
        return path.is_file() and self.validate_checkpoint(
            path.read_bytes(), reference.owner_identity, reference.schema_version
        )


def _frame() -> FrameDescriptor:
    return FrameDescriptor(
        frame_id=FRAME_ID,
        parent_frame_id="world",
        origin_m=(0.0, 0.0, 0.0),
        x_axis_in_parent=(1.0, 0.0, 0.0),
        y_axis_in_parent=(0.0, 1.0, 0.0),
        z_axis_in_parent=(0.0, 0.0, 1.0),
        handedness="RIGHT_HANDED",
        up_axis="Z",
    )


def _spec() -> ArticulatedBodySpec:
    return ArticulatedBodySpec(
        body_kind="acceptance-fixed-base-hinge",
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
        body_material=SurfaceMaterial(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        fixed_base=True,
    )


def _reference(directory: Path, owner: str, identity: str, schema: str, tick: int, payload: bytes) -> CheckpointReference:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{owner}-{identity.replace('/', '_')}-{tick}.checkpoint"
    path.write_bytes(payload)
    return CheckpointReference(
        owner=owner,
        owner_identity=identity,
        checkpoint_id=f"{owner}-{tick}",
        schema_version=schema,
        locator=str(path),
        sha256=hashlib.sha256(payload).hexdigest(),
    )


def _run(directory: Path):
    frame = _frame()
    spec = _spec()
    morphology = ActiveArticulatedMorphology.from_spec(
        body_id=BODY_ID,
        embodiment_episode_id=EPISODE_ID,
        morphology_revision=MORPHOLOGY_REVISION,
        spec=spec,
    )
    pose = BodyPoseSample(
        body_id=BODY_ID,
        reference_frame=FRAME_ID,
        body_frame="body.root",
        position_m=(0.0, 0.0, 0.0),
        orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
        sample_kind="genesis",
        sample_value=0,
    )
    root = EmbodimentPhysicalState(
        world_id=WORLD_ID,
        causal_tick=0,
        body_id=BODY_ID,
        embodiment_episode_id=EPISODE_ID,
        morphology_revision=MORPHOLOGY_REVISION,
        pose=pose,
        linear_velocity_m_per_s=(0.0, 0.0, 0.0),
        angular_velocity_rad_per_s=(0.0, 0.0, 0.0),
    )
    body = ArticulatedEmbodimentPhysicalState(
        root=root,
        morphology=morphology,
        joints=(JointKinematicState("joint.hinge", 0.0, 0.0),),
    )
    env_owner = StaticEnvironmentOwner(
        StaticPhysicalScene(
            environment_id=ENVIRONMENT_ID,
            environment_revision=ENVIRONMENT_REVISION,
            frame=frame,
            gravity_m_s2=(0.0, 0.0, 0.0),
            objects=(),
        ),
        owner_id=ENVIRONMENT_OWNER_ID,
        world_id=WORLD_ID,
    )
    field = IdentifiedFiniteScalarField(
        resource_id=RESOURCE_ID,
        field_revision=FIELD_REVISION,
        environment_id=ENVIRONMENT_ID,
        environment_revision=ENVIRONMENT_REVISION,
        frame_id=FRAME_ID,
        state=FiniteResourceState(
            position=(1.0, 0.0, 0.0), radius=0.1, field_radius=2.0,
            remaining=10.0, transfer_per_tick=1.0,
        ),
    )
    cognition = CognitiveRuntimeCheckpoint(
        state=CognitiveTransitionState.initial(ORGANISM_ID),
        graph=CognitiveGraph(
            nodes=(PlasticNode("opaque.scalar.signal", NodeKind.SENSE),),
            edges=(),
            kernel_limits=KernelLimits(),
        ),
    )
    organism = OrganismTransitionState(
        owner_checkpoint=cognition.to_organism_checkpoint(),
        organism_step=0,
        embodiment=EmbodimentContextRef(BODY_ID, EPISODE_ID, MORPHOLOGY_REVISION),
        lifecycle=OrganismLifecycle.ACTIVE,
    )
    contracts = {
        ("organism", str(ORGANISM_ID)): _Contract(
            str(ORGANISM_ID), {
                "organism-transition-v2": lambda data: _valid_transition_checkpoint(data, str(ORGANISM_ID)),
                ORGANISM_ACTION_CHECKPOINT_SCHEMA: lambda data: _valid_organism_checkpoint(data, str(ORGANISM_ID)),
            },
        ),
        ("embodiment", BODY_ID): _Contract(
            BODY_ID, {ARTICULATED_PHYSICAL_CHECKPOINT_SCHEMA: lambda data: _valid_body_checkpoint(data, morphology)},
        ),
        ("environment", ENVIRONMENT_OWNER_ID): _Contract(
            ENVIRONMENT_OWNER_ID, {StaticEnvironmentOwner.CHECKPOINT_SCHEMA: lambda data: StaticEnvironmentOwner.validate_checkpoint(
                data, ENVIRONMENT_OWNER_ID, StaticEnvironmentOwner.CHECKPOINT_SCHEMA
            )},
        ),
        ("environment", RESOURCE_ID): _Contract(
            RESOURCE_ID, {IdentifiedFiniteScalarField.OWNER_SCHEMA: lambda data: IdentifiedFiniteScalarField.validate_checkpoint(
                data, RESOURCE_ID, IdentifiedFiniteScalarField.OWNER_SCHEMA
            )},
        ),
    }
    owner_dir = directory / "owners"
    initial_refs = (
        _reference(owner_dir, "organism", str(ORGANISM_ID), "organism-transition-v2", 0, organism.to_bytes()),
        _reference(owner_dir, "embodiment", BODY_ID, ARTICULATED_PHYSICAL_CHECKPOINT_SCHEMA, 0, body.checkpoint_bytes()),
        _reference(owner_dir, "environment", ENVIRONMENT_OWNER_ID, StaticEnvironmentOwner.CHECKPOINT_SCHEMA, 0, env_owner.checkpoint_bytes()),
        _reference(owner_dir, "environment", RESOURCE_ID, IdentifiedFiniteScalarField.OWNER_SCHEMA, 0, field.checkpoint_bytes()),
    )
    world = DurableWorldRuntime.initialize(
        WorldGenesisSpec(
            world_id=WorldId(WORLD_ID),
            initial_presence_ids=(PRESENCE_ID,),
            registered_owner_ids=tuple(sorted(contracts)),
            initial_owner_references=initial_refs,
            owner_checkpoint_contracts=contracts,
            presence_owner_links=(PresenceOwnerLink(PresenceId(PRESENCE_ID), "organism", str(ORGANISM_ID)),),
        ),
        PRESENCE_ID,
        WorldCausalManifestStore(directory / "manifest.json"),
        DurableJournalSnapshotStore(directory / "journal", WorldId(WORLD_ID)),
        owner_checkpoint_verifier=lambda refs: all(
            contracts[(ref.owner, ref.owner_identity)].validate_checkpoint(
                Path(ref.locator).read_bytes(), ref.owner_identity, ref.schema_version
            ) for ref in refs if ref.owner != "world-journal"
        ),
    )
    surface = ActuatorSurface.from_effector_ids(
        ("actuator.hinge.positive", "actuator.hinge.negative"),
        fingerprint_material=SURFACE_FINGERPRINT,
    )
    actuator_context = ActuatorEvidenceContext(BODY_ID, EPISODE_ID, MORPHOLOGY_REVISION, surface.contract_fingerprint)
    engine = CompetenceDevelopmentEngine(
        surface.actuator_ids,
        organism_id=str(ORGANISM_ID),
        embodiment_fingerprint=actuator_context.fingerprint,
    )
    # These are independently addressable channels. The Embodiment-owned
    # differential binding defines their combined physical effect; they are
    # not declared mutually exclusive to Organism, so a complete command may
    # include both without World synthesizing an omitted channel.
    composer = OrganismExplorationComposer(organism_id=str(ORGANISM_ID), engine=engine)
    receptor = EmbodimentReceptor(
        receptor_id=RECEPTOR_ID, body_id=BODY_ID,
        embodiment_episode_id=EPISODE_ID, morphology_revision=MORPHOLOGY_REVISION,
        body_frame="body.root", position_in_body_frame_m=(0.0, 0.0, 0.0),
    )
    motor_binding = EmbodimentDifferentialMotorBinding(
        binding_id="binding.acceptance.differential-r1",
        body_id=BODY_ID, embodiment_episode_id=EPISODE_ID,
        morphology_revision=MORPHOLOGY_REVISION,
        source_surface_fingerprint=surface.contract_fingerprint,
        pairs=(DifferentialMotorPair(
            joint_id="joint.hinge",
            positive_actuator_id="actuator.hinge.positive",
            negative_actuator_id="actuator.hinge.negative",
            scale=0.5,
        ),),
    )
    torque_binding = EmbodimentJointTorqueBinding(
        binding_id="binding.acceptance.torque-r1",
        body_id=BODY_ID, embodiment_episode_id=EPISODE_ID,
        morphology_revision=MORPHOLOGY_REVISION,
        source_surface_fingerprint=surface.contract_fingerprint,
        channels=(JointTorqueChannelBinding("actuator.hinge", "joint.hinge", (0.0, 1.0, 0.0)),),
    )
    solver = ArticulatedMujocoConfiguration(
        revision=SOLVER_REVISION, timestep_s=0.01, iterations=50,
        tolerance=1e-10, solref=(0.02, 1.0),
        solimp=(0.9, 0.95, 0.001, 0.5, 2.0),
        contact_policy="max-sliding-body-torsion-rolling/no-self-collision-v1",
    )

    plane = ObservationPlane()

    result = run_scalar_sensorimotor_cycle_and_publish(
        world, plane,
        commit_id="commit.acceptance.sensorimotor.tick-1",
        physical_step_id="physical-step.acceptance.sensorimotor.tick-1",
        environment_revision=ENVIRONMENT_REVISION,
        solver_configuration=solver,
        current_organism=organism,
        action_composer=composer,
        receptor=receptor,
        embodiment_state=root,
        environment_field=field,
        sampling_frame=frame,
        transduction=DimensionlessScalarFieldIdentityV1(),
        sense_binding=SenseChannelBinding(CHANNEL_ID, RECEPTOR_ID, "opaque.scalar.signal"),
        kernel_limits=KernelLimits(),
        actuator_context=actuator_context,
        actuator_surface=surface,
        articulated_body=body,
        active_morphology=morphology,
        differential_binding=motor_binding,
        torque_binding=torque_binding,
        environment_owner=env_owner,
        owner_checkpoint_store=FileOwnerCheckpointStore(owner_dir / "committed"),
    )
    return result, plane


def _valid_organism_checkpoint(data: bytes, identity: str) -> bool:
    try:
        import base64
        import json

        payload = json.loads(data)
        expected_fields = {
            "schema", "organism_id", "generation", "world_id",
            "source_causal_tick", "target_causal_tick", "physical_step_id",
            "commit_id", "transition_state_base64", "engine", "action_domain",
        }
        if (
            not isinstance(payload, dict)
            or set(payload) != expected_fields
            or payload["schema"] != "biopeia-organism-exploration-checkpoint-v3"
            or payload["organism_id"] != identity
            or type(payload["generation"]) is not int
            or payload["generation"] < 1
            or type(payload["source_causal_tick"]) is not int
            or type(payload["target_causal_tick"]) is not int
            or payload["target_causal_tick"] != payload["source_causal_tick"] + 1
            or not isinstance(payload["world_id"], str)
            or not isinstance(payload["physical_step_id"], str)
            or not isinstance(payload["commit_id"], str)
            or not isinstance(payload["engine"], dict)
            or not isinstance(payload["action_domain"], dict)
        ):
            return False
        transition_bytes = base64.b64decode(payload["transition_state_base64"], validate=True)
        state = OrganismTransitionState.from_bytes(transition_bytes)
    except (TypeError, ValueError, KeyError, UnicodeDecodeError):
        return False
    return (
        str(state.identity) == identity
        and state.world_correlation is not None
        and state.world_correlation.world_id == payload["world_id"]
        and state.world_correlation.causal_tick == payload["source_causal_tick"]
    )


def _valid_transition_checkpoint(data: bytes, identity: str) -> bool:
    try:
        state = OrganismTransitionState.from_bytes(data)
    except (TypeError, ValueError):
        return False
    return str(state.identity) == identity


def _valid_body_checkpoint(data: bytes, morphology: ActiveArticulatedMorphology) -> bool:
    try:
        ArticulatedEmbodimentPhysicalState.restore_checkpoint(data, active_morphology=morphology)
    except (TypeError, ValueError):
        return False
    return True


def test_real_target_sensorimotor_cycle_commits_publishes_and_repeats(tmp_path: Path) -> None:
    first, first_plane = _run(tmp_path / "run-a")
    second, second_plane = _run(tmp_path / "run-b")

    for result, plane in ((first, first_plane), (second, second_plane)):
        assert result.context.source_causal_tick == 0
        assert result.context.target_causal_tick == result.committed_cut.causal_tick == 1
        assert result.sensory_input.sample.value == pytest.approx(0.25)
        assert result.sensory_input.signal.channel_id == CHANNEL_ID
        assert result.sensory_input.signal.receptor_id == RECEPTOR_ID
        assert result.action_candidate.proposal is not None
        assert result.action_candidate.motor_command is not None
        command = result.action_candidate.motor_command
        assert command.issued_at_tick == 0
        assert set(dict(command.channels)) == {
            "actuator.hinge.positive",
            "actuator.hinge.negative",
        }
        assert result.physical_outcome.body_state.joints[0].position_rad != 0.0
        assert len(result.committed_cut.owner_checkpoint_references) == 4
        assert len(result.observation_publication.delivery_sequences) == 2
        messages = plane.current_messages()
        assert len(messages) == 2
        assert all(message.observation.context_time for message in messages)

    assert first.committed_cut.commit_id == second.committed_cut.commit_id
    assert first.committed_cut.causal_tick == second.committed_cut.causal_tick
    assert first.sensory_input.signal.value == second.sensory_input.signal.value
    assert first.action_candidate.motor_command == second.action_candidate.motor_command
    assert first.physical_outcome.body_state.joints == pytest.approx(
        second.physical_outcome.body_state.joints
    )
    assert tuple(message.observation for message in first_plane.current_messages()) == tuple(
        message.observation for message in second_plane.current_messages()
    )
