"""Two-run acceptance of the production TARGET sensorimotor composition.

This is bounded integration evidence, not a scientific efficacy, calibration,
or production-scale durability claim. Every semantic join is supplied by the
domain APIs and World production composition, never by a test-only causal
shortcut.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import mujoco
import pytest
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
from observation.plane import ObservationPlane
from organism.actuation import (
    ORGANISM_EXPLORATION_CHECKPOINT_SCHEMA,
    ActuatorEvidenceContext,
    ActuatorSurface,
    CompetenceDevelopmentEngine,
    OrganismCheckpointContract,
    OrganismExplorationComposer,
)
from organism.cognition import CognitiveRuntimeCheckpoint, SenseChannelBinding
from organism.checkpoint import OrganismCheckpoint
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
from world.causal_cycle import (
    restore_scalar_sensorimotor_runtime,
    run_scalar_sensorimotor_cycle_and_publish,
)
from world.durable_runtime import (
    DurableJournalSnapshotStore,
    DurableWorldRuntime,
    PresenceOwnerLink,
    WorldGenesisSpec,
)
from world.manifest import CheckpointReference, ManifestError, WorldCausalManifestStore
from world.owner_checkpoint_store import FileOwnerCheckpointStore
from world.runtime import PresenceId, WorldId

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


def _run(
    directory: Path,
    *,
    stop_after_first: bool = False,
    continue_from_live_state: bool = False,
    malformed_organism_checkpoint: bool = False,
):
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
        ("organism", str(ORGANISM_ID)): OrganismCheckpointContract(),
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
    manifest_store = WorldCausalManifestStore(directory / "manifest.json")
    journal_store = DurableJournalSnapshotStore(directory / "journal", WorldId(WORLD_ID))
    owner_checkpoint_store = FileOwnerCheckpointStore(owner_dir / "committed")
    if malformed_organism_checkpoint:
        # Deliberately construct an adversarial committed fixture: keep the
        # World reference digest and Organism envelope integrity valid while
        # making the private cognitive payload semantically invalid. The
        # lenient validator is used only to seed this negative fixture; restore
        # below always uses the real owner contract.
        strict_organism_contract = contracts[("organism", str(ORGANISM_ID))]
        class _FixtureOrganismContract:
            def validate_checkpoint(self, encoded, owner_identity, schema_version):
                if strict_organism_contract.validate_checkpoint(
                    encoded, owner_identity, schema_version
                ):
                    return True
                if schema_version != ORGANISM_EXPLORATION_CHECKPOINT_SCHEMA:
                    return False
                try:
                    envelope = json.loads(encoded)
                    transition = OrganismTransitionState.from_bytes(
                        base64.b64decode(
                            envelope["transition_state_base64"], validate=True
                        )
                    )
                    return (
                        envelope["organism_id"] == owner_identity
                        and str(transition.identity) == owner_identity
                        and OrganismCheckpoint.from_bytes(
                            transition.owner_checkpoint.to_bytes()
                        )
                        == transition.owner_checkpoint
                    )
                except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                    return False

            def verify_restoration(self, reference):
                try:
                    encoded = Path(reference.locator).read_bytes()
                except OSError:
                    return False
                return self.validate_checkpoint(
                    encoded, reference.owner_identity, reference.schema_version
                )

        contracts[("organism", str(ORGANISM_ID))] = _FixtureOrganismContract()
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
        manifest_store,
        journal_store,
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
    if malformed_organism_checkpoint:
        original_checkpoint_for_candidate = composer.checkpoint_for_candidate

        def checkpoint_with_invalid_private_payload(*args, **kwargs):
            encoded = original_checkpoint_for_candidate(*args, **kwargs)
            document = json.loads(encoded)
            transition_bytes = base64.b64decode(
                document["transition_state_base64"], validate=True
            )
            transition = OrganismTransitionState.from_bytes(transition_bytes)
            invalid_private_state = OrganismCheckpoint(
                identity=transition.identity,
                private_state=b"not-a-valid-cognitive-runtime-checkpoint",
                dependencies=transition.owner_checkpoint.dependencies,
            )
            invalid_transition = replace(transition, owner_checkpoint=invalid_private_state)
            document["transition_state_base64"] = base64.b64encode(
                invalid_transition.to_bytes()
            ).decode("ascii")
            return json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")

        composer.checkpoint_for_candidate = checkpoint_with_invalid_private_payload
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
        owner_checkpoint_store=owner_checkpoint_store,
    )
    first_messages = plane.current_messages()
    if stop_after_first:
        return result, first_messages, None, None

    if continue_from_live_state:
        # This is the independent uninterrupted control: retain the original
        # in-memory World, Organism composer, Embodiment result and Environment
        # owner after the first accepted commit. Do not restore from a checkpoint.
        next_world = world
        next_organism = result.sensory_input.state
        next_composer = composer
        next_body = result.physical_outcome.body_state
        next_environment = env_owner
        next_field = field
        assert next_world.causal_tick == result.committed_cut.causal_tick == 1
        assert next_body.checkpoint_bytes() == Path(
            next(
                ref.locator
                for ref in result.committed_cut.owner_checkpoint_references
                if ref.owner == "embodiment"
            )
        ).read_bytes()
        assert next_environment.last_adopted_tick == result.context.source_causal_tick
        assert next_environment.last_accepted_correlation is not None
        assert next_environment.last_accepted_correlation.physical_step_id == (
            result.context.physical_step_id
        )
    else:
        restored_runtime = restore_scalar_sensorimotor_runtime(
            WorldId(WORLD_ID), PRESENCE_ID, manifest_store, journal_store,
            owner_checkpoint_verifier=lambda refs: all(
                contracts[(ref.owner, ref.owner_identity)].validate_checkpoint(
                    Path(ref.locator).read_bytes(), ref.owner_identity, ref.schema_version
                ) for ref in refs if ref.owner != "world-journal"
            ),
            owner_checkpoint_contracts=contracts,
            active_morphology=morphology,
            expected_organism_id=str(ORGANISM_ID),
            expected_physical_step_id=result.context.physical_step_id,
            expected_environment_owner_id=ENVIRONMENT_OWNER_ID,
            expected_environment_id=ENVIRONMENT_ID,
            expected_environment_revision=ENVIRONMENT_REVISION,
            expected_field_id=RESOURCE_ID,
            expected_field_revision=FIELD_REVISION,
            actuator_context=actuator_context,
            actuator_surface=surface,
        )
        next_world = restored_runtime.world
        next_organism = restored_runtime.organism.transition_state
        next_composer = restored_runtime.organism.action_composer
        next_body = restored_runtime.articulated_body
        next_environment = restored_runtime.environment_owner
        next_field = restored_runtime.environment_field
        assert next_world.causal_tick == result.committed_cut.causal_tick == 1
        assert next_body.checkpoint_bytes() == Path(
            next(ref.locator for ref in result.committed_cut.owner_checkpoint_references if ref.owner == "embodiment")
        ).read_bytes()
        # The static Environment owner records the source-state correlation used
        # for this physical step, while the World cut advances to the target tick.
        assert next_environment.last_adopted_tick == result.context.source_causal_tick
        assert next_environment.last_accepted_correlation is not None
        assert next_environment.last_accepted_correlation.physical_step_id == (
            result.context.physical_step_id
        )

    resumed = run_scalar_sensorimotor_cycle_and_publish(
        next_world,
        plane,
        commit_id="commit.acceptance.sensorimotor.tick-2",
        physical_step_id="physical-step.acceptance.sensorimotor.tick-2",
        environment_revision=ENVIRONMENT_REVISION,
        solver_configuration=solver,
        current_organism=next_organism,
        action_composer=next_composer,
        receptor=receptor,
        embodiment_state=next_body.root,
        environment_field=next_field,
        sampling_frame=frame,
        transduction=DimensionlessScalarFieldIdentityV1(),
        sense_binding=SenseChannelBinding(CHANNEL_ID, RECEPTOR_ID, "opaque.scalar.signal"),
        kernel_limits=KernelLimits(),
        actuator_context=actuator_context,
        actuator_surface=surface,
        articulated_body=next_body,
        active_morphology=morphology,
        differential_binding=motor_binding,
        torque_binding=torque_binding,
        environment_owner=next_environment,
        owner_checkpoint_store=owner_checkpoint_store,
    )
    resumed_messages = plane.current_messages()
    return result, first_messages, resumed, resumed_messages


def _restore_runtime_from_directory(
    directory: Path,
    *,
    expected_field_revision: str = FIELD_REVISION,
    actuator_context: ActuatorEvidenceContext | None = None,
):
    morphology = ActiveArticulatedMorphology.from_spec(
        body_id=BODY_ID,
        embodiment_episode_id=EPISODE_ID,
        morphology_revision=MORPHOLOGY_REVISION,
        spec=_spec(),
    )
    surface = ActuatorSurface.from_effector_ids(
        ("actuator.hinge.positive", "actuator.hinge.negative"),
        fingerprint_material=SURFACE_FINGERPRINT,
    )
    current_actuator_context = actuator_context or ActuatorEvidenceContext(
        BODY_ID, EPISODE_ID, MORPHOLOGY_REVISION, surface.contract_fingerprint
    )
    contracts = {
        ("organism", str(ORGANISM_ID)): OrganismCheckpointContract(),
        ("embodiment", BODY_ID): _Contract(
            BODY_ID,
            {ARTICULATED_PHYSICAL_CHECKPOINT_SCHEMA: lambda data: _valid_body_checkpoint(data, morphology)},
        ),
        ("environment", ENVIRONMENT_OWNER_ID): _Contract(
            ENVIRONMENT_OWNER_ID,
            {StaticEnvironmentOwner.CHECKPOINT_SCHEMA: lambda data: StaticEnvironmentOwner.validate_checkpoint(
                data, ENVIRONMENT_OWNER_ID, StaticEnvironmentOwner.CHECKPOINT_SCHEMA
            )},
        ),
        ("environment", RESOURCE_ID): _Contract(
            RESOURCE_ID,
            {IdentifiedFiniteScalarField.OWNER_SCHEMA: lambda data: IdentifiedFiniteScalarField.validate_checkpoint(
                data, RESOURCE_ID, IdentifiedFiniteScalarField.OWNER_SCHEMA
            )},
        ),
    }
    manifest_store = WorldCausalManifestStore(directory / "manifest.json")
    journal_store = DurableJournalSnapshotStore(directory / "journal", WorldId(WORLD_ID))

    def verify_owner_checkpoints(refs: tuple[CheckpointReference, ...]) -> bool:
        try:
            return all(
                contracts[(ref.owner, ref.owner_identity)].validate_checkpoint(
                    Path(ref.locator).read_bytes(), ref.owner_identity, ref.schema_version
                )
                for ref in refs
                if ref.owner != "world-journal"
            )
        except OSError:
            return False

    restored = restore_scalar_sensorimotor_runtime(
        WorldId(WORLD_ID), PRESENCE_ID, manifest_store, journal_store,
        owner_checkpoint_verifier=verify_owner_checkpoints,
        owner_checkpoint_contracts=contracts,
        active_morphology=morphology,
        expected_organism_id=str(ORGANISM_ID),
        expected_physical_step_id="physical-step.acceptance.sensorimotor.tick-1",
        expected_environment_owner_id=ENVIRONMENT_OWNER_ID,
        expected_environment_id=ENVIRONMENT_ID,
        expected_environment_revision=ENVIRONMENT_REVISION,
        expected_field_id=RESOURCE_ID,
        expected_field_revision=expected_field_revision,
        actuator_context=current_actuator_context,
        actuator_surface=surface,
    )
    return restored, morphology


def _resume_in_fresh_process(directory: Path, output_path: Path) -> None:
    """Child-process entrypoint: restore and continue using World production APIs."""
    frame = _frame()
    restored, morphology = _restore_runtime_from_directory(directory)
    surface = ActuatorSurface.from_effector_ids(
        ("actuator.hinge.positive", "actuator.hinge.negative"),
        fingerprint_material=SURFACE_FINGERPRINT,
    )
    actuator_context = ActuatorEvidenceContext(
        BODY_ID, EPISODE_ID, MORPHOLOGY_REVISION, surface.contract_fingerprint
    )
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
    cycle = run_scalar_sensorimotor_cycle_and_publish(
        restored.world,
        plane,
        commit_id="commit.acceptance.sensorimotor.tick-2",
        physical_step_id="physical-step.acceptance.sensorimotor.tick-2",
        environment_revision=ENVIRONMENT_REVISION,
        solver_configuration=solver,
        current_organism=restored.organism.transition_state,
        action_composer=restored.organism.action_composer,
        receptor=receptor,
        embodiment_state=restored.articulated_body.root,
        environment_field=restored.environment_field,
        sampling_frame=frame,
        transduction=DimensionlessScalarFieldIdentityV1(),
        sense_binding=SenseChannelBinding(CHANNEL_ID, RECEPTOR_ID, "opaque.scalar.signal"),
        kernel_limits=KernelLimits(),
        actuator_context=actuator_context,
        actuator_surface=surface,
        articulated_body=restored.articulated_body,
        active_morphology=morphology,
        differential_binding=motor_binding,
        torque_binding=torque_binding,
        environment_owner=restored.environment_owner,
        owner_checkpoint_store=FileOwnerCheckpointStore(directory / "owners" / "committed"),
    )
    messages = plane.current_messages()
    output_path.write_text(
        json.dumps(
            {"process_id": os.getpid(), "cycle": _summary(cycle, messages)},
            sort_keys=True,
        )
    )


def _summary(cycle, messages) -> dict[str, object]:
    command = cycle.action_candidate.motor_command
    state = cycle.sensory_input.state
    return {
        "world_id": cycle.committed_cut.world_id,
        "causal_tick": cycle.committed_cut.causal_tick,
        "commit_id": cycle.committed_cut.commit_id,
        "organism_id": str(state.identity),
        "organism_step": state.organism_step,
        "signal_value": cycle.sensory_input.signal.value,
        "command_channels": repr(command.channels),
        "joint_state": repr(cycle.physical_outcome.body_state.joints),
        "owner_refs": sorted(
            (ref.owner, ref.owner_identity, ref.schema_version, ref.sha256)
            for ref in cycle.committed_cut.owner_checkpoint_references
        ),
        "observations": [repr(message.observation) for message in messages],
    }


def test_committed_cycle_restores_and_continues_in_fresh_process(tmp_path: Path) -> None:
    uninterrupted, _, uninterrupted_next, uninterrupted_messages = _run(
        tmp_path / "uninterrupted", continue_from_live_state=True
    )
    assert uninterrupted.committed_cut.causal_tick == 1
    assert uninterrupted_next is not None

    persisted, _, no_resume, _ = _run(
        tmp_path / "fresh-process", stop_after_first=True
    )
    assert persisted.committed_cut.causal_tick == 1
    assert no_resume is None

    output_path = tmp_path / "fresh-process-result.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--resume",
            str(tmp_path / "fresh-process"),
            str(output_path),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, (
        f"fresh-process restore failed ({completed.returncode})\n"
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
    fresh_process_result = json.loads(output_path.read_text())

    uninterrupted_summary = json.loads(
        json.dumps(_summary(uninterrupted_next, uninterrupted_messages), sort_keys=True)
    )
    assert fresh_process_result["process_id"] != os.getpid()
    assert fresh_process_result["cycle"] == uninterrupted_summary
    assert fresh_process_result["cycle"]["causal_tick"] == 2
    assert fresh_process_result["cycle"]["commit_id"] == "commit.acceptance.sensorimotor.tick-2"


def test_fresh_process_restore_rejects_corrupt_committed_owner_state(tmp_path: Path) -> None:
    persisted, _, no_resume, _ = _run(
        tmp_path / "corrupt-owner", stop_after_first=True
    )
    assert no_resume is None
    organism_reference = next(
        reference
        for reference in persisted.committed_cut.owner_checkpoint_references
        if reference.owner == "organism"
    )
    Path(organism_reference.locator).write_bytes(b"corrupt owner checkpoint")

    manifest_store = WorldCausalManifestStore(tmp_path / "corrupt-owner" / "manifest.json")
    committed_before_restore = manifest_store.read()
    assert committed_before_restore.causal_tick == 1

    with pytest.raises(ManifestError):
        _restore_runtime_from_directory(tmp_path / "corrupt-owner")

    committed_after_restore = manifest_store.read()
    assert committed_after_restore.causal_tick == committed_before_restore.causal_tick == 1
    assert committed_after_restore.commit_id == committed_before_restore.commit_id


def test_restore_rejects_digest_valid_semantically_malformed_organism_checkpoint(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "malformed-organism-owner"
    persisted, _, no_resume, _ = _run(
        directory,
        stop_after_first=True,
        malformed_organism_checkpoint=True,
    )
    assert no_resume is None
    organism_reference = next(
        reference
        for reference in persisted.committed_cut.owner_checkpoint_references
        if reference.owner == "organism"
    )
    assert organism_reference.owner_identity == str(ORGANISM_ID)
    assert organism_reference.schema_version == ORGANISM_EXPLORATION_CHECKPOINT_SCHEMA
    encoded = Path(organism_reference.locator).read_bytes()
    assert hashlib.sha256(encoded).hexdigest() == organism_reference.sha256
    envelope = json.loads(encoded)
    transition = OrganismTransitionState.from_bytes(
        base64.b64decode(envelope["transition_state_base64"], validate=True)
    )
    # The nested owner envelope is itself digest-valid, while its private
    # cognitive bytes fail the Organism semantic decoder.
    assert OrganismCheckpoint.from_bytes(
        transition.owner_checkpoint.to_bytes()
    ) == transition.owner_checkpoint
    with pytest.raises(ValueError):
        CognitiveRuntimeCheckpoint.from_bytes(
            transition.owner_checkpoint.private_state,
            kernel_limits=KernelLimits(),
        )

    manifest_store = WorldCausalManifestStore(directory / "manifest.json")
    committed_before_restore = manifest_store.read()
    assert committed_before_restore.causal_tick == persisted.committed_cut.causal_tick == 1

    with pytest.raises(ManifestError):
        _restore_runtime_from_directory(directory)

    committed_after_restore = manifest_store.read()
    assert committed_after_restore.causal_tick == committed_before_restore.causal_tick == 1
    assert committed_after_restore.commit_id == committed_before_restore.commit_id
    assert committed_after_restore.checkpoint_references == committed_before_restore.checkpoint_references


@pytest.mark.parametrize(
    ("owner", "owner_identity"),
    (("embodiment", BODY_ID), ("organism", str(ORGANISM_ID))),
)
def test_fresh_process_restore_rejects_missing_committed_owner_checkpoint(
    tmp_path: Path, owner: str, owner_identity: str
) -> None:
    directory = tmp_path / f"missing-{owner}-owner"
    persisted, _, no_resume, _ = _run(directory, stop_after_first=True)
    assert no_resume is None
    owner_reference = next(
        reference
        for reference in persisted.committed_cut.owner_checkpoint_references
        if reference.owner == owner and reference.owner_identity == owner_identity
    )
    Path(owner_reference.locator).unlink()

    manifest_store = WorldCausalManifestStore(directory / "manifest.json")
    committed_before_restore = manifest_store.read()
    assert committed_before_restore.causal_tick == 1

    with pytest.raises(ManifestError):
        _restore_runtime_from_directory(directory)

    committed_after_restore = manifest_store.read()
    assert committed_after_restore.causal_tick == committed_before_restore.causal_tick == 1
    assert committed_after_restore.commit_id == committed_before_restore.commit_id


def test_fresh_process_restore_rejects_stale_owner_revision_without_publishing_cut(
    tmp_path: Path,
) -> None:
    persisted, _, no_resume, _ = _run(
        tmp_path / "stale-owner-revision", stop_after_first=True
    )
    assert no_resume is None
    manifest_store = WorldCausalManifestStore(
        tmp_path / "stale-owner-revision" / "manifest.json"
    )
    committed_before_restore = manifest_store.read()
    assert committed_before_restore.causal_tick == persisted.committed_cut.causal_tick == 1

    with pytest.raises(ManifestError, match="scalar field identity or revision"):
        _restore_runtime_from_directory(
            tmp_path / "stale-owner-revision",
            expected_field_revision="field.acceptance.stale-r0",
        )

    committed_after_restore = manifest_store.read()
    assert committed_after_restore.causal_tick == committed_before_restore.causal_tick == 1
    assert committed_after_restore.commit_id == committed_before_restore.commit_id


def test_fresh_process_restore_rejects_stale_actuator_applicability_without_publishing_cut(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "stale-actuator-applicability"
    persisted, _, no_resume, _ = _run(directory, stop_after_first=True)
    assert no_resume is None
    manifest_store = WorldCausalManifestStore(directory / "manifest.json")
    committed_before_restore = manifest_store.read()
    assert committed_before_restore.causal_tick == persisted.committed_cut.causal_tick == 1

    stale_context = ActuatorEvidenceContext(
        BODY_ID,
        "episode.acceptance.stale-r0",
        MORPHOLOGY_REVISION,
        SURFACE_FINGERPRINT,
    )
    with pytest.raises(ValueError, match="active Embodiment morphology and surface"):
        _restore_runtime_from_directory(directory, actuator_context=stale_context)

    committed_after_restore = manifest_store.read()
    assert committed_after_restore.causal_tick == committed_before_restore.causal_tick == 1
    assert committed_after_restore.commit_id == committed_before_restore.commit_id
    assert committed_after_restore.checkpoint_references == committed_before_restore.checkpoint_references


def _valid_body_checkpoint(data: bytes, morphology: ActiveArticulatedMorphology) -> bool:
    try:
        ArticulatedEmbodimentPhysicalState.restore_checkpoint(data, active_morphology=morphology)
    except (TypeError, ValueError):
        return False
    return True


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] != "--resume":
        raise SystemExit("usage: test_end_to_end_causal_cycle_acceptance.py --resume DIR OUTPUT")
    _resume_in_fresh_process(Path(sys.argv[2]), Path(sys.argv[3]))


def test_real_target_sensorimotor_cycle_commits_publishes_and_repeats(tmp_path: Path) -> None:
    first, first_messages, first_resumed, first_resumed_messages = _run(
        tmp_path / "run-a"
    )
    second, second_messages, second_resumed, second_resumed_messages = _run(
        tmp_path / "run-b"
    )

    for result, messages in (
        (first, first_messages),
        (second, second_messages),
    ):
        observations = tuple(message.observation for message in messages)
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
        organism_references = tuple(
            reference
            for reference in result.committed_cut.owner_checkpoint_references
            if reference.owner == "organism" and reference.owner_identity == str(ORGANISM_ID)
        )
        assert len(organism_references) == 1
        organism_reference = organism_references[0]
        assert organism_reference.schema_version == ORGANISM_EXPLORATION_CHECKPOINT_SCHEMA
        # Owner-side payload validation is not proof of restored-state equivalence
        # or binding to this exact committed World cut.
        assert OrganismCheckpointContract().validate_checkpoint(
            Path(organism_reference.locator).read_bytes(),
            organism_reference.owner_identity,
            organism_reference.schema_version,
        )
        assert len(result.observation_publication.delivery_sequences) == 2
        assert result.observation_publication.committed_cut == result.committed_cut
        assert tuple(message.sequence for message in messages) == (
            result.observation_publication.delivery_sequences
        )
        assert len(observations) == 2
        body_observation, field_observation = observations
        assert body_observation.claim.family.path == ("body_pose",)
        assert body_observation.context_time[0].value == result.committed_cut.causal_tick
        assert field_observation.claim.family.path == ("finite_scalar_field",)
        assert field_observation.context_time[-1].value == result.committed_cut.causal_tick
        assert any(
            reference.owner == "embodiment"
            and reference.sha256
            in {
                parent.reference
                for parent in body_observation.lineage.parents
            }
            for reference in result.committed_cut.owner_checkpoint_references
        )
        assert any(
            reference.owner == "environment"
            and reference.owner_identity == RESOURCE_ID
            and reference.sha256
            in {
                parent.reference
                for parent in field_observation.lineage.parents
            }
            for reference in result.committed_cut.owner_checkpoint_references
        )

    for resumed, messages in (
        (first_resumed, first_resumed_messages),
        (second_resumed, second_resumed_messages),
    ):
        observations = tuple(message.observation for message in messages)
        assert resumed.context.source_causal_tick == 1
        assert resumed.context.target_causal_tick == resumed.committed_cut.causal_tick == 2
        assert resumed.action_candidate.motor_command is not None
        assert resumed.physical_outcome.body_state.joints[0].position_rad != 0.0
        assert len(resumed.observation_publication.delivery_sequences) == 2
        assert resumed.observation_publication.committed_cut == resumed.committed_cut
        assert len(observations) == 2
        assert tuple(message.sequence for message in messages) == (
            resumed.observation_publication.delivery_sequences
        )
        body_observation, field_observation = observations
        assert body_observation.claim.family.path == ("body_pose",)
        assert field_observation.claim.family.path == ("finite_scalar_field",)
        for observation in observations:
            assert any(
                domain_time.domain.value == "world"
                and domain_time.kind
                in {"world_target_causal_tick", "world_causal_tick"}
                and domain_time.value == resumed.committed_cut.causal_tick
                for domain_time in observation.context_time
            )
        assert any(
            reference.owner == "embodiment"
            and reference.sha256
            in {parent.reference for parent in body_observation.lineage.parents}
            for reference in resumed.committed_cut.owner_checkpoint_references
        )
        assert any(
            reference.owner == "environment"
            and reference.owner_identity == RESOURCE_ID
            and reference.sha256
            in {parent.reference for parent in field_observation.lineage.parents}
            for reference in resumed.committed_cut.owner_checkpoint_references
        )

    assert first.committed_cut.commit_id == second.committed_cut.commit_id
    assert first.committed_cut.causal_tick == second.committed_cut.causal_tick
    assert first.sensory_input.signal.value == second.sensory_input.signal.value
    assert first.action_candidate.motor_command == second.action_candidate.motor_command
    assert first.physical_outcome.body_state.joints == pytest.approx(
        second.physical_outcome.body_state.joints
    )
    assert tuple(message.observation for message in first_messages) == tuple(
        message.observation for message in second_messages
    )
    assert first_resumed.committed_cut.commit_id == second_resumed.committed_cut.commit_id
    assert first_resumed.committed_cut.causal_tick == second_resumed.committed_cut.causal_tick == 2
    assert first_resumed.action_candidate.motor_command == second_resumed.action_candidate.motor_command
    assert tuple(message.observation for message in first_resumed_messages) == tuple(
        message.observation for message in second_resumed_messages
    )
    assert first_resumed.physical_outcome.body_state.joints == pytest.approx(
        second_resumed.physical_outcome.body_state.joints
    )
