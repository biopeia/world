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
import urllib.request
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import mujoco
import pytest
from embodiment.observation import BodyPoseSample
from embodiment.checkpoint import EmbodimentAdoptionBindingV1, EmbodimentCheckpoint
from embodiment.metabolism import MetabolicLedger
from embodiment.physiology import LivingBodyState
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
from organism.actuation import (
    ORGANISM_EXPLORATION_CHECKPOINT_SCHEMA,
    ORGANISM_EXPLORATION_CHECKPOINT_V5_SCHEMA,
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
    CausalCycleResult,
    ScalarSensorimotorCycleInputs,
    ScalarSensorimotorSessionStatus,
    restore_scalar_sensorimotor_runtime,
    run_scalar_sensorimotor_cycle_and_publish,
    run_scalar_sensorimotor_session,
)
from world.durable_runtime import (
    DurableJournalSnapshotStore,
    DurableWorldRuntime,
    PresenceOwnerLink,
    WorldGenesisSpec,
)
from world.manifest import CheckpointReference, ManifestError, WorldCausalManifestStore
from world.manifest import ManifestStatus
from organism.actuation.action_history import HistoryIndex
from world.checkpoint_dependency import resolve_dependency
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

    def verify_checkpoint_set(self, references, *, world_id, source_causal_tick, target_causal_tick, commit_id):
        return EmbodimentAdoptionBindingV1.verify_checkpoint_set(
            tuple(references), world_id=world_id,
            source_causal_tick=source_causal_tick,
            target_causal_tick=target_causal_tick, commit_id=commit_id,
        )


def _reserve_checkpoint(body_id: str, episode_id: str) -> EmbodimentCheckpoint:
    body = LivingBodyState(energy_reserve=1.2, max_energy=2.0)
    return EmbodimentCheckpoint(body_id, episode_id, body, MetabolicLedger(body_state=body))


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


def _committed_resolver(directory: Path):
    """Digest-verified reads from the committed owner store (M3 dependencies)."""
    root = directory / "owners" / "committed"
    return lambda digest: resolve_dependency(root, digest)


def _run(
    directory: Path,
    *,
    stop_after_first: bool = False,
    continue_from_live_state: bool = False,
    malformed_organism_checkpoint: bool = False,
    session_cycles: int | None = None,
    use_session: bool = False,
    session_stop_after: int | None = None,
    on_cycle_result: Callable[[CausalCycleResult], None] | None = None,
    retain_cycle_results: bool = True,
    environment_revision_override: str | None = None,
    kernel_limits: KernelLimits = KernelLimits(),
    actuator_surface_material: str = SURFACE_FINGERPRINT,
    initial_joint_velocity_rad_per_s: float = 0.0,
    velocity_regulation_policy: str = "none",
    organism_v5: bool = False,
    after_first_commit: Callable[
        [DurableWorldRuntime, CausalCycleResult, ObservationPlane], None
    ]
    | None = None,
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
        joints=(JointKinematicState("joint.hinge", 0.0, initial_joint_velocity_rad_per_s),),
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
            kernel_limits=kernel_limits,
        ),
    )
    organism = OrganismTransitionState(
        owner_checkpoint=cognition.to_organism_checkpoint(),
        organism_step=0,
        embodiment=EmbodimentContextRef(BODY_ID, EPISODE_ID, MORPHOLOGY_REVISION),
        lifecycle=OrganismLifecycle.ACTIVE,
    )
    contracts = {
        ("organism", str(ORGANISM_ID)): OrganismCheckpointContract(
            kernel_limits=kernel_limits,
            read_dependency=_committed_resolver(directory) if organism_v5 else None,
        ),
        ("embodiment", BODY_ID): _Contract(
            BODY_ID, {
                ARTICULATED_PHYSICAL_CHECKPOINT_SCHEMA: lambda data: _valid_body_checkpoint(data, morphology),
                EmbodimentCheckpoint.OWNER_SCHEMA: lambda data: EmbodimentCheckpoint.validate_checkpoint(
                    data, BODY_ID, EmbodimentCheckpoint.OWNER_SCHEMA
                ),
                EmbodimentAdoptionBindingV1.OWNER_SCHEMA: lambda data: EmbodimentAdoptionBindingV1.validate_checkpoint(
                    data, BODY_ID, EmbodimentAdoptionBindingV1.OWNER_SCHEMA
                ),
            },
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
        fingerprint_material=actuator_surface_material,
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
    composer = OrganismExplorationComposer(
        organism_id=str(ORGANISM_ID), engine=engine,
        history=HistoryIndex() if organism_v5 else None,
    )
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
        velocity_regulation_policy=velocity_regulation_policy,
    )

    plane = ObservationPlane()

    cycle_inputs = ScalarSensorimotorCycleInputs(
        environment_revision=environment_revision_override or ENVIRONMENT_REVISION,
        solver_configuration=solver,
        current_organism=organism,
        action_composer=composer,
        receptor=receptor,
        embodiment_state=root,
        environment_field=field,
        sampling_frame=frame,
        transduction=DimensionlessScalarFieldIdentityV1(),
        sense_binding=SenseChannelBinding(CHANNEL_ID, RECEPTOR_ID, "opaque.scalar.signal"),
        kernel_limits=kernel_limits,
        actuator_context=actuator_context,
        actuator_surface=surface,
        articulated_body=body,
        reserve_checkpoint=_reserve_checkpoint(BODY_ID, EPISODE_ID),
        active_morphology=morphology,
        differential_binding=motor_binding,
        torque_binding=torque_binding,
        environment_owner=env_owner,
        owner_checkpoint_store=owner_checkpoint_store,
        embodiment_geometry_revision="geometry.acceptance.sensorimotor-r1",
    )
    if not use_session:
        result = run_scalar_sensorimotor_cycle_and_publish(
            world, plane,
            commit_id="commit.acceptance.sensorimotor.tick-1",
            physical_step_id="physical-step.acceptance.sensorimotor.tick-1",
            **cycle_inputs.as_cycle_arguments(),
        )
        session_result = None
    else:
        session_result = run_scalar_sensorimotor_session(
            world,
            plane,
            run_id="acceptance-run",
            max_cycles=session_cycles,
            cycle_inputs=cycle_inputs,
            stop_requested=(
                (lambda: world.causal_tick >= session_stop_after)
                if session_stop_after is not None
                else None
            ),
            on_cycle_result=on_cycle_result,
            retain_cycle_results=retain_cycle_results,
            source_revisions={"world": "test-commit"},
        )
        result = session_result.cycles[0] if session_result.cycles else None
    first_messages = plane.current_messages()
    if after_first_commit is not None and result is not None:
        after_first_commit(world, result, plane)
    if stop_after_first:
        return result, first_messages, None, None

    if session_result is not None:
        return session_result, first_messages, None, None

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
        next_reserve_checkpoint = cycle_inputs.reserve_checkpoint
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
            kernel_limits=kernel_limits,
            actuator_context=actuator_context,
            actuator_surface=surface,
        )
        next_world = restored_runtime.world
        next_organism = restored_runtime.organism.transition_state
        next_composer = restored_runtime.organism.action_composer
        next_body = restored_runtime.articulated_body
        next_environment = restored_runtime.environment_owner
        next_field = restored_runtime.environment_field
        next_reserve_checkpoint = restored_runtime.reserve_checkpoint
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
        reserve_checkpoint=next_reserve_checkpoint,
        embodiment_geometry_revision="geometry.acceptance.sensorimotor-r1",
        environment_field=next_field,
        sampling_frame=frame,
        transduction=DimensionlessScalarFieldIdentityV1(),
        sense_binding=SenseChannelBinding(CHANNEL_ID, RECEPTOR_ID, "opaque.scalar.signal"),
        kernel_limits=kernel_limits,
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
    kernel_limits: KernelLimits | None = KernelLimits(),
    actuator_context: ActuatorEvidenceContext | None = None,
    actuator_surface_material: str = SURFACE_FINGERPRINT,
    organism_v5: bool = False,
):
    morphology = ActiveArticulatedMorphology.from_spec(
        body_id=BODY_ID,
        embodiment_episode_id=EPISODE_ID,
        morphology_revision=MORPHOLOGY_REVISION,
        spec=_spec(),
    )
    surface = ActuatorSurface.from_effector_ids(
        ("actuator.hinge.positive", "actuator.hinge.negative"),
        fingerprint_material=actuator_surface_material,
    )
    current_actuator_context = actuator_context or ActuatorEvidenceContext(
        BODY_ID, EPISODE_ID, MORPHOLOGY_REVISION, surface.contract_fingerprint
    )
    contracts = {
        ("organism", str(ORGANISM_ID)): OrganismCheckpointContract(
            kernel_limits=kernel_limits or KernelLimits(),
            read_dependency=_committed_resolver(directory) if organism_v5 else None,
        ),
        ("embodiment", BODY_ID): _Contract(
            BODY_ID,
            {
                ARTICULATED_PHYSICAL_CHECKPOINT_SCHEMA: lambda data: _valid_body_checkpoint(data, morphology),
                EmbodimentCheckpoint.OWNER_SCHEMA: lambda data: EmbodimentCheckpoint.validate_checkpoint(
                    data, BODY_ID, EmbodimentCheckpoint.OWNER_SCHEMA
                ),
                EmbodimentAdoptionBindingV1.OWNER_SCHEMA: lambda data: EmbodimentAdoptionBindingV1.validate_checkpoint(
                    data, BODY_ID, EmbodimentAdoptionBindingV1.OWNER_SCHEMA
                ),
            },
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
        kernel_limits=kernel_limits,
        actuator_context=current_actuator_context,
        actuator_surface=surface,
    )
    return restored, morphology


def _resume_in_fresh_process(directory: Path, output_path: Path, *, organism_v5: bool = False) -> None:
    """Child-process entrypoint: restore and continue using World production APIs."""
    frame = _frame()
    restored, morphology = _restore_runtime_from_directory(directory, organism_v5=organism_v5)
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
    session = run_scalar_sensorimotor_session(
        restored.world,
        plane,
        run_id="acceptance-restored-run",
        source_revisions={"world": "test-commit"},
        cycle_inputs=ScalarSensorimotorCycleInputs(
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
            reserve_checkpoint=_reserve_checkpoint(
                restored.articulated_body.root.body_id,
                restored.articulated_body.root.embodiment_episode_id,
            ),
            active_morphology=morphology,
            differential_binding=motor_binding,
            torque_binding=torque_binding,
            environment_owner=restored.environment_owner,
            owner_checkpoint_store=FileOwnerCheckpointStore(directory / "owners" / "committed"),
            embodiment_geometry_revision="geometry.acceptance.sensorimotor-r1",
        ),
        max_cycles=1,
        commit_id_prefix="commit.acceptance.sensorimotor.tick-",
        physical_step_id_prefix="physical-step.acceptance.sensorimotor.tick-",
    )
    assert session.status is ScalarSensorimotorSessionStatus.COMPLETED
    cycle = session.cycles[0]
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


def test_adoption_binding_verifier_rejects_incomplete_corrupt_and_cross_cut_sets(
    tmp_path: Path,
) -> None:
    result, _, no_resume, _ = _run(tmp_path / "binding-negative", stop_after_first=True)
    assert no_resume is None
    references = tuple(
        reference for reference in result.committed_cut.owner_checkpoint_references
        if reference.owner == "embodiment" and reference.owner_identity == BODY_ID
    )
    context = dict(
        world_id=result.context.world_id,
        source_causal_tick=result.context.source_causal_tick,
        target_causal_tick=result.context.target_causal_tick,
        commit_id=result.committed_cut.commit_id,
    )
    verify = EmbodimentAdoptionBindingV1.verify_checkpoint_set
    assert verify(references, **context)

    # Missing references (including an unbound historical cut) and duplicate
    # references cannot be promoted to an adoption assertion.
    for index in range(len(references)):
        assert not verify(references[:index] + references[index + 1 :], **context)
    binding_reference = next(
        ref for ref in references if ref.schema_version == EmbodimentAdoptionBindingV1.OWNER_SCHEMA
    )
    assert not verify(references + (binding_reference,), **context)

    # A valid-looking owner set still cannot be replayed under another World,
    # transition tick, or commit identity.
    for field, wrong in (
        ("world_id", "world.other"),
        ("source_causal_tick", context["source_causal_tick"] + 1),
        ("target_causal_tick", context["target_causal_tick"] + 1),
        ("commit_id", "commit.other"),
    ):
        assert not verify(references, **(context | {field: wrong}))

    # Every persisted member is independently integrity checked.
    for reference in references:
        path = Path(reference.locator)
        original = path.read_bytes()
        path.write_bytes(original + b"corrupt")
        assert not verify(references, **context)
        path.write_bytes(original)

    # Recompute a structurally valid binding with another episode; the owner
    # digest is valid, but it cannot agree with the adopted physical/reserve state.
    encoded_binding = Path(binding_reference.locator).read_bytes()
    wrong_episode = replace(
        EmbodimentAdoptionBindingV1.from_bytes(encoded_binding), episode_id="episode.other"
    ).to_bytes()
    wrong_binding_path = tmp_path / "wrong-episode-binding.json"
    wrong_binding_path.write_bytes(wrong_episode)
    wrong_binding_reference = replace(
        binding_reference,
        locator=str(wrong_binding_path),
        sha256=hashlib.sha256(wrong_episode).hexdigest(),
    )
    wrong_episode_references = tuple(
        wrong_binding_reference if ref == binding_reference else ref for ref in references
    )
    assert not verify(wrong_episode_references, **context)

    # A caller cannot make mismatched identities look like one owner set by
    # supplying the expected body id separately.
    wrong_identity_references = tuple(
        replace(ref, owner_identity="body.other") for ref in references
    )
    assert not verify(wrong_identity_references, **context)

    # A binding from another morphology or physical step is rejected even if
    # serialized canonically and its own digest is updated.
    valid_binding = EmbodimentAdoptionBindingV1.from_bytes(encoded_binding)
    for changes in ({"morphology_revision": "morphology.other"},
                    {"physical_step_id": "physical-step.other"}):
        changed_bytes = replace(valid_binding, **changes).to_bytes()
        changed_path = tmp_path / f"wrong-{next(iter(changes))}.json"
        changed_path.write_bytes(changed_bytes)
        changed_reference = replace(
            binding_reference,
            locator=str(changed_path),
            sha256=hashlib.sha256(changed_bytes).hexdigest(),
        )
        changed_references = tuple(
            changed_reference if ref == binding_reference else ref for ref in references
        )
        assert not verify(changed_references, **context)


def test_non_default_kernel_limits_restore_and_continue_fail_closed(tmp_path: Path) -> None:
    configured_limits = replace(KernelLimits(), consolidation_interval_ticks=31)
    uninterrupted, _, uninterrupted_next, uninterrupted_messages = _run(
        tmp_path / "non-default-uninterrupted",
        continue_from_live_state=True,
        kernel_limits=configured_limits,
    )
    persisted, _, resumed, resumed_messages = _run(
        tmp_path / "non-default-restored",
        kernel_limits=configured_limits,
    )

    assert resumed is not None
    assert uninterrupted_next is not None
    assert uninterrupted.committed_cut.causal_tick == persisted.committed_cut.causal_tick == 1
    assert _summary(resumed, resumed_messages) == _summary(
        uninterrupted_next, uninterrupted_messages
    )

    directory = tmp_path / "non-default-rejected-restore"
    first, _, no_resume, _ = _run(
        directory,
        stop_after_first=True,
        kernel_limits=configured_limits,
    )
    assert no_resume is None
    manifest_store = WorldCausalManifestStore(directory / "manifest.json")
    committed_before_restore = manifest_store.read()
    assert committed_before_restore.causal_tick == first.committed_cut.causal_tick == 1

    for supplied_limits in (None, replace(configured_limits, consolidation_interval_ticks=32)):
        with pytest.raises(ManifestError):
            _restore_runtime_from_directory(directory, kernel_limits=supplied_limits)
        committed_after_restore = manifest_store.read()
        assert committed_after_restore.causal_tick == committed_before_restore.causal_tick == 1
        assert committed_after_restore.commit_id == committed_before_restore.commit_id


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
    persisted, _, no_resume, _ = _run(
        directory,
        stop_after_first=True,
        actuator_surface_material="surface.saved-r1",
    )
    assert no_resume is None
    manifest_store = WorldCausalManifestStore(directory / "manifest.json")
    committed_before_restore = manifest_store.read()
    assert committed_before_restore.causal_tick == persisted.committed_cut.causal_tick == 1

    # The current morphology and context agree with each other, but the
    # committed Organism checkpoint was produced against another surface.
    current_surface = ActuatorSurface.from_effector_ids(
        ("actuator.hinge.positive", "actuator.hinge.negative"),
        fingerprint_material="surface.current-r2",
    )
    current_context = ActuatorEvidenceContext(
        BODY_ID,
        EPISODE_ID,
        MORPHOLOGY_REVISION,
        current_surface.contract_fingerprint,
    )
    with pytest.raises(ManifestError, match="Organism checkpoint does not restore"):
        _restore_runtime_from_directory(
            directory,
            actuator_context=current_context,
            actuator_surface_material="surface.current-r2",
        )

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
    if len(sys.argv) != 4 or sys.argv[1] not in ("--resume", "--resume-v5"):
        raise SystemExit("usage: test_end_to_end_causal_cycle_acceptance.py --resume|--resume-v5 DIR OUTPUT")
    _resume_in_fresh_process(Path(sys.argv[2]), Path(sys.argv[3]), organism_v5=sys.argv[1] == "--resume-v5")


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
        observations = tuple(message.observation for message in messages if message.observation.context_time)
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
        assert len(result.committed_cut.owner_checkpoint_references) == 6
        embodiment_references = tuple(
            reference for reference in result.committed_cut.owner_checkpoint_references
            if reference.owner == "embodiment" and reference.owner_identity == BODY_ID
        )
        assert {reference.schema_version for reference in embodiment_references} == {
            ARTICULATED_PHYSICAL_CHECKPOINT_SCHEMA,
            EmbodimentCheckpoint.OWNER_SCHEMA,
            EmbodimentAdoptionBindingV1.OWNER_SCHEMA,
        }
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
        assert len(result.observation_publication.delivery_sequences) == 3
        assert result.observation_publication.committed_cut == result.committed_cut
        assert tuple(message.sequence for message in messages if message.observation.context_time) == (
            result.observation_publication.delivery_sequences
        )
        assert len(observations) == 3
        body_observation, field_observation, reserve_observation = observations
        assert body_observation.claim.family.path == ("body_pose",)
        assert body_observation.context_time[0].value == result.committed_cut.causal_tick
        assert field_observation.claim.family.path == ("finite_scalar_field",)
        assert field_observation.context_time[-1].value == result.committed_cut.causal_tick
        assert reserve_observation.claim.family.path == ("reserve_state_snapshot",)
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
        observations = tuple(message.observation for message in messages if any(
            ref.domain.value == "world" and ref.value == resumed.committed_cut.causal_tick
            for ref in message.observation.context_time
        ))
        assert resumed.context.source_causal_tick == 1
        assert resumed.context.target_causal_tick == resumed.committed_cut.causal_tick == 2
        assert resumed.action_candidate.motor_command is not None
        assert resumed.physical_outcome.body_state.joints[0].position_rad != 0.0
        assert len(resumed.observation_publication.delivery_sequences) == 3
        assert resumed.observation_publication.committed_cut == resumed.committed_cut
        assert len(observations) == 3
        assert tuple(message.sequence for message in messages if any(
            ref.domain.value == "world" and ref.value == resumed.committed_cut.causal_tick
            for ref in message.observation.context_time
        )) == (
            resumed.observation_publication.delivery_sequences
        )
        body_observation, field_observation, reserve_observation = observations
        assert body_observation.claim.family.path == ("body_pose",)
        assert field_observation.claim.family.path == ("finite_scalar_field",)
        assert reserve_observation.claim.family.path == ("reserve_state_snapshot",)
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
    def claims(messages):
        return {
            f"{message.observation.claim.family.authority.value}."
            f"{'.'.join(message.observation.claim.family.path) }": message.observation.claim.value
            for message in messages
        }

    first_claims, second_claims = claims(first_messages), claims(second_messages)
    assert first_claims.keys() == second_claims.keys()
    for family in ("embodiment.body_pose", "environment.finite_scalar_field", "embodiment.reserve_state_snapshot", "embodiment.render_morphology", "environment.render_scene"):
        assert first_claims[family] == second_claims[family]
    assert first_claims["embodiment.reserve_state_snapshot"] == second_claims[
        "embodiment.reserve_state_snapshot"
    ]
    assert first_resumed.committed_cut.commit_id == second_resumed.committed_cut.commit_id
    assert first_resumed.committed_cut.causal_tick == second_resumed.committed_cut.causal_tick == 2
    assert first_resumed.action_candidate.motor_command == second_resumed.action_candidate.motor_command
    first_resumed_claims = claims(first_resumed_messages)
    second_resumed_claims = claims(second_resumed_messages)
    assert first_resumed_claims.keys() == second_resumed_claims.keys()
    for family in ("embodiment.body_pose", "environment.finite_scalar_field"):
        assert first_resumed_claims[family] == second_resumed_claims[family]
    assert first_resumed_claims["embodiment.reserve_state_snapshot"] == second_resumed_claims[
        "embodiment.reserve_state_snapshot"
    ]
    assert first_resumed.physical_outcome.body_state.joints == pytest.approx(
        second_resumed.physical_outcome.body_state.joints
    )


def test_explicit_no_actuation_advances_physics_and_commits_cut(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import world.adapters.articulated_mujoco as articulated_mujoco
    from organism.actuation import CompetenceDevelopmentEngine, ExplorationOutcome

    def completed_without_intents(self, tick):
        self._last_output_source = "exploration"
        self._last_output_primitive_id = None
        return ()

    def reject_torque_search(*args, **kwargs):
        raise AssertionError("no-actuation must not invoke torque search")

    monkeypatch.setattr(CompetenceDevelopmentEngine, "motor_intents", completed_without_intents)
    monkeypatch.setattr(articulated_mujoco, "_select_compliant_torque", reject_torque_search)
    result, messages, _, _ = _run(
        tmp_path / "explicit-no-actuation",
        stop_after_first=True,
        initial_joint_velocity_rad_per_s=0.25,
        velocity_regulation_policy="m1-fixed-single-hinge-grid-v2",
    )

    assert result.action_candidate.outcome is ExplorationOutcome.NO_ACTUATION
    assert result.action_candidate.proposal is None
    assert result.action_candidate.motor_command is None
    assert result.committed_cut.causal_tick == result.physical_outcome.record.causal_tick == 1
    assert result.physical_outcome.body_state.joints[0].position_rad != 0.0
    assert result.physical_outcome.solver_evidence["requested_torque_nm"] == 0.0
    assert result.physical_outcome.solver_evidence["applied_torque_nm"] == 0.0
    assert len(result.observation_publication.delivery_sequences) == 3
    assert len(tuple(message for message in messages if message.observation.context_time)) == 3


def test_no_actuation_rejects_infeasible_zero_torque_without_committing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    import numpy as np
    import world.adapters.articulated_mujoco as articulated_mujoco
    from organism.actuation import CompetenceDevelopmentEngine

    directory = tmp_path / "no-actuation-infeasible"

    def completed_without_intents(self, tick):
        self._last_output_source = "exploration"
        self._last_output_primitive_id = None
        return ()

    def out_of_range_zero_torque_trial(mujoco, model, qpos, qvel, qfrc, dof, applied):
        assert applied == 0.0
        return SimpleNamespace(
            qpos=np.array((1.001,)), qvel=np.array((0.25,)), warning=()
        )

    def reject_torque_search(*args, **kwargs):
        raise AssertionError("no-actuation must fail closed, not search for another torque")

    monkeypatch.setattr(CompetenceDevelopmentEngine, "motor_intents", completed_without_intents)
    monkeypatch.setattr(articulated_mujoco, "_mujoco_trial", out_of_range_zero_torque_trial)
    monkeypatch.setattr(articulated_mujoco, "_select_compliant_torque", reject_torque_search)

    with pytest.raises(ValueError, match="no-actuation zero-torque candidate violates"):
        _run(
            directory,
            stop_after_first=True,
            initial_joint_velocity_rad_per_s=0.25,
            velocity_regulation_policy="m1-fixed-single-hinge-grid-v2",
        )

    manifest = WorldCausalManifestStore(directory / "manifest.json").read()
    assert manifest.status is ManifestStatus.RECOVERY_REQUIRED
    assert manifest.causal_tick == 0
    assert not any(
        reference.schema_version == EmbodimentAdoptionBindingV1.OWNER_SCHEMA
        for reference in manifest.checkpoint_references
    )


def test_articulated_solver_revalidates_no_actuation_at_physics_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The low-level physical join must not trust a forged projection object."""
    import world.adapters.articulated_mujoco as articulated_mujoco
    from organism.actuation import CompetenceDevelopmentEngine

    original_project = articulated_mujoco.project_no_actuation
    calls = 0

    def completed_without_intents(self, tick):
        self._last_output_source = "exploration"
        self._last_output_primitive_id = None
        return ()

    def reject_at_solver_boundary(binding, morphology, *, issued_at_tick):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValueError("solver-boundary no-actuation revalidation rejected")
        return original_project(binding, morphology, issued_at_tick=issued_at_tick)

    monkeypatch.setattr(articulated_mujoco, "project_no_actuation", reject_at_solver_boundary)
    monkeypatch.setattr(CompetenceDevelopmentEngine, "motor_intents", completed_without_intents)
    with pytest.raises(ValueError, match="solver-boundary no-actuation revalidation"):
        _run(tmp_path / "solver-boundary-revalidation", stop_after_first=True)

    assert calls == 2
    manifest = WorldCausalManifestStore(
        tmp_path / "solver-boundary-revalidation" / "manifest.json"
    ).read()
    assert manifest.status is ManifestStatus.RECOVERY_REQUIRED
    assert manifest.causal_tick == 0


@pytest.mark.parametrize("invalid_result", ["missing_outcome", "terminated"])
def test_invalid_or_terminated_organism_result_preserves_last_cut(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    invalid_result: str,
) -> None:
    from organism.actuation import OrganismExplorationComposer
    from organism.transition import OrganismLifecycle

    original_prepare = OrganismExplorationComposer.prepare

    def prepare_invalid(self, transition, **kwargs):
        if invalid_result == "terminated":
            transition = replace(transition, lifecycle=OrganismLifecycle.TERMINATED)
            return original_prepare(self, transition, **kwargs)
        candidate = original_prepare(self, transition, **kwargs)
        # Exercise World as an adversarial structural consumer: a malformed
        # candidate must not be reinterpreted as a valid no-actuation result.
        object.__setattr__(candidate, "outcome", None)
        return candidate

    monkeypatch.setattr(OrganismExplorationComposer, "prepare", prepare_invalid)
    directory = tmp_path / invalid_result
    with pytest.raises((TypeError, ValueError)):
        _run(directory, stop_after_first=True)

    manifest = WorldCausalManifestStore(directory / "manifest.json").read()
    assert manifest.status is ManifestStatus.COMMITTED
    assert manifest.causal_tick == 0


def test_world_owned_session_runs_three_production_cycles_and_publishes_each(
    tmp_path: Path,
) -> None:
    session, _messages, _, _ = _run(
        tmp_path / "three-cycle-session", session_cycles=3, use_session=True
    )

    assert session.status is ScalarSensorimotorSessionStatus.COMPLETED
    assert session.world_id == WORLD_ID
    assert session.run_id == "acceptance-run"
    assert session.source_revisions == {"world": "test-commit"}
    assert tuple(result.committed_cut.causal_tick for result in session.cycles) == (1, 2, 3)
    assert tuple(result.context.source_causal_tick for result in session.cycles) == (0, 1, 2)
    assert len({result.committed_cut.commit_id for result in session.cycles}) == 3
    for result in session.cycles:
        assert result.observation_publication.committed_cut == result.committed_cut
        assert len(result.observation_publication.delivery_sequences) == 3
    assert session.last_committed_cut == session.cycles[-1].committed_cut
    assert session.error is None


def test_world_owned_session_honors_stop_between_committed_cycles(
    tmp_path: Path,
) -> None:
    session, _messages, _, _ = _run(
        tmp_path / "stopped-session",
        use_session=True,
        session_stop_after=3,
    )

    assert session.status is ScalarSensorimotorSessionStatus.STOPPED
    assert len(session.cycles) == 3
    assert session.last_committed_cut == session.cycles[-1].committed_cut
    assert session.last_committed_cut.causal_tick == 3
    assert session.error is None


def test_continuous_world_session_can_stream_without_retaining_cycle_results(
    tmp_path: Path,
) -> None:
    observed: list[CausalCycleResult] = []
    session, _messages, _, _ = _run(
        tmp_path / "streaming-continuous-session",
        use_session=True,
        session_cycles=None,
        session_stop_after=3,
        on_cycle_result=observed.append,
        retain_cycle_results=False,
    )

    assert session.status is ScalarSensorimotorSessionStatus.STOPPED
    assert session.completed_cycle_count == 3
    assert session.cycles == ()
    assert tuple(result.committed_cut.causal_tick for result in observed) == (1, 2, 3)
    assert session.last_committed_cut == observed[-1].committed_cut


def test_world_owned_session_reports_initial_cut_verification_failure_when_stopped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_cut_verification(*args, **kwargs):
        raise ManifestError("injected cut verification failure")

    monkeypatch.setattr(DurableWorldRuntime, "committed_cut_for", fail_cut_verification)
    session, _messages, _, _ = _run(
        tmp_path / "initial-cut-verification-failure",
        use_session=True,
        session_stop_after=0,
    )

    assert session.status is ScalarSensorimotorSessionStatus.STOPPED
    assert session.cycles == ()
    assert session.last_committed_cut is None
    assert session.error is not None
    assert "initial committed-cut verification failed" in session.error
    assert "injected cut verification failure" in session.error


def test_world_owned_session_reports_cycle_failure_without_claiming_uncommitted_cut(
    tmp_path: Path,
) -> None:
    session, _messages, _, _ = _run(
        tmp_path / "failed-session",
        session_cycles=3,
        use_session=True,
        environment_revision_override="environment.acceptance.stale-r0",
    )

    assert session.status is ScalarSensorimotorSessionStatus.FAILED
    assert session.cycles == ()
    assert session.error is not None
    assert session.last_committed_cut is None or session.last_committed_cut.causal_tick == 0


def test_world_owned_session_preserves_last_verified_cut_when_recovery_is_required(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import world.causal_cycle as causal_cycle

    real_cycle = causal_cycle.run_scalar_sensorimotor_cycle_and_publish
    attempted = 0

    def fail_after_pending_transition(world, *args, **kwargs):
        nonlocal attempted
        attempted += 1
        if attempted == 1:
            return real_cycle(world, *args, **kwargs)
        pending = replace(
            world.manifest,
            status=ManifestStatus.RECOVERY_REQUIRED,
            pending_transition_id="acceptance-run:commit:2",
            pending_tick=2,
            failure="injected interrupted transition",
        )
        world._manifest = pending
        raise RuntimeError("injected failure after recovery became required")

    monkeypatch.setattr(
        causal_cycle,
        "run_scalar_sensorimotor_cycle_and_publish",
        fail_after_pending_transition,
    )
    session, _messages, _, _ = _run(
        tmp_path / "preserve-last-verified-cut-on-recovery",
        use_session=True,
        session_cycles=3,
    )

    assert session.status is ScalarSensorimotorSessionStatus.FAILED
    assert session.completed_cycle_count == 1
    assert session.last_committed_cut is not None
    assert session.last_committed_cut.causal_tick == 1
    assert session.last_committed_cut.commit_id == "acceptance-run:commit:1"
    assert "injected failure after recovery became required" in (session.error or "")


def test_world_owned_session_reports_committed_cut_when_observation_delivery_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FailsOnSecondDeliveryPlane(ObservationPlane):
        def __init__(self) -> None:
            super().__init__()
            self.delivery_count = 0

        def publish(self, observation) -> int:
            self.delivery_count += 1
            if self.delivery_count == 2:
                raise RuntimeError("injected Observation delivery failure")
            return super().publish(observation)

    monkeypatch.setattr(
        sys.modules[__name__], "ObservationPlane", FailsOnSecondDeliveryPlane
    )
    session, messages, _, _ = _run(
        tmp_path / "post-commit-observation-failure",
        session_cycles=1,
        use_session=True,
    )

    assert session.status is ScalarSensorimotorSessionStatus.FAILED
    assert session.cycles == ()
    assert session.error is not None
    assert "ObservationPublicationFailure" in session.error
    assert session.last_committed_cut is not None
    assert session.last_committed_cut.causal_tick == 1
    assert session.last_committed_cut.commit_id == "acceptance-run:commit:1"
    assert session.failure_kind == "ObservationPublicationFailure"
    assert session.publication_delivery_status_uncertain is True
    # One item was acknowledged before the second delivery raised; the
    # publication as a whole is not returned as a completed cycle.
    assert len(messages) == 1


def test_world_owned_session_emits_cycle_result_only_after_full_publication(
    tmp_path: Path,
) -> None:
    observed: list[CausalCycleResult] = []
    session, _messages, _, _ = _run(
        tmp_path / "cycle-result-callback",
        session_cycles=2,
        use_session=True,
        on_cycle_result=observed.append,
    )

    assert session.status is ScalarSensorimotorSessionStatus.COMPLETED
    assert len(session.cycles) == len(observed) == 2
    assert [result.committed_cut for result in observed] == [
        result.committed_cut for result in session.cycles
    ]
    assert all(result.observation_publication.complete for result in observed)


def test_committed_world_publications_are_readable_through_local_observation_bridge(
    tmp_path: Path,
) -> None:
    reads: list[dict[str, object]] = []

    def read_after_commit(
        world: DurableWorldRuntime,
        result: CausalCycleResult,
        plane: ObservationPlane,
    ) -> None:
        committed_cut = result.committed_cut
        publication = result.observation_publication
        assert publication.committed_cut == committed_cut
        world_before = world.causal_tick
        manifest_before = WorldCausalManifestStore(
            tmp_path / "bridge-read" / "manifest.json"
        ).read()
        retained_before = plane.current_messages()
        with LocalObservationReadBridge.create(
            plane,
            host="127.0.0.1",
            port=0,
            allowed_origin="http://127.0.0.1:4173",
            max_response_bytes=65536,
        ) as bridge:
            host, port = bridge.address
            request = urllib.request.Request(
                f"http://{host}:{port}/v1/current",
                headers={"Origin": "http://127.0.0.1:4173"},
                method="GET",
            )
            try:
                with urllib.request.urlopen(request, timeout=2) as response:
                    assert response.status == 200
                    assert response.headers["Cache-Control"] == "no-store"
                    first = json.loads(response.read())
            except urllib.error.HTTPError as error:
                raise AssertionError(error.read().decode()) from error
            with urllib.request.urlopen(request, timeout=2) as response:
                second = json.loads(response.read())

        assert first == second
        assert first["schema"] == "biopeia.observation.current.v1"
        assert first["scope"] == "bounded_in_memory_current_subset"
        assert first["absence_semantics"] == "not_evidence_of_source_absence"
        assert len(first["messages"]) == 4
        assert [item["delivery_sequence"] for item in first["messages"]] == [
            message.sequence for message in retained_before
            if f"{message.observation.claim.family.authority.value}."
            f"{'.'.join(message.observation.claim.family.path)}"
            in {
                "embodiment.body_pose", "environment.finite_scalar_field",
                "embodiment.render_morphology", "environment.render_scene",
            }
        ]
        observations = [item["observation"] for item in first["messages"]]
        family_names = {
            f"{observation['claim']['family']['authority']}."
            f"{'.'.join(observation['claim']['family']['path'])}"
            for observation in observations
        }
        assert family_names == {
            "embodiment.body_pose",
            "environment.finite_scalar_field",
            "embodiment.render_morphology",
            "environment.render_scene",
        }
        dynamic_observations = [observation for observation in observations if observation["context_time"]]
        assert [item["delivery_sequence"] for item in first["messages"] if item["observation"]["context_time"]] == [
            sequence for sequence, observation in zip(
                publication.delivery_sequences,
                (message.observation for message in retained_before if message.sequence in publication.delivery_sequences),
            )
            if observation.claim.family.qualified_name in {
                "embodiment.body_pose", "environment.finite_scalar_field"
            }
        ]
        assert len(dynamic_observations) == 2
        assert all(
            any(
                item["domain"] == "world"
                and item["kind"] in {"world_causal_tick", "world_target_causal_tick"}
                and item["value"] == committed_cut.causal_tick
                for item in observation["context_time"]
            )
            for observation in dynamic_observations
        )
        body_observation = next(
            observation
            for observation in observations
            if observation["claim"]["family"]["authority"] == "embodiment"
        )
        field_observation = next(
            observation
            for observation in observations
            if observation["claim"]["family"]["authority"] == "environment"
        )
        assert body_observation["claim"]["family"]["contract"] == {
            "contract_id": "embodiment.body_pose",
            "revision": "v1",
        }
        assert field_observation["claim"]["family"]["contract"] == {
            "contract_id": "biopeia.environment.finite-scalar-field",
            "revision": "1",
        }
        body_projection = dict(
            body_observation["lineage"]["transformation"]["effective_configuration"]
        )
        field_projection = dict(
            field_observation["lineage"]["transformation"]["effective_configuration"]
        )
        assert body_projection["world_id"] == committed_cut.world_id
        assert body_projection["causal_tick"] == str(committed_cut.causal_tick)
        assert body_projection["body_id"] == BODY_ID
        assert body_projection["embodiment_episode_id"] == EPISODE_ID
        assert body_projection["morphology_revision"] == MORPHOLOGY_REVISION
        assert field_projection["world_id"] == committed_cut.world_id
        assert field_projection["physical_step_id"] == "physical-step.acceptance.sensorimotor.tick-1"
        assert any(
            reference.owner == "embodiment"
            and reference.sha256
            in {
                parent["reference"]
                for observation in observations
                if observation["claim"]["family"]["authority"] == "embodiment"
                and observation["claim"]["family"]["path"] == ["body_pose"]
                for parent in observation["lineage"]["parents"]
            }
            for reference in committed_cut.owner_checkpoint_references
        )
        assert any(
            reference.owner == "environment"
            and reference.owner_identity == RESOURCE_ID
            and reference.sha256
            in {
                parent["reference"]
                for observation in observations
                if observation["claim"]["family"]["authority"] == "environment"
                and observation["claim"]["family"]["path"] == ["finite_scalar_field"]
                for parent in observation["lineage"]["parents"]
            }
            for reference in committed_cut.owner_checkpoint_references
        )
        assert world.causal_tick == world_before == committed_cut.causal_tick == 1
        assert WorldCausalManifestStore(
            tmp_path / "bridge-read" / "manifest.json"
        ).read() == manifest_before
        assert plane.current_messages() == retained_before
        reads.append(first)

    result, messages, resumed, _ = _run(
        tmp_path / "bridge-read",
        stop_after_first=True,
        continue_from_live_state=True,
        after_first_commit=read_after_commit,
    )
    assert resumed is None
    assert result.committed_cut.causal_tick == 1
    assert len(messages) == 5
    assert len(reads) == 1


# ---------------------------------------------------------------- M3 (v5)
def _organism_ref(result: CausalCycleResult) -> CheckpointReference:
    return next(ref for ref in result.committed_cut.owner_checkpoint_references if ref.owner == "organism")


def test_m3_compact_checkpoint_session_matches_v4_cut_for_cut(tmp_path: Path) -> None:
    """Same inputs, v4 vs v5 Organism checkpoints, through the production cycle.

    Every committed cut must have identical physical/environment owner state,
    and the v5 Organism checkpoint (with its World-verified dependency) must
    reconstruct exactly the v4 bytes of the same cut.
    """
    from organism.actuation.action_history import exploration_v5_to_v4

    cycles = 12
    v4_results: list[CausalCycleResult] = []
    v5_results: list[CausalCycleResult] = []
    _run(tmp_path / "v4", use_session=True, session_cycles=cycles, on_cycle_result=v4_results.append)
    _run(tmp_path / "v5", use_session=True, session_cycles=cycles, on_cycle_result=v5_results.append,
         organism_v5=True)
    assert len(v4_results) == len(v5_results) == cycles
    resolve = _committed_resolver(tmp_path / "v5")
    with_dependency = 0
    for r4, r5 in zip(v4_results, v5_results):
        assert r4.committed_cut.causal_tick == r5.committed_cut.causal_tick
        others4 = {(ref.owner, ref.owner_identity): ref.sha256 for ref in r4.committed_cut.owner_checkpoint_references
                   if ref.owner not in ("organism", "world-journal")}
        others5 = {(ref.owner, ref.owner_identity): ref.sha256 for ref in r5.committed_cut.owner_checkpoint_references
                   if ref.owner not in ("organism", "world-journal")}
        assert others4 == others5                                   # same physics and environment
        o4, o5 = _organism_ref(r4), _organism_ref(r5)
        assert o4.schema_version == ORGANISM_EXPLORATION_CHECKPOINT_SCHEMA
        assert o5.schema_version == ORGANISM_EXPLORATION_CHECKPOINT_V5_SCHEMA
        assert exploration_v5_to_v4(Path(o5.locator).read_bytes(), resolve=resolve) == Path(o4.locator).read_bytes()
        with_dependency += o5.dependency_sha256 is not None
    assert with_dependency > 0                                      # the dependency path was exercised


def test_m3_compact_checkpoint_restores_and_continues_in_fresh_process(tmp_path: Path) -> None:
    uninterrupted, _, uninterrupted_next, uninterrupted_messages = _run(
        tmp_path / "uninterrupted", continue_from_live_state=True, organism_v5=True
    )
    persisted, _, no_resume, _ = _run(tmp_path / "fresh-process", stop_after_first=True, organism_v5=True)
    assert persisted.committed_cut.causal_tick == 1 and no_resume is None
    assert _organism_ref(persisted).schema_version == ORGANISM_EXPLORATION_CHECKPOINT_V5_SCHEMA

    output_path = tmp_path / "fresh-process-result.json"
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--resume-v5", str(tmp_path / "fresh-process"),
         str(output_path)],
        check=False, capture_output=True, text=True, timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    fresh = json.loads(output_path.read_text())
    expected = json.loads(json.dumps(_summary(uninterrupted_next, uninterrupted_messages), sort_keys=True))
    assert fresh["process_id"] != os.getpid()
    assert fresh["cycle"] == expected


def test_m3_restore_refuses_missing_organism_history(tmp_path: Path) -> None:
    cycles: list[CausalCycleResult] = []
    _run(tmp_path / "v5", use_session=True, session_cycles=4, on_cycle_result=cycles.append, organism_v5=True)
    head = _organism_ref(cycles[-1]).dependency_sha256
    assert head is not None
    (tmp_path / "v5" / "owners" / "committed" / head[:2] / f"{head}.blob").unlink()
    # Refused by World's dependency check (R1), before any Organism restore.
    with pytest.raises(ManifestError, match="dependency closure is not verified"):
        _restore_runtime_from_directory(tmp_path / "v5", organism_v5=True)
