"""Cross-repository World commit of Environment-generated sphere geometry.

This fixture exercises one deterministic Environment genesis, explicit local
identity-frame projection into MuJoCo, Environment owner adoption, durable
World publication, owner restore, and one continued physical step. It is a
contract integration test, not evidence of ecological quality, calibration,
physical material properties, or production crash durability.
"""

from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import mujoco
import pytest
from embodiment.observation import BodyPoseSample
from embodiment.physics3d.articulated_spec import ArticulatedBodySpec
from embodiment.physics3d.articulated_state import (
    ActiveArticulatedMorphology,
    ArticulatedEmbodimentPhysicalState,
)
from embodiment.physics3d.mechanics import SegmentSpec
from embodiment.physics3d.physical_projection import project_body_physical_input
from embodiment.physics3d.physical_transition import (
    EmbodimentPhysicalState,
    PhysicalStepCorrelation,
    SolverBodyCandidate,
)
from environment import (
    GENERATOR_ID,
    GENERATOR_REVISION,
    EnvironmentGenesisConfig,
    EnvironmentGenesisSpec,
    FrameDescriptor,
    StaticEnvironmentOwner,
    StaticGroundPlane,
    StaticPhysicalScene,
    StaticSceneCandidate,
    generate_procedural_environment,
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
from world.runtime import PresenceId, WorldId

WORLD_ID = "world.procedural-sphere-e2e"
PRESENCE_ID = "presence.procedural-sphere-e2e"
BODY_ID = "body.procedural-sphere-e2e"
EPISODE_ID = "episode.procedural-sphere-e2e"
MORPHOLOGY_REVISION = "morphology.procedural-sphere-e2e-r1"
ENVIRONMENT_ID = "environment.procedural-sphere-e2e"
ENVIRONMENT_REVISION = "environment.procedural-sphere-e2e-r1"
FRAME_ID = "physics3d.world"
ENVIRONMENT_OWNER_ID = ENVIRONMENT_ID
WORLD_FRAME = FrameDescriptor(
    frame_id=FRAME_ID,
    parent_frame_id="world",
    origin_m=(0.0, 0.0, 0.0),
    x_axis_in_parent=(1.0, 0.0, 0.0),
    y_axis_in_parent=(0.0, 1.0, 0.0),
    z_axis_in_parent=(0.0, 0.0, 1.0),
    handedness="RIGHT_HANDED",
    up_axis="Z",
)
SOLVER_CONFIGURATION_REVISION = (
    f"mujoco-{mujoco.__version__}-world-sphere-euler-dt0.01-newton-50-"
    "tol1e-10-condim1-friction0-solref0.02x1-solimp0.9x0.95x0.001-r1"
)


@dataclass(frozen=True)
class _FixtureOwnerContract:
    owner_identity: str
    schema_version: str
    validate: Callable[[bytes], bool]

    def validate_checkpoint(self, encoded: bytes, owner_identity: str, schema_version: str) -> bool:
        return (
            owner_identity == self.owner_identity
            and schema_version == self.schema_version
            and self.validate(encoded)
        )

    def verify_restoration(self, encoded: bytes, owner_identity: str, schema_version: str) -> bool:
        return self.validate_checkpoint(encoded, owner_identity, schema_version)


@dataclass
class _Fixture:
    runtime: DurableWorldRuntime
    environment: StaticEnvironmentOwner
    body: EmbodimentPhysicalState
    contracts: dict[tuple[str, str], _FixtureOwnerContract]
    manifest_store: WorldCausalManifestStore
    journal_store: DurableJournalSnapshotStore
    owner_directory: Path


def _generated_scene() -> StaticPhysicalScene:
    """Build a deterministic scene with a controlled sphere/body contact case."""
    config = EnvironmentGenesisConfig(
        environment_id=ENVIRONMENT_ID,
        environment_revision=ENVIRONMENT_REVISION,
        frame=WORLD_FRAME,
        gravity_m_s2=(0.0, 0.0, -9.81),
        ground_plane=StaticGroundPlane("ground.z0", (0.0, 0.0, 1.0), 0.0),
        # Small but non-degenerate bounds make the generated geometry stable
        # and nearby while leaving its coordinates genuinely generator-owned.
        bounds_min_m=(-0.05, -0.05, 0.30),
        bounds_max_m=(0.05, 0.05, 0.31),
        sphere_count=2,
        minimum_radius_m=0.20,
        maximum_radius_m=0.20,
    )
    spec = EnvironmentGenesisSpec(
        environment_seed=82731,
        generator_id=GENERATOR_ID,
        generator_revision=GENERATOR_REVISION,
        configuration=config,
    )
    generated = generate_procedural_environment(spec).scene
    repeated = generate_procedural_environment(spec).scene
    assert generated == repeated
    assert generated.genesis is not None
    assert generated.genesis.environment_seed == 82731
    assert generated.genesis.generator_id == GENERATOR_ID
    assert generated.genesis.generator_revision == GENERATOR_REVISION
    assert generated.genesis.configuration["sphere_count"] == 2
    assert tuple(sphere.object_id for sphere in generated.spheres) == (
        "sphere-000000",
        "sphere-000001",
    )
    return generated


def _body_spec() -> ArticulatedBodySpec:
    return ArticulatedBodySpec(
        body_kind="procedural-sphere-contact-fixture",
        state_schema_version=1,
        spawn_height=0.0,
        base_link_name="body.root",
        joint_specs=(),
        joint_topology=(),
        segments={
            "body.root": SegmentSpec(
                1.0,
                (0.2, 0.2, 0.2),
                (0.0, 0.0, 0.0),
                (0.5, 0.5, 0.5, 1.0),
            )
        },
        contact_link_names=("body.root",),
        structural_neighbour_names=(),
    )


def _initial_body(scene: StaticPhysicalScene) -> EmbodimentPhysicalState:
    sphere = scene.spheres[0]
    # The box bottom starts with 1 cm overlap of the generated sphere. The
    # solver therefore exercises an actual contact rather than merely placing
    # sphere geometry in the model.
    pose = BodyPoseSample(
        body_id=BODY_ID,
        reference_frame=FRAME_ID,
        body_frame="body.root",
        position_m=(sphere.center_m[0], sphere.center_m[1], sphere.center_m[2] + sphere.radius_m + 0.09),
        orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
        sample_kind="genesis",
        sample_value=0,
    )
    return EmbodimentPhysicalState(
        world_id=WORLD_ID,
        causal_tick=0,
        body_id=BODY_ID,
        embodiment_episode_id=EPISODE_ID,
        morphology_revision=MORPHOLOGY_REVISION,
        pose=pose,
        linear_velocity_m_per_s=(0.0, 0.0, 0.0),
        angular_velocity_rad_per_s=(0.0, 0.0, 0.0),
    )


def _projectable_body_state(
    root: EmbodimentPhysicalState, spec: ArticulatedBodySpec
) -> ArticulatedEmbodimentPhysicalState:
    """Bind the fixture's zero-joint root to its active morphology."""
    morphology = ActiveArticulatedMorphology.from_spec(
        body_id=root.body_id,
        embodiment_episode_id=root.embodiment_episode_id,
        morphology_revision=root.morphology_revision,
        spec=spec,
    )
    return ArticulatedEmbodimentPhysicalState(root=root, morphology=morphology, joints=())


def _checkpoint_reference(
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


def _fixture(root: Path) -> _Fixture:
    scene = _generated_scene()
    environment = StaticEnvironmentOwner(
        scene, owner_id=ENVIRONMENT_OWNER_ID, world_id=WORLD_ID
    )
    body = _initial_body(environment.scene)
    owner_directory = root / "owner-checkpoints"
    initial_references = (
        _checkpoint_reference(
            owner_directory,
            owner="embodiment",
            owner_identity=BODY_ID,
            checkpoint_id="tick-0",
            schema="embodiment-physical-state-v2",
            encoded=body.checkpoint_bytes(),
        ),
        _checkpoint_reference(
            owner_directory,
            owner="environment",
            owner_identity=ENVIRONMENT_OWNER_ID,
            checkpoint_id="tick-0",
            schema=StaticEnvironmentOwner.CHECKPOINT_SCHEMA,
            encoded=environment.checkpoint_bytes(),
        ),
    )
    contracts = {
        ("embodiment", BODY_ID): _FixtureOwnerContract(
            BODY_ID,
            "embodiment-physical-state-v2",
            lambda encoded: EmbodimentPhysicalState.restore_checkpoint(encoded).body_id == BODY_ID,
        ),
        ("environment", ENVIRONMENT_OWNER_ID): _FixtureOwnerContract(
            ENVIRONMENT_OWNER_ID,
            StaticEnvironmentOwner.CHECKPOINT_SCHEMA,
            lambda encoded: StaticEnvironmentOwner.validate_checkpoint(
                encoded,
                ENVIRONMENT_OWNER_ID,
                StaticEnvironmentOwner.CHECKPOINT_SCHEMA,
            ),
        ),
    }
    world_id = WorldId(WORLD_ID)
    manifest_store = WorldCausalManifestStore(root / "world-manifest.json")
    journal_store = DurableJournalSnapshotStore(root / "world-journal", world_id)
    runtime = DurableWorldRuntime.initialize(
        WorldGenesisSpec(
            world_id=world_id,
            initial_presence_ids=(PRESENCE_ID,),
            registered_owner_ids=tuple(sorted(contracts)),
            initial_owner_references=initial_references,
            owner_checkpoint_contracts=contracts,
            presence_owner_links=(
                PresenceOwnerLink(PresenceId(PRESENCE_ID), "embodiment", BODY_ID),
            ),
        ),
        PRESENCE_ID,
        manifest_store,
        journal_store,
        owner_checkpoint_verifier=lambda references: all(
            contracts[(reference.owner, reference.owner_identity)].validate_checkpoint(
                Path(reference.locator).read_bytes(),
                reference.owner_identity,
                reference.schema_version,
            )
            for reference in references
            if reference.owner != "world-journal"
        ),
    )
    return _Fixture(
        runtime,
        environment,
        body,
        contracts,
        manifest_store,
        journal_store,
        owner_directory,
    )


def _require_identity_world_frame(scene: StaticPhysicalScene) -> None:
    frame = scene.frame
    if (
        frame.frame_id != FRAME_ID
        or frame.parent_frame_id != "world"
        or frame.origin_m != (0.0, 0.0, 0.0)
        or frame.x_axis_in_parent != (1.0, 0.0, 0.0)
        or frame.y_axis_in_parent != (0.0, 1.0, 0.0)
        or frame.z_axis_in_parent != (0.0, 0.0, 1.0)
        or frame.handedness != "RIGHT_HANDED"
        or frame.up_axis != "Z"
    ):
        raise ValueError("fixture solver supports only the declared identity physics3d.world frame")
    if any(sphere.frame_id != frame.frame_id for sphere in scene.spheres):
        raise ValueError("fixture solver rejects sphere geometry outside the declared scene frame")


def _sphere_geom_name(object_id: str) -> str:
    digest = hashlib.sha256(object_id.encode("utf-8")).hexdigest()
    return f"environment.sphere.{digest[:24]}"


def _execute_mujoco_step(body_input, scene: StaticPhysicalScene):
    """Project Environment-owned plane/spheres without implicit transforms."""
    _require_identity_world_frame(scene)
    if (
        body_input.body_kind != "procedural-sphere-contact-fixture"
        or body_input.reference_frame != FRAME_ID
    ):
        raise ValueError("fixture solver requires the declared free-root body projection")
    if len(scene.objects) != 1 or scene.objects[0].normal != (0.0, 0.0, 1.0):
        raise ValueError("fixture solver requires exactly one +Z ground plane")
    spheres = tuple(sorted(scene.spheres, key=lambda sphere: sphere.object_id))
    names = tuple((sphere.object_id, _sphere_geom_name(sphere.object_id)) for sphere in spheres)
    if len({name for _, name in names}) != len(names):
        raise ValueError("sphere object identities do not map to unique solver geometry names")

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
    gravity = " ".join(repr(value) for value in scene.gravity_m_s2)
    sphere_geoms = "\n".join(
        f'<geom name="{name}" type="sphere" pos="{" ".join(repr(value) for value in sphere.center_m)}" '
        f'size="{repr(sphere.radius_m)}" contype="1" conaffinity="1" condim="1" '
        'friction="0 0 0" solref="0.02 1" solimp="0.9 0.95 0.001" margin="0" gap="0"/>'
        for sphere, (_object_id, name) in zip(spheres, names, strict=True)
    )
    xml = f"""
    <mujoco model="biopeia-procedural-sphere-world-fixture">
      <compiler angle="radian" coordinate="local"/>
      <option timestep="0.01" integrator="Euler" solver="Newton" iterations="50"
              tolerance="1e-10" cone="pyramidal" gravity="{gravity}"/>
      <worldbody>
        <geom name="environment.ground.{hashlib.sha256(plane.object_id.encode('utf-8')).hexdigest()[:24]}"
              type="plane" pos="0 0 {repr(plane.offset_m)}" size="10 10 0.1"
              contype="1" conaffinity="1" condim="1" friction="0 0 0"
              solref="0.02 1" solimp="0.9 0.95 0.001" margin="0" gap="0"/>
        {sphere_geoms}
        <body name="fixture-body" pos="{' '.join(repr(value) for value in body_input.position_m)}">
          <freejoint name="root.free"/>
          <geom name="body.root.collider" type="box" size="0.1 0.1 0.1" mass="1"
                contype="1" conaffinity="1" condim="1" friction="0 0 0"
                solref="0.02 1" solimp="0.9 0.95 0.001" margin="0" gap="0"/>
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
    mujoco.mj_step(model, data)

    geom_id_by_object_id = {
        object_id: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        for object_id, geom_name in names
    }
    sphere_geom_ids = set(geom_id_by_object_id.values())
    contacted_ids = {
        int(geom_id)
        for contact_index in range(data.ncon)
        for geom_id in data.contact[contact_index].geom
        if int(geom_id) in sphere_geom_ids
    }
    contacted_object_ids = tuple(
        sorted(object_id for object_id, geom_id in geom_id_by_object_id.items() if geom_id in contacted_ids)
    )
    qpos = tuple(float(value) for value in data.qpos[qpos_address : qpos_address + 7])
    position = qpos[:3]
    quaternion_wxyz = qpos[3:]
    linear_velocity = tuple(float(value) for value in data.qvel[qvel_address : qvel_address + 3])
    angular_velocity = tuple(float(value) for value in data.qvel[qvel_address + 3 : qvel_address + 6])
    return (
        position,
        (quaternion_wxyz[1], quaternion_wxyz[2], quaternion_wxyz[3], quaternion_wxyz[0]),
        linear_velocity,
        angular_velocity,
        contacted_object_ids,
        names,
    )


def _world_verifier(contracts: dict[tuple[str, str], _FixtureOwnerContract]):
    def verify(references: tuple[CheckpointReference, ...]) -> bool:
        for reference in references:
            if reference.owner == "world-journal":
                continue
            contract = contracts.get((reference.owner, reference.owner_identity))
            try:
                encoded = Path(reference.locator).read_bytes()
            except OSError:
                return False
            if contract is None or not contract.validate_checkpoint(
                encoded, reference.owner_identity, reference.schema_version
            ):
                return False
        return True

    return verify


def _advance_and_commit(fixture: _Fixture, *, world_tick: int):
    staged: dict[str, object] = {}

    def persist(causal_tick: int) -> tuple[CheckpointReference, ...]:
        assert causal_tick == world_tick
        correlation = PhysicalStepCorrelation(
            world_id=WORLD_ID,
            causal_tick=causal_tick - 1,
            physical_step_id=f"physical-step-{causal_tick}",
            body_id=BODY_ID,
            embodiment_episode_id=EPISODE_ID,
            morphology_revision=MORPHOLOGY_REVISION,
            environment_revision=ENVIRONMENT_REVISION,
            solver_config_revision=SOLVER_CONFIGURATION_REVISION,
        )
        assert correlation.causal_tick == fixture.body.causal_tick
        body_input = project_body_physical_input(
            _projectable_body_state(fixture.body, _body_spec()),
            correlation=correlation,
        )
        (
            position_m,
            orientation_xyzw,
            linear_velocity_m_per_s,
            angular_velocity_rad_per_s,
            contacted_sphere_ids,
            geom_names,
        ) = _execute_mujoco_step(body_input, fixture.environment.scene)
        body_candidate = SolverBodyCandidate(
            correlation=correlation,
            reference_frame=body_input.reference_frame,
            body_frame=body_input.body_frame,
            position_m=position_m,
            orientation_xyzw=orientation_xyzw,
            linear_velocity_m_per_s=linear_velocity_m_per_s,
            angular_velocity_rad_per_s=angular_velocity_rad_per_s,
        )
        next_body = fixture.body.adopt_candidate(body_candidate, expected_step=correlation)
        environment_correlation = EnvironmentStepCorrelation(
            world_id=correlation.world_id,
            causal_tick=correlation.causal_tick,
            physical_step_id=correlation.physical_step_id,
            environment_id=fixture.environment.scene.environment_id,
            environment_revision=fixture.environment.scene.environment_revision,
        )
        fixture.environment.validate_and_adopt(
            StaticSceneCandidate(environment_correlation, project_static_scene(fixture.environment.scene)),
            expected_correlation=environment_correlation,
        )
        assert next_body.causal_tick == causal_tick
        assert fixture.environment.last_adopted_tick == causal_tick - 1
        staged.update(
            correlation=correlation,
            body=next_body,
            contacted_sphere_ids=contacted_sphere_ids,
            geom_names=geom_names,
        )
        return (
            _checkpoint_reference(
                fixture.owner_directory,
                owner="embodiment",
                owner_identity=BODY_ID,
                checkpoint_id=f"tick-{causal_tick}",
                schema="embodiment-physical-state-v2",
                encoded=next_body.checkpoint_bytes(),
            ),
            _checkpoint_reference(
                fixture.owner_directory,
                owner="environment",
                owner_identity=ENVIRONMENT_OWNER_ID,
                checkpoint_id=f"tick-{causal_tick}",
                schema=StaticEnvironmentOwner.CHECKPOINT_SCHEMA,
                encoded=fixture.environment.checkpoint_bytes(),
            ),
        )

    record = fixture.runtime.tick(f"physical-sphere-commit-{world_tick}", persist)
    assert fixture.runtime.status is ManifestStatus.COMMITTED
    assert fixture.runtime.causal_tick == world_tick == record.causal_tick
    return staged


def _restore_committed_owners(fixture: _Fixture) -> tuple[EmbodimentPhysicalState, StaticEnvironmentOwner]:
    restored: dict[str, object] = {}

    def restore_owners(references: tuple[CheckpointReference, ...]) -> None:
        for reference in references:
            encoded = Path(reference.locator).read_bytes()
            if reference.owner == "embodiment":
                restored["body"] = EmbodimentPhysicalState.restore_checkpoint(encoded)
            elif reference.owner == "environment":
                restored["environment"] = StaticEnvironmentOwner.from_checkpoint_bytes(encoded)
            else:
                raise AssertionError(f"unexpected semantic owner {reference.owner}")

    runtime = DurableWorldRuntime.restore(
        WorldId(WORLD_ID),
        PRESENCE_ID,
        fixture.manifest_store,
        fixture.journal_store,
        owner_checkpoint_verifier=_world_verifier(fixture.contracts),
        owner_checkpoint_contracts=fixture.contracts,
        owner_checkpoint_restorer=restore_owners,
    )
    assert runtime.status is ManifestStatus.COMMITTED
    assert runtime.causal_tick == fixture.runtime.causal_tick
    fixture.runtime = runtime
    body = restored.get("body")
    environment = restored.get("environment")
    assert isinstance(body, EmbodimentPhysicalState)
    assert isinstance(environment, StaticEnvironmentOwner)
    return body, environment


def test_procedural_spheres_are_adopted_committed_restored_and_reused(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    original_scene = fixture.environment.scene
    first = _advance_and_commit(fixture, world_tick=1)

    assert fixture.runtime.manifest.status is ManifestStatus.COMMITTED
    assert fixture.runtime.manifest.causal_tick == 1
    assert first["contacted_sphere_ids"]
    assert original_scene.spheres[0].object_id in first["contacted_sphere_ids"]
    assert set(first["contacted_sphere_ids"]) <= {
        sphere.object_id for sphere in original_scene.spheres
    }
    first_pose = first["body"].pose.position_m
    assert first_pose != fixture.body.pose.position_m
    correlation = PhysicalStepCorrelation(
        world_id=WORLD_ID,
        causal_tick=0,
        physical_step_id="physical-step-contact-control",
        body_id=BODY_ID,
        embodiment_episode_id=EPISODE_ID,
        morphology_revision=MORPHOLOGY_REVISION,
        environment_revision=ENVIRONMENT_REVISION,
        solver_config_revision=SOLVER_CONFIGURATION_REVISION,
    )
    initial_body = _initial_body(original_scene)
    body_input = project_body_physical_input(
        _projectable_body_state(initial_body, _body_spec()),
        correlation=correlation,
    )
    with_spheres = _execute_mujoco_step(body_input, original_scene)
    without_spheres = _execute_mujoco_step(body_input, replace(original_scene, spheres=()))
    assert (with_spheres[0], with_spheres[2]) != (without_spheres[0], without_spheres[2])

    environment_reference = next(
        reference for reference in fixture.runtime.manifest.checkpoint_references
        if reference.owner == "environment"
    )
    persisted_environment = StaticEnvironmentOwner.from_checkpoint_bytes(
        Path(environment_reference.locator).read_bytes()
    )
    assert persisted_environment.scene == original_scene
    assert persisted_environment.scene.genesis == original_scene.genesis
    assert tuple(sphere.object_id for sphere in persisted_environment.scene.spheres) == tuple(
        sphere.object_id for sphere in original_scene.spheres
    )

    body, restored_environment = _restore_committed_owners(fixture)
    assert body == first["body"]
    assert restored_environment.scene == original_scene
    assert restored_environment.last_adopted_tick == 0
    fixture.body = body
    fixture.environment = restored_environment

    second = _advance_and_commit(fixture, world_tick=2)
    assert fixture.runtime.manifest.status is ManifestStatus.COMMITTED
    assert fixture.runtime.manifest.causal_tick == 2
    assert second["body"].causal_tick == 2
    assert second["geom_names"] == first["geom_names"]
    assert second["contacted_sphere_ids"]
    assert restored_environment.scene == original_scene


def test_sphere_projection_is_stable_under_collection_permutation() -> None:
    scene = _generated_scene()
    permuted = replace(scene, spheres=tuple(reversed(scene.spheres)))
    body = _initial_body(scene)
    correlation = PhysicalStepCorrelation(
        world_id=WORLD_ID,
        causal_tick=0,
        physical_step_id="physical-step-permutation",
        body_id=BODY_ID,
        embodiment_episode_id=EPISODE_ID,
        morphology_revision=MORPHOLOGY_REVISION,
        environment_revision=ENVIRONMENT_REVISION,
        solver_config_revision=SOLVER_CONFIGURATION_REVISION,
    )
    body_input = project_body_physical_input(
        _projectable_body_state(body, _body_spec()),
        correlation=correlation,
    )
    projected = _execute_mujoco_step(body_input, scene)
    permuted_projected = _execute_mujoco_step(body_input, permuted)
    assert projected == permuted_projected


def test_solver_rejects_non_identity_environment_frame_without_transforming() -> None:
    scene = _generated_scene()
    unsupported = replace(
        scene,
        frame=replace(scene.frame, origin_m=(1.0, 0.0, 0.0)),
    )
    body = _initial_body(scene)
    correlation = PhysicalStepCorrelation(
        world_id=WORLD_ID,
        causal_tick=0,
        physical_step_id="physical-step-unsupported-frame",
        body_id=BODY_ID,
        embodiment_episode_id=EPISODE_ID,
        morphology_revision=MORPHOLOGY_REVISION,
        environment_revision=ENVIRONMENT_REVISION,
        solver_config_revision=SOLVER_CONFIGURATION_REVISION,
    )
    body_input = project_body_physical_input(
        _projectable_body_state(body, _body_spec()),
        correlation=correlation,
    )
    with pytest.raises(ValueError, match="identity physics3d.world frame"):
        _execute_mujoco_step(body_input, unsupported)


def test_changed_environment_geometry_rejection_cannot_commit_next_cut(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    original_scene = fixture.environment.scene
    changed_spheres = (
        replace(original_scene.spheres[0], radius_m=original_scene.spheres[0].radius_m + 0.05),
        *original_scene.spheres[1:],
    )
    changed_scene = replace(original_scene, spheres=changed_spheres)
    before = fixture.body

    def reject_changed_geometry(causal_tick: int) -> tuple[CheckpointReference, ...]:
        correlation = EnvironmentStepCorrelation(
            world_id=WORLD_ID,
            causal_tick=causal_tick - 1,
            physical_step_id="physical-step-changed-environment",
            environment_id=ENVIRONMENT_ID,
            environment_revision=ENVIRONMENT_REVISION,
        )
        fixture.environment.validate_and_adopt(
            StaticSceneCandidate(correlation, project_static_scene(changed_scene)),
            expected_correlation=correlation,
        )
        raise AssertionError("changed Environment geometry must be rejected before writing references")

    with pytest.raises(ValueError, match="rejects changed scene candidates"):
        fixture.runtime.tick("physical-sphere-invalid-geometry", reject_changed_geometry)

    manifest = fixture.manifest_store.read()
    assert manifest.status is ManifestStatus.RECOVERY_REQUIRED
    assert manifest.causal_tick == 0
    assert manifest.pending_tick == 1
    assert fixture.body == before
    assert fixture.environment.scene == original_scene
    assert fixture.environment.last_adopted_tick == -1


def test_stale_morphology_candidate_cannot_commit_sphere_step(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    expected = PhysicalStepCorrelation(
        world_id=WORLD_ID,
        causal_tick=0,
        physical_step_id="physical-step-stale-morphology",
        body_id=BODY_ID,
        embodiment_episode_id=EPISODE_ID,
        morphology_revision=MORPHOLOGY_REVISION,
        environment_revision=ENVIRONMENT_REVISION,
        solver_config_revision=SOLVER_CONFIGURATION_REVISION,
    )
    stale = replace(expected, morphology_revision="morphology.previous")
    body_before = fixture.body
    scene_before = fixture.environment.scene
    adopted: list[EmbodimentPhysicalState] = []

    def reject_stale_candidate(causal_tick: int) -> tuple[CheckpointReference, ...]:
        assert causal_tick == 1
        candidate = SolverBodyCandidate(
            correlation=stale,
            reference_frame=FRAME_ID,
            body_frame="body.root",
            position_m=(0.0, 0.0, 0.5),
            orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
            linear_velocity_m_per_s=(0.0, 0.0, 0.0),
            angular_velocity_rad_per_s=(0.0, 0.0, 0.0),
        )
        adopted.append(fixture.body.adopt_candidate(candidate, expected_step=expected))
        raise AssertionError("stale morphology candidate must be rejected before checkpoint writes")

    with pytest.raises(ValueError, match="authorized physical step"):
        fixture.runtime.tick("physical-sphere-stale-morphology", reject_stale_candidate)

    manifest = fixture.manifest_store.read()
    assert manifest.status is ManifestStatus.RECOVERY_REQUIRED
    assert manifest.causal_tick == 0
    assert manifest.pending_tick == 1
    assert fixture.body == body_before
    assert fixture.environment.scene == scene_before
    assert fixture.environment.last_adopted_tick == -1
    assert adopted == []


def _committed_cut_observations(fixture: _Fixture) -> tuple[tuple[object, ...], tuple[SemanticObservation[object], ...]]:
    """Build fixture-local semantic projections from an already committed cut.

    This intentionally consumes only approved in-memory Observation concepts.
    The family names/contracts below are integration-fixture identifiers, not
    production semantic-family approval, a record schema, or a transport API.
    """
    manifest_before = fixture.runtime.manifest
    assert manifest_before.status is ManifestStatus.COMMITTED
    assert manifest_before.causal_tick == 1
    references = manifest_before.checkpoint_references
    live_body_before = fixture.body
    live_environment_before = fixture.environment.scene
    environment_adoption_tick_before = fixture.environment.last_adopted_tick
    owner_bytes_before = tuple(
        (reference.owner, hashlib.sha256(Path(reference.locator).read_bytes()).hexdigest())
        for reference in references
        if reference.owner != "world-journal"
    )

    restored: dict[str, object] = {}
    for reference in references:
        if reference.owner == "embodiment":
            restored[reference.owner] = EmbodimentPhysicalState.restore_checkpoint(
                Path(reference.locator).read_bytes()
            )
        elif reference.owner == "environment":
            restored[reference.owner] = StaticEnvironmentOwner.from_checkpoint_bytes(
                Path(reference.locator).read_bytes()
            )
    body = restored.get("embodiment")
    environment = restored.get("environment")
    assert isinstance(body, EmbodimentPhysicalState)
    assert isinstance(environment, StaticEnvironmentOwner)

    embodiment = ActorRef("domain", "embodiment")
    environment_actor = ActorRef("domain", "environment")
    world_actor = ActorRef("domain", "world")
    adapter = ActorRef("component", "fixture-observation-projection")
    body_identity = IdentityAssertion(
        IdentityRef("body_id", BODY_ID, embodiment, scope=EPISODE_ID),
        ActorRef("boundary", "embodiment-owner"),
    )
    environment_identity = IdentityAssertion(
        IdentityRef("environment_id", ENVIRONMENT_ID, environment_actor),
        ActorRef("boundary", "environment-owner"),
    )
    world_identity = IdentityAssertion(
        IdentityRef("world_id", WORLD_ID, world_actor),
        ActorRef("boundary", "world-owner"),
    )

    body_observation = SemanticObservation(
        claim=SemanticClaim(
            family=SemanticFamily(
                AuthorityDomain.EMBODIMENT,
                ("body_pose",),
                SemanticContractRef("integration_fixture_body_pose", "v1"),
            ),
            subject=SubjectScope("body", identities=(body_identity,)),
            epistemic_kind="authoritative_state",
            value={
                "position_m": body.pose.position_m,
                "orientation_xyzw": body.pose.orientation_xyzw,
                "linear_velocity_m_per_s": body.linear_velocity_m_per_s,
                "angular_velocity_rad_per_s": body.angular_velocity_rad_per_s,
                "reference_frame": body.pose.reference_frame,
                "body_frame": body.pose.body_frame,
            },
        ),
        acquisition=AcquisitionBasis("authoritative_state"),
        asserter=ActorRef("boundary", "embodiment-owner"),
        producer=adapter,
        domain_time=(
            DomainTimeRef(
                AuthorityDomain.EMBODIMENT,
                "morphology_revision",
                body.morphology_revision,
            ),
        ),
        context_time=(
            DomainTimeRef(
                AuthorityDomain.WORLD, "causal_tick", manifest_before.causal_tick
            ),
        ),
    )
    scene = environment.scene
    environment_observation = SemanticObservation(
        claim=SemanticClaim(
            family=SemanticFamily(
                AuthorityDomain.ENVIRONMENT,
                ("static_scene",),
                SemanticContractRef("integration_fixture_static_scene", "v1"),
            ),
            subject=SubjectScope("environment", identities=(environment_identity,)),
            epistemic_kind="authoritative_state",
            value={
                "environment_revision": scene.environment_revision,
                "frame_id": scene.frame.frame_id,
                "gravity_m_s2": scene.gravity_m_s2,
                "sphere_geometry": tuple(
                    (sphere.object_id, sphere.center_m, sphere.radius_m, sphere.frame_id)
                    for sphere in sorted(scene.spheres, key=lambda item: item.object_id)
                ),
                "genesis_seed": scene.genesis.environment_seed if scene.genesis else None,
                "generator_id": scene.genesis.generator_id if scene.genesis else None,
                "generator_revision": scene.genesis.generator_revision
                if scene.genesis
                else None,
            },
        ),
        acquisition=AcquisitionBasis("authoritative_state"),
        asserter=ActorRef("boundary", "environment-owner"),
        producer=adapter,
        domain_time=(
            DomainTimeRef(
                AuthorityDomain.ENVIRONMENT,
                "environment_revision",
                scene.environment_revision,
            ),
        ),
        context_time=(
            DomainTimeRef(
                AuthorityDomain.WORLD, "causal_tick", manifest_before.causal_tick
            ),
        ),
    )
    world_observation = SemanticObservation(
        claim=SemanticClaim(
            family=SemanticFamily(
                AuthorityDomain.WORLD,
                ("causal_cut",),
                SemanticContractRef("integration_fixture_causal_cut", "v1"),
            ),
            subject=SubjectScope("world", identities=(world_identity,)),
            epistemic_kind="authoritative_state",
            value={
                "commit_id": manifest_before.commit_id,
                "causal_tick": manifest_before.causal_tick,
                "journal_prefix_sha256": manifest_before.journal_prefix_sha256,
                "owner_checkpoint_sha256": tuple(
                    (reference.owner, reference.owner_identity, reference.sha256)
                    for reference in references
                    if reference.owner != "world-journal"
                ),
            },
        ),
        acquisition=AcquisitionBasis("authoritative_state"),
        asserter=ActorRef("boundary", "world-owner"),
        producer=adapter,
        domain_time=(
            DomainTimeRef(
                AuthorityDomain.WORLD, "causal_tick", manifest_before.causal_tick
            ),
        ),
    )

    # The projection reads adopted owner checkpoints after publication and
    # cannot write back to either owners or World.
    assert fixture.runtime.manifest is manifest_before
    assert fixture.body is live_body_before
    assert fixture.environment.scene is live_environment_before
    assert fixture.environment.last_adopted_tick == environment_adoption_tick_before
    assert tuple(
        (reference.owner, hashlib.sha256(Path(reference.locator).read_bytes()).hexdigest())
        for reference in references
        if reference.owner != "world-journal"
    ) == owner_bytes_before
    normalized_cut = (
        manifest_before.world_id,
        manifest_before.causal_tick,
        manifest_before.commit_id,
        manifest_before.status,
        manifest_before.presence_ids,
        manifest_before.registered_owner_ids,
        manifest_before.journal_prefix_sha256,
        tuple(
            (
                ref.owner,
                ref.owner_identity,
                ref.checkpoint_id,
                ref.schema_version,
                ref.sha256,
            )
            for ref in references
        ),
    )
    return normalized_cut, (
        world_observation,
        body_observation,
        environment_observation,
    )


def test_identical_runs_have_equal_adopted_cut_and_passive_semantic_claims(
    tmp_path: Path,
) -> None:
    runs: list[
        tuple[
            tuple[object, ...],
            tuple[SemanticObservation[object], ...],
            tuple[object, ...],
        ]
    ] = []
    for name in ("run-a", "run-b"):
        fixture = _fixture(tmp_path / name)
        _advance_and_commit(fixture, world_tick=1)
        assert fixture.runtime.manifest.status is ManifestStatus.COMMITTED

        before = fixture.runtime.manifest
        normalized_cut, observations = _committed_cut_observations(fixture)
        after = fixture.runtime.manifest
        assert after is before
        owner_hashes = tuple(
            (reference.owner, reference.sha256)
            for reference in after.checkpoint_references
            if reference.owner != "world-journal"
        )
        runs.append((normalized_cut, observations, owner_hashes))

    assert runs[0] == runs[1]
