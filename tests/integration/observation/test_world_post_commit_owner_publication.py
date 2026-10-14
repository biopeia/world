"""World publishes an actual Environment projection after a committed cut.

This tests the production World physical-step/publication composition with the
real Environment owner projection, Observation family adapter, and in-process
ObservationPlane. The physical solver and Embodiment port are deterministic
test capabilities; this is a bounded #103 integration slice, not #86's full
causal-cycle acceptance evidence.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from threading import Thread

import pytest
from embodiment.observation import BodyPoseSample
from embodiment.physics3d.physical_transition import (
    EmbodimentPhysicalState,
    SolverBodyCandidate,
)
from embodiment.physics3d.physical_transition import (
    PhysicalStepCorrelation as EmbodimentStepCorrelation,
)
from environment import (
    EnvironmentObservationContext,
    FiniteResourceState,
    FrameDescriptor,
    IdentifiedFiniteScalarField,
)
from observation.embodiment import embodiment_body_pose_observation
from observation.environment import environment_finite_scalar_field_observation
from observation.plane import ObservationPlane, PlaneRebaseRequired

from world.durable_runtime import (
    DurableJournalSnapshotStore,
    DurableWorldRuntime,
    PresenceOwnerLink,
    WorldGenesisSpec,
)
from world.manifest import CheckpointReference, ManifestError, WorldCausalManifestStore
from world.observation_publication import OwnerObservation
from world.physical_step import (
    PhysicalStepCandidate,
    PhysicalStepContext,
    PhysicalStepPorts,
    run_physical_step,
)
from world.physical_step_publication import run_physical_step_and_publish
from world.runtime import PresenceId, TickInProgressError, WorldId

WORLD_ID = "world.post-commit-observation"
PRESENCE_ID = "presence.environment"
RESOURCE_ID = "resource.scalar-field"
ENVIRONMENT_ID = "environment.scalar-field"
ENVIRONMENT_REVISION = "environment-r1"
FIELD_REVISION = "field-r1"
FRAME_ID = "environment-frame"


def _frame() -> FrameDescriptor:
    return FrameDescriptor(
        frame_id=FRAME_ID,
        parent_frame_id="world-frame",
        origin_m=(0.0, 0.0, 0.0),
        x_axis_in_parent=(1.0, 0.0, 0.0),
        y_axis_in_parent=(0.0, 1.0, 0.0),
        z_axis_in_parent=(0.0, 0.0, 1.0),
        handedness="RIGHT_HANDED",
        up_axis="Z",
    )


def _field() -> IdentifiedFiniteScalarField:
    return IdentifiedFiniteScalarField(
        resource_id=RESOURCE_ID,
        field_revision=FIELD_REVISION,
        environment_id=ENVIRONMENT_ID,
        environment_revision=ENVIRONMENT_REVISION,
        frame_id=FRAME_ID,
        state=FiniteResourceState(
            position=(1.0, 0.0, 0.0),
            radius=0.1,
            field_radius=2.0,
            remaining=10.0,
            transfer_per_tick=1.0,
        ),
    )


class _FieldCheckpointContract:
    def validate_checkpoint(
        self, encoded: bytes, owner_identity: str, schema_version: str
    ) -> bool:
        return IdentifiedFiniteScalarField.validate_checkpoint(
            encoded, owner_identity, schema_version
        )

    def verify_restoration(self, reference: CheckpointReference) -> bool:
        return Path(reference.locator).is_file()


def _reference(directory: Path, *, tick: int, encoded: bytes) -> CheckpointReference:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"environment-{tick}.checkpoint"
    path.write_bytes(encoded)
    return CheckpointReference(
        owner="environment",
        owner_identity=RESOURCE_ID,
        checkpoint_id=f"environment-cut-{tick}",
        schema_version=IdentifiedFiniteScalarField.OWNER_SCHEMA,
        locator=str(path),
        sha256=hashlib.sha256(encoded).hexdigest(),
    )


def _world(tmp_path: Path, field: IdentifiedFiniteScalarField) -> DurableWorldRuntime:
    initial = _reference(tmp_path / "owners", tick=0, encoded=field.checkpoint_bytes())
    contract = _FieldCheckpointContract()
    world_id = WorldId(WORLD_ID)
    return DurableWorldRuntime.initialize(
        WorldGenesisSpec(
            world_id=world_id,
            initial_presence_ids=(PRESENCE_ID,),
            registered_owner_ids=(("environment", RESOURCE_ID),),
            initial_owner_references=(initial,),
            owner_checkpoint_contracts={("environment", RESOURCE_ID): contract},
            presence_owner_links=(
                PresenceOwnerLink(PresenceId(PRESENCE_ID), "environment", RESOURCE_ID),
            ),
        ),
        PRESENCE_ID,
        WorldCausalManifestStore(tmp_path / "manifest.json"),
        DurableJournalSnapshotStore(tmp_path / "journal", world_id),
        owner_checkpoint_verifier=lambda refs: all(
            contract.validate_checkpoint(
                Path(reference.locator).read_bytes(),
                reference.owner_identity,
                reference.schema_version,
            )
            for reference in refs
        ),
    )


def _ports(field: IdentifiedFiniteScalarField, directory: Path, *, fail: bool = False):
    def execute(context, body_projection, environment_projection):
        if fail:
            raise RuntimeError("deterministic solver failure")
        return PhysicalStepCandidate(body_projection, environment_projection)

    def write_checkpoint(context: PhysicalStepContext):
        return (
            _reference(
                directory,
                tick=context.target_causal_tick,
                encoded=field.checkpoint_bytes(),
            ),
        )

    return PhysicalStepPorts(
        project_body=lambda context: (0.0, 0.0, 0.0),
        project_environment=lambda context: field,
        execute_solver=execute,
        adopt_body=lambda context, candidate: None,
        adopt_environment=lambda context, candidate: None,
        write_checkpoints=write_checkpoint,
    )


def _run(
    tmp_path: Path,
    *,
    fail: bool = False,
    plane: ObservationPlane | None = None,
    on_projection: Callable[[DurableWorldRuntime], None] | None = None,
):
    field = _field()
    world = _world(tmp_path, field)
    plane = plane or ObservationPlane()
    frame = _frame()

    def project(context: PhysicalStepContext, committed_cut):
        if on_projection is not None:
            on_projection(world)
        owner_projection = field.project_observation(
            (0.0, 0.0, 0.0),
            frame=frame,
            context=EnvironmentObservationContext(
                world_id=context.world_id,
                source_causal_tick=context.source_causal_tick,
                target_causal_tick=context.target_causal_tick,
                physical_step_id=context.physical_step_id,
                environment_id=ENVIRONMENT_ID,
                environment_revision=ENVIRONMENT_REVISION,
                resource_id=RESOURCE_ID,
                field_revision=FIELD_REVISION,
            ),
        )
        observation = environment_finite_scalar_field_observation(owner_projection)
        reference = next(
            ref
            for ref in committed_cut.owner_checkpoint_references
            if ref.owner == "environment" and ref.owner_identity == RESOURCE_ID
        )
        return (OwnerObservation(reference, observation),)

    if fail:
        # The callback is wrapped to prove that failed/uncommitted work is not
        # projected or delivered.
        def forbidden_projection(context, committed_cut):
            pytest.fail("publication projection must not run after solver failure")

        project = forbidden_projection

    result = run_physical_step_and_publish(
        world,
        plane,
        commit_id="commit-1",
        physical_step_id="physical-step-1",
        body_id="body.fixture",
        embodiment_episode_id="episode.fixture",
        morphology_revision="morphology-r1",
        environment_revision=ENVIRONMENT_REVISION,
        solver_config_revision="solver.fixture-r1",
        ports=_ports(field, tmp_path / "owners", fail=fail),
        project_owner_observations=project,
    )
    return world, plane, result


def test_actual_owner_projection_is_published_only_from_committed_world_cut(tmp_path):
    competing_tick_errors = []

    def attempt_competing_tick(world: DurableWorldRuntime):
        try:
            world.tick("competing-transition", lambda _tick: ())
        except TickInProgressError as error:
            competing_tick_errors.append(error)

    def attempt_during_publication(world: DurableWorldRuntime):
        thread = Thread(target=attempt_competing_tick, args=(world,))
        thread.start()
        thread.join(timeout=2)
        assert not thread.is_alive()

    world, plane, result = _run(tmp_path, on_projection=attempt_during_publication)

    assert world.causal_tick == 1
    assert result.committed_cut.causal_tick == 1
    assert result.publication.delivery_sequences == (1,)
    messages = plane.current_messages()
    assert len(messages) == 1
    observation = messages[0].observation
    assert observation.claim.authority.value == "environment"
    assert observation.context_time[-1].kind == "world_target_causal_tick"
    assert observation.context_time[-1].value == 1
    assert observation.lineage.parents[0].reference == next(
        ref.sha256 for ref in result.committed_cut.owner_checkpoint_references
    )
    assert len(competing_tick_errors) == 1
    assert isinstance(competing_tick_errors[0], TickInProgressError)


def test_reentrant_tick_during_projection_is_rejected(tmp_path):
    reentrant_tick_errors = []

    def attempt_reentrant_tick(world: DurableWorldRuntime):
        try:
            world.tick("reentrant-transition", lambda _tick: ())
        except TickInProgressError as error:
            reentrant_tick_errors.append(error)

    world, plane, result = _run(tmp_path, on_projection=attempt_reentrant_tick)

    assert len(reentrant_tick_errors) == 1
    assert world.causal_tick == 1
    assert result.committed_cut.causal_tick == 1
    assert result.publication.delivery_sequences == (1,)
    assert plane.latest_sequence == 1


def test_observation_sink_failure_leaves_committed_world_cut_authoritative(tmp_path):
    """Characterize the post-commit boundary; this does not implement retry."""
    committed_worlds = []

    class FailingObservationPlane:
        def publish(self, observation):
            raise RuntimeError("synthetic sink unavailable")

    def capture_committed_world(world: DurableWorldRuntime) -> None:
        committed_worlds.append(world)

    with pytest.raises(RuntimeError, match="synthetic sink unavailable"):
        _run(
            tmp_path,
            plane=FailingObservationPlane(),
            on_projection=capture_committed_world,
        )

    assert len(committed_worlds) == 1
    world = committed_worlds[0]
    assert world.causal_tick == 1
    assert world.status.value == "COMMITTED"
    assert world.manifest.status.value == "COMMITTED"
    assert world.manifest.commit_id == "commit-1"
    assert any(record.causal_tick == 1 for record in world.records)
    committed_cut = world.committed_cut_for(
        world_id=WORLD_ID,
        causal_tick=1,
        commit_id="commit-1",
    )
    assert committed_cut.causal_tick == 1
    assert committed_cut.commit_id == "commit-1"
    assert any(
        reference.owner == "environment" and reference.owner_identity == RESOURCE_ID
        for reference in committed_cut.owner_checkpoint_references
    )


def test_solver_failure_commits_no_cut_and_publishes_nothing(tmp_path):
    plane = ObservationPlane()
    with pytest.raises(RuntimeError, match="solver failure"):
        _run(tmp_path, fail=True, plane=plane)
    assert plane.latest_sequence == 0


def test_stale_owner_reference_after_commit_is_rejected_without_publication(tmp_path):
    field = _field()
    world = _world(tmp_path, field)
    plane = ObservationPlane()
    frame = _frame()

    def project(context: PhysicalStepContext, committed_cut):
        projection = field.project_observation(
            (0.0, 0.0, 0.0),
            frame=frame,
            context=EnvironmentObservationContext(
                world_id=context.world_id,
                source_causal_tick=context.source_causal_tick,
                target_causal_tick=context.target_causal_tick,
                physical_step_id=context.physical_step_id,
                environment_id=ENVIRONMENT_ID,
                environment_revision=ENVIRONMENT_REVISION,
                resource_id=RESOURCE_ID,
                field_revision=FIELD_REVISION,
            ),
        )
        observation = environment_finite_scalar_field_observation(projection)
        committed_reference = next(
            ref
            for ref in committed_cut.owner_checkpoint_references
            if ref.owner == "environment"
        )
        stale_reference = replace(committed_reference, sha256="f" * 64)
        return (OwnerObservation(stale_reference, observation),)

    with pytest.raises(ManifestError, match="not in the committed World cut"):
        run_physical_step_and_publish(
            world,
            plane,
            commit_id="commit-1",
            physical_step_id="physical-step-1",
            body_id="body.fixture",
            embodiment_episode_id="episode.fixture",
            morphology_revision="morphology-r1",
            environment_revision=ENVIRONMENT_REVISION,
            solver_config_revision="solver.fixture-r1",
            ports=_ports(field, tmp_path / "owners"),
            project_owner_observations=project,
        )

    assert world.causal_tick == 1
    assert plane.latest_sequence == 0


def test_real_body_pose_projection_uses_committed_checkpoint_digest(tmp_path):
    body_id = "body.publication"
    episode_id = "episode.publication"
    morphology_revision = "morphology.publication-r1"
    world_id = WorldId("world.body-pose-publication")
    field = _field()
    initial = EmbodimentPhysicalState(
        world_id=world_id.value,
        causal_tick=0,
        body_id=body_id,
        embodiment_episode_id=episode_id,
        morphology_revision=morphology_revision,
        pose=BodyPoseSample(
            body_id=body_id,
            reference_frame="world-frame",
            body_frame="body-root",
            position_m=(0.0, 0.0, 0.5),
            orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
            sample_kind="genesis",
            sample_value=0,
        ),
        linear_velocity_m_per_s=(0.0, 0.0, 0.0),
        angular_velocity_rad_per_s=(0.0, 0.0, 0.0),
    )

    class BodyContract:
        def validate_checkpoint(self, encoded, owner_identity, schema_version):
            restored = EmbodimentPhysicalState.restore_checkpoint(encoded)
            return (
                owner_identity == body_id
                and schema_version == "embodiment-physical-state-v2"
                and restored.body_id == body_id
                and restored.embodiment_episode_id == episode_id
            )

        def verify_restoration(self, reference):
            return Path(reference.locator).is_file()

    body_contract = BodyContract()
    field_contract = _FieldCheckpointContract()
    owner_contracts = {
        ("embodiment", body_id): body_contract,
        ("environment", RESOURCE_ID): field_contract,
    }

    def store_reference(owner, identity, schema, tick, encoded):
        directory = tmp_path / "body-pose-owners"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{owner}-{tick}.checkpoint"
        path.write_bytes(encoded)
        return CheckpointReference(
            owner=owner,
            owner_identity=identity,
            checkpoint_id=f"{owner}-cut-{tick}",
            schema_version=schema,
            locator=str(path),
            sha256=hashlib.sha256(encoded).hexdigest(),
        )

    initial_references = (
        store_reference(
            "embodiment", body_id, "embodiment-physical-state-v2", 0,
            initial.checkpoint_bytes(),
        ),
        store_reference(
            "environment", RESOURCE_ID, IdentifiedFiniteScalarField.OWNER_SCHEMA, 0,
            field.checkpoint_bytes(),
        ),
    )
    world = DurableWorldRuntime.initialize(
        WorldGenesisSpec(
            world_id=world_id,
            initial_presence_ids=("presence.body", "presence.environment"),
            registered_owner_ids=tuple(sorted(owner_contracts)),
            initial_owner_references=initial_references,
            owner_checkpoint_contracts=owner_contracts,
            presence_owner_links=(
                PresenceOwnerLink(PresenceId("presence.body"), "embodiment", body_id),
                PresenceOwnerLink(PresenceId("presence.environment"), "environment", RESOURCE_ID),
            ),
        ),
        "presence.body",
        WorldCausalManifestStore(tmp_path / "body-pose-manifest.json"),
        DurableJournalSnapshotStore(tmp_path / "body-pose-journal", world_id),
        owner_checkpoint_verifier=lambda refs: all(
            owner_contracts[(ref.owner, ref.owner_identity)].validate_checkpoint(
                Path(ref.locator).read_bytes(), ref.owner_identity, ref.schema_version
            )
            for ref in refs
        ),
    )
    staged = {"body": initial}

    def execute(context, body_projection, _environment_projection):
        assert body_projection is initial
        correlation = EmbodimentStepCorrelation(
            world_id=context.world_id,
            causal_tick=context.source_causal_tick,
            physical_step_id=context.physical_step_id,
            body_id=context.body_id,
            embodiment_episode_id=context.embodiment_episode_id,
            morphology_revision=context.morphology_revision,
            environment_revision=context.environment_revision,
            solver_config_revision=context.solver_config_revision,
        )
        return PhysicalStepCandidate(
            SolverBodyCandidate(
                correlation=correlation,
                reference_frame="world-frame",
                body_frame="body-root",
                position_m=(0.1, 0.0, 0.5),
                orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
                linear_velocity_m_per_s=(0.1, 0.0, 0.0),
                angular_velocity_rad_per_s=(0.0, 0.0, 0.0),
            ),
            None,
        )

    def write_checkpoints(context):
        body_state = staged["body"]
        return (
            store_reference(
                "embodiment", body_id, "embodiment-physical-state-v2",
                context.target_causal_tick, body_state.checkpoint_bytes(),
            ),
            store_reference(
                "environment", RESOURCE_ID, IdentifiedFiniteScalarField.OWNER_SCHEMA,
                context.target_causal_tick, field.checkpoint_bytes(),
            ),
        )

    def project_body_pose(_context, cut):
        reference = next(
            ref for ref in cut.owner_checkpoint_references
            if ref.owner == "embodiment" and ref.owner_identity == body_id
        )
        encoded = Path(reference.locator).read_bytes()
        assert hashlib.sha256(encoded).hexdigest() == reference.sha256
        restored = EmbodimentPhysicalState.restore_checkpoint(encoded)
        observation = embodiment_body_pose_observation(
            restored.body_pose_observation_projection()
        )
        return (OwnerObservation(reference, observation),)

    plane = ObservationPlane()
    result = run_physical_step_and_publish(
        world,
        plane,
        commit_id="body-pose-commit-1",
        physical_step_id="body-pose-step-1",
        body_id=body_id,
        embodiment_episode_id=episode_id,
        morphology_revision=morphology_revision,
        environment_revision=ENVIRONMENT_REVISION,
        solver_config_revision="solver.publication-r1",
        ports=PhysicalStepPorts(
            project_body=lambda _context: initial,
            project_environment=lambda _context: field,
            execute_solver=execute,
            adopt_body=lambda context, candidate: staged.update(
                body=initial.adopt_candidate(candidate, world_context=context)
            ),
            adopt_environment=lambda _context, _candidate: None,
            write_checkpoints=write_checkpoints,
        ),
        project_owner_observations=project_body_pose,
    )

    reference = next(
        ref for ref in result.committed_cut.owner_checkpoint_references
        if ref.owner == "embodiment" and ref.owner_identity == body_id
    )
    messages = plane.current_messages()
    assert world.causal_tick == result.committed_cut.causal_tick == 1
    assert len(messages) == 1
    observation = messages[0].observation
    assert observation.claim.authority.value == "embodiment"
    assert observation.context_time[0].value == result.committed_cut.causal_tick
    assert observation.lineage.parents[0].namespace == "embodiment_state_sha256"
    assert observation.lineage.parents[0].reference == reference.sha256
    assert reference.sha256 == hashlib.sha256(Path(reference.locator).read_bytes()).hexdigest()


def _run_nontrivial_body_step(
    tmp_path: Path,
    *,
    publish_observation: bool,
    overflow_subscriber: bool = False,
):
    """Run the same adopted Embodiment transition with publication enabled/disabled."""
    body_id = "body.noninterference"
    episode_id = "episode.noninterference"
    morphology_revision = "morphology.noninterference-r1"
    world_id = WorldId("world.observation-noninterference")
    field = _field()
    initial_body = EmbodimentPhysicalState(
        world_id=world_id.value,
        causal_tick=0,
        body_id=body_id,
        embodiment_episode_id=episode_id,
        morphology_revision=morphology_revision,
        pose=BodyPoseSample(
            body_id=body_id,
            reference_frame="world-frame",
            body_frame="body-root",
            position_m=(0.0, 0.0, 0.5),
            orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
            sample_kind="genesis",
            sample_value=0,
        ),
        linear_velocity_m_per_s=(0.0, 0.0, 0.0),
        angular_velocity_rad_per_s=(0.0, 0.0, 0.0),
    )

    class BodyContract:
        def validate_checkpoint(self, encoded, owner_identity, schema_version):
            restored = EmbodimentPhysicalState.restore_checkpoint(encoded)
            return (
                owner_identity == body_id
                and schema_version == "embodiment-physical-state-v2"
                and restored.body_id == body_id
                and restored.embodiment_episode_id == episode_id
                and restored.world_id == world_id.value
            )

        def verify_restoration(self, reference):
            return Path(reference.locator).is_file()

    body_contract = BodyContract()
    field_contract = _FieldCheckpointContract()
    owner_contracts = {
        ("embodiment", body_id): body_contract,
        ("environment", RESOURCE_ID): field_contract,
    }

    def store_reference(owner, identity, schema, tick, encoded):
        directory = tmp_path / "owners"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{owner}-{tick}.checkpoint"
        path.write_bytes(encoded)
        return CheckpointReference(
            owner=owner,
            owner_identity=identity,
            checkpoint_id=f"{owner}-cut-{tick}",
            schema_version=schema,
            locator=str(path),
            sha256=hashlib.sha256(encoded).hexdigest(),
        )

    initial_references = (
        store_reference(
            "embodiment", body_id, "embodiment-physical-state-v2", 0,
            initial_body.checkpoint_bytes(),
        ),
        store_reference(
            "environment", RESOURCE_ID, IdentifiedFiniteScalarField.OWNER_SCHEMA, 0,
            field.checkpoint_bytes(),
        ),
    )
    world = DurableWorldRuntime.initialize(
        WorldGenesisSpec(
            world_id=world_id,
            initial_presence_ids=("presence.body", "presence.environment"),
            registered_owner_ids=tuple(sorted(owner_contracts)),
            initial_owner_references=initial_references,
            owner_checkpoint_contracts=owner_contracts,
            presence_owner_links=(
                PresenceOwnerLink(PresenceId("presence.body"), "embodiment", body_id),
                PresenceOwnerLink(PresenceId("presence.environment"), "environment", RESOURCE_ID),
            ),
        ),
        "presence.body",
        WorldCausalManifestStore(tmp_path / "manifest.json"),
        DurableJournalSnapshotStore(tmp_path / "journal", world_id),
        owner_checkpoint_verifier=lambda refs: all(
            owner_contracts[(ref.owner, ref.owner_identity)].validate_checkpoint(
                Path(ref.locator).read_bytes(), ref.owner_identity, ref.schema_version
            )
            for ref in refs
        ),
    )
    staged_body = {"value": initial_body}

    def execute(context, body_projection, environment_projection):
        assert body_projection is initial_body
        assert environment_projection is field
        correlation = EmbodimentStepCorrelation(
            world_id=context.world_id,
            causal_tick=context.source_causal_tick,
            physical_step_id=context.physical_step_id,
            body_id=context.body_id,
            embodiment_episode_id=context.embodiment_episode_id,
            morphology_revision=context.morphology_revision,
            environment_revision=context.environment_revision,
            solver_config_revision=context.solver_config_revision,
        )
        return PhysicalStepCandidate(
            SolverBodyCandidate(
                correlation=correlation,
                reference_frame="world-frame",
                body_frame="body-root",
                position_m=(0.1, 0.0, 0.5),
                orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
                linear_velocity_m_per_s=(0.1, 0.0, 0.0),
                angular_velocity_rad_per_s=(0.0, 0.0, 0.0),
            ),
            None,
        )

    checkpoint_references = []

    def write_checkpoints(context):
        references = (
            store_reference(
                "embodiment", body_id, "embodiment-physical-state-v2",
                context.target_causal_tick, staged_body["value"].checkpoint_bytes(),
            ),
            store_reference(
                "environment", RESOURCE_ID, IdentifiedFiniteScalarField.OWNER_SCHEMA,
                context.target_causal_tick, field.checkpoint_bytes(),
            ),
        )
        checkpoint_references[:] = references
        return references

    ports = PhysicalStepPorts(
        project_body=lambda _context: initial_body,
        project_environment=lambda _context: field,
        execute_solver=execute,
        adopt_body=lambda context, candidate: staged_body.update(
            value=initial_body.adopt_candidate(candidate, world_context=context)
        ),
        adopt_environment=lambda _context, _candidate: None,
        write_checkpoints=write_checkpoints,
    )
    common = dict(
        commit_id="noninterference-commit-1",
        physical_step_id="noninterference-step-1",
        body_id=body_id,
        embodiment_episode_id=episode_id,
        morphology_revision=morphology_revision,
        environment_revision=ENVIRONMENT_REVISION,
        solver_config_revision="solver.noninterference-r1",
        ports=ports,
    )
    plane = (
        ObservationPlane(subscriber_queue_size=1)
        if publish_observation and overflow_subscriber
        else ObservationPlane() if publish_observation else None
    )
    subscription = plane.open_stream() if overflow_subscriber and plane is not None else None
    if publish_observation:
        frame = _frame()

        def project_body_pose(context, committed_cut):
            reference = next(
                ref for ref in committed_cut.owner_checkpoint_references
                if ref.owner == "embodiment" and ref.owner_identity == body_id
            )
            encoded = Path(reference.locator).read_bytes()
            assert hashlib.sha256(encoded).hexdigest() == reference.sha256
            restored = EmbodimentPhysicalState.restore_checkpoint(encoded)
            projection = restored.body_pose_observation_projection()
            observation = embodiment_body_pose_observation(projection)
            owner_observations = [OwnerObservation(reference, observation)]
            if overflow_subscriber:
                environment_reference = next(
                    ref for ref in committed_cut.owner_checkpoint_references
                    if ref.owner == "environment" and ref.owner_identity == RESOURCE_ID
                )
                environment_projection = field.project_observation(
                    (0.0, 0.0, 0.0),
                    frame=frame,
                    context=EnvironmentObservationContext(
                        world_id=context.world_id,
                        source_causal_tick=context.source_causal_tick,
                        target_causal_tick=context.target_causal_tick,
                        physical_step_id=context.physical_step_id,
                        environment_id=ENVIRONMENT_ID,
                        environment_revision=ENVIRONMENT_REVISION,
                        resource_id=RESOURCE_ID,
                        field_revision=FIELD_REVISION,
                    ),
                )
                owner_observations.append(
                    OwnerObservation(
                        environment_reference,
                        environment_finite_scalar_field_observation(environment_projection),
                    )
                )
            return tuple(owner_observations)

        result = run_physical_step_and_publish(
            world,
            plane,
            **common,
            project_owner_observations=project_body_pose,
        )
        record = result.causal_record
        cut = result.committed_cut
    else:
        record = run_physical_step(world, **common)
        cut = world.committed_cut_for(
            world_id=record.world_id,
            causal_tick=record.causal_tick,
            commit_id="noninterference-commit-1",
            expected_owner_references=tuple(checkpoint_references),
        )
    body_ref = next(
        ref for ref in cut.owner_checkpoint_references
        if ref.owner == "embodiment" and ref.owner_identity == body_id
    )
    env_ref = next(
        ref for ref in cut.owner_checkpoint_references
        if ref.owner == "environment" and ref.owner_identity == RESOURCE_ID
    )
    adopted_body = EmbodimentPhysicalState.restore_checkpoint(
        Path(body_ref.locator).read_bytes()
    )
    rebase = subscription.queue.get_nowait() if subscription is not None else None
    if subscription is not None:
        plane.close_stream(subscription)
    return world, plane, record, cut, adopted_body, env_ref, rebase


def test_passive_observation_publication_does_not_change_adopted_world_result(
    tmp_path,
):
    """Publication does not alter a nontrivial solver result or owner adoption."""
    observed = _run_nontrivial_body_step(
        tmp_path / "observed", publish_observation=True
    )
    unobserved = _run_nontrivial_body_step(
        tmp_path / "unobserved", publish_observation=False
    )
    observed_world, plane, observed_record, observed_cut, observed_body, observed_env_ref, _ = observed
    unobserved_world, _, unobserved_record, unobserved_cut, unobserved_body, unobserved_env_ref, _ = unobserved

    assert observed_world.causal_tick == unobserved_world.causal_tick == 1
    assert observed_record == unobserved_record
    assert observed_cut.causal_tick == unobserved_cut.causal_tick == 1
    assert observed_body == unobserved_body
    assert observed_body.pose.position_m == (0.1, 0.0, 0.5)
    assert observed_body.linear_velocity_m_per_s == (0.1, 0.0, 0.0)
    assert observed_env_ref.sha256 == unobserved_env_ref.sha256

    def stable_reference(reference):
        return (
            reference.owner,
            reference.owner_identity,
            reference.checkpoint_id,
            reference.schema_version,
            reference.sha256,
        )

    assert tuple(map(stable_reference, observed_cut.owner_checkpoint_references)) == tuple(
        map(stable_reference, unobserved_cut.owner_checkpoint_references)
    )
    assert plane is not None
    messages = plane.current_messages()
    assert len(messages) == 1
    observation = messages[0].observation
    assert observation.claim.authority.value == "embodiment"
    assert observation.lineage.parents[0].reference == next(
        reference.sha256
        for reference in observed_cut.owner_checkpoint_references
        if reference.owner == "embodiment"
    )


def test_observation_subscriber_overflow_does_not_change_world_causal_result(tmp_path):
    """A bounded subscriber rebase cannot alter World commit or owner adoption."""
    saturated = _run_nontrivial_body_step(
        tmp_path / "saturated", publish_observation=True, overflow_subscriber=True
    )
    unobserved = _run_nontrivial_body_step(
        tmp_path / "unobserved", publish_observation=False
    )
    world, plane, record, cut, body, environment_ref, rebase = saturated
    baseline_world, _, baseline_record, baseline_cut, baseline_body, baseline_environment_ref, _ = unobserved

    assert isinstance(rebase, PlaneRebaseRequired)
    assert rebase.reason == "consumer_overflow"
    assert rebase.sequence == 2
    assert plane is not None
    assert plane.consumer_overflows == 1

    assert world.causal_tick == baseline_world.causal_tick == 1
    assert record == baseline_record
    assert cut.causal_tick == baseline_cut.causal_tick == 1
    assert cut.commit_id == baseline_cut.commit_id == "noninterference-commit-1"
    assert body == baseline_body
    assert body.pose.position_m == (0.1, 0.0, 0.5)
    assert environment_ref.sha256 == baseline_environment_ref.sha256

    def stable_reference(reference):
        return (
            reference.owner,
            reference.owner_identity,
            reference.checkpoint_id,
            reference.schema_version,
            reference.sha256,
        )

    assert tuple(map(stable_reference, cut.owner_checkpoint_references)) == tuple(
        map(stable_reference, baseline_cut.owner_checkpoint_references)
    )
    assert len(plane.current_messages()) == 2
