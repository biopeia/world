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
from environment import (
    EnvironmentObservationContext,
    FiniteResourceState,
    FrameDescriptor,
    IdentifiedFiniteScalarField,
)
from observation.environment import environment_finite_scalar_field_observation
from observation.plane import ObservationPlane

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
