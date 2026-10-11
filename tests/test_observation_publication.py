from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from world.durable_runtime import CommittedWorldCut
from world.manifest import CheckpointReference, ManifestError
from world.observation_publication import (
    OwnerObservation,
    publish_committed_owner_observations,
)


@dataclass
class _World:
    cut: CommittedWorldCut

    def committed_cut_for(
        self, *, world_id, causal_tick, commit_id, expected_owner_references
    ):
        if (
            world_id != self.cut.world_id
            or causal_tick != self.cut.causal_tick
            or commit_id != self.cut.commit_id
            or tuple(expected_owner_references) != self.cut.owner_checkpoint_references
        ):
            raise ManifestError("cut identity or references do not match")
        return self.cut


class _Plane:
    def __init__(self):
        self.observations = []

    def publish(self, observation):
        self.observations.append(observation)
        return len(self.observations)


def _reference(*, owner="environment", identity="resource-1", digest="a" * 64):
    return CheckpointReference(
        owner=owner,
        owner_identity=identity,
        checkpoint_id="checkpoint-1",
        schema_version="schema-v1",
        locator="opaque://checkpoint-1",
        sha256=digest,
    )


def _cut(reference=None):
    reference = reference or _reference()
    return CommittedWorldCut(
        world_id="world-1",
        causal_tick=4,
        commit_id="commit-4",
        participant_ids=((reference.owner, reference.owner_identity),),
        presence_owner_links=(),
        owner_checkpoint_references=(reference,),
    )


def _observation(
    reference=None, *, tick=4, world_id="world-1", authority="environment"
):
    reference = reference or _reference()
    parent = SimpleNamespace(
        namespace="owner_checkpoint_sha256", reference=reference.sha256
    )
    transformation = SimpleNamespace(effective_configuration=(("world_id", world_id),))
    return SimpleNamespace(
        claim=SimpleNamespace(authority=SimpleNamespace(value=authority)),
        lineage=SimpleNamespace(parents=(parent,), transformation=transformation),
        context_time=(
            SimpleNamespace(
                domain=SimpleNamespace(value="world"),
                kind="world_target_causal_tick",
                value=tick,
            ),
        ),
    )


def test_publishes_only_after_receipt_and_owner_provenance_match():
    reference = _reference()
    cut = _cut(reference)
    world = _World(cut)
    plane = _Plane()
    observation = _observation(reference)

    result = publish_committed_owner_observations(
        world,  # type: ignore[arg-type]
        plane,
        committed_cut=cut,
        owner_observations=(OwnerObservation(reference, observation),),
    )

    assert result.committed_cut == cut
    assert result.delivery_sequences == (1,)
    assert plane.observations == [observation]


@pytest.mark.parametrize(
    "observation",
    [
        _observation(tick=3),
        _observation(world_id="another-world"),
        _observation(authority="organism"),
    ],
)
def test_rejects_wrong_cut_correlation_or_authority_before_publication(observation):
    reference = _reference()
    cut = _cut(reference)
    plane = _Plane()

    with pytest.raises(ManifestError):
        publish_committed_owner_observations(
            _World(cut),  # type: ignore[arg-type]
            plane,
            committed_cut=cut,
            owner_observations=(OwnerObservation(reference, observation),),
        )

    assert plane.observations == []


def test_rejects_stale_owner_reference_without_publication():
    committed_reference = _reference()
    stale_reference = _reference(digest="b" * 64)
    cut = _cut(committed_reference)
    plane = _Plane()

    with pytest.raises(ManifestError):
        publish_committed_owner_observations(
            _World(cut),  # type: ignore[arg-type]
            plane,
            committed_cut=cut,
            owner_observations=(
                OwnerObservation(stale_reference, _observation(stale_reference)),
            ),
        )

    assert plane.observations == []


def test_validates_entire_batch_before_first_publication():
    reference = _reference()
    cut = _cut(reference)
    plane = _Plane()

    with pytest.raises(ManifestError):
        publish_committed_owner_observations(
            _World(cut),  # type: ignore[arg-type]
            plane,
            committed_cut=cut,
            owner_observations=(
                OwnerObservation(reference, _observation(reference)),
                OwnerObservation(reference, _observation(reference, tick=3)),
            ),
        )

    assert plane.observations == []


def test_empty_observation_batch_is_not_a_successful_publication():
    cut = _cut()
    plane = _Plane()
    with pytest.raises(ValueError, match="at least one"):
        publish_committed_owner_observations(
            _World(cut),  # type: ignore[arg-type]
            plane,
            committed_cut=cut,
            owner_observations=(),
        )
    assert plane.observations == []
