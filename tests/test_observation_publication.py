from __future__ import annotations

from dataclasses import dataclass, replace
from types import SimpleNamespace

import pytest

from world.durable_runtime import CommittedWorldCut
from world.manifest import CheckpointReference, ManifestError
from world.observation_publication import (
    OwnerObservation,
    OwnerProjectionIdentity,
    associate_committed_owner_observations,
    publish_committed_owner_observations,
    verify_associated_owner_observations,
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
    reference=None,
    *,
    tick=4,
    world_id="world-1",
    authority="environment",
    lineage_namespace="owner_checkpoint_sha256",
):
    reference = reference or _reference()
    parent = SimpleNamespace(
        namespace=lineage_namespace, reference=reference.sha256
    )
    transformation = SimpleNamespace(effective_configuration=(("world_id", world_id),))
    return SimpleNamespace(
        claim=SimpleNamespace(
            authority=SimpleNamespace(value=authority),
            family=SimpleNamespace(
                qualified_name=f"{authority}.scene",
                contract=SimpleNamespace(contract_id=f"{authority}.scene", revision="v1"),
            ),
        ),
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


def test_rejects_caller_fabricated_structural_cut_receipt_before_publication():
    reference = _reference()
    actual_cut = _cut(reference)
    forged_cut = replace(actual_cut, commit_id="caller-invented-commit")
    plane = _Plane()

    with pytest.raises(ManifestError, match="cut identity or references"):
        publish_committed_owner_observations(
            _World(actual_cut),  # type: ignore[arg-type]
            plane,
            committed_cut=forged_cut,
            owner_observations=(OwnerObservation(reference, _observation(reference)),),
        )

    assert plane.observations == []


def test_preserves_owner_specific_lineage_namespace_when_digest_matches_cut():
    reference = _reference(owner="embodiment", identity="body-1")
    cut = _cut(reference)
    plane = _Plane()
    observation = _observation(
        reference,
        authority="embodiment",
        lineage_namespace="embodiment_state_sha256",
    )

    result = publish_committed_owner_observations(
        _World(cut),  # type: ignore[arg-type]
        plane,
        committed_cut=cut,
        owner_observations=(OwnerObservation(reference, observation),),
    )

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


def test_rejects_owner_lineage_digest_mismatch_without_publication():
    reference = _reference(owner="embodiment", identity="body-1")
    cut = _cut(reference)
    observation = _observation(
        reference,
        authority="embodiment",
        lineage_namespace="embodiment_state_sha256",
    )
    observation.lineage.parents = (
        SimpleNamespace(
            namespace="embodiment_state_sha256", reference="b" * 64
        ),
    )
    plane = _Plane()

    with pytest.raises(ManifestError, match="provenance"):
        publish_committed_owner_observations(
            _World(cut),  # type: ignore[arg-type]
            plane,
            committed_cut=cut,
            owner_observations=(OwnerObservation(reference, observation),),
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


def _projection_identity(reference, *, family=None, claim="scene-1"):
    family = family or f"{reference.owner}.scene"
    return OwnerProjectionIdentity(
        owner_id=reference.owner,
        owner_identity=reference.owner_identity,
        family_id=family,
        owner_contract_id=family,
        owner_contract_revision="v1",
        projection_revision="r1",
        claim_identity=claim,
    )


def _associated(*, reference=None, commit_id="commit-4", identity=None):
    reference = reference or _reference()
    cut = replace(_cut(reference), commit_id=commit_id)
    observation = _observation(reference)
    item = OwnerObservation(
        reference, observation,
        projection_identity=identity or _projection_identity(reference),
    )
    association, projections = associate_committed_owner_observations(
        _World(cut), committed_cut=cut, owner_observations=(item,)
    )
    return association, projections


def test_association_binds_exact_cut_members_and_consumer_verifies_them():
    association, projections = _associated()

    assert association.world_id == "world-1"
    assert association.causal_tick == 4
    assert association.commit_id == "commit-4"
    assert len(association.members) == 1
    assert association.members[0].checkpoint_sha256 == "a" * 64
    assert verify_associated_owner_observations(association, projections)


def test_consumer_rejects_same_tick_projection_bound_to_different_commit():
    association, projections = _associated()
    other_commit_association = replace(association, commit_id="different-commit-4")
    different = replace(projections[0], association=other_commit_association)

    assert other_commit_association.causal_tick == association.causal_tick
    assert not verify_associated_owner_observations(association, (different,))


@pytest.mark.parametrize(
    "mutation",
    ["authority", "family", "contract", "contract_revision", "world", "tick"],
)
def test_consumer_rejects_cross_claim_or_cross_cut_observation(mutation):
    association, projections = _associated()
    observation = _observation()
    if mutation == "authority":
        observation = _observation(authority="organism")
    elif mutation == "family":
        observation.claim.family.qualified_name = "environment.other"
    elif mutation == "contract":
        observation.claim.family.contract.contract_id = "environment.other"
    elif mutation == "contract_revision":
        observation.claim.family.contract.revision = "stale"
    elif mutation == "world":
        observation = _observation(world_id="different-world")
    elif mutation == "tick":
        observation = _observation(tick=5)
    tampered = replace(projections[0], observation=observation)

    assert not verify_associated_owner_observations(association, (tampered,))


def test_consumer_rejects_observation_with_wrong_checkpoint_lineage_digest():
    association, projections = _associated()
    observation = _observation()
    observation = SimpleNamespace(
        claim=observation.claim,
        lineage=SimpleNamespace(
            parents=(SimpleNamespace(namespace="opaque", reference="b" * 64),),
            transformation=observation.lineage.transformation,
        ),
        context_time=observation.context_time,
    )
    assert not verify_associated_owner_observations(
        association, (replace(projections[0], observation=observation),)
    )


def test_verifies_multiple_owner_members_in_the_same_committed_cut():
    first = _reference()
    second = _reference(owner="embodiment", identity="body-1", digest="b" * 64)
    cut = CommittedWorldCut(
        world_id="world-1",
        causal_tick=4,
        commit_id="commit-4",
        participant_ids=(("environment", "resource-1"), ("embodiment", "body-1")),
        presence_owner_links=(),
        owner_checkpoint_references=(first, second),
    )
    items = (
        OwnerObservation(first, _observation(first), _projection_identity(first)),
        OwnerObservation(second, _observation(second, authority="embodiment"), _projection_identity(second)),
    )
    association, projections = associate_committed_owner_observations(
        _World(cut), committed_cut=cut, owner_observations=items
    )
    assert len(association.members) == 2
    assert verify_associated_owner_observations(association, projections)
    wrong_second = replace(
        projections[1], observation=_observation(second, authority="environment")
    )
    assert not verify_associated_owner_observations(
        association, (projections[0], wrong_second)
    )


@pytest.mark.parametrize("mutation", ["missing_member", "wrong_checkpoint", "ambiguous_duplicate"])
def test_consumer_rejects_missing_mismatched_or_ambiguous_references(mutation):
    association, projections = _associated()
    projection = projections[0]
    if mutation == "missing_member":
        bad = replace(projection, member=replace(projection.member, checkpoint_sha256="b" * 64))
        candidates = (bad,)
    elif mutation == "wrong_checkpoint":
        bad_reference = replace(projection.owner_reference, sha256="b" * 64)
        candidates = (replace(projection, owner_reference=bad_reference),)
    else:
        candidates = (projection, projection)

    assert not verify_associated_owner_observations(association, candidates)


def test_association_issuer_rejects_missing_and_duplicate_exact_projection_identity():
    reference = _reference()
    cut = _cut(reference)
    missing = OwnerObservation(reference, _observation(reference))
    duplicate = OwnerObservation(
        reference, _observation(reference), projection_identity=_projection_identity(reference)
    )

    with pytest.raises(ManifestError, match="identity is required"):
        associate_committed_owner_observations(
            _World(cut), committed_cut=cut, owner_observations=(missing,)
        )
    with pytest.raises(ManifestError, match="duplicate exact"):
        associate_committed_owner_observations(
            _World(cut), committed_cut=cut, owner_observations=(duplicate, duplicate)
        )


@pytest.mark.parametrize(
    "field",
    [
        "owner_id",
        "owner_identity",
        "family_id",
        "owner_contract_id",
        "owner_contract_revision",
        "projection_revision",
        "claim_identity",
    ],
)
def test_association_issuer_rejects_empty_projection_identity_fields(field):
    reference = _reference()
    cut = _cut(reference)
    identity = replace(_projection_identity(reference), **{field: " "})
    with pytest.raises(ManifestError, match="non-empty strings"):
        associate_committed_owner_observations(
            _World(cut),
            committed_cut=cut,
            owner_observations=(
                OwnerObservation(
                    reference,
                    _observation(reference),
                    projection_identity=identity,
                ),
            ),
        )


def test_association_rejects_duplicate_claim_identity_even_with_different_checkpoint_refs():
    first_ref = _reference()
    second_ref = replace(first_ref, checkpoint_id="checkpoint-2", sha256="b" * 64)
    cut = CommittedWorldCut(
        world_id="world-1",
        causal_tick=4,
        commit_id="commit-4",
        participant_ids=(("environment", "resource-1"),),
        presence_owner_links=(),
        owner_checkpoint_references=(first_ref, second_ref),
    )
    identity = _projection_identity(first_ref)
    items = (
        OwnerObservation(first_ref, _observation(first_ref), projection_identity=identity),
        OwnerObservation(second_ref, _observation(second_ref), projection_identity=identity),
    )

    with pytest.raises(ManifestError, match="duplicate exact"):
        associate_committed_owner_observations(
            _World(cut), committed_cut=cut, owner_observations=items
        )
