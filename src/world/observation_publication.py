"""Post-commit publication of owner-authored observations.

World verifies causal-cut identity and checkpoint correlation here. It does
not construct, decode, or reinterpret semantic claims; each input observation
must already have been produced through its domain's approved Observation
adapter.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol

from .durable_runtime import CommittedWorldCut, DurableWorldRuntime
from .manifest import CheckpointReference, ManifestError


class _ParentRef(Protocol):
    namespace: str
    reference: str


class _Transformation(Protocol):
    effective_configuration: Sequence[tuple[str, str]]


class _Lineage(Protocol):
    parents: Sequence[_ParentRef]
    transformation: _Transformation | None


class _Claim(Protocol):
    authority: object
    family: object


class OwnerObservationLike(Protocol):
    """Structural subset of the transport-neutral Observation record."""

    claim: _Claim
    lineage: _Lineage
    context_time: Sequence[object]


class ObservationPlaneLike(Protocol):
    def publish(self, observation: OwnerObservationLike) -> int: ...


@dataclass(frozen=True, slots=True)
class OwnerObservation:
    """An owner-authorized semantic observation tied to its source checkpoint."""

    owner_reference: CheckpointReference
    observation: OwnerObservationLike
    # Owner-defined identity of this exact projection claim. Existing callers
    # may omit it for ordinary per-message publication; cut association fails
    # closed unless every candidate claim supplies it.
    projection_identity: "OwnerProjectionIdentity | None" = None


@dataclass(frozen=True, slots=True, order=True)
class OwnerProjectionIdentity:
    """Owner-defined identity fields needed to distinguish exact projections.

    World treats these strings as opaque identifiers. It does not define
    family, contract, revision, or claim semantics.
    """

    owner_id: str
    owner_identity: str
    family_id: str
    owner_contract_id: str
    owner_contract_revision: str
    projection_revision: str
    claim_identity: str


@dataclass(frozen=True, slots=True, order=True)
class CommittedCutMember:
    """One exact owner projection and its committed checkpoint reference."""

    projection_identity: OwnerProjectionIdentity
    checkpoint_id: str
    checkpoint_schema_version: str
    checkpoint_sha256: str


@dataclass(frozen=True, slots=True)
class CommittedCutAssociation:
    """In-memory World-issued association; no digest or persistence is implied.

    The complete immutable value is the association identity. Carrying this
    value alongside a projection avoids an ambiguous client-side "current
    association" pointer and does not invent a hash/canonical byte format.
    """

    kind: str
    contract_revision: str
    issuer: str
    world_id: str
    causal_tick: int
    commit_id: str
    members: tuple[CommittedCutMember, ...]


@dataclass(frozen=True, slots=True)
class AssociatedOwnerObservation:
    """Owner projection bound to the complete association and exact member."""

    association: CommittedCutAssociation
    member: CommittedCutMember
    projection_identity: OwnerProjectionIdentity
    owner_reference: CheckpointReference
    observation: OwnerObservationLike


def associate_committed_owner_observations(
    world: DurableWorldRuntime,
    *,
    committed_cut: CommittedWorldCut,
    owner_observations: Iterable[OwnerObservation],
) -> tuple[CommittedCutAssociation, tuple[AssociatedOwnerObservation, ...]]:
    """Issue an ephemeral association for validated projections in one cut.

    This re-verifies the existing durable committed cut, then validates each
    owner projection using the same checks as ordinary publication. It does
    not publish, persist, decode checkpoint bytes, or attest claim meaning.
    """

    verified = _verify_committed_cut(world, committed_cut)
    items = tuple(owner_observations)
    if not items:
        raise ValueError("at least one owner observation is required for association")

    members: list[CommittedCutMember] = []
    for item in items:
        _validate_owner_observation(verified, item)
        identity = item.projection_identity
        if identity is None:
            raise ManifestError("owner projection identity is required for cut association")
        if not _valid_projection_identity(identity):
            raise ManifestError("projection identity fields must be non-empty strings")
        reference = item.owner_reference
        if (
            identity.owner_id != reference.owner
            or identity.owner_identity != reference.owner_identity
        ):
            raise ManifestError("projection identity does not match its committed owner reference")
        family = getattr(item.observation.claim, "family", None)
        contract = getattr(family, "contract", None)
        if (
            getattr(family, "qualified_name", None) != identity.family_id
            or getattr(contract, "contract_id", None) != identity.owner_contract_id
            or getattr(contract, "revision", None) != identity.owner_contract_revision
        ):
            raise ManifestError("projection identity does not match its owner claim family contract")
        members.append(
            CommittedCutMember(
                identity,
                reference.checkpoint_id,
                reference.schema_version,
                reference.sha256,
            )
        )

    projection_identities = [member.projection_identity for member in members]
    if len(set(projection_identities)) != len(projection_identities):
        raise ManifestError("duplicate exact projection member makes association ambiguous")
    association = CommittedCutAssociation(
        kind="world.committed_cut_association",
        contract_revision="candidate-1",
        issuer="world",
        world_id=verified.world_id,
        causal_tick=verified.causal_tick,
        commit_id=verified.commit_id,
        members=tuple(sorted(members)),
    )
    associated = tuple(
        AssociatedOwnerObservation(
            association=association,
            member=member,
            projection_identity=member.projection_identity,
            owner_reference=item.owner_reference,
            observation=item.observation,
        )
        for item, member in zip(items, members, strict=True)
    )
    return association, associated


def verify_associated_owner_observations(
    association: CommittedCutAssociation,
    projections: Iterable[AssociatedOwnerObservation],
) -> bool:
    """Consumer-side fail-closed verification of membership in one value.

    This verifies generic claim/provenance fields against the bound member and
    cut. Owner-specific projection/claim identifiers remain opaque: World can
    require them and preserve them, but cannot verify their owner semantics.
    This does not prove completeness, source truth, or semantic compatibility.
    """

    if not isinstance(association, CommittedCutAssociation):
        return False
    items = tuple(projections)
    if (
        association.kind != "world.committed_cut_association"
        or association.contract_revision != "candidate-1"
        or association.issuer != "world"
        or not isinstance(association.world_id, str)
        or not association.world_id.strip()
        or not isinstance(association.commit_id, str)
        or not association.commit_id.strip()
        or type(association.causal_tick) is not int
        or association.causal_tick < 0
        or not association.members
        or not items
        or any(not _valid_member(member) for member in association.members)
        or tuple(sorted(association.members)) != association.members
        or len({member.projection_identity for member in association.members})
        != len(association.members)
    ):
        return False
    seen: set[OwnerProjectionIdentity] = set()
    for item in items:
        if not isinstance(item, AssociatedOwnerObservation) or item.association != association:
            return False
        member = item.member
        if (
            association.members.count(member) != 1
            or item.projection_identity != member.projection_identity
            or member.projection_identity in seen
        ):
            return False
        reference = item.owner_reference
        if (
            member.projection_identity.owner_id != reference.owner
            or member.projection_identity.owner_identity != reference.owner_identity
            or member.checkpoint_id != reference.checkpoint_id
            or member.checkpoint_schema_version != reference.schema_version
            or member.checkpoint_sha256 != reference.sha256
            or not _matches_associated_observation(item.observation, association, member)
        ):
            return False
        seen.add(member.projection_identity)
    return True


def _valid_projection_identity(identity: object) -> bool:
    if not isinstance(identity, OwnerProjectionIdentity):
        return False
    return all(
        isinstance(value, str) and bool(value.strip())
        for value in (
            identity.owner_id,
            identity.owner_identity,
            identity.family_id,
            identity.owner_contract_id,
            identity.owner_contract_revision,
            identity.projection_revision,
            identity.claim_identity,
        )
    )


def _valid_member(member: object) -> bool:
    if not isinstance(member, CommittedCutMember) or not _valid_projection_identity(
        member.projection_identity
    ):
        return False
    return (
        isinstance(member.checkpoint_id, str)
        and bool(member.checkpoint_id.strip())
        and isinstance(member.checkpoint_schema_version, str)
        and bool(member.checkpoint_schema_version.strip())
        and isinstance(member.checkpoint_sha256, str)
        and len(member.checkpoint_sha256) == 64
        and all(char in "0123456789abcdef" for char in member.checkpoint_sha256)
    )


def _matches_associated_observation(
    observation: OwnerObservationLike,
    association: CommittedCutAssociation,
    member: CommittedCutMember,
) -> bool:
    """Check all generic owner claim/correlation links exposed by Observation."""
    identity = member.projection_identity
    try:
        claim = observation.claim
        authority = getattr(getattr(claim, "authority", None), "value", None)
        family = getattr(claim, "family", None)
        contract = getattr(family, "contract", None)
        if (
            authority != identity.owner_id
            or getattr(family, "qualified_name", None) != identity.family_id
            or getattr(contract, "contract_id", None) != identity.owner_contract_id
            or getattr(contract, "revision", None) != identity.owner_contract_revision
        ):
            return False

        parents = getattr(observation.lineage, "parents", ())
        if not any(
            getattr(parent, "reference", None) == member.checkpoint_sha256
            for parent in parents
        ):
            return False

        if not any(
            getattr(getattr(time_ref, "domain", None), "value", None) == "world"
            and getattr(time_ref, "kind", None)
            in {"world_target_causal_tick", "world_causal_tick"}
            and getattr(time_ref, "value", None) == association.causal_tick
            for time_ref in observation.context_time
        ):
            return False

        transformation = getattr(observation.lineage, "transformation", None)
        configuration = dict(getattr(transformation, "effective_configuration", ()))
        return configuration.get("world_id") == association.world_id
    except (AttributeError, TypeError, ValueError):
        return False


@dataclass(frozen=True, slots=True)
class ObservationPublication:
    """World verification receipt plus the Observation delivery sequences."""

    committed_cut: CommittedWorldCut
    delivery_sequences: tuple[int, ...]
    complete: bool = True
    failed_index: int | None = None
    delivery_status_uncertain: bool = False
    failure: str | None = None


class ObservationPublicationFailure(RuntimeError):
    """A sequential delivery failed after zero or more acknowledged messages.

    The failing call's side effect cannot be inferred from its exception, so
    ``delivery_status_uncertain`` is true whenever a publish call was entered.
    Earlier returned sequence numbers are in-process acknowledgements only.
    """

    def __init__(
        self,
        *,
        committed_cut: CommittedWorldCut,
        delivery_sequences: tuple[int, ...],
        failed_index: int,
        failure: str,
    ) -> None:
        super().__init__(failure)
        self.committed_cut = committed_cut
        self.delivery_sequences = delivery_sequences
        self.failed_index = failed_index
        self.delivery_status_uncertain = True
        self.failure = failure


def publish_committed_owner_observations(
    world: DurableWorldRuntime,
    plane: ObservationPlaneLike,
    *,
    committed_cut: CommittedWorldCut,
    owner_observations: Iterable[OwnerObservation],
) -> ObservationPublication:
    """Publish owner projections only after re-verifying their exact World cut.

    The function validates every item before publishing any item. World checks
    only generic authority/correlation/provenance links and never interprets
    the claim value or reads owner checkpoint bytes.
    """

    if not isinstance(committed_cut, CommittedWorldCut):
        raise TypeError("publication requires a World-issued committed-cut receipt")
    verified = _verify_committed_cut(world, committed_cut)

    items = tuple(owner_observations)
    if not items:
        raise ValueError("at least one owner observation is required for publication")
    for item in items:
        _validate_owner_observation(verified, item)

    # All contract failures are detected above before the first publication.
    # Delivery remains the Observation Plane's in-process, transport-neutral
    # operation; this API does not claim transactional multi-message delivery.
    sequences: list[int] = []
    for index, item in enumerate(items):
        try:
            sequences.append(plane.publish(item.observation))
        except Exception as exc:
            raise ObservationPublicationFailure(
                committed_cut=verified,
                delivery_sequences=tuple(sequences),
                failed_index=index,
                failure=f"{type(exc).__name__}: {exc}",
            ) from exc
    return ObservationPublication(verified, tuple(sequences))


def _verify_committed_cut(
    world: DurableWorldRuntime, committed_cut: CommittedWorldCut
) -> CommittedWorldCut:
    if not isinstance(committed_cut, CommittedWorldCut):
        raise TypeError("operation requires a World-issued committed-cut receipt")
    verified = world.committed_cut_for(
        world_id=committed_cut.world_id,
        causal_tick=committed_cut.causal_tick,
        commit_id=committed_cut.commit_id,
        expected_owner_references=committed_cut.owner_checkpoint_references,
    )
    if verified != committed_cut:
        raise ManifestError("committed-cut receipt differs from durable World history")
    return verified


def _validate_owner_observation(cut: CommittedWorldCut, item: OwnerObservation) -> None:
    if not isinstance(item, OwnerObservation):
        raise TypeError("publication items must be OwnerObservation values")
    reference = item.owner_reference
    if reference not in cut.owner_checkpoint_references:
        raise ManifestError(
            "observation owner reference is not in the committed World cut"
        )

    observation = item.observation
    authority = getattr(observation.claim, "authority", None)
    authority_name = getattr(authority, "value", None)
    if authority_name != reference.owner:
        raise ManifestError("observation authority does not match its committed owner")

    parents = getattr(observation.lineage, "parents", ())
    if not any(
        # The namespace is owned by the semantic source family (for example,
        # ``embodiment_state_sha256``). World verifies the opaque digest link
        # to its committed checkpoint without rewriting that domain lineage.
        getattr(parent, "reference", None) == reference.sha256
        for parent in parents
    ):
        raise ManifestError(
            "observation provenance does not reference the committed owner checkpoint"
        )

    context_time = getattr(observation, "context_time", ())
    if not any(
        getattr(getattr(time_ref, "domain", None), "value", None) == "world"
        and getattr(time_ref, "kind", None)
        in {"world_target_causal_tick", "world_causal_tick"}
        and getattr(time_ref, "value", None) == cut.causal_tick
        for time_ref in context_time
    ):
        raise ManifestError(
            "observation target causal tick does not match the committed World cut"
        )

    lineage = observation.lineage
    transformation = getattr(lineage, "transformation", None)
    configuration = dict(getattr(transformation, "effective_configuration", ()))
    if configuration.get("world_id") != cut.world_id:
        raise ManifestError(
            "observation provenance does not identify the committed World"
        )


__all__ = [
    "AssociatedOwnerObservation",
    "CommittedCutAssociation",
    "CommittedCutMember",
    "ObservationPublication",
    "OwnerObservation",
    "OwnerProjectionIdentity",
    "associate_committed_owner_observations",
    "publish_committed_owner_observations",
    "verify_associated_owner_observations",
]
