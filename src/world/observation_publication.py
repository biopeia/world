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


@dataclass(frozen=True, slots=True)
class ObservationPublication:
    """World verification receipt plus the Observation delivery sequences."""

    committed_cut: CommittedWorldCut
    delivery_sequences: tuple[int, ...]


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
    verified = world.committed_cut_for(
        world_id=committed_cut.world_id,
        causal_tick=committed_cut.causal_tick,
        commit_id=committed_cut.commit_id,
        expected_owner_references=committed_cut.owner_checkpoint_references,
    )
    if verified != committed_cut:
        raise ManifestError("committed-cut receipt differs from durable World history")

    items = tuple(owner_observations)
    if not items:
        raise ValueError("at least one owner observation is required for publication")
    for item in items:
        _validate_owner_observation(verified, item)

    # All contract failures are detected above before the first publication.
    # Delivery remains the Observation Plane's in-process, transport-neutral
    # operation; this API does not claim transactional multi-message delivery.
    sequences = tuple(plane.publish(item.observation) for item in items)
    return ObservationPublication(verified, sequences)


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
    "ObservationPublication",
    "OwnerObservation",
    "publish_committed_owner_observations",
]
