"""Transient World binding for Embodiment-owned render pose candidates.

This module does not publish, store or interpret geometry. It links a
solver-produced candidate to the exact Embodiment checkpoint and committed
World cut that accepted the corresponding physical transition.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from .durable_runtime import CommittedWorldCut, DurableWorldRuntime
from .manifest import CheckpointReference, ManifestError


@dataclass(frozen=True, slots=True)
class CommittedRenderPoseProjection:
    """A transient candidate plus exact cut/checkpoint and canonical digest.

    The digest covers only the render pose candidate bytes; it is not added to
    the World journal or checkpoint and makes no claim that a consumer has
    verified the physical state by independently decoding the checkpoint.
    """

    committed_cut: CommittedWorldCut
    owner_checkpoint_reference: CheckpointReference
    projection_sha256: str
    candidate: object


def associate_render_pose_candidate(
    world: DurableWorldRuntime,
    committed_cut: CommittedWorldCut,
    owner_reference: CheckpointReference,
    candidate: object,
) -> CommittedRenderPoseProjection:
    """Verify one Embodiment pose candidate against a current committed cut."""
    if not isinstance(committed_cut, CommittedWorldCut):
        raise TypeError("render projection requires a World-issued committed-cut receipt")
    verified = world.committed_cut_for(
        world_id=committed_cut.world_id,
        causal_tick=committed_cut.causal_tick,
        commit_id=committed_cut.commit_id,
        expected_owner_references=committed_cut.owner_checkpoint_references,
    )
    if verified != committed_cut:
        raise ManifestError("committed-cut receipt differs from active World cut")
    if not isinstance(owner_reference, CheckpointReference):
        raise TypeError("render projection requires an exact Embodiment checkpoint reference")
    matching = tuple(
        reference
        for reference in verified.owner_checkpoint_references
        if reference.owner == "embodiment"
        and reference.owner_identity == owner_reference.owner_identity
    )
    if len(matching) != 1 or matching[0] != owner_reference:
        raise ManifestError("render projection checkpoint is not the exact Embodiment member of this cut")
    for field, expected in (
        ("source_world_id", verified.world_id),
        ("source_causal_tick", verified.causal_tick - 1),
        ("target_causal_tick", verified.causal_tick),
        ("body_id", owner_reference.owner_identity),
    ):
        if getattr(candidate, field, None) != expected:
            raise ManifestError(f"render pose candidate {field} differs from committed cut/checkpoint")
    canonical_bytes = getattr(candidate, "canonical_bytes", None)
    if not callable(canonical_bytes):
        raise TypeError("render pose candidate must provide its owner-defined canonical bytes")
    encoded = canonical_bytes()
    if not isinstance(encoded, bytes) or not encoded:
        raise ValueError("render pose candidate canonical bytes must be non-empty")
    return CommittedRenderPoseProjection(
        committed_cut=verified,
        owner_checkpoint_reference=owner_reference,
        projection_sha256=hashlib.sha256(encoded).hexdigest(),
        candidate=candidate,
    )


def verify_committed_render_pose_projection(
    world: DurableWorldRuntime, projection: CommittedRenderPoseProjection
) -> bool:
    """Re-verify candidate digest, exact checkpoint membership and current cut."""
    if not isinstance(projection, CommittedRenderPoseProjection):
        raise TypeError("projection must be a CommittedRenderPoseProjection")
    verified = associate_render_pose_candidate(
        world,
        projection.committed_cut,
        projection.owner_checkpoint_reference,
        projection.candidate,
    )
    if verified.projection_sha256 != projection.projection_sha256:
        raise ManifestError("render pose candidate digest does not match its committed association")
    return True

