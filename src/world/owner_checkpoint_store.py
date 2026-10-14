"""Opaque, immutable owner-checkpoint storage for World causal cuts.

World stores exact owner-produced bytes and records their identities and hashes.
It does not decode checkpoint payloads or decide whether they are valid; those
checks remain with each owner's registered checkpoint contract.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol, runtime_checkable

from .manifest import CheckpointReference
from .physical_step import PhysicalStepContext


@dataclass(frozen=True, slots=True)
class OwnerCheckpointPayload:
    """Exact checkpoint bytes supplied by one semantic owner."""

    owner: str
    owner_identity: str
    schema_version: str
    encoded: bytes

    def __post_init__(self) -> None:
        for name in ("owner", "owner_identity", "schema_version"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if not isinstance(self.encoded, bytes) or not self.encoded:
            raise ValueError("encoded owner checkpoint must be non-empty bytes")


@runtime_checkable
class OwnerCheckpointStore(Protocol):
    """Durable byte-store port; it does not confer checkpoint ownership."""

    def write(
        self,
        context: PhysicalStepContext,
        payloads: tuple[OwnerCheckpointPayload, ...],
    ) -> tuple[CheckpointReference, ...]: ...


class FileOwnerCheckpointStore:
    """Append immutable content-addressed checkpoint blobs on the local store.

    Each new blob is fsynced and linked into its final content-addressed path
    before a reference is returned. Existing bytes are never replaced. This is
    a local-filesystem implementation; it makes no cross-filesystem or
    distributed durability guarantee.
    """

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root).resolve()

    def write(
        self,
        context: PhysicalStepContext,
        payloads: tuple[OwnerCheckpointPayload, ...],
    ) -> tuple[CheckpointReference, ...]:
        if not isinstance(context, PhysicalStepContext):
            raise TypeError("context must be a World-issued PhysicalStepContext")
        payloads = tuple(payloads)
        keys = tuple((item.owner, item.owner_identity) for item in payloads)
        if not payloads or len(keys) != len(set(keys)):
            raise ValueError("checkpoint payloads must be non-empty and uniquely identified")

        return self._write_payloads(
            payloads,
            checkpoint_id=lambda payload, digest: (
                f"{context.target_causal_tick}:{digest[:24]}"
            ),
        )

    def write_genesis(
        self,
        payloads: tuple[OwnerCheckpointPayload, ...],
    ) -> tuple[CheckpointReference, ...]:
        """Persist immutable owner-authored bytes referenced by an initial cut.

        Genesis has no World-issued physical-step context. This narrow method
        therefore records content-addressed owner checkpoints without inventing
        a transition identity; World genesis names them as initial references.
        """
        payloads = tuple(payloads)
        keys = tuple((item.owner, item.owner_identity) for item in payloads)
        if not payloads or len(keys) != len(set(keys)):
            raise ValueError("genesis checkpoint payloads must be non-empty and uniquely identified")
        if any(not isinstance(item, OwnerCheckpointPayload) for item in payloads):
            raise TypeError("genesis payloads must be OwnerCheckpointPayload values")
        return self._write_payloads(
            payloads,
            checkpoint_id=lambda payload, digest: f"genesis:{digest[:24]}",
        )

    def _write_payloads(
        self,
        payloads: tuple[OwnerCheckpointPayload, ...],
        *,
        checkpoint_id: Callable[[OwnerCheckpointPayload, str], str],
    ) -> tuple[CheckpointReference, ...]:
        references: list[CheckpointReference] = []
        for payload in payloads:
            if not isinstance(payload, OwnerCheckpointPayload):
                raise TypeError("payloads must be OwnerCheckpointPayload values")
            digest = hashlib.sha256(payload.encoded).hexdigest()
            target = self.root / digest[:2] / f"{digest}.blob"
            self._publish_blob(target, payload.encoded, digest)
            references.append(
                CheckpointReference(
                    owner=payload.owner,
                    owner_identity=payload.owner_identity,
                    checkpoint_id=checkpoint_id(payload, digest),
                    schema_version=payload.schema_version,
                    locator=str(target),
                    sha256=digest,
                )
            )
        return tuple(references)

    def _publish_blob(self, target: Path, encoded: bytes, digest: str) -> None:
        _mkdir_durable(target.parent)
        if target.exists():
            if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
                raise OSError("content-addressed checkpoint path contains different bytes")
            return

        fd, temporary_name = tempfile.mkstemp(prefix=".checkpoint-", dir=target.parent)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temporary, target)
            except FileExistsError:
                if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
                    raise OSError("content-addressed checkpoint path contains different bytes")
            _fsync_directory(target.parent)
        finally:
            temporary.unlink(missing_ok=True)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _mkdir_durable(path: Path) -> None:
    """Create missing directory entries and sync each containing directory."""
    missing: list[Path] = []
    current = path
    while not current.exists():
        missing.append(current)
        current = current.parent
    for directory in reversed(missing):
        directory.mkdir()
        _fsync_directory(directory.parent)


__all__ = ["FileOwnerCheckpointStore", "OwnerCheckpointPayload", "OwnerCheckpointStore"]
