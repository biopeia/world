"""Durable World causal-cut manifest publication primitives.

The manifest records opaque owner checkpoint references. It coordinates a
causal cut without decoding or taking semantic ownership of referenced state.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SCHEMA_VERSION = 1


class ManifestStatus(str, Enum):
    COMPLETE = "COMPLETE"
    RECOVERY_REQUIRED = "RECOVERY_REQUIRED"


class ManifestError(RuntimeError):
    """Base error for invalid or unavailable causal-cut state."""


class RecoveryRequiredError(ManifestError):
    """Raised when a pending causal transition prevents ordinary execution."""


@dataclass(frozen=True, slots=True)
class CheckpointReference:
    """Opaque reference to one exact, durable owner checkpoint."""

    owner: str
    owner_identity: str
    checkpoint_id: str
    schema_version: str
    locator: str
    sha256: str

    def __post_init__(self) -> None:
        for name in ("owner", "owner_identity", "checkpoint_id", "schema_version", "locator"):
            value = getattr(self, name)
            if not value or not value.strip():
                raise ValueError(f"{name} must not be empty")
        if not _SHA256.fullmatch(self.sha256):
            raise ValueError("checkpoint sha256 must be 64 lowercase hexadecimal characters")


@dataclass(frozen=True, slots=True)
class WorldCausalManifest:
    """World-owned description of a committed cut or pending transition."""

    world_id: str
    causal_tick: int
    checkpoint_references: tuple[CheckpointReference, ...]
    journal_last_sequence: int | None
    journal_prefix_sha256: str
    status: ManifestStatus = ManifestStatus.COMPLETE
    pending_transition_id: str | None = None
    pending_tick: int | None = None
    failure: str | None = None
    schema_version: int = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.world_id or not self.world_id.strip():
            raise ValueError("world_id must not be empty")
        if self.causal_tick < 0:
            raise ValueError("causal_tick must be non-negative")
        if self.schema_version != _SCHEMA_VERSION:
            raise ValueError(f"unsupported manifest schema version: {self.schema_version}")
        if self.journal_last_sequence is not None and self.journal_last_sequence < 0:
            raise ValueError("journal_last_sequence must be non-negative or null")
        if self.journal_last_sequence is None:
            if self.journal_prefix_sha256 != "":
                raise ValueError("empty journal must use an empty prefix digest")
        elif not _SHA256.fullmatch(self.journal_prefix_sha256):
            raise ValueError("non-empty journal must have a SHA-256 prefix digest")
        if self.status is ManifestStatus.COMPLETE:
            if self.pending_transition_id is not None or self.pending_tick is not None or self.failure is not None:
                raise ValueError("complete manifest cannot contain pending transition details")
        else:
            if not self.pending_transition_id or self.pending_tick != self.causal_tick + 1:
                raise ValueError("recovery manifest must identify the next pending causal tick")
        refs = tuple(self.checkpoint_references)
        keys = [(ref.owner, ref.owner_identity) for ref in refs]
        if len(keys) != len(set(keys)):
            raise ValueError("manifest cannot contain duplicate owner identity references")
        object.__setattr__(self, "checkpoint_references", refs)


DependencyVerifier = Callable[[tuple[CheckpointReference, ...]], bool]


class WorldCausalManifestStore:
    """Publish manifests using same-directory atomic replacement and fsync.

    Call ``begin_transition`` and wait for it to return before mutating causal
    state. It durably records RECOVERY_REQUIRED first. A successful
    ``commit_transition`` is the only operation that clears that state.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def initialize(self, manifest: WorldCausalManifest) -> None:
        if manifest.status is not ManifestStatus.COMPLETE or manifest.causal_tick != 0:
            raise ValueError("initial manifest must be a complete tick-zero causal cut")
        if self.path.exists():
            raise FileExistsError(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._write(manifest)

    def read(self) -> WorldCausalManifest:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return _manifest_from_data(data)
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ManifestError(f"cannot read valid World causal manifest at {self.path}") from exc

    def require_active(self) -> WorldCausalManifest:
        manifest = self.read()
        if manifest.status is ManifestStatus.RECOVERY_REQUIRED:
            raise RecoveryRequiredError(
                f"World requires explicit recovery for transition {manifest.pending_transition_id}"
            )
        return manifest

    def begin_transition(self, transition_id: str) -> WorldCausalManifest:
        if not transition_id or not transition_id.strip():
            raise ValueError("transition_id must not be empty")
        current = self.require_active()
        pending = WorldCausalManifest(
            world_id=current.world_id,
            causal_tick=current.causal_tick,
            checkpoint_references=current.checkpoint_references,
            journal_last_sequence=current.journal_last_sequence,
            journal_prefix_sha256=current.journal_prefix_sha256,
            status=ManifestStatus.RECOVERY_REQUIRED,
            pending_transition_id=transition_id,
            pending_tick=current.causal_tick + 1,
        )
        self._write(pending)
        return pending

    def record_failure(self, transition_id: str, failure: str) -> WorldCausalManifest:
        pending = self.read()
        if pending.status is not ManifestStatus.RECOVERY_REQUIRED:
            raise ValueError("there is no pending transition to mark failed")
        if pending.pending_transition_id != transition_id:
            raise ValueError("failure transition_id does not match pending transition")
        failed = WorldCausalManifest(
            world_id=pending.world_id,
            causal_tick=pending.causal_tick,
            checkpoint_references=pending.checkpoint_references,
            journal_last_sequence=pending.journal_last_sequence,
            journal_prefix_sha256=pending.journal_prefix_sha256,
            status=ManifestStatus.RECOVERY_REQUIRED,
            pending_transition_id=transition_id,
            pending_tick=pending.pending_tick,
            failure=failure or "unspecified transition failure",
        )
        self._write(failed)
        return failed

    def commit_transition(
        self,
        manifest: WorldCausalManifest,
        *,
        dependency_verifier: DependencyVerifier,
    ) -> WorldCausalManifest:
        pending = self.read()
        if pending.status is not ManifestStatus.RECOVERY_REQUIRED:
            raise ValueError("commit requires a durable pending transition")
        if manifest.status is not ManifestStatus.COMPLETE:
            raise ValueError("committed causal cut must have COMPLETE status")
        if (manifest.world_id, manifest.causal_tick) != (pending.world_id, pending.pending_tick):
            raise ValueError("committed causal cut does not match the pending World transition")
        if not dependency_verifier(manifest.checkpoint_references):
            raise ManifestError("causal cut checkpoint dependency closure is not verified")
        self._write(manifest)
        return manifest

    def _write(self, manifest: WorldCausalManifest) -> None:
        encoded = _canonical_bytes(_manifest_to_data(manifest))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(prefix=f".{self.path.name}.", dir=self.path.parent)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_name, self.path)
            _fsync_directory(self.path.parent)
        finally:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass


def _canonical_bytes(data: dict) -> bytes:
    return (json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _manifest_to_data(manifest: WorldCausalManifest) -> dict:
    return {
        "schema_version": manifest.schema_version,
        "world_id": manifest.world_id,
        "causal_tick": manifest.causal_tick,
        "checkpoint_references": [
            {
                "owner": ref.owner,
                "owner_identity": ref.owner_identity,
                "checkpoint_id": ref.checkpoint_id,
                "schema_version": ref.schema_version,
                "locator": ref.locator,
                "sha256": ref.sha256,
            }
            for ref in manifest.checkpoint_references
        ],
        "journal_last_sequence": manifest.journal_last_sequence,
        "journal_prefix_sha256": manifest.journal_prefix_sha256,
        "status": manifest.status.value,
        "pending_transition_id": manifest.pending_transition_id,
        "pending_tick": manifest.pending_tick,
        "failure": manifest.failure,
    }


def _manifest_from_data(data: dict) -> WorldCausalManifest:
    expected = {
        "schema_version", "world_id", "causal_tick", "checkpoint_references",
        "journal_last_sequence", "journal_prefix_sha256", "status",
        "pending_transition_id", "pending_tick", "failure",
    }
    if not isinstance(data, dict) or set(data) != expected:
        raise ValueError("manifest fields do not match schema")
    references = data["checkpoint_references"]
    if not isinstance(references, list):
        raise ValueError("checkpoint_references must be a list")
    refs = []
    ref_keys = {"owner", "owner_identity", "checkpoint_id", "schema_version", "locator", "sha256"}
    for ref in references:
        if not isinstance(ref, dict) or set(ref) != ref_keys:
            raise ValueError("checkpoint reference fields do not match schema")
        refs.append(CheckpointReference(**ref))
    return WorldCausalManifest(
        world_id=data["world_id"],
        causal_tick=data["causal_tick"],
        checkpoint_references=tuple(refs),
        journal_last_sequence=data["journal_last_sequence"],
        journal_prefix_sha256=data["journal_prefix_sha256"],
        status=ManifestStatus(data["status"]),
        pending_transition_id=data["pending_transition_id"],
        pending_tick=data["pending_tick"],
        failure=data["failure"],
        schema_version=data["schema_version"],
    )


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
