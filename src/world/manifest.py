"""Durable World causal-cut manifest publication primitives.

The manifest records opaque owner checkpoint references. It coordinates a
causal cut without decoding or taking semantic ownership of referenced state.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from contextlib import contextmanager
from typing import Callable


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SCHEMA_VERSION = 3
_JOURNAL_OWNER = "world-journal"


class ManifestStatus(str, Enum):
    PREPARING = "PREPARING"
    COMMITTED = "COMMITTED"
    RECOVERY_REQUIRED = "RECOVERY_REQUIRED"


class ManifestError(RuntimeError):
    """Base error for invalid or unavailable causal-cut state."""


class RecoveryRequiredError(ManifestError):
    """Raised when a pending causal transition prevents ordinary execution."""


@dataclass(frozen=True, slots=True)
class CheckpointReference:
    """Opaque reference to one owner checkpoint identified by its owner contract."""

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
class OwnerRecoveryReceipt:
    """Trusted owner-component attestation bound to one exact checkpoint ref."""

    owner: str
    owner_identity: str
    checkpoint_id: str
    schema_version: str
    locator: str
    sha256: str
    recoverable: bool

    def matches(self, reference: CheckpointReference) -> bool:
        return self == OwnerRecoveryReceipt(
            owner=reference.owner,
            owner_identity=reference.owner_identity,
            checkpoint_id=reference.checkpoint_id,
            schema_version=reference.schema_version,
            locator=reference.locator,
            sha256=reference.sha256,
            recoverable=True,
        )


@dataclass(frozen=True, slots=True)
class WorldCausalManifest:
    """World-owned description of a committed cut or pending transition."""

    world_id: str
    causal_tick: int
    presence_ids: tuple[str, ...]
    checkpoint_references: tuple[CheckpointReference, ...]
    journal_last_sequence: int | None
    journal_prefix_sha256: str
    commit_id: str = ""
    registered_owner_ids: tuple[tuple[str, str], ...] = ()
    step_participant_ids: tuple[tuple[str, str], ...] = ()
    pending_registered_owner_ids: tuple[tuple[str, str], ...] = ()
    pending_step_participant_ids: tuple[tuple[str, str], ...] = ()
    pending_presence_ids: tuple[str, ...] = ()
    status: ManifestStatus = ManifestStatus.COMMITTED
    pending_transition_id: str | None = None
    pending_tick: int | None = None
    failure: str | None = None
    schema_version: int = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.world_id or not self.world_id.strip():
            raise ValueError("world_id must not be empty")
        if self.causal_tick < 0:
            raise ValueError("causal_tick must be non-negative")
        presence_ids = tuple(self.presence_ids)
        if any(not presence_id or not presence_id.strip() for presence_id in presence_ids):
            raise ValueError("presence ids must not be empty")
        if len(presence_ids) != len(set(presence_ids)):
            raise ValueError("World manifest cannot contain duplicate presence ids")
        if self.schema_version != _SCHEMA_VERSION:
            raise ValueError(f"unsupported manifest schema version: {self.schema_version}")
        if self.journal_last_sequence is not None and self.journal_last_sequence < 0:
            raise ValueError("journal_last_sequence must be non-negative or null")
        if self.journal_last_sequence is None:
            if self.journal_prefix_sha256 != "":
                raise ValueError("empty journal must use an empty prefix digest")
        elif not _SHA256.fullmatch(self.journal_prefix_sha256):
            raise ValueError("non-empty journal must have a SHA-256 prefix digest")
        if self.status is ManifestStatus.COMMITTED:
            if not self.commit_id or not self.commit_id.strip():
                raise ValueError("committed manifest must identify its commit")
            if self.pending_transition_id is not None or self.pending_tick is not None or self.failure is not None:
                raise ValueError("committed manifest cannot contain pending transition details")
            if self.pending_registered_owner_ids or self.pending_step_participant_ids or self.pending_presence_ids:
                raise ValueError("committed manifest cannot contain pending participant rosters")
        else:
            if not self.pending_transition_id or self.pending_tick != self.causal_tick + 1:
                raise ValueError("pending manifest must identify the next causal tick")
        refs = tuple(self.checkpoint_references)
        keys = [(ref.owner, ref.owner_identity) for ref in refs]
        if len(keys) != len(set(keys)):
            raise ValueError("manifest cannot contain duplicate owner identity references")
        registered = tuple(tuple(pair) for pair in self.registered_owner_ids)
        step = tuple(tuple(pair) for pair in self.step_participant_ids)
        pending_registered = tuple(tuple(pair) for pair in self.pending_registered_owner_ids)
        pending_step = tuple(tuple(pair) for pair in self.pending_step_participant_ids)
        pending_presence = tuple(self.pending_presence_ids)
        if any(len(pair) != 2 or any(not value or not value.strip() for value in pair) for pair in (*registered, *step, *pending_registered, *pending_step)):
            raise ValueError("participant owner identities must contain non-empty owner and identity")
        if any(len(set(group)) != len(group) for group in (registered, step, pending_registered, pending_step)):
            raise ValueError("participant owner identities cannot be duplicated")
        if set(registered) & set(step) or set(pending_registered) & set(pending_step):
            raise ValueError("step-only participants cannot also be registered owners")
        if len(pending_presence) != len(set(pending_presence)) or any(not item or not item.strip() for item in pending_presence):
            raise ValueError("pending presence roster must contain unique non-empty identities")
        expected = set(registered) | set(step)
        actual = {(ref.owner, ref.owner_identity) for ref in refs if ref.owner != _JOURNAL_OWNER}
        if actual != expected:
            raise ValueError("owner checkpoint references must exactly match the World participant roster")
        object.__setattr__(self, "checkpoint_references", refs)
        object.__setattr__(self, "presence_ids", presence_ids)
        object.__setattr__(self, "registered_owner_ids", registered)
        object.__setattr__(self, "step_participant_ids", step)
        object.__setattr__(self, "pending_registered_owner_ids", pending_registered)
        object.__setattr__(self, "pending_step_participant_ids", pending_step)
        object.__setattr__(self, "pending_presence_ids", pending_presence)


DependencyVerifier = Callable[[tuple[CheckpointReference, ...]], bool]


class WorldCausalManifestStore:
    """Internal persistence backend for :class:`DurableWorldRuntime`.

    Public API is intentionally read-only. The underscored mutation methods
    are implementation details and do not provide an independent commit
    authority. They are not a security boundary against arbitrary trusted
    in-process code or direct filesystem access.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def _initialize(self, manifest: WorldCausalManifest) -> None:
        if manifest.status is not ManifestStatus.COMMITTED or manifest.causal_tick != 0:
            raise ValueError("initial manifest must be a committed tick-zero causal cut")
        _require_owner_state(manifest)
        with self._writer_lock():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            encoded = _canonical_bytes(_manifest_to_data(manifest))
            fd, temp_name = tempfile.mkstemp(prefix=f".{self.path.name}.", dir=self.path.parent)
            try:
                with os.fdopen(fd, "wb") as stream:
                    stream.write(encoded)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.link(temp_name, self.path)
                _fsync_directory(self.path.parent)
            finally:
                try:
                    os.unlink(temp_name)
                except FileNotFoundError:
                    pass

    def read(self) -> WorldCausalManifest:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return _manifest_from_data(data)
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ManifestError(f"cannot read valid World causal manifest at {self.path}") from exc

    def _require_active(self) -> WorldCausalManifest:
        with self._writer_lock():
            manifest = self.read()
            if manifest.status is ManifestStatus.PREPARING:
                manifest = WorldCausalManifest(
                    world_id=manifest.world_id,
                    causal_tick=manifest.causal_tick,
                    presence_ids=manifest.presence_ids,
                    checkpoint_references=manifest.checkpoint_references,
                    journal_last_sequence=manifest.journal_last_sequence,
                    journal_prefix_sha256=manifest.journal_prefix_sha256,
                    commit_id=manifest.commit_id,
                    registered_owner_ids=manifest.registered_owner_ids,
                    step_participant_ids=manifest.step_participant_ids,
                    pending_registered_owner_ids=manifest.pending_registered_owner_ids,
                    pending_step_participant_ids=manifest.pending_step_participant_ids,
                    pending_presence_ids=manifest.pending_presence_ids,
                    status=ManifestStatus.RECOVERY_REQUIRED,
                    pending_transition_id=manifest.pending_transition_id,
                    pending_tick=manifest.pending_tick,
                    failure="startup found an interrupted PREPARING causal cut",
                )
                self._write(manifest)
            if manifest.status is ManifestStatus.RECOVERY_REQUIRED:
                raise RecoveryRequiredError(
                    f"World requires explicit recovery for transition {manifest.pending_transition_id}"
                )
            return manifest

    def _begin_transition(
        self,
        transition_id: str,
        *,
        registered_owner_ids: tuple[tuple[str, str], ...] | None = None,
        step_participant_ids: tuple[tuple[str, str], ...] = (),
        presence_ids: tuple[str, ...] | None = None,
    ) -> WorldCausalManifest:
        if not transition_id or not transition_id.strip():
            raise ValueError("transition_id must not be empty")
        with self._writer_lock():
            current = self.read()
            if current.status is not ManifestStatus.COMMITTED:
                raise RecoveryRequiredError(
                    f"World requires recovery for transition {current.pending_transition_id}"
                )
            pending = WorldCausalManifest(
                world_id=current.world_id,
                causal_tick=current.causal_tick,
                presence_ids=current.presence_ids,
                checkpoint_references=current.checkpoint_references,
                journal_last_sequence=current.journal_last_sequence,
                journal_prefix_sha256=current.journal_prefix_sha256,
                commit_id=current.commit_id,
                registered_owner_ids=current.registered_owner_ids,
                step_participant_ids=current.step_participant_ids,
                pending_registered_owner_ids=(
                    current.registered_owner_ids
                    if registered_owner_ids is None
                    else registered_owner_ids
                ),
                pending_step_participant_ids=step_participant_ids,
                pending_presence_ids=current.presence_ids if presence_ids is None else presence_ids,
                status=ManifestStatus.PREPARING,
                pending_transition_id=transition_id,
                pending_tick=current.causal_tick + 1,
            )
            self._write(pending)
            return pending

    def _record_failure(self, transition_id: str, failure: str) -> WorldCausalManifest:
        with self._writer_lock():
            pending = self.read()
            if pending.status not in (ManifestStatus.PREPARING, ManifestStatus.RECOVERY_REQUIRED):
                raise ValueError("there is no pending transition to mark failed")
            if pending.pending_transition_id != transition_id:
                raise ValueError("failure transition_id does not match pending transition")
            failed = WorldCausalManifest(
                world_id=pending.world_id,
                causal_tick=pending.causal_tick,
                presence_ids=pending.presence_ids,
                checkpoint_references=pending.checkpoint_references,
                journal_last_sequence=pending.journal_last_sequence,
                journal_prefix_sha256=pending.journal_prefix_sha256,
                commit_id=pending.commit_id,
                registered_owner_ids=pending.registered_owner_ids,
                step_participant_ids=pending.step_participant_ids,
                pending_registered_owner_ids=pending.pending_registered_owner_ids,
                pending_step_participant_ids=pending.pending_step_participant_ids,
                pending_presence_ids=pending.pending_presence_ids,
                status=ManifestStatus.RECOVERY_REQUIRED,
                pending_transition_id=transition_id,
                pending_tick=pending.pending_tick,
                failure=failure or "unspecified transition failure",
            )
            self._write(failed)
            return failed

    def _commit_transition(
        self,
        manifest: WorldCausalManifest,
        *,
        dependency_verifier: DependencyVerifier,
    ) -> WorldCausalManifest:
        with self._writer_lock():
            pending = self.read()
            if pending.status is not ManifestStatus.PREPARING:
                raise ValueError("commit requires a pending transition in PREPARING state")
            if manifest.status is not ManifestStatus.COMMITTED:
                raise ValueError("committed causal cut must have COMMITTED status")
            if (manifest.world_id, manifest.causal_tick) != (pending.world_id, pending.pending_tick):
                raise ValueError("committed causal cut does not match the pending World transition")
            if manifest.commit_id != pending.pending_transition_id:
                raise ValueError("committed causal cut must use the pending transition as commit_id")
            _require_owner_state(manifest)
            if (manifest.registered_owner_ids, manifest.step_participant_ids) != (
                pending.pending_registered_owner_ids, pending.pending_step_participant_ids
            ):
                raise ManifestError("committed participant roster differs from the World-prepared roster")
            if manifest.presence_ids != pending.pending_presence_ids:
                raise ManifestError("committed presence roster differs from the World-prepared roster")
            if not dependency_verifier(manifest.checkpoint_references):
                raise ManifestError("causal cut checkpoint dependency closure is not verified")
            self._write(manifest)
            return manifest

    def _complete_recovery(self, transition_id: str) -> WorldCausalManifest:
        """Publish the prior coherent cut after caller-validated restoration.

        The runtime calls this only after restoring every non-World checkpoint
        through its semantic owner's recovery contract and verifying references.
        This method records a durable receipt before clearing
        ``RECOVERY_REQUIRED`` so an interrupted publication can be retried
        without losing which transition was recovered.
        """
        if not transition_id or not transition_id.strip():
            raise ValueError("transition_id must not be empty")
        with self._writer_lock():
            return self._complete_recovery_locked(transition_id)

    def _complete_recovery_locked(self, transition_id: str) -> WorldCausalManifest:
        pending = self.read()
        if pending.status is not ManifestStatus.RECOVERY_REQUIRED:
            raise ValueError("recovery requires a durable pending transition")
        if pending.pending_transition_id != transition_id:
            raise ValueError("recovery transition_id does not match pending transition")

        recovered = WorldCausalManifest(
            world_id=pending.world_id,
            causal_tick=pending.causal_tick,
            presence_ids=pending.presence_ids,
            checkpoint_references=pending.checkpoint_references,
            journal_last_sequence=pending.journal_last_sequence,
            journal_prefix_sha256=pending.journal_prefix_sha256,
            commit_id=pending.commit_id,
            registered_owner_ids=pending.registered_owner_ids,
            step_participant_ids=pending.step_participant_ids,
        )
        pending_bytes = _canonical_bytes(_manifest_to_data(pending))
        recovered_bytes = _canonical_bytes(_manifest_to_data(recovered))
        receipt = _canonical_bytes(
            {
                "world_id": pending.world_id,
                "transition_id": transition_id,
                "restored_causal_tick": pending.causal_tick,
                "pending_manifest_sha256": hashlib.sha256(pending_bytes).hexdigest(),
                "recovered_manifest_sha256": hashlib.sha256(recovered_bytes).hexdigest(),
            }
        )
        receipt_dir = self.path.parent / "recoveries"
        receipt_dir.mkdir(parents=True, exist_ok=True)
        # Persist the new directory entry before making a receipt inside it
        # eligible for manifest publication.
        _fsync_directory(receipt_dir.parent)
        receipt_name = hashlib.sha256(transition_id.encode("utf-8")).hexdigest() + ".json"
        _write_immutable(receipt_dir / receipt_name, receipt)
        self._write(recovered)
        return recovered

    def _write(self, manifest: WorldCausalManifest) -> None:
        if manifest.status is ManifestStatus.COMMITTED:
            _require_owner_state(manifest)
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

    @contextmanager
    def _writer_lock(self):
        """Serialize manifest transitions across processes on POSIX filesystems."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.path.with_name(f".{self.path.name}.writer.lock")
        fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)


def _canonical_bytes(data: dict) -> bytes:
    return (json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _require_owner_state(manifest: WorldCausalManifest) -> None:
    """A committed World cut must identify semantic-owner state as well as its journal."""
    if not any(reference.owner != _JOURNAL_OWNER for reference in manifest.checkpoint_references):
        raise ManifestError("committed causal cut must identify at least one durable owner checkpoint reference")


def _owner_keys(references: tuple[CheckpointReference, ...]) -> frozenset[tuple[str, str]]:
    """Return declared semantic-owner identities, excluding World journal state."""
    return frozenset(
        (reference.owner, reference.owner_identity)
        for reference in references
        if reference.owner != _JOURNAL_OWNER
    )


def _manifest_to_data(manifest: WorldCausalManifest) -> dict:
    return {
        "schema_version": manifest.schema_version,
        "world_id": manifest.world_id,
        "causal_tick": manifest.causal_tick,
        "presence_ids": list(manifest.presence_ids),
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
        "commit_id": manifest.commit_id,
        "registered_owner_ids": [list(pair) for pair in manifest.registered_owner_ids],
        "step_participant_ids": [list(pair) for pair in manifest.step_participant_ids],
        "pending_registered_owner_ids": [list(pair) for pair in manifest.pending_registered_owner_ids],
        "pending_step_participant_ids": [list(pair) for pair in manifest.pending_step_participant_ids],
        "pending_presence_ids": list(manifest.pending_presence_ids),
        "status": manifest.status.value,
        "pending_transition_id": manifest.pending_transition_id,
        "pending_tick": manifest.pending_tick,
        "failure": manifest.failure,
    }


def _manifest_from_data(data: dict) -> WorldCausalManifest:
    expected = {
        "schema_version", "world_id", "causal_tick", "presence_ids", "checkpoint_references",
        "journal_last_sequence", "journal_prefix_sha256", "commit_id",
        "registered_owner_ids", "step_participant_ids", "pending_registered_owner_ids", "pending_step_participant_ids", "pending_presence_ids", "status",
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
        presence_ids=tuple(data["presence_ids"]),
        checkpoint_references=tuple(refs),
        journal_last_sequence=data["journal_last_sequence"],
        journal_prefix_sha256=data["journal_prefix_sha256"],
        commit_id=data["commit_id"],
        registered_owner_ids=tuple(tuple(pair) for pair in data["registered_owner_ids"]),
        step_participant_ids=tuple(tuple(pair) for pair in data["step_participant_ids"]),
        pending_registered_owner_ids=tuple(tuple(pair) for pair in data["pending_registered_owner_ids"]),
        pending_step_participant_ids=tuple(tuple(pair) for pair in data["pending_step_participant_ids"]),
        pending_presence_ids=tuple(data["pending_presence_ids"]),
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


def _write_immutable(path: Path, encoded: bytes) -> None:
    """Durably create a content-checked immutable receipt without overwrite."""
    if path.exists():
        if path.read_bytes() != encoded:
            raise ManifestError("existing recovery receipt differs from candidate")
    else:
        fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temp_name, path)
            except FileExistsError:
                if path.read_bytes() != encoded:
                    raise ManifestError("recovery receipt path collision")
        finally:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass

    # This must also run for an identical preexisting receipt: recovery may
    # have been interrupted after receipt creation but before manifest publish.
    _fsync_directory(path.parent)
