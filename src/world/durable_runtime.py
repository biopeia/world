"""Durable single-World tick publication over opaque owner checkpoints.

This API coordinates publication only. The supplied transition callback must
use owner APIs to persist its new state and return exact references for the
resulting causal cut. World validates references but never decodes owner data.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import fcntl
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from enum import Enum
from math import isfinite
from pathlib import Path
from threading import Lock, get_ident
from types import MappingProxyType
from typing import Protocol

from .journal import CausalRecord, EventJournal, _freeze_json, _thaw_json
from .manifest import (
    CheckpointReference,
    DependencyVerifier,
    ManifestError,
    ManifestStatus,
    SingleWriterPreconditionError,
    WorldCausalManifest,
    WorldCausalManifestStore,
)
from .runtime import PresenceId, RecoveryRequiredError, TickInProgressError, WorldId

_JOURNAL_OWNER = "world-journal"
_JOURNAL_SCHEMA = "event-journal-v3"
_CAUSAL_CUT_RECORD_TYPE = "world.causal_cut.committed"


def _reference_descriptor(reference: CheckpointReference) -> dict[str, str]:
    """Serialize an opaque checkpoint reference without inspecting its bytes."""
    return {
        "owner_id": reference.owner,
        "owner_identity": reference.owner_identity,
        "checkpoint_id": reference.checkpoint_id,
        "schema_version": reference.schema_version,
        "locator": reference.locator,
        "sha256": reference.sha256,
    }


def _causal_cut_record(
    *,
    world_id: str,
    causal_tick: int,
    commit_id: str,
    references: tuple[CheckpointReference, ...],
) -> CausalRecord:
    owner_references = tuple(
        sorted(
            (ref for ref in references if ref.owner != _JOURNAL_OWNER),
            key=lambda ref: (ref.owner, ref.owner_identity),
        )
    )
    participants = tuple(sorted((ref.owner, ref.owner_identity) for ref in owner_references))
    if len(participants) != len(set(participants)):
        raise ManifestError("causal-cut checkpoint references contain duplicate participants")
    return CausalRecord(
        record_id=f"{world_id}:causal-cut:{causal_tick}",
        world_id=world_id,
        causal_tick=causal_tick,
        record_type=_CAUSAL_CUT_RECORD_TYPE,
        actor_id="world",
        payload={
            "commit_id": commit_id,
            "participants": [list(item) for item in participants],
            "owner_checkpoint_references": [
                _reference_descriptor(reference) for reference in owner_references
            ],
        },
    )


def _validate_causal_cut_record(record: CausalRecord) -> None:
    """Fail closed on malformed per-cut roster/reference history."""
    if (
        record.record_type != _CAUSAL_CUT_RECORD_TYPE
        or record.actor_id != "world"
        or record.causal_tick < 0
        or record.record_id != f"{record.world_id}:causal-cut:{record.causal_tick}"
    ):
        raise ManifestError("persisted causal-cut record has an invalid identity")
    payload = record.payload
    if set(payload) != {"commit_id", "participants", "owner_checkpoint_references"}:
        raise ManifestError("persisted causal-cut record has an invalid envelope")
    commit_id = payload["commit_id"]
    participants = payload["participants"]
    references = payload["owner_checkpoint_references"]
    if not isinstance(commit_id, str) or not commit_id.strip():
        raise ManifestError("persisted causal-cut record has an invalid commit identity")
    if not isinstance(participants, tuple) or not isinstance(references, tuple):
        raise ManifestError("persisted causal-cut record has invalid participant references")
    normalized_participants: list[tuple[str, str]] = []
    for participant in participants:
        if (
            not isinstance(participant, tuple)
            or len(participant) != 2
            or any(not isinstance(item, str) or not item.strip() for item in participant)
        ):
            raise ManifestError("persisted causal-cut record has an invalid participant identity")
        normalized_participants.append((participant[0], participant[1]))
    if normalized_participants != sorted(set(normalized_participants)):
        raise ManifestError("persisted causal-cut participants are duplicated or unordered")
    required_reference_fields = {
        "owner_id", "owner_identity", "checkpoint_id", "schema_version", "locator", "sha256"
    }
    normalized_references: list[tuple[str, str]] = []
    for reference in references:
        if not isinstance(reference, Mapping) or set(reference) != required_reference_fields:
            raise ManifestError("persisted causal-cut record has an invalid checkpoint reference")
        values = tuple(reference[field] for field in (
            "owner_id", "owner_identity", "checkpoint_id", "schema_version", "locator", "sha256"
        ))
        if any(not isinstance(value, str) or not value.strip() for value in values):
            raise ManifestError("persisted causal-cut record has an invalid checkpoint reference")
        # Reuse the canonical descriptor validation without changing ownership.
        try:
            CheckpointReference(*values)
        except ValueError as exc:
            raise ManifestError("persisted causal-cut record has an invalid checkpoint reference") from exc
        normalized_references.append((values[0], values[1]))
    if (
        normalized_references != sorted(set(normalized_references))
        or tuple(normalized_references) != tuple(normalized_participants)
    ):
        raise ManifestError("causal-cut participants and checkpoint references do not match exactly")


@dataclass(frozen=True, slots=True)
class PresenceOwnerLink:
    """Identify the one registered causal owner responsible for a presence."""

    presence_id: PresenceId
    owner_id: str
    owner_identity: str

    def __post_init__(self) -> None:
        if not isinstance(self.presence_id, PresenceId):
            raise ValueError("presence linkage requires a typed PresenceId")
        if not self.owner_id or not self.owner_id.strip() or not self.owner_identity or not self.owner_identity.strip():
            raise ValueError("presence linkage owner identity must be non-empty")


@dataclass(frozen=True, slots=True)
class PresenceLifecycleTransition:
    """World-owned lifecycle intent and separately attributed trigger data."""

    transition_type: str
    presence_id: PresenceId
    trigger_provenance: Mapping[str, object]

    def __post_init__(self) -> None:
        if self.transition_type not in {"REGISTER", "TERMINATE"}:
            raise ValueError("presence lifecycle type must be REGISTER or TERMINATE")
        if not isinstance(self.presence_id, PresenceId):
            raise ValueError("presence lifecycle requires a typed PresenceId")
        if not isinstance(self.trigger_provenance, Mapping) or not self.trigger_provenance:
            raise ValueError("presence lifecycle requires explicit trigger provenance")
        _validate_json_object(self.trigger_provenance)
        object.__setattr__(self, "trigger_provenance", _freeze_json(self.trigger_provenance))


@dataclass(frozen=True, slots=True)
class WorldGenesisSpec:
    """Authoritative creation input against which tick-zero closure is checked.

    This is an explicit declaration by the World creator, not a discovery
    mechanism for authorities external to the accepted specification.
    """

    world_id: WorldId
    initial_presence_ids: tuple[str, ...]
    registered_owner_ids: tuple[tuple[str, str], ...]
    initial_owner_references: tuple[CheckpointReference, ...]
    owner_checkpoint_contracts: Mapping[tuple[str, str], OwnerCheckpointContract]
    presence_owner_links: tuple[PresenceOwnerLink, ...] = ()

    def __post_init__(self) -> None:
        presences = tuple(self.initial_presence_ids)
        if not presences or any(not item or not item.strip() for item in presences):
            raise ValueError("World genesis requires at least one non-empty initial presence")
        if len(set(presences)) != len(presences):
            raise ValueError("World genesis cannot contain duplicate presences")
        owners = tuple(tuple(item) for item in self.registered_owner_ids)
        if any(len(item) != 2 or any(not value or not value.strip() for value in item) for item in owners):
            raise ValueError("World genesis owner identities must be non-empty pairs")
        if len(set(owners)) != len(owners):
            raise ValueError("World genesis cannot contain duplicate owner identities")
        object.__setattr__(self, "initial_presence_ids", presences)
        object.__setattr__(self, "registered_owner_ids", tuple(sorted(owners)))
        object.__setattr__(self, "initial_owner_references", tuple(self.initial_owner_references))
        object.__setattr__(self, "owner_checkpoint_contracts", MappingProxyType(dict(self.owner_checkpoint_contracts)))
        links = tuple(self.presence_owner_links)
        link_presence_values = tuple(link.presence_id.value for link in links)
        if set(link_presence_values) != set(presences) or len(link_presence_values) != len(set(link_presence_values)):
            raise ValueError("presence owner links must cover each genesis presence exactly once")
        owner_keys = set(owners)
        if any((link.owner_id, link.owner_identity) not in owner_keys for link in links):
            raise ValueError("presence owner link must target a registered genesis owner")
        object.__setattr__(self, "presence_owner_links", links)


class DurableJournalSnapshotStore:
    """Persist committed journal deltas in one append-oriented World log.

    A reference identifies a committed byte prefix. Appending a later
    candidate never rewrites that prefix, so a failed candidate can be
    discarded by truncating only bytes beyond the last committed reference.
    The reference digest is the EventJournal prefix-chain digest, not a hash
    of the entire physical log. Before extending a committed reference, World
    verifies every record in that committed prefix. This keeps append bytes
    bounded for fixed-size records, while prefix validation costs O(n) reads.
    """

    def __init__(self, directory: str | Path, world_id: WorldId) -> None:
        self.directory = Path(directory)
        self.world_id = world_id.value
        # A live single-writer can reuse a prefix it just verified and wrote.
        # The cache is deliberately process-local: restore/new store instances
        # still perform a complete hash-chain verification before extending.
        # File metadata is only an optimization witness under the declared
        # local-filesystem/single-writer model, not a cryptographic tamper proof.
        self._verified_reference: CheckpointReference | None = None
        self._verified_file_signature: tuple[int, int, int, int, int] | None = None

    @staticmethod
    def _file_signature(path: Path) -> tuple[int, int, int, int, int]:
        stat = path.stat()
        return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)

    def _previous_prefix_is_cached(self, reference: CheckpointReference, path: Path) -> bool:
        if self._verified_reference != reference or self._verified_file_signature is None:
            return False
        try:
            return self._file_signature(path) == self._verified_file_signature
        except OSError:
            return False

    def _cache_verified_prefix(self, reference: CheckpointReference, path: Path) -> None:
        self._verified_reference = reference
        self._verified_file_signature = self._file_signature(path)

    @staticmethod
    def _position(reference: CheckpointReference) -> tuple[int, int, int]:
        match = re.fullmatch(r"journal-cut-(\d+)-sequence-(-?\d+)-offset-(\d+)", reference.checkpoint_id)
        if match is None:
            raise ManifestError("World journal reference has an invalid checkpoint identity")
        tick, sequence, offset = (int(value) for value in match.groups())
        if sequence < -1 or offset < 1:
            raise ManifestError("World journal reference has an invalid prefix position")
        return tick, sequence, offset

    @staticmethod
    def _header(world_id: str) -> bytes:
        return (json.dumps(
            {"schema": _JOURNAL_SCHEMA, "world_id": world_id},
            sort_keys=True,
            separators=(",", ":"),
        ) + "\n").encode("utf-8")

    @staticmethod
    def _record_line(record: CausalRecord, previous: str) -> tuple[bytes, str]:
        encoded = EventJournal._encode(record)
        digest = hashlib.sha256(previous.encode("ascii") + b"\0" + encoded).hexdigest()
        line = (json.dumps(
            {"record": EventJournal._snapshot_record(record), "prefix_sha256": digest},
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ) + "\n").encode("utf-8")
        return line, digest

    @staticmethod
    def _record_from_data(value: object) -> CausalRecord:
        fields = {
            "record_id",
            "world_id",
            "causal_tick",
            "record_type",
            "actor_id",
            "payload",
            "causal_parent_ids",
            "contributing_record_ids",
        }
        if not isinstance(value, dict) or set(value) != fields:
            raise ManifestError("World journal record does not match the closed v3 schema")
        if any(not isinstance(value[key], str) for key in ("record_id", "world_id", "record_type")):
            raise ManifestError("World journal record identity fields must be strings")
        tick = value["causal_tick"]
        if isinstance(tick, bool) or not isinstance(tick, int):
            raise ManifestError("World journal causal_tick must be an integer")
        actor_id = value["actor_id"]
        payload = value["payload"]
        parent_ids = value["causal_parent_ids"]
        contributing_ids = value["contributing_record_ids"]
        if actor_id is not None and not isinstance(actor_id, str):
            raise ManifestError("World journal actor_id must be a string or null")
        if not isinstance(payload, dict):
            raise ManifestError("World journal payload must be an object")
        if not isinstance(parent_ids, list) or not all(isinstance(item, str) for item in parent_ids):
            raise ManifestError("World journal causal_parent_ids must be a string array")
        if not isinstance(contributing_ids, list) or not all(isinstance(item, str) for item in contributing_ids):
            raise ManifestError("World journal contributing_record_ids must be a string array")
        try:
            return CausalRecord(
                record_id=value["record_id"],
                world_id=value["world_id"],
                causal_tick=tick,
                record_type=value["record_type"],
                actor_id=actor_id,
                payload=payload,
                causal_parent_ids=tuple(parent_ids),
                contributing_record_ids=tuple(contributing_ids),
            )
        except (TypeError, ValueError) as exc:
            raise ManifestError("World journal record is invalid") from exc

    @staticmethod
    def _read_line_ending_at(stream, offset: int) -> bytes:
        if offset <= 0:
            raise ManifestError("World journal reference has an invalid byte offset")
        stream.seek(offset - 1)
        if stream.read(1) != b"\n":
            raise ManifestError("World journal committed prefix is not line-aligned")
        start = offset - 1
        while start > 0:
            stream.seek(start - 1)
            if stream.read(1) == b"\n":
                break
            start -= 1
        stream.seek(start)
        return stream.read(offset - start)

    @classmethod
    def _verify_prefix(cls, reference: CheckpointReference) -> bool:
        if reference.owner != _JOURNAL_OWNER or reference.schema_version != _JOURNAL_SCHEMA:
            return False
        try:
            tick, sequence, offset = cls._position(reference)
            path = Path(reference.locator)
            with path.open("rb") as stream:
                if path.stat().st_size < offset:
                    return False
                header = stream.readline()
                expected_header = cls._header(reference.owner_identity)
                if header != expected_header or len(header) >= offset:
                    return False
                if sequence == -1:
                    return offset == len(header) and reference.sha256 == hashlib.sha256(b"").hexdigest()
                stream.seek(len(header))
                previous = ""
                previous_tick = -1
                actual_sequence = 0
                actual_tick = -1
                while stream.tell() < offset:
                    line = stream.readline()
                    if not line or not line.endswith(b"\n") or stream.tell() > offset:
                        return False
                    data = json.loads(line)
                    if not isinstance(data, dict) or set(data) != {"record", "prefix_sha256"}:
                        return False
                    record = cls._record_from_data(data["record"])
                    if (
                        record.world_id != reference.owner_identity
                        or record.causal_tick < previous_tick
                        or record.causal_tick > tick
                    ):
                        return False
                    encoded = EventJournal._encode(record)
                    actual = hashlib.sha256(previous.encode("ascii") + b"\0" + encoded).hexdigest()
                    if data["prefix_sha256"] != actual:
                        return False
                    previous = actual
                    previous_tick = record.causal_tick
                    actual_tick = record.causal_tick
                    actual_sequence += 1
                return (
                    stream.tell() == offset
                    and actual_sequence == sequence + 1
                    and actual_tick == tick
                    and previous == reference.sha256
                )
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError, ManifestError):
            return False

    def write(
        self,
        journal: EventJournal,
        *,
        causal_tick: int,
        previous_reference: CheckpointReference | None = None,
    ) -> CheckpointReference:
        record_count = len(journal)
        self.directory.mkdir(parents=True, exist_ok=True)
        target = self.directory / "world-causal-journal-v3.jsonl"
        if previous_reference is None:
            if target.exists():
                raise ManifestError("World journal already exists without its committed reference")
            previous_count = 0
            previous_offset = 0
            previous_digest = ""
            prefix = self._header(self.world_id)
        else:
            if (
                previous_reference.owner != _JOURNAL_OWNER
                or previous_reference.owner_identity != self.world_id
                or previous_reference.schema_version != _JOURNAL_SCHEMA
                or Path(previous_reference.locator) != target
            ):
                raise ManifestError("previous committed World journal reference is invalid")
            # Sequential writes through this store have already validated and
            # fsynced the exact committed prefix. Avoid rescanning the entire
            # history on every tick, but reverify whenever this store instance
            # is new or the backing file's identity/metadata changed.
            if not self._previous_prefix_is_cached(previous_reference, target):
                if not self._verify_prefix(previous_reference):
                    raise ManifestError("previous committed World journal reference is invalid")
                self._cache_verified_prefix(previous_reference, target)
            _, previous_sequence, previous_offset = self._position(previous_reference)
            previous_count = previous_sequence + 1
            if previous_count < 0 or previous_count > record_count:
                raise ManifestError("candidate journal does not extend its committed prefix")
            previous_digest = "" if previous_sequence == -1 else previous_reference.sha256
            if journal.prefix_digest(previous_count) != previous_digest:
                raise ManifestError("candidate journal differs from its committed prefix")
            prefix = b""

        delta_records = tuple(journal.record_at(index) for index in range(previous_count, record_count))
        previous_tick = journal.record_at(previous_count - 1).causal_tick if previous_count else -1
        for record in delta_records:
            if record.world_id != self.world_id:
                raise ManifestError("candidate journal contains a record for another World")
            if record.causal_tick < previous_tick or record.causal_tick > causal_tick:
                raise ManifestError("candidate journal causal order is invalid")
            previous_tick = record.causal_tick
        if delta_records and delta_records[-1].causal_tick != causal_tick:
            raise ManifestError("candidate journal tail does not match the requested causal tick")
        appended: list[bytes] = []
        digest = previous_digest
        for record in delta_records:
            line, digest = self._record_line(record, digest)
            appended.append(line)
        if not delta_records:
            digest = hashlib.sha256(b"").hexdigest() if not record_count else journal.prefix_digest()
        encoded_delta = prefix + b"".join(appended)
        if previous_reference is None:
            fd, temporary_name = tempfile.mkstemp(prefix=".journal-", dir=self.directory)
            try:
                with os.fdopen(fd, "wb") as stream:
                    stream.write(encoded_delta)
                    stream.flush()
                    os.fsync(stream.fileno())
                try:
                    os.link(temporary_name, target)
                except FileExistsError as exc:
                    raise ManifestError("World journal was concurrently initialized") from exc
                _fsync_directory(self.directory)
            finally:
                try:
                    os.unlink(temporary_name)
                except FileNotFoundError:
                    pass
            previous_offset = 0
        else:
            with target.open("r+b") as stream:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
                if target.stat().st_size < previous_offset:
                    raise ManifestError("World journal is shorter than its committed reference")
                stream.truncate(previous_offset)
                stream.seek(previous_offset)
                stream.write(encoded_delta)
                stream.flush()
                os.fsync(stream.fileno())
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        offset = previous_offset + len(encoded_delta)
        reference = CheckpointReference(
            owner=_JOURNAL_OWNER,
            owner_identity=self.world_id,
            checkpoint_id=f"journal-cut-{causal_tick}-sequence-{record_count - 1}-offset-{offset}",
            schema_version=_JOURNAL_SCHEMA,
            locator=str(target),
            sha256=digest,
        )
        self._cache_verified_prefix(reference, target)
        return reference

    def read(self, reference: CheckpointReference) -> EventJournal:
        try:
            if (
                reference.owner != _JOURNAL_OWNER
                or reference.owner_identity != self.world_id
                or reference.schema_version != _JOURNAL_SCHEMA
                or Path(reference.locator) != self.directory / "world-causal-journal-v3.jsonl"
            ):
                raise ManifestError("journal reference belongs to another World")
            expected_tick, expected_sequence, offset = self._position(reference)
            with Path(reference.locator).open("rb") as stream:
                if stream.readline() != self._header(self.world_id):
                    raise ManifestError("World journal header is invalid")
                journal = EventJournal()
                previous = ""
                previous_tick = -1
                while stream.tell() < offset:
                    line = stream.readline()
                    if not line or stream.tell() > offset:
                        raise ManifestError("World journal committed prefix is truncated")
                    if not line.endswith(b"\n"):
                        raise ManifestError("World journal committed record is not LF-terminated")
                    data = json.loads(line)
                    if not isinstance(data, dict) or set(data) != {"record", "prefix_sha256"}:
                        raise ManifestError("World journal record envelope is invalid")
                    record = self._record_from_data(data["record"])
                    if record.world_id != self.world_id:
                        raise ManifestError("World journal contains a record for another World")
                    if record.causal_tick < previous_tick or record.causal_tick > expected_tick:
                        raise ManifestError("World journal causal order is invalid")
                    _, actual = self._record_line(record, previous)
                    if data["prefix_sha256"] != actual:
                        raise ManifestError("World journal integrity verification failed")
                    journal.append(record)
                    if journal.prefix_digest() != actual:
                        raise ManifestError("World journal prefix digest is inconsistent")
                    previous = actual
                    previous_tick = record.causal_tick
                if stream.tell() != offset or len(journal) != expected_sequence + 1:
                    raise ManifestError("World journal reference does not identify its exact prefix")
                if journal and journal.tail(1)[0].causal_tick != expected_tick:
                    raise ManifestError("World journal reference causal tick does not match its prefix")
                expected_digest = hashlib.sha256(b"").hexdigest() if not len(journal) else journal.prefix_digest()
                if reference.sha256 != expected_digest:
                    raise ManifestError("World journal reference digest does not match its prefix")
                return journal
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ManifestError("cannot read valid World journal snapshot") from exc

    def _discard_uncommitted_tail(self, committed_reference: CheckpointReference) -> int:
        """Truncate World-owned journal bytes beyond a verified committed cut.

        Owner checkpoint artifacts are deliberately outside this cleanup
        boundary. This recovery-only helper must be called under the manifest
        writer lock after owner restoration and verification.
        """
        target = self.directory / "world-causal-journal-v3.jsonl"
        if (
            committed_reference.owner != _JOURNAL_OWNER
            or committed_reference.owner_identity != self.world_id
            or committed_reference.schema_version != _JOURNAL_SCHEMA
            or Path(committed_reference.locator) != target
            or not self._verify_prefix(committed_reference)
        ):
            raise ManifestError("cannot clean journal tail without a valid committed World reference")
        _, _, committed_offset = self._position(committed_reference)
        try:
            with target.open("r+b") as stream:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
                current_size = os.fstat(stream.fileno()).st_size
                if current_size < committed_offset:
                    raise ManifestError("World journal is shorter than its committed reference")
                discarded_bytes = current_size - committed_offset
                if discarded_bytes:
                    stream.truncate(committed_offset)
                    stream.flush()
                    os.fsync(stream.fileno())
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
                return discarded_bytes
        except OSError as exc:
            raise ManifestError("cannot discard uncommitted World journal tail") from exc

    @staticmethod
    def verify(reference: CheckpointReference) -> bool:
        """Verify the complete committed prefix, including its hash chain."""
        return DurableJournalSnapshotStore._verify_prefix(reference)


OwnerCheckpointWriter = Callable[[int], Iterable[CheckpointReference]]
OwnerCheckpointVerifier = Callable[[tuple[CheckpointReference, ...]], bool]
OwnerCheckpointRestorer = Callable[[tuple[CheckpointReference, ...]], None]


@dataclass(frozen=True, slots=True)
class CommittedWorldCut:
    """World-verified identity and opaque owner references for one committed cut.

    This is a correlation receipt for post-commit consumers, not an owner-state
    decoder or a semantic observation. Only ``DurableWorldRuntime`` creates it
    after matching the requested identity against the durable manifest and
    journal.
    """

    world_id: str
    causal_tick: int
    commit_id: str
    participant_ids: tuple[tuple[str, str], ...]
    presence_owner_links: tuple[tuple[str, str, str], ...]
    owner_checkpoint_references: tuple[CheckpointReference, ...]


class TransitionCommitState(str, Enum):
    """Read-only classification of one transition against current manifest evidence."""

    NOT_COMMITTED = "NOT_COMMITTED"
    COMMITTED = "COMMITTED"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class TransitionCommitInspection:
    """Read-only transition classification; it does not perform recovery."""

    state: TransitionCommitState
    committed_cut: CommittedWorldCut | None = None


@dataclass(frozen=True, slots=True)
class OwnerCausalRecord:
    """Opaque owner-authored record committed in a World causal journal.

    ``owner_record_id`` is stable within the owner identity and World. World
    binds the record to the causal tick and stores it without interpreting the
    payload's domain meaning.
    """

    owner_id: str
    owner_identity: str
    owner_record_id: str
    payload: Mapping[str, object]

    def __post_init__(self) -> None:
        for field_name in ("owner_id", "owner_identity", "owner_record_id"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        if not isinstance(self.payload, Mapping):
            raise TypeError("owner causal record payload must be a JSON object")
        _validate_json_object(self.payload)
        # Validate strict JSON before freezing. CausalRecord performs the
        # immutable nested copy when the journal record is materialized.
        json.dumps(self.payload, allow_nan=False)


OwnerCausalRecordWriter = Callable[
    [int, tuple[CheckpointReference, ...]], Iterable[OwnerCausalRecord]
]


_OWNER_CAUSAL_RECORD_TYPE = "world.owner_record.committed"


def _validate_json_object(value: object) -> None:
    """Reject non-JSON values and non-string object keys without coercion."""
    if not isinstance(value, Mapping):
        raise TypeError("owner causal record payload must be a JSON object")
    for key, item in value.items():
        if not isinstance(key, str):
            raise TypeError("owner causal record payload keys must be strings")
        _validate_json_value(item)


def _validate_json_value(value: object) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("owner causal record payload keys must be strings")
            _validate_json_value(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _validate_json_value(item)
    elif value is None or isinstance(value, (str, bool, int)) or (
        isinstance(value, float) and isfinite(value)
    ):
        return
    else:
        raise TypeError(f"unsupported owner causal record payload value: {type(value).__name__}")


class OwnerCheckpointContract(Protocol):
    """Owner codec capability used to validate checkpoint bytes in process."""

    def validate_checkpoint(self, encoded: bytes, owner_identity: str, schema_version: str) -> bool: ...

    def verify_restoration(self, reference: CheckpointReference) -> bool: ...


class DurableWorldRuntime:
    """Publish each World tick only after its journal and owner refs are durable.

    A complete cut must contain at least one semantic-owner checkpoint. The
    World persists an active-presence roster and registered causal-owner
    roster. Each tick may explicitly register/unregister either roster and
    name step-only participants. Owner identities are resource identities,
    not World presence IDs.

    A transition first durably records ``PREPARING`` while retaining the last
    committed cut and proposed roster. Its callback then persists owner-owned
    state and returns opaque references. World writes an immutable journal
    prefix and finally publishes the committed manifest. Any ordinary exception latches
    the in-process coordinator and leaves explicit recovery required; recovery
    itself is deliberately outside this initial API.
    """

    def __init__(
        self,
        world_id: WorldId,
        presence_id: str | None,
        manifest_store: WorldCausalManifestStore,
        journal_store: DurableJournalSnapshotStore,
        manifest: WorldCausalManifest,
        journal: EventJournal,
        *,
        owner_checkpoint_verifier: OwnerCheckpointVerifier,
        owner_checkpoint_contracts: Mapping[tuple[str, str], OwnerCheckpointContract],
    ) -> None:
        if presence_id is not None and (not presence_id or not presence_id.strip()):
            raise ValueError("presence id must not be empty")
        if manifest.world_id != world_id.value:
            raise ValueError("manifest World identity does not match runtime")
        if presence_id is not None and presence_id not in manifest.presence_ids:
            raise ValueError("runtime presence is not part of the durable World cut")
        if presence_id is None and manifest.presence_ids:
            raise ValueError("a World with active presences requires an active runtime presence")
        if journal_store.world_id != world_id.value:
            raise ValueError("journal store World identity does not match runtime")
        if manifest.status is not ManifestStatus.COMMITTED:
            raise RecoveryRequiredError("World manifest requires explicit recovery")
        self.world_id = world_id
        self.presence_id = presence_id
        self._manifest_store = manifest_store
        self._journal_store = journal_store
        self._manifest = manifest
        self._journal = journal
        self._owner_checkpoint_verifier = owner_checkpoint_verifier
        self._owner_checkpoint_contracts = dict(owner_checkpoint_contracts)
        self._tick_guard = Lock()
        self._publication_guard_thread_id: int | None = None
        self._publication_guard_tick_used = False
        self._latched_failure: str | None = None
        self._validate_cut(manifest, journal)

    @classmethod
    def initialize(
        cls,
        genesis: WorldGenesisSpec,
        active_presence_id: str,
        manifest_store: WorldCausalManifestStore,
        journal_store: DurableJournalSnapshotStore,
        *,
        owner_checkpoint_verifier: OwnerCheckpointVerifier,
    ) -> DurableWorldRuntime:
        world_id = genesis.world_id
        presence_id = active_presence_id
        journal = EventJournal()
        if presence_id not in genesis.initial_presence_ids:
            raise ManifestError("active presence must be included in the accepted World genesis spec")
        owner_references = genesis.initial_owner_references
        registered_owner_ids = genesis.registered_owner_ids
        contracts = dict(genesis.owner_checkpoint_contracts)
        if set(contracts) != set(registered_owner_ids):
            raise ManifestError("initial owner roster requires one registered recoverability contract per owner")
        cls._require_owner_references(owner_references, "initial causal cut")
        if cls._owner_keys(owner_references) != frozenset(registered_owner_ids):
            raise ManifestError("initial owner checkpoints must exactly match the World-registered owner roster")
        genesis_commit_id = f"genesis:{world_id.value}"
        journal.append(
            _causal_cut_record(
                world_id=world_id.value,
                causal_tick=0,
                commit_id=genesis_commit_id,
                references=owner_references,
            )
        )
        journal_reference = journal_store.write(journal, causal_tick=0)
        references = (*owner_references, journal_reference)
        if not cls._verify_references(references, owner_checkpoint_verifier, contracts):
            raise ManifestError("initial causal cut checkpoint dependency closure is not verified")
        manifest = WorldCausalManifest(
            world_id=world_id.value,
            causal_tick=0,
            presence_ids=genesis.initial_presence_ids,
            checkpoint_references=references,
            journal_last_sequence=len(journal) - 1,
            journal_prefix_sha256=journal.prefix_digest(),
            commit_id=genesis_commit_id,
            registered_owner_ids=registered_owner_ids,
            presence_owner_links=tuple(
                (link.presence_id.value, link.owner_id, link.owner_identity)
                for link in genesis.presence_owner_links
            ),
        )
        manifest_store._initialize(manifest)
        return cls(
            world_id,
            presence_id,
            manifest_store,
            journal_store,
            manifest,
            journal,
            owner_checkpoint_verifier=owner_checkpoint_verifier,
            owner_checkpoint_contracts=contracts,
        )

    @classmethod
    def restore(
        cls,
        world_id: WorldId,
        presence_id: str | None,
        manifest_store: WorldCausalManifestStore,
        journal_store: DurableJournalSnapshotStore,
        *,
        owner_checkpoint_verifier: OwnerCheckpointVerifier,
        owner_checkpoint_contracts: Mapping[tuple[str, str], OwnerCheckpointContract],
        owner_checkpoint_restorer: OwnerCheckpointRestorer,
    ) -> DurableWorldRuntime:
        manifest = manifest_store._require_active()
        if manifest.world_id != world_id.value:
            raise ManifestError("stored World identity does not match requested World")
        if presence_id is None:
            if manifest.presence_ids:
                raise ManifestError("active World cut requires an explicit runtime presence")
        elif presence_id not in manifest.presence_ids:
            raise ManifestError("requested presence is not part of the stored World cut")
        references = manifest.checkpoint_references
        cls._require_owner_references(references, "stored causal cut")
        journal_refs = tuple(ref for ref in references if ref.owner == _JOURNAL_OWNER)
        if len(journal_refs) != 1 or journal_refs[0].owner_identity != world_id.value:
            raise ManifestError("causal cut must reference exactly one World journal snapshot")
        journal = journal_store.read(journal_refs[0])
        cls._validate_cut(manifest, journal)
        contracts = dict(owner_checkpoint_contracts)
        if not cls._contracts_cover(manifest, contracts):
            raise ManifestError("restore requires registered owner recoverability contracts for the stored roster")
        if not cls._verify_references(references, owner_checkpoint_verifier, contracts):
            raise ManifestError("causal cut checkpoint dependency closure is not verified")
        owner_references = tuple(ref for ref in references if ref.owner != _JOURNAL_OWNER)
        owner_checkpoint_restorer(owner_references)
        if not cls._validate_owner_checkpoints(
            owner_references,
            contracts,
            method="verify_restoration",
        ):
            raise ManifestError("owner restoration validation failed for one or more checkpoint references")
        if not cls._verify_references(references, owner_checkpoint_verifier, contracts):
            raise ManifestError("causal cut checkpoint dependency closure changed during restore")
        return cls(
            world_id,
            presence_id,
            manifest_store,
            journal_store,
            manifest,
            journal,
            owner_checkpoint_verifier=owner_checkpoint_verifier,
            owner_checkpoint_contracts=contracts,
        )

    @classmethod
    def recover_previous_cut(
        cls,
        world_id: WorldId,
        presence_id: str | None,
        manifest_store: WorldCausalManifestStore,
        journal_store: DurableJournalSnapshotStore,
        *,
        owner_checkpoint_verifier: OwnerCheckpointVerifier,
        owner_checkpoint_contracts: Mapping[tuple[str, str], OwnerCheckpointContract],
        owner_checkpoint_restorer: OwnerCheckpointRestorer,
    ) -> DurableWorldRuntime:
        """Explicitly restore the last committed cut after an interrupted tick.

        Recovery delegates private-state restoration to each semantic owner.
        The prior durable references must verify before and after restoration;
        only then is an immutable recovery receipt written and the previous
        cut republished as COMMITTED. This is not crash-consistency proof or a
        multi-process recovery protocol. The owner restorer must be safe to
        retry: a partial restoration may leave the manifest RECOVERY_REQUIRED.
        """
        pending = manifest_store.read()
        if pending.status is not ManifestStatus.RECOVERY_REQUIRED:
            raise ManifestError("explicit recovery requires RECOVERY_REQUIRED state")
        if pending.world_id != world_id.value:
            raise ManifestError("pending World identity does not match requested World")
        if not pending.pending_transition_id or pending.pending_tick != pending.causal_tick + 1:
            raise ManifestError("pending transition metadata is incomplete or inconsistent")
        if presence_id is None:
            if pending.presence_ids:
                raise ManifestError("active World cut requires an explicit runtime presence")
        elif presence_id not in pending.presence_ids:
            raise ManifestError("requested presence is not part of the pending World cut")

        references = pending.checkpoint_references
        contracts = dict(owner_checkpoint_contracts)
        if not cls._contracts_cover(pending, contracts):
            raise ManifestError("recovery requires registered owner recoverability contracts for the prior roster")
        cls._require_owner_references(references, "pending causal cut")
        journal_refs = tuple(ref for ref in references if ref.owner == _JOURNAL_OWNER)
        if len(journal_refs) != 1 or journal_refs[0].owner_identity != world_id.value:
            raise ManifestError("pending causal cut must reference exactly one World journal snapshot")
        journal = journal_store.read(journal_refs[0])
        cls._validate_cut(pending, journal)
        if not cls._verify_references(references, owner_checkpoint_verifier, contracts):
            raise ManifestError("previous causal cut dependency closure is not verified")

        owner_references = tuple(ref for ref in references if ref.owner != _JOURNAL_OWNER)
        owner_checkpoint_restorer(owner_references)
        if not cls._validate_owner_checkpoints(owner_references, contracts, method="verify_restoration"):
            raise ManifestError("owner restoration validation failed for one or more checkpoint references")
        if not cls._verify_references(references, owner_checkpoint_verifier, contracts):
            raise ManifestError("previous causal cut dependency closure changed during recovery")

        # Finalization and tail cleanup are serialized under the manifest
        # writer lock. A stale concurrent recovery must not truncate journal
        # bytes from a later committed transition.
        recovered = manifest_store._complete_recovery(
            pending.pending_transition_id,
            before_publish=lambda: journal_store._discard_uncommitted_tail(journal_refs[0]),
        )
        return cls(
            world_id,
            presence_id,
            manifest_store,
            journal_store,
            recovered,
            journal,
            owner_checkpoint_verifier=owner_checkpoint_verifier,
            owner_checkpoint_contracts=contracts,
        )

    @property
    def causal_tick(self) -> int:
        return self._manifest.causal_tick

    @property
    def status(self) -> ManifestStatus:
        if self._latched_failure is not None:
            return ManifestStatus.RECOVERY_REQUIRED
        try:
            store_manifest = self._manifest_store.read()
            if store_manifest.status is ManifestStatus.RECOVERY_REQUIRED:
                return ManifestStatus.RECOVERY_REQUIRED
        except ManifestError:
            pass
        return self._manifest.status

    @property
    def records(self) -> tuple[CausalRecord, ...]:
        return self._journal.replay()

    @property
    def manifest(self) -> WorldCausalManifest:
        return self._manifest

    @contextmanager
    def committed_cut_publication_guard(self) -> Iterator[None]:
        """Keep a committed cut current while its post-commit publication runs.

        Physical-step composition holds this scope across commit receipt
        verification, owner projection, and Observation delivery. Reentrancy
        permits that same thread to perform the World tick and receipt lookup,
        but a second tick from that thread is rejected until publication leaves
        the scope. Competing threads fail with ``TickInProgressError``.
        """
        if self._publication_guard_thread_id is not None:
            raise TickInProgressError("a World causal publication is already in progress")
        if not self._tick_guard.acquire(blocking=False):
            raise TickInProgressError("a World causal tick is already in progress")
        self._publication_guard_thread_id = get_ident()
        self._publication_guard_tick_used = False
        try:
            yield
        finally:
            self._publication_guard_thread_id = None
            self._publication_guard_tick_used = False
            self._tick_guard.release()

    def committed_cut_for(
        self,
        *,
        world_id: str,
        causal_tick: int,
        commit_id: str,
        expected_owner_references: Iterable[CheckpointReference] | None = None,
    ) -> CommittedWorldCut:
        """Verify an exact current cut against durable World-owned history.

        This is intended for post-commit publication joins. It checks the active
        committed manifest against this already-validated runtime snapshot and
        returns the exact opaque owner references World committed. Full journal
        integrity validation occurs on commit and restore, not on every receipt
        lookup; this keeps publication lookup bounded as the journal grows. It
        never opens or decodes owner checkpoint bytes. A pending/recovery state,
        stale identity, or stale owner reference fails closed.
        """
        publication_owner = self._publication_guard_thread_id == get_ident()
        with nullcontext() if publication_owner else self._tick_guard:
            # Unlike startup restore, a receipt lookup must not reinterpret a
            # live PREPARING transition as abandoned work. _read_for_tick rejects
            # that state without mutating the manifest.
            manifest = self._manifest_store._read_for_tick()
            if manifest != self._manifest:
                raise SingleWriterPreconditionError(
                    "World runtime is stale relative to the active committed cut"
                )
            if (
                world_id != manifest.world_id
                or type(causal_tick) is not int
                or causal_tick != manifest.causal_tick
                or commit_id != manifest.commit_id
            ):
                raise ManifestError("requested identity does not match the current committed World cut")
            journal_references = tuple(
                reference
                for reference in manifest.checkpoint_references
                if reference.owner == _JOURNAL_OWNER
            )
            if len(journal_references) != 1:
                raise ManifestError("committed World cut must identify exactly one journal prefix")
            journal_reference = journal_references[0]
            journal_tick, journal_sequence, _offset = self._journal_store._position(journal_reference)
            if (
                journal_tick != manifest.causal_tick
                or journal_sequence != manifest.journal_last_sequence
                or journal_reference.sha256 != manifest.journal_prefix_sha256
                or self._journal.prefix_digest() != manifest.journal_prefix_sha256
            ):
                raise ManifestError("committed World manifest and validated runtime journal identity differ")
            owner_references = tuple(
                sorted(
                    (
                        reference
                        for reference in manifest.checkpoint_references
                        if reference.owner != _JOURNAL_OWNER
                    ),
                    key=lambda reference: (reference.owner, reference.owner_identity),
                )
            )
            if expected_owner_references is not None:
                expected = tuple(
                    sorted(
                        tuple(expected_owner_references),
                        key=lambda reference: (reference.owner, reference.owner_identity),
                    )
                )
                if expected != owner_references:
                    raise ManifestError(
                        "owner checkpoint references do not match the committed World cut"
                    )
            return CommittedWorldCut(
                world_id=manifest.world_id,
                causal_tick=manifest.causal_tick,
                commit_id=manifest.commit_id,
                participant_ids=tuple(sorted((*manifest.registered_owner_ids, *manifest.step_participant_ids))),
                presence_owner_links=manifest.presence_owner_links,
                owner_checkpoint_references=owner_references,
            )

    def inspect_transition_commit(
        self,
        *,
        transition_id: str,
        expected_source: WorldCausalManifest,
    ) -> TransitionCommitInspection:
        """Inspect commit evidence without retrying, recovering, or mutating World.

        A committed source cut still current proves NOT_COMMITTED for this
        sample. A pending manifest is ambiguous and remains UNKNOWN. A
        committed target is accepted only when its manifest, referenced
        journal snapshot, and recorded predecessor cut agree. Any unreadable,
        stale, or conflicting state is UNKNOWN. This does not claim
        crash-durability or cross-process recovery.
        """
        if not transition_id or not transition_id.strip():
            raise ValueError("transition_id must not be empty")
        try:
            current = self._manifest_store.read()
            if current.world_id != self.world_id.value or expected_source.world_id != self.world_id.value:
                return TransitionCommitInspection(TransitionCommitState.UNKNOWN)

            source_identity = (
                expected_source.world_id,
                expected_source.causal_tick,
                expected_source.presence_ids,
                expected_source.checkpoint_references,
                expected_source.journal_last_sequence,
                expected_source.journal_prefix_sha256,
                expected_source.commit_id,
                expected_source.registered_owner_ids,
                expected_source.presence_owner_links,
                expected_source.step_participant_ids,
            )
            current_identity = (
                current.world_id,
                current.causal_tick,
                current.presence_ids,
                current.checkpoint_references,
                current.journal_last_sequence,
                current.journal_prefix_sha256,
                current.commit_id,
                current.registered_owner_ids,
                current.presence_owner_links,
                current.step_participant_ids,
            )
            if current_identity == source_identity:
                if current.status is ManifestStatus.COMMITTED:
                    return TransitionCommitInspection(TransitionCommitState.NOT_COMMITTED)

            if (
                current.status is not ManifestStatus.COMMITTED
                or current.causal_tick != expected_source.causal_tick + 1
                or current.commit_id != transition_id
            ):
                return TransitionCommitInspection(TransitionCommitState.UNKNOWN)

            journal_references = tuple(
                reference
                for reference in current.checkpoint_references
                if reference.owner == _JOURNAL_OWNER
            )
            if len(journal_references) != 1 or journal_references[0].owner_identity != self.world_id.value:
                return TransitionCommitInspection(TransitionCommitState.UNKNOWN)
            journal = self._journal_store.read(journal_references[0])
            self._validate_cut(current, journal)
            source_cut_records = tuple(
                record
                for record in journal.records_at_tick(expected_source.causal_tick)
                if record.record_type == _CAUSAL_CUT_RECORD_TYPE
            )
            if len(source_cut_records) != 1:
                return TransitionCommitInspection(TransitionCommitState.UNKNOWN)
            source_cut_record = source_cut_records[0]
            _validate_causal_cut_record(source_cut_record)
            source_cut_sequence = next(
                index
                for index, record in enumerate(journal.replay())
                if record is source_cut_record
            )
            source_prefix_count = (expected_source.journal_last_sequence + 1) if (
                expected_source.journal_last_sequence is not None
            ) else 0
            if (
                expected_source.journal_last_sequence is None
                or source_prefix_count <= source_cut_sequence
                or source_prefix_count > len(journal)
                or journal.prefix_digest(source_prefix_count) != expected_source.journal_prefix_sha256
                or any(
                    record.causal_tick > expected_source.causal_tick
                    for record in journal.replay()[:source_prefix_count]
                )
                or journal.record_at(source_prefix_count - 1).causal_tick != expected_source.causal_tick
            ):
                return TransitionCommitInspection(TransitionCommitState.UNKNOWN)

            # Reconstruct the source presence projection by reversing only
            # committed lifecycle records after that cut. The journal does not
            # make presence identity inferable from owner references alone.
            source_presence = set(current.presence_ids)
            source_presence_links = {
                presence: (owner, identity)
                for presence, owner, identity in current.presence_owner_links
            }
            later_lifecycle_records = tuple(
                record
                for record in journal.replay()
                if record.record_type == "world.presence.lifecycle.committed"
                and expected_source.causal_tick < record.causal_tick <= current.causal_tick
            )
            for lifecycle_record in reversed(later_lifecycle_records):
                payload = lifecycle_record.payload
                if (
                    lifecycle_record.world_id != current.world_id
                    or lifecycle_record.actor_id != "world"
                    or not isinstance(payload, Mapping)
                    or payload.get("authority") != "world"
                    or payload.get("source_tick") != lifecycle_record.causal_tick - 1
                    or payload.get("target_tick") != lifecycle_record.causal_tick
                ):
                    return TransitionCommitInspection(TransitionCommitState.UNKNOWN)
                affected_presence = payload.get("affected_presence_id")
                affected_owner = payload.get("affected_owner")
                transition_type = payload.get("transition_type")
                if (
                    not isinstance(affected_presence, str)
                    or not affected_presence
                    or not isinstance(affected_owner, Mapping)
                    or set(affected_owner) != {"owner_id", "owner_identity"}
                    or not all(
                        isinstance(affected_owner.get(field), str) and affected_owner[field]
                        for field in ("owner_id", "owner_identity")
                    )
                    or transition_type not in {"REGISTER", "TERMINATE"}
                ):
                    return TransitionCommitInspection(TransitionCommitState.UNKNOWN)
                recorded_resulting_presence = payload.get("resulting_active_presence_ids")
                recorded_resulting_links = payload.get("resulting_presence_owner_links")
                if (
                    not isinstance(recorded_resulting_presence, tuple)
                    or not isinstance(recorded_resulting_links, tuple)
                    or tuple(sorted(source_presence)) != tuple(recorded_resulting_presence)
                    or tuple(
                        (presence, *source_presence_links[presence])
                        for presence in sorted(source_presence_links)
                    ) != tuple(tuple(link) for link in recorded_resulting_links)
                ):
                    return TransitionCommitInspection(TransitionCommitState.UNKNOWN)
                owner_link = (affected_owner["owner_id"], affected_owner["owner_identity"])
                if transition_type == "REGISTER":
                    if source_presence_links.get(affected_presence) != owner_link:
                        return TransitionCommitInspection(TransitionCommitState.UNKNOWN)
                    source_presence.discard(affected_presence)
                    source_presence_links.pop(affected_presence, None)
                else:
                    if affected_presence in source_presence or affected_presence in source_presence_links:
                        return TransitionCommitInspection(TransitionCommitState.UNKNOWN)
                    source_presence.add(affected_presence)
                    source_presence_links[affected_presence] = owner_link

            if (
                tuple(sorted(source_presence)) != tuple(sorted(expected_source.presence_ids))
                or tuple(
                    (presence, *source_presence_links[presence])
                    for presence in sorted(source_presence_links)
                ) != tuple(sorted(expected_source.presence_owner_links))
            ):
                return TransitionCommitInspection(TransitionCommitState.UNKNOWN)
            source_owner_references = tuple(
                sorted(
                    (
                        reference
                        for reference in expected_source.checkpoint_references
                        if reference.owner != _JOURNAL_OWNER
                    ),
                    key=lambda reference: (reference.owner, reference.owner_identity),
                )
            )
            source_participants = tuple(
                sorted((*expected_source.registered_owner_ids, *expected_source.step_participant_ids))
            )
            source_payload = source_cut_record.payload
            if (
                expected_source.status is not ManifestStatus.COMMITTED
                or source_cut_record.world_id != expected_source.world_id
                or source_payload["commit_id"] != expected_source.commit_id
                or tuple(tuple(item) for item in source_payload["participants"])
                != source_participants
                or tuple(source_payload["owner_checkpoint_references"])
                != tuple(_reference_descriptor(reference) for reference in source_owner_references)
            ):
                return TransitionCommitInspection(TransitionCommitState.UNKNOWN)
            owner_references = tuple(
                sorted(
                    (reference for reference in current.checkpoint_references if reference.owner != _JOURNAL_OWNER),
                    key=lambda reference: (reference.owner, reference.owner_identity),
                )
            )
            cut = CommittedWorldCut(
                world_id=current.world_id,
                causal_tick=current.causal_tick,
                commit_id=current.commit_id,
                participant_ids=tuple(sorted((*current.registered_owner_ids, *current.step_participant_ids))),
                presence_owner_links=current.presence_owner_links,
                owner_checkpoint_references=owner_references,
            )
            return TransitionCommitInspection(TransitionCommitState.COMMITTED, cut)
        except Exception:
            return TransitionCommitInspection(TransitionCommitState.UNKNOWN)

    def tick(
        self,
        transition_id: str,
        write_owner_checkpoints: OwnerCheckpointWriter,
        *,
        write_owner_records: OwnerCausalRecordWriter | None = None,
        register_owners: Iterable[tuple[str, str]] = (),
        unregister_owners: Iterable[tuple[str, str]] = (),
        step_participants: Iterable[tuple[str, str]] = (),
        register_presence: Iterable[str] = (),
        unregister_presence: Iterable[str] = (),
        presence_owner_updates: Iterable[PresenceOwnerLink] = (),
        register_owner_contracts: Mapping[tuple[str, str], OwnerCheckpointContract] | None = None,
        presence_lifecycle: PresenceLifecycleTransition | None = None,
    ) -> CausalRecord:
        publication_owner = self._publication_guard_thread_id == get_ident()
        if publication_owner:
            if self._publication_guard_tick_used:
                raise TickInProgressError(
                    "a World causal publication cannot start another tick before delivery completes"
                )
            self._publication_guard_tick_used = True
        elif not self._tick_guard.acquire(blocking=False):
            raise TickInProgressError("a World causal tick is already in progress")
        try:
            if self._latched_failure is not None:
                raise RecoveryRequiredError("World runtime requires explicit recovery")
            current = self._manifest_store._read_for_tick()
            if current != self._manifest:
                raise SingleWriterPreconditionError(
                    "runtime is stale relative to the current committed World cut "
                    f"(runtime tick {self._manifest.causal_tick} commit {self._manifest.commit_id!r}; "
                    f"stored tick {current.causal_tick} commit {current.commit_id!r})"
                )
            proposed_tick = current.causal_tick + 1
            registered = set(current.registered_owner_ids)
            additions_input = tuple(tuple(item) for item in register_owners)
            removals_input = tuple(tuple(item) for item in unregister_owners)
            step_input = tuple(tuple(item) for item in step_participants)
            additions = set(additions_input)
            removals = set(removals_input)
            step_only = set(step_input)
            if len(additions) != len(additions_input) or len(removals) != len(removals_input) or len(step_only) != len(step_input):
                raise ManifestError("World participant lifecycle input contains duplicate identities")
            if additions & registered or not removals <= registered:
                raise ManifestError("World owner registration transition is inconsistent with the current roster")
            registered = (registered | additions) - removals
            if registered & step_only:
                raise ManifestError("step participants must not duplicate registered World owners")
            if not registered and not step_only:
                raise ManifestError("a committed causal cut requires at least one participating owner")
            presence_additions_input = tuple(register_presence)
            presence_removals_input = tuple(unregister_presence)
            presence_additions = set(presence_additions_input)
            presence_removals = set(presence_removals_input)
            active_presence = set(current.presence_ids)
            changed_presences = presence_additions | presence_removals
            if bool(changed_presences) != (presence_lifecycle is not None):
                raise ManifestError("World presence roster changes require an explicit lifecycle transition")
            if presence_lifecycle is not None:
                lifecycle_presence = presence_lifecycle.presence_id.value
                expected_set = (
                    presence_additions
                    if presence_lifecycle.transition_type == "REGISTER"
                    else presence_removals
                )
                other_set = (
                    presence_removals
                    if presence_lifecycle.transition_type == "REGISTER"
                    else presence_additions
                )
                if expected_set != {lifecycle_presence} or other_set:
                    raise ManifestError("one lifecycle transition must change exactly its declared presence")
            if (
                len(presence_additions) != len(presence_additions_input)
                or len(presence_removals) != len(presence_removals_input)
                or any(not item or not item.strip() for item in (*presence_additions, *presence_removals))
                or presence_additions & active_presence
                or not presence_removals <= active_presence
            ):
                raise ManifestError("World presence lifecycle input is inconsistent with the current roster")
            active_presence = (active_presence | presence_additions) - presence_removals
            if self.presence_id is not None and self.presence_id not in active_presence and presence_lifecycle is None:
                raise ManifestError("the current runtime presence cannot be unregistered by its own tick")
            if self.presence_id is None and presence_lifecycle is None:
                raise ManifestError("a World without active presence requires an explicit lifecycle transition")
            active_links = {
                presence: (owner, identity)
                for presence, owner, identity in current.presence_owner_links
                if presence not in presence_removals
            }
            updates = tuple(presence_owner_updates)
            updated_presence_ids = tuple(link.presence_id.value for link in updates)
            if len(updated_presence_ids) != len(set(updated_presence_ids)):
                raise ManifestError("presence owner updates contain duplicate presences")
            for link in updates:
                presence = link.presence_id.value
                if presence not in active_presence:
                    raise ManifestError("presence owner update must target an active presence")
                active_links[presence] = (link.owner_id, link.owner_identity)
            lifecycle_owner: tuple[str, str] | None = None
            if presence_lifecycle is not None:
                if presence_lifecycle.transition_type == "TERMINATE":
                    lifecycle_owner = next(
                        (owner, identity)
                        for presence, owner, identity in current.presence_owner_links
                        if presence == presence_lifecycle.presence_id.value
                    )
                else:
                    lifecycle_owner = active_links.get(presence_lifecycle.presence_id.value)
            if set(active_links) != active_presence:
                raise ManifestError("each active presence must retain exactly one registered causal owner")
            if presence_lifecycle is not None and lifecycle_owner is None:
                raise ManifestError("registered presence must have exactly one declared owner")
            if any(owner_key not in registered for owner_key in active_links.values()):
                raise ManifestError("each active presence owner must be in the proposed registered owner roster")
            if lifecycle_owner is not None and lifecycle_owner not in registered and lifecycle_owner not in step_only:
                raise ManifestError("presence lifecycle owner must participate in the lifecycle commit")
            if presence_lifecycle is not None and any(
                item.record_type == "world.presence.lifecycle.committed"
                and item.payload.get("lifecycle_transition_id") == transition_id
                for item in self._journal.replay()
            ):
                raise ManifestError("presence lifecycle transition identity is already committed")
            registered_tuple = tuple(sorted(registered))
            step_tuple = tuple(sorted(step_only))
            expected_owners = frozenset(registered | step_only)
            proposed_contracts = dict(self._owner_checkpoint_contracts)
            new_contracts = dict(register_owner_contracts or {})
            if set(new_contracts) != additions | step_only:
                raise ManifestError("each newly registered or step-only owner requires its owner checkpoint contract")
            proposed_contracts.update(new_contracts)
            for key in removals:
                if key not in step_only:
                    proposed_contracts.pop(key, None)
            pending = self._manifest_store._begin_transition(
                transition_id,
                expected_manifest=current,
                registered_owner_ids=registered_tuple,
                presence_owner_links=tuple(
                    (presence, owner, identity)
                    for presence, (owner, identity) in sorted(active_links.items())
                ),
                step_participant_ids=step_tuple,
                presence_ids=tuple(sorted(active_presence)),
            )
            try:
                owner_references = tuple(write_owner_checkpoints(proposed_tick))
                candidate_owners = self._owner_keys(owner_references)
                if candidate_owners != expected_owners:
                    raise ManifestError(
                        "transition owner checkpoint identities must match the World-prepared roster"
                    )
                if not self._contracts_cover_keys(expected_owners, proposed_contracts):
                    raise ManifestError("transition lacks an owner recoverability contract for every participant")
                if not self._validate_owner_checkpoints(owner_references, proposed_contracts):
                    raise ManifestError("owner checkpoint validation failed for one or more checkpoint references")
                owner_records = tuple(
                    write_owner_records(proposed_tick, owner_references)
                    if write_owner_records is not None
                    else ()
                )
                owner_record_keys = tuple(
                    (item.owner_id, item.owner_identity, item.owner_record_id)
                    for item in owner_records
                )
                if len(set(owner_record_keys)) != len(owner_record_keys):
                    raise ManifestError("tick contains duplicate owner causal record identities")
                if any(
                    (item.owner_id, item.owner_identity) not in expected_owners
                    for item in owner_records
                ):
                    raise ManifestError("owner causal record must belong to a participating owner")
                record = CausalRecord(
                    record_id=f"{self.world_id.value}:tick:{proposed_tick}",
                    world_id=self.world_id.value,
                    causal_tick=proposed_tick,
                    record_type="world.causal_tick.committed",
                    actor_id="world" if presence_lifecycle is not None else self.presence_id,
                    payload={"presence_id": self.presence_id},
                )
                candidate_journal = EventJournal.from_snapshot(self._journal.snapshot())
                existing_owner_record_keys = self._owner_causal_record_keys(
                    candidate_journal.replay()
                )
                if existing_owner_record_keys.intersection(owner_record_keys):
                    raise ManifestError("owner causal record identity was already committed in this World")
                candidate_journal.append(record)
                candidate_journal.append(
                    _causal_cut_record(
                        world_id=self.world_id.value,
                        causal_tick=proposed_tick,
                        commit_id=transition_id,
                        references=owner_references,
                    )
                )
                if presence_lifecycle is not None:
                    owner_refs = tuple(
                        {
                            "owner_id": reference.owner,
                            "owner_identity": reference.owner_identity,
                            "checkpoint_id": reference.checkpoint_id,
                            "schema_version": reference.schema_version,
                            "locator": reference.locator,
                            "sha256": reference.sha256,
                        }
                        for reference in owner_references
                        if (reference.owner, reference.owner_identity) == lifecycle_owner
                    )
                    if not owner_refs:
                        raise ManifestError("lifecycle owner must contribute a checkpoint reference to the commit")
                    candidate_journal.append(
                        CausalRecord(
                            record_id=f"{self.world_id.value}:presence-lifecycle:{proposed_tick}:{transition_id}",
                            world_id=self.world_id.value,
                            causal_tick=proposed_tick,
                            record_type="world.presence.lifecycle.committed",
                            actor_id="world",
                            payload={
                                "authority": "world",
                                "lifecycle_transition_id": transition_id,
                                "transition_type": presence_lifecycle.transition_type,
                                "affected_presence_id": presence_lifecycle.presence_id.value,
                                "affected_owner": {
                                    "owner_id": lifecycle_owner[0],
                                    "owner_identity": lifecycle_owner[1],
                                },
                                "source_tick": current.causal_tick,
                                "target_tick": proposed_tick,
                                "trigger_provenance": _thaw_json(presence_lifecycle.trigger_provenance),
                                "required_participants": [list(key) for key in sorted(expected_owners)],
                                "owner_checkpoint_references": list(owner_refs),
                                "resulting_active_presence_ids": sorted(active_presence),
                                "resulting_presence_owner_links": [
                                    [presence, owner, identity]
                                    for presence, (owner, identity) in sorted(active_links.items())
                                ],
                            },
                        )
                    )
                for owner_record in owner_records:
                    encoded_identity = json.dumps(
                        [owner_record.owner_id, owner_record.owner_identity, owner_record.owner_record_id],
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ).encode("utf-8")
                    owner_record_event_id = hashlib.sha256(encoded_identity).hexdigest()
                    # Round-trip into a detached JSON tree so the journal never
                    # retains caller-owned mutable payload containers.
                    payload = json.loads(json.dumps(owner_record.payload, allow_nan=False))
                    candidate_journal.append(
                        CausalRecord(
                            record_id=(
                                f"{self.world_id.value}:owner-record:{proposed_tick}:{owner_record_event_id}"
                            ),
                            world_id=self.world_id.value,
                            causal_tick=proposed_tick,
                            record_type=_OWNER_CAUSAL_RECORD_TYPE,
                            payload={
                                "owner_id": owner_record.owner_id,
                                "owner_identity": owner_record.owner_identity,
                                "owner_record_id": owner_record.owner_record_id,
                                "data": payload,
                            },
                        )
                    )
                committed_journal_refs = tuple(
                    reference
                    for reference in current.checkpoint_references
                    if reference.owner == _JOURNAL_OWNER
                )
                if len(committed_journal_refs) != 1:
                    raise ManifestError("committed World cut must identify exactly one journal prefix")
                journal_reference = self._journal_store.write(
                    candidate_journal,
                    causal_tick=proposed_tick,
                    previous_reference=committed_journal_refs[0],
                )
                references = (*owner_references, journal_reference)
                candidate = WorldCausalManifest(
                    world_id=self.world_id.value,
                    causal_tick=proposed_tick,
                    presence_ids=tuple(sorted(active_presence)),
                    checkpoint_references=references,
                    journal_last_sequence=len(candidate_journal) - 1,
                    journal_prefix_sha256=candidate_journal.prefix_digest(),
                    commit_id=transition_id,
                    registered_owner_ids=registered_tuple,
                    presence_owner_links=tuple(
                        (presence, owner, identity)
                        for presence, (owner, identity) in sorted(active_links.items())
                    ),
                    step_participant_ids=step_tuple,
                )
                self._manifest_store._commit_transition(
                    candidate,
                    dependency_verifier=lambda refs: self._verify_references(
                        refs,
                        self._owner_checkpoint_verifier,
                        proposed_contracts,
                    ),
                )
                self._journal = candidate_journal
                self._manifest = candidate
                self._owner_checkpoint_contracts = {
                    key: proposed_contracts[key] for key in registered
                }
                if (
                    presence_lifecycle is not None
                    and presence_lifecycle.transition_type == "REGISTER"
                    and self.presence_id is None
                ):
                    self.presence_id = presence_lifecycle.presence_id.value
                elif (
                    presence_lifecycle is not None
                    and presence_lifecycle.transition_type == "TERMINATE"
                    and self.presence_id == presence_lifecycle.presence_id.value
                ):
                    self.presence_id = None
                return record
            except Exception as exc:
                self._latched_failure = f"{type(exc).__name__}: {exc}"
                try:
                    self._manifest = self._manifest_store._record_failure(transition_id, self._latched_failure)
                except Exception:
                    # Publication outcome can be ambiguous after an OS-level write error.
                    # Keep execution latched even if the marker cannot be re-written.
                    pass
                raise
        finally:
            if not publication_owner:
                self._tick_guard.release()

    @staticmethod
    def _verify_references(
        references: tuple[CheckpointReference, ...],
        owner_checkpoint_verifier: OwnerCheckpointVerifier,
        owner_checkpoint_contracts: Mapping[tuple[str, str], OwnerCheckpointContract],
    ) -> bool:
        journal_refs = tuple(ref for ref in references if ref.owner == _JOURNAL_OWNER)
        owner_refs = tuple(ref for ref in references if ref.owner != _JOURNAL_OWNER)
        if len(journal_refs) != 1 or not DurableJournalSnapshotStore.verify(journal_refs[0]):
            return False
        return bool(owner_checkpoint_verifier(owner_refs)) and DurableWorldRuntime._validate_owner_checkpoints(
            owner_refs, owner_checkpoint_contracts
        )

    @staticmethod
    def _validate_owner_checkpoints(
        references: tuple[CheckpointReference, ...],
        contracts: Mapping[tuple[str, str], OwnerCheckpointContract],
        *,
        method: str = "validate_checkpoint",
    ) -> bool:
        for reference in references:
            if reference.owner == _JOURNAL_OWNER:
                continue
            contract = contracts.get((reference.owner, reference.owner_identity))
            if contract is None:
                return False
            try:
                encoded = Path(reference.locator).read_bytes()
                if hashlib.sha256(encoded).hexdigest() != reference.sha256:
                    return False
                if method == "verify_restoration":
                    validated = contract.verify_restoration(reference)
                else:
                    validated = contract.validate_checkpoint(
                        encoded, reference.owner_identity, reference.schema_version
                    )
            except Exception:
                return False
            if validated is not True:
                return False
        return True

    @staticmethod
    def _contracts_cover(manifest: WorldCausalManifest, contracts: Mapping[tuple[str, str], OwnerCheckpointContract]) -> bool:
        return DurableWorldRuntime._contracts_cover_keys(
            frozenset(manifest.registered_owner_ids) | frozenset(manifest.step_participant_ids), contracts
        )

    @staticmethod
    def _contracts_cover_keys(keys: frozenset[tuple[str, str]], contracts: Mapping[tuple[str, str], OwnerCheckpointContract]) -> bool:
        return keys <= set(contracts)

    @staticmethod
    def _owner_keys(references: tuple[CheckpointReference, ...]) -> frozenset[tuple[str, str]]:
        """Return the explicitly participating semantic owner identities."""
        return frozenset((reference.owner, reference.owner_identity) for reference in references)

    @staticmethod
    def _validate_cut(manifest: WorldCausalManifest, journal: EventJournal) -> None:
        DurableWorldRuntime._require_owner_references(manifest.checkpoint_references, "causal cut")
        records = journal.replay()
        if any(record.world_id != manifest.world_id for record in records):
            raise ManifestError("journal contains a record for another World")
        commit_ticks = tuple(
            record.causal_tick
            for record in records
            if record.record_type == "world.causal_tick.committed"
        )
        if commit_ticks != tuple(range(1, manifest.causal_tick + 1)):
            raise ManifestError("journal does not cover the manifest causal tick prefix")
        cut_records = tuple(
            record for record in records if record.record_type == _CAUSAL_CUT_RECORD_TYPE
        )
        if tuple(record.causal_tick for record in cut_records) != tuple(
            range(0, manifest.causal_tick + 1)
        ):
            raise ManifestError("journal does not preserve owner references for every committed causal cut")
        for record in cut_records:
            _validate_causal_cut_record(record)
        current_cut = cut_records[-1]
        expected_current_refs = tuple(
            _reference_descriptor(ref)
            for ref in sorted(
                (ref for ref in manifest.checkpoint_references if ref.owner != _JOURNAL_OWNER),
                key=lambda ref: (ref.owner, ref.owner_identity),
            )
        )
        current_payload = current_cut.payload
        if (
            current_cut.world_id != manifest.world_id
            or current_payload["commit_id"] != manifest.commit_id
            or tuple(tuple(item) for item in current_payload["participants"])
            != tuple(sorted((*manifest.registered_owner_ids, *manifest.step_participant_ids)))
            or tuple(current_payload["owner_checkpoint_references"]) != expected_current_refs
        ):
            raise ManifestError("latest causal-cut journal record does not match the committed manifest")
        if any(record.causal_tick < 0 or record.causal_tick > manifest.causal_tick for record in records):
            raise ManifestError("journal contains a record outside the committed causal tick prefix")
        DurableWorldRuntime._owner_causal_record_keys(records)
        expected_last = len(records) - 1 if records else None
        if manifest.journal_last_sequence != expected_last:
            raise ManifestError("manifest journal sequence does not match persisted record prefix")
        if manifest.journal_prefix_sha256 != journal.prefix_digest():
            raise ManifestError("manifest journal digest does not match persisted record prefix")

    @staticmethod
    def _require_owner_references(
        references: tuple[CheckpointReference, ...], cut_description: str
    ) -> tuple[CheckpointReference, ...]:
        """Reject cuts that contain only World journal data, not owner state."""
        owner_references = tuple(ref for ref in references if ref.owner != _JOURNAL_OWNER)
        if not owner_references:
            raise ManifestError(f"{cut_description} must identify at least one durable owner checkpoint")
        return owner_references

    @staticmethod
    def _owner_causal_record_keys(
        records: Iterable[CausalRecord],
    ) -> set[tuple[str, str, str]]:
        """Validate and return unique World-scoped owner record identities."""
        keys: set[tuple[str, str, str]] = set()
        for record in records:
            if record.record_type != _OWNER_CAUSAL_RECORD_TYPE:
                continue
            payload = record.payload
            if set(payload) != {"owner_id", "owner_identity", "owner_record_id", "data"}:
                raise ManifestError("persisted owner causal record has an invalid envelope")
            owner_id = payload["owner_id"]
            owner_identity = payload["owner_identity"]
            owner_record_id = payload["owner_record_id"]
            if any(not isinstance(value, str) or not value.strip() for value in (owner_id, owner_identity, owner_record_id)):
                raise ManifestError("persisted owner causal record has an invalid stable identity")
            key = (owner_id, owner_identity, owner_record_id)
            if key in keys:
                raise ManifestError("journal contains a duplicate owner causal record identity")
            keys.add(key)
        return keys


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
