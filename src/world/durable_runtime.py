"""Durable single-World tick publication over opaque owner checkpoints.

This API coordinates publication only. The supplied transition callback must
use owner APIs to persist its new state and return exact references for the
resulting causal cut. World validates references but never decodes owner data.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from threading import Lock
from typing import Callable, Iterable

from .journal import CausalRecord, EventJournal
from .manifest import (
    CheckpointReference,
    DependencyVerifier,
    ManifestError,
    ManifestStatus,
    WorldCausalManifest,
    WorldCausalManifestStore,
)
from .runtime import RecoveryRequiredError, TickInProgressError, WorldId


_JOURNAL_OWNER = "world-journal"
_JOURNAL_SCHEMA = "event-journal-v1"


class DurableJournalSnapshotStore:
    """Persist immutable journal prefixes and return opaque World references."""

    def __init__(self, directory: str | Path, world_id: WorldId) -> None:
        self.directory = Path(directory)
        self.world_id = world_id.value

    def write(self, journal: EventJournal, *, causal_tick: int) -> CheckpointReference:
        records = journal.snapshot()
        encoded = (json.dumps(records, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
        digest = hashlib.sha256(encoded).hexdigest()
        self.directory.mkdir(parents=True, exist_ok=True)
        target = self.directory / f"journal-{causal_tick}-{digest}.json"
        if target.exists():
            if target.read_bytes() != encoded:
                raise ManifestError("existing immutable journal snapshot differs from candidate")
        else:
            fd, temporary_name = tempfile.mkstemp(prefix=".journal-", dir=self.directory)
            try:
                with os.fdopen(fd, "wb") as stream:
                    stream.write(encoded)
                    stream.flush()
                    os.fsync(stream.fileno())
                try:
                    os.link(temporary_name, target)
                except FileExistsError:
                    if target.read_bytes() != encoded:
                        raise ManifestError("journal snapshot path collision")
                _fsync_directory(self.directory)
            finally:
                try:
                    os.unlink(temporary_name)
                except FileNotFoundError:
                    pass

        return CheckpointReference(
            owner=_JOURNAL_OWNER,
            owner_identity=self.world_id,
            checkpoint_id=f"journal-prefix-{causal_tick}",
            schema_version=_JOURNAL_SCHEMA,
            locator=str(target),
            sha256=digest,
        )

    def read(self, reference: CheckpointReference) -> EventJournal:
        self.verify(reference)
        try:
            records = json.loads(Path(reference.locator).read_bytes())
            if not isinstance(records, list):
                raise ValueError("journal snapshot must be a list")
            return EventJournal.from_snapshot(records)
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ManifestError("cannot read valid World journal snapshot") from exc

    @staticmethod
    def verify(reference: CheckpointReference) -> bool:
        if reference.owner != _JOURNAL_OWNER or reference.schema_version != _JOURNAL_SCHEMA:
            return False
        path = Path(reference.locator)
        try:
            return path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == reference.sha256
        except OSError:
            return False


OwnerCheckpointWriter = Callable[[int], Iterable[CheckpointReference]]
OwnerCheckpointVerifier = Callable[[tuple[CheckpointReference, ...]], bool]
OwnerCheckpointRestorer = Callable[[tuple[CheckpointReference, ...]], None]


class DurableWorldRuntime:
    """Publish each World tick only after its journal and owner refs are durable.

    A transition first durably marks the previous complete cut
    ``RECOVERY_REQUIRED``. Its callback then persists owner-owned state and
    returns opaque references. World writes an immutable journal prefix and
    finally publishes the complete manifest. Any ordinary exception latches
    the in-process coordinator and leaves explicit recovery required; recovery
    itself is deliberately outside this initial API.
    """

    def __init__(
        self,
        world_id: WorldId,
        presence_id: str,
        manifest_store: WorldCausalManifestStore,
        journal_store: DurableJournalSnapshotStore,
        manifest: WorldCausalManifest,
        journal: EventJournal,
        *,
        owner_checkpoint_verifier: OwnerCheckpointVerifier,
    ) -> None:
        if not presence_id or not presence_id.strip():
            raise ValueError("presence id must not be empty")
        if manifest.world_id != world_id.value:
            raise ValueError("manifest World identity does not match runtime")
        if presence_id not in manifest.presence_ids:
            raise ValueError("runtime presence is not part of the durable World cut")
        if journal_store.world_id != world_id.value:
            raise ValueError("journal store World identity does not match runtime")
        if manifest.status is not ManifestStatus.COMPLETE:
            raise RecoveryRequiredError("World manifest requires explicit recovery")
        self.world_id = world_id
        self.presence_id = presence_id
        self._manifest_store = manifest_store
        self._journal_store = journal_store
        self._manifest = manifest
        self._journal = journal
        self._owner_checkpoint_verifier = owner_checkpoint_verifier
        self._tick_guard = Lock()
        self._latched_failure: str | None = None
        self._validate_cut(manifest, journal)

    @classmethod
    def initialize(
        cls,
        world_id: WorldId,
        presence_id: str,
        manifest_store: WorldCausalManifestStore,
        journal_store: DurableJournalSnapshotStore,
        *,
        initial_owner_references: Iterable[CheckpointReference],
        owner_checkpoint_verifier: OwnerCheckpointVerifier,
    ) -> DurableWorldRuntime:
        journal = EventJournal()
        journal_reference = journal_store.write(journal, causal_tick=0)
        references = (*tuple(initial_owner_references), journal_reference)
        if not cls._verify_references(references, owner_checkpoint_verifier):
            raise ManifestError("initial causal cut checkpoint dependency closure is not verified")
        manifest = WorldCausalManifest(
            world_id=world_id.value,
            causal_tick=0,
            presence_ids=(presence_id,),
            checkpoint_references=references,
            journal_last_sequence=None,
            journal_prefix_sha256="",
        )
        manifest_store.initialize(manifest)
        return cls(
            world_id,
            presence_id,
            manifest_store,
            journal_store,
            manifest,
            journal,
            owner_checkpoint_verifier=owner_checkpoint_verifier,
        )

    @classmethod
    def restore(
        cls,
        world_id: WorldId,
        presence_id: str,
        manifest_store: WorldCausalManifestStore,
        journal_store: DurableJournalSnapshotStore,
        *,
        owner_checkpoint_verifier: OwnerCheckpointVerifier,
        owner_checkpoint_restorer: OwnerCheckpointRestorer,
    ) -> DurableWorldRuntime:
        manifest = manifest_store.require_active()
        if manifest.world_id != world_id.value:
            raise ManifestError("stored World identity does not match requested World")
        if presence_id not in manifest.presence_ids:
            raise ManifestError("requested presence is not part of the stored World cut")
        references = manifest.checkpoint_references
        journal_refs = tuple(ref for ref in references if ref.owner == _JOURNAL_OWNER)
        if len(journal_refs) != 1 or journal_refs[0].owner_identity != world_id.value:
            raise ManifestError("causal cut must reference exactly one World journal snapshot")
        if not cls._verify_references(references, owner_checkpoint_verifier):
            raise ManifestError("causal cut checkpoint dependency closure is not verified")
        journal = journal_store.read(journal_refs[0])
        cls._validate_cut(manifest, journal)
        owner_checkpoint_restorer(tuple(ref for ref in references if ref.owner != _JOURNAL_OWNER))
        return cls(
            world_id,
            presence_id,
            manifest_store,
            journal_store,
            manifest,
            journal,
            owner_checkpoint_verifier=owner_checkpoint_verifier,
        )

    @classmethod
    def recover_previous_cut(
        cls,
        world_id: WorldId,
        presence_id: str,
        manifest_store: WorldCausalManifestStore,
        journal_store: DurableJournalSnapshotStore,
        *,
        owner_checkpoint_verifier: OwnerCheckpointVerifier,
        owner_checkpoint_restorer: OwnerCheckpointRestorer,
    ) -> DurableWorldRuntime:
        """Explicitly restore the last committed cut after an interrupted tick.

        Recovery delegates private-state restoration to each semantic owner.
        The prior durable references must verify before and after restoration;
        only then is an immutable recovery receipt written and the previous
        cut republished as COMPLETE. This is not crash-consistency proof or a
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
        if presence_id not in pending.presence_ids:
            raise ManifestError("requested presence is not part of the pending World cut")

        references = pending.checkpoint_references
        journal_refs = tuple(ref for ref in references if ref.owner == _JOURNAL_OWNER)
        if len(journal_refs) != 1 or journal_refs[0].owner_identity != world_id.value:
            raise ManifestError("pending causal cut must reference exactly one World journal snapshot")
        if not cls._verify_references(references, owner_checkpoint_verifier):
            raise ManifestError("previous causal cut dependency closure is not verified")
        journal = journal_store.read(journal_refs[0])
        cls._validate_cut(pending, journal)

        owner_references = tuple(ref for ref in references if ref.owner != _JOURNAL_OWNER)
        owner_checkpoint_restorer(owner_references)
        if not cls._verify_references(references, owner_checkpoint_verifier):
            raise ManifestError("previous causal cut dependency closure changed during recovery")

        recovered = manifest_store._complete_recovery(pending.pending_transition_id)
        return cls(
            world_id,
            presence_id,
            manifest_store,
            journal_store,
            recovered,
            journal,
            owner_checkpoint_verifier=owner_checkpoint_verifier,
        )

    @property
    def causal_tick(self) -> int:
        return self._manifest.causal_tick

    @property
    def status(self) -> ManifestStatus:
        if self._latched_failure is not None:
            return ManifestStatus.RECOVERY_REQUIRED
        return self._manifest.status

    @property
    def records(self) -> tuple[CausalRecord, ...]:
        return self._journal.replay()

    @property
    def manifest(self) -> WorldCausalManifest:
        return self._manifest

    def tick(self, transition_id: str, write_owner_checkpoints: OwnerCheckpointWriter) -> CausalRecord:
        if not self._tick_guard.acquire(blocking=False):
            raise TickInProgressError("a World causal tick is already in progress")
        try:
            if self._latched_failure is not None:
                raise RecoveryRequiredError("World runtime requires explicit recovery")
            current = self._manifest_store.require_active()
            if current != self._manifest:
                self._latched_failure = "durable manifest changed outside this runtime"
                raise RecoveryRequiredError(self._latched_failure)
            proposed_tick = current.causal_tick + 1
            pending = self._manifest_store.begin_transition(transition_id)
            try:
                owner_references = tuple(write_owner_checkpoints(proposed_tick))
                record = CausalRecord(
                    record_id=f"{self.world_id.value}:tick:{proposed_tick}",
                    world_id=self.world_id.value,
                    causal_tick=proposed_tick,
                    record_type="world.causal_tick.committed",
                    actor_id=self.presence_id,
                    payload={"presence_id": self.presence_id},
                )
                candidate_journal = EventJournal.from_snapshot(self._journal.snapshot())
                candidate_journal.append(record)
                journal_reference = self._journal_store.write(candidate_journal, causal_tick=proposed_tick)
                references = (*owner_references, journal_reference)
                candidate = WorldCausalManifest(
                    world_id=self.world_id.value,
                    causal_tick=proposed_tick,
                    presence_ids=current.presence_ids,
                    checkpoint_references=references,
                    journal_last_sequence=len(candidate_journal) - 1,
                    journal_prefix_sha256=candidate_journal.prefix_digest(),
                )
                self._manifest_store.commit_transition(
                    candidate,
                    dependency_verifier=lambda refs: self._verify_references(
                        refs, self._owner_checkpoint_verifier
                    ),
                )
                self._journal = candidate_journal
                self._manifest = candidate
                return record
            except Exception as exc:
                self._latched_failure = f"{type(exc).__name__}: {exc}"
                try:
                    self._manifest = self._manifest_store.record_failure(transition_id, self._latched_failure)
                except Exception:
                    # Publication outcome can be ambiguous after an OS-level write error.
                    # Keep execution latched even if the marker cannot be re-written.
                    pass
                raise
        finally:
            self._tick_guard.release()

    @staticmethod
    def _verify_references(
        references: tuple[CheckpointReference, ...],
        owner_checkpoint_verifier: OwnerCheckpointVerifier,
    ) -> bool:
        journal_refs = tuple(ref for ref in references if ref.owner == _JOURNAL_OWNER)
        owner_refs = tuple(ref for ref in references if ref.owner != _JOURNAL_OWNER)
        if len(journal_refs) != 1 or not DurableJournalSnapshotStore.verify(journal_refs[0]):
            return False
        return bool(owner_checkpoint_verifier(owner_refs))

    @staticmethod
    def _validate_cut(manifest: WorldCausalManifest, journal: EventJournal) -> None:
        records = journal.replay()
        if any(record.world_id != manifest.world_id for record in records):
            raise ManifestError("journal contains a record for another World")
        if tuple(record.causal_tick for record in records) != tuple(range(1, manifest.causal_tick + 1)):
            raise ManifestError("journal does not cover the manifest causal tick prefix")
        expected_last = len(records) - 1 if records else None
        if manifest.journal_last_sequence != expected_last:
            raise ManifestError("manifest journal sequence does not match persisted record prefix")
        if manifest.journal_prefix_sha256 != journal.prefix_digest():
            raise ManifestError("manifest journal digest does not match persisted record prefix")


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
