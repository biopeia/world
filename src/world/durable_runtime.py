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
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from threading import Lock
from typing import Callable, Iterable, Mapping, Protocol

from .journal import CausalRecord, EventJournal
from .manifest import (
    CheckpointReference,
    DependencyVerifier,
    ManifestError,
    ManifestStatus,
    OwnerRecoveryReceipt,
    WorldCausalManifest,
    WorldCausalManifestStore,
)
from .runtime import PresenceId, RecoveryRequiredError, TickInProgressError, WorldId


_JOURNAL_OWNER = "world-journal"
_JOURNAL_SCHEMA = "event-journal-v1"


@dataclass(frozen=True, slots=True)
class PresenceOwnerLink:
    """Validate an explicit World-genesis presence-to-owner association.

    This association is currently a genesis input only; the durable manifest
    does not preserve it for later lifecycle transitions or restoration.
    """

    presence_id: PresenceId
    owner_id: str
    owner_identity: str

    def __post_init__(self) -> None:
        if not isinstance(self.presence_id, PresenceId):
            raise ValueError("presence linkage requires a typed PresenceId")
        if not self.owner_id or not self.owner_id.strip() or not self.owner_identity or not self.owner_identity.strip():
            raise ValueError("presence linkage owner identity must be non-empty")


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
        if links:
            if set(link_presence_values) != set(presences) or len(link_presence_values) != len(set(link_presence_values)):
                raise ValueError("presence owner links must cover each genesis presence exactly once")
            owner_keys = set(owners)
            if any((link.owner_id, link.owner_identity) not in owner_keys for link in links):
                raise ValueError("presence owner link must target a registered genesis owner")
        object.__setattr__(self, "presence_owner_links", links)


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


class OwnerCheckpointContract(Protocol):
    """Owner codec capability; World, not the owner adapter, issues receipts."""

    def validate_checkpoint(self, encoded: bytes, owner_identity: str, schema_version: str) -> bool: ...

    def verify_restoration(self, encoded: bytes, owner_identity: str, schema_version: str) -> bool: ...


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
        presence_id: str,
        manifest_store: WorldCausalManifestStore,
        journal_store: DurableJournalSnapshotStore,
        manifest: WorldCausalManifest,
        journal: EventJournal,
        *,
        owner_checkpoint_verifier: OwnerCheckpointVerifier,
        owner_checkpoint_contracts: Mapping[tuple[str, str], OwnerCheckpointContract],
    ) -> None:
        if not presence_id or not presence_id.strip():
            raise ValueError("presence id must not be empty")
        if manifest.world_id != world_id.value:
            raise ValueError("manifest World identity does not match runtime")
        if presence_id not in manifest.presence_ids:
            raise ValueError("runtime presence is not part of the durable World cut")
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
        journal_reference = journal_store.write(journal, causal_tick=0)
        references = (*owner_references, journal_reference)
        if not cls._verify_references(references, owner_checkpoint_verifier, contracts):
            raise ManifestError("initial causal cut checkpoint dependency closure is not verified")
        manifest = WorldCausalManifest(
            world_id=world_id.value,
            causal_tick=0,
            presence_ids=genesis.initial_presence_ids,
            checkpoint_references=references,
            journal_last_sequence=None,
            journal_prefix_sha256="",
            commit_id=f"genesis:{world_id.value}",
            registered_owner_ids=registered_owner_ids,
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
            owner_checkpoint_contracts=contracts,
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
        owner_checkpoint_contracts: Mapping[tuple[str, str], OwnerCheckpointContract],
        owner_checkpoint_restorer: OwnerCheckpointRestorer,
    ) -> DurableWorldRuntime:
        manifest = manifest_store.require_active()
        if manifest.world_id != world_id.value:
            raise ManifestError("stored World identity does not match requested World")
        if presence_id not in manifest.presence_ids:
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
        owner_checkpoint_restorer(tuple(ref for ref in references if ref.owner != _JOURNAL_OWNER))
        if not cls._verify_owner_receipts(
            tuple(ref for ref in references if ref.owner != _JOURNAL_OWNER),
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
        presence_id: str,
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
        if presence_id not in pending.presence_ids:
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
        if not cls._verify_owner_receipts(owner_references, contracts, method="verify_restoration"):
            raise ManifestError("owner restoration validation failed for one or more checkpoint references")
        if not cls._verify_references(references, owner_checkpoint_verifier, contracts):
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
            owner_checkpoint_contracts=contracts,
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

    def tick(
        self,
        transition_id: str,
        write_owner_checkpoints: OwnerCheckpointWriter,
        *,
        register_owners: Iterable[tuple[str, str]] = (),
        unregister_owners: Iterable[tuple[str, str]] = (),
        step_participants: Iterable[tuple[str, str]] = (),
        register_presence: Iterable[str] = (),
        unregister_presence: Iterable[str] = (),
        register_owner_contracts: Mapping[tuple[str, str], OwnerCheckpointContract] | None = None,
    ) -> CausalRecord:
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
                raise ManifestError("a committed causal cut requires at least one registered or step owner")
            presence_additions_input = tuple(register_presence)
            presence_removals_input = tuple(unregister_presence)
            presence_additions = set(presence_additions_input)
            presence_removals = set(presence_removals_input)
            active_presence = set(current.presence_ids)
            if (
                len(presence_additions) != len(presence_additions_input)
                or len(presence_removals) != len(presence_removals_input)
                or any(not item or not item.strip() for item in (*presence_additions, *presence_removals))
                or presence_additions & active_presence
                or not presence_removals <= active_presence
            ):
                raise ManifestError("World presence lifecycle input is inconsistent with the current roster")
            active_presence = (active_presence | presence_additions) - presence_removals
            if self.presence_id not in active_presence:
                raise ManifestError("the current runtime presence cannot be unregistered by its own tick")
            registered_tuple = tuple(sorted(registered))
            step_tuple = tuple(sorted(step_only))
            expected_owners = frozenset(registered | step_only)
            proposed_contracts = dict(self._owner_checkpoint_contracts)
            new_contracts = dict(register_owner_contracts or {})
            if set(new_contracts) != additions | step_only:
                raise ManifestError("each newly registered or step-only owner requires its owner checkpoint contract")
            proposed_contracts.update(new_contracts)
            for key in removals:
                proposed_contracts.pop(key, None)
            pending = self._manifest_store.begin_transition(
                transition_id,
                registered_owner_ids=registered_tuple,
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
                if not self._verify_owner_receipts(owner_references, proposed_contracts):
                    raise ManifestError("owner checkpoint validation failed for one or more checkpoint references")
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
                    presence_ids=tuple(sorted(active_presence)),
                    checkpoint_references=references,
                    journal_last_sequence=len(candidate_journal) - 1,
                    journal_prefix_sha256=candidate_journal.prefix_digest(),
                    commit_id=transition_id,
                    registered_owner_ids=registered_tuple,
                    step_participant_ids=step_tuple,
                )
                self._manifest_store.commit_transition(
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
        owner_checkpoint_contracts: Mapping[tuple[str, str], OwnerCheckpointContract],
    ) -> bool:
        journal_refs = tuple(ref for ref in references if ref.owner == _JOURNAL_OWNER)
        owner_refs = tuple(ref for ref in references if ref.owner != _JOURNAL_OWNER)
        if len(journal_refs) != 1 or not DurableJournalSnapshotStore.verify(journal_refs[0]):
            return False
        return bool(owner_checkpoint_verifier(owner_refs)) and DurableWorldRuntime._verify_owner_receipts(
            owner_refs, owner_checkpoint_contracts
        )

    @staticmethod
    def _verify_owner_receipts(
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
                validated = getattr(contract, method)(
                    encoded, reference.owner_identity, reference.schema_version
                )
            except Exception:
                return False
            if validated is not True:
                return False
            # Only World constructs the receipt after verifying the exact bytes
            # and receiving a positive validation result from the owner codec.
            receipt = OwnerRecoveryReceipt(
                owner=reference.owner,
                owner_identity=reference.owner_identity,
                checkpoint_id=reference.checkpoint_id,
                schema_version=reference.schema_version,
                locator=reference.locator,
                sha256=reference.sha256,
                recoverable=True,
            )
            if not receipt.matches(reference):
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
        if tuple(record.causal_tick for record in records) != tuple(range(1, manifest.causal_tick + 1)):
            raise ManifestError("journal does not cover the manifest causal tick prefix")
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


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
