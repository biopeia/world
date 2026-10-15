"""Append-only journal for committed World causal records.

The journal records opaque event types; domain-specific event meaning remains
owned by the authority that defines each event contract. Staging is an in-memory
transaction aid, not a durable commit protocol.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("causal record payload keys must be strings")
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported causal record payload value: {type(value).__name__}")


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class CausalRecord:
    """Read-only record envelope for one World-scoped causal event."""

    record_id: str
    world_id: str
    causal_tick: int
    record_type: str
    actor_id: str | None = None
    payload: Mapping[str, Any] = field(default_factory=dict)
    causal_parent_ids: tuple[str, ...] = ()
    contributing_record_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.record_id:
            raise ValueError("record_id must not be empty")
        if not self.world_id:
            raise ValueError("world_id must not be empty")
        if self.causal_tick < 0:
            raise ValueError("causal_tick must be non-negative")
        if not self.record_type:
            raise ValueError("record_type must not be empty")
        object.__setattr__(self, "payload", _freeze_json(self.payload))
        object.__setattr__(self, "causal_parent_ids", tuple(self.causal_parent_ids))
        object.__setattr__(self, "contributing_record_ids", tuple(self.contributing_record_ids))


class EventJournal:
    """Append-only committed record journal with staged batch support.

    This class provides in-memory ordering and prefix digests. It does not claim
    durable storage, atomic filesystem publication, or World-wide recovery.
    """

    def __init__(self) -> None:
        self._records: list[CausalRecord] = []
        self._staged: list[CausalRecord] = []
        self._first_index_by_id: dict[str, int] = {}
        self._records_by_tick: dict[int, list[CausalRecord]] = {}
        self._prefix_digests: list[str] = []

    @staticmethod
    def _snapshot_record(record: CausalRecord) -> dict[str, Any]:
        return {
            "record_id": record.record_id,
            "world_id": record.world_id,
            "causal_tick": record.causal_tick,
            "record_type": record.record_type,
            "actor_id": record.actor_id,
            "payload": _thaw_json(record.payload),
            "causal_parent_ids": list(record.causal_parent_ids),
            "contributing_record_ids": list(record.contributing_record_ids),
        }

    @staticmethod
    def _encode(record: CausalRecord) -> bytes:
        return json.dumps(
            EventJournal._snapshot_record(record),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")

    def _append_encoded(self, record: CausalRecord, encoded: bytes) -> None:
        previous = self._prefix_digests[-1] if self._prefix_digests else ""
        digest = hashlib.sha256(previous.encode("ascii") + b"\0" + encoded).hexdigest()
        index = len(self._records)
        self._records.append(record)
        self._first_index_by_id.setdefault(record.record_id, index)
        self._records_by_tick.setdefault(record.causal_tick, []).append(record)
        self._prefix_digests.append(digest)

    def append(self, record: CausalRecord) -> None:
        # Serialize before mutating journal indexes so invalid payloads cannot
        # leave a visible record without its digest/index entries.
        encoded = self._encode(record)
        self._append_encoded(record, encoded)

    def candidate(self) -> JournalCandidate:
        """Create an append-only transaction view without copying the prefix."""
        if self._staged:
            raise ValueError("cannot create a journal candidate while a batch is staged")
        return JournalCandidate(self)

    def stage(self, record: CausalRecord) -> None:
        self._staged.append(record)

    def commit_staged(self) -> tuple[CausalRecord, ...]:
        committed = tuple(self._staged)
        # Validate and encode the complete batch before publishing any member.
        encoded = tuple(self._encode(record) for record in committed)
        for record, payload in zip(committed, encoded, strict=True):
            self._append_encoded(record, payload)
        self._staged.clear()
        return committed

    def abort_staged(self) -> None:
        self._staged.clear()

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self):
        return iter(self._records)

    def replay(self) -> tuple[CausalRecord, ...]:
        return tuple(self._records)

    def records_at_tick(self, causal_tick: int) -> tuple[CausalRecord, ...]:
        return tuple(self._records_by_tick.get(int(causal_tick), ()))

    def tail(self, limit: int) -> tuple[CausalRecord, ...]:
        if limit < 0:
            raise ValueError("tail limit must be non-negative")
        if limit == 0:
            return ()
        return tuple(self._records[-limit:])

    def record_at(self, index: int) -> CausalRecord:
        return self._records[index]

    def prefix_digest(self, count: int | None = None) -> str:
        resolved = len(self._records) if count is None else int(count)
        if resolved < 0 or resolved > len(self._records):
            raise ValueError("journal prefix count out of bounds")
        return "" if resolved == 0 else self._prefix_digests[resolved - 1]

    def snapshot_range(self, start: int = 0, stop: int | None = None) -> list[dict[str, Any]]:
        return [self._snapshot_record(record) for record in self._records[start:stop]]

    def page_after(
        self,
        after_record_id: str | None = None,
        *,
        limit: int = 256,
    ) -> tuple[tuple[CausalRecord, ...], str | None, bool]:
        if limit < 1:
            raise ValueError("record page limit must be positive")
        start = 0
        if after_record_id is not None:
            try:
                start = self._first_index_by_id[after_record_id] + 1
            except KeyError as exc:
                raise ValueError("unknown after_record_id") from exc
        page = tuple(self._records[start : start + limit])
        cursor = page[-1].record_id if page else after_record_id
        return page, cursor, start + len(page) < len(self._records)

    def snapshot(self) -> list[dict[str, Any]]:
        return self.snapshot_range()

    @classmethod
    def from_snapshot(cls, records_data: list[dict[str, Any]]) -> EventJournal:
        journal = cls()
        for data in records_data:
            journal.append(
                CausalRecord(
                    record_id=data["record_id"],
                    world_id=data["world_id"],
                    causal_tick=int(data["causal_tick"]),
                    record_type=data["record_type"],
                    actor_id=data.get("actor_id"),
                    payload=data.get("payload", {}),
                    causal_parent_ids=tuple(data.get("causal_parent_ids", ())),
                    contributing_record_ids=tuple(data.get("contributing_record_ids", ())),
                )
            )
        return journal


class JournalCandidate:
    """Small append overlay over a committed journal, used before durable commit."""

    def __init__(self, base: EventJournal) -> None:
        self._base = base
        self._records: list[CausalRecord] = []
        self._encoded_records: list[bytes] = []
        self._digests: list[str] = []

    def __len__(self) -> int:
        return len(self._base) + len(self._records)

    def append(self, record: CausalRecord) -> None:
        encoded = EventJournal._encode(record)
        previous = self._digests[-1] if self._digests else self._base.prefix_digest()
        self._digests.append(hashlib.sha256(previous.encode("ascii") + b"\0" + encoded).hexdigest())
        self._records.append(record)
        self._encoded_records.append(encoded)

    def replay(self) -> tuple[CausalRecord, ...]:
        return (*self._base.replay(), *self._records)

    def record_at(self, index: int) -> CausalRecord:
        base_length = len(self._base)
        resolved = index if index >= 0 else len(self) + index
        if resolved < 0 or resolved >= len(self):
            raise IndexError("journal record index out of range")
        return self._base.record_at(resolved) if resolved < base_length else self._records[resolved - base_length]

    def prefix_digest(self, count: int | None = None) -> str:
        resolved = len(self) if count is None else int(count)
        base_length = len(self._base)
        if resolved < 0 or resolved > len(self):
            raise ValueError("journal prefix count out of bounds")
        if resolved <= base_length:
            return self._base.prefix_digest(resolved)
        return self._digests[resolved - base_length - 1]

    def commit(self) -> EventJournal:
        for record, encoded in zip(self._records, self._encoded_records, strict=True):
            self._base._append_encoded(record, encoded)
        self._records.clear()
        self._encoded_records.clear()
        self._digests.clear()
        return self._base
