"""Minimal in-memory World causal coordinator.

This prototype serializes one synchronous transition at a time. It coordinates
an opaque callback and publishes a journal record only after the callback
returns successfully. Recovery state is in-memory only; this is not durable
recovery and does not by itself satisfy the TARGET post-failure contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from threading import Lock
from typing import Callable

from .journal import CausalRecord, EventJournal


@dataclass(frozen=True, slots=True)
class WorldId:
    """Canonical identifier for this World runtime instance."""

    value: str

    def __post_init__(self) -> None:
        if not self.value or not self.value.strip():
            raise ValueError("world id must not be empty")


class RuntimeStatus(str, Enum):
    ACTIVE = "ACTIVE"
    RECOVERY_REQUIRED = "RECOVERY_REQUIRED"


class RecoveryRequiredError(RuntimeError):
    """Raised when a failed causal transition prevents further ticks."""


class TickInProgressError(RuntimeError):
    """Raised when a tick is invoked while another tick is executing."""


Transition = Callable[[int], None]


class WorldRuntime:
    """Coordinate one active presence through explicit synchronous commits.

    ``causal_tick`` counts successfully committed transitions and begins at
    zero. The callback receives the proposed, one-based tick. It is opaque to
    World: the callback's state ownership and rollback semantics are outside
    this package. If it or journal publication raises, the runtime enters
    ``RECOVERY_REQUIRED`` and refuses later transitions. Calls made while a
    transition is already executing fail fast before invoking their callback;
    this applies to both concurrent and reentrant calls and does not itself
    alter runtime status.
    """

    def __init__(self, world_id: WorldId, presence_id: str) -> None:
        if not presence_id or not presence_id.strip():
            raise ValueError("presence id must not be empty")
        self.world_id = world_id
        self.presence_id = presence_id
        self._causal_tick = 0
        self._status = RuntimeStatus.ACTIVE
        self._failed_tick: int | None = None
        self._failure: str | None = None
        self._journal = EventJournal()
        self._tick_guard = Lock()

    @property
    def causal_tick(self) -> int:
        return self._causal_tick

    @property
    def status(self) -> RuntimeStatus:
        return self._status

    @property
    def failed_tick(self) -> int | None:
        return self._failed_tick

    @property
    def failure(self) -> str | None:
        return self._failure

    @property
    def records(self) -> tuple[CausalRecord, ...]:
        """Return an immutable snapshot of committed in-memory records."""

        return self._journal.replay()

    def tick(self, transition: Transition) -> CausalRecord:
        """Run a transition and publish its causal record on success only."""

        if not self._tick_guard.acquire(blocking=False):
            raise TickInProgressError("a World causal tick is already in progress")

        try:
            if self._status is RuntimeStatus.RECOVERY_REQUIRED:
                raise RecoveryRequiredError(
                    f"World runtime requires recovery after failed tick {self._failed_tick}"
                )

            proposed_tick = self._causal_tick + 1
            try:
                transition(proposed_tick)
                record = CausalRecord(
                    record_id=f"{self.world_id.value}:tick:{proposed_tick}",
                    world_id=self.world_id.value,
                    causal_tick=proposed_tick,
                    record_type="world.causal_tick.committed",
                    actor_id=self.presence_id,
                    payload={"presence_id": self.presence_id},
                )
                self._journal.append(record)
            except Exception as exc:
                self._status = RuntimeStatus.RECOVERY_REQUIRED
                self._failed_tick = proposed_tick
                self._failure = f"{type(exc).__name__}: {exc}"
                raise

            self._causal_tick = proposed_tick
            return record
        finally:
            self._tick_guard.release()
