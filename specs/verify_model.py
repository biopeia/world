"""Exhaustive finite-state model checker for WorldCommitRecovery.tla.

This script executes a breadth-first exploration of all reachable states
in the transition system specified by WorldCommitRecovery.tla and validates
the safety invariants I1–I5 on every reachable state.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass
from typing import FrozenSet, Tuple


@dataclass(frozen=True)
class CheckpointRef:
    owner: str
    tick: int


@dataclass(frozen=True)
class TempManifest:
    status: str
    tick: int
    pending_tick: int
    refs: FrozenSet[CheckpointRef]


@dataclass(frozen=True)
class RecoveryReceipt:
    transition: int
    restored_tick: int


@dataclass(frozen=True)
class State:
    manifest_status: str
    manifest_tick: int
    manifest_pending_tick: int
    manifest_refs: FrozenSet[CheckpointRef]
    temp_manifest: TempManifest | None
    journal_committed_len: int
    journal_file_len: int
    owner_blobs: FrozenSet[CheckpointRef]
    recovery_receipts: FrozenSet[RecoveryReceipt]
    proc_phase: str
    candidate_tick: int
    candidate_refs: FrozenSet[CheckpointRef]


def get_initial_state() -> State:
    genesis_ref = CheckpointRef("organism", 0)
    return State(
        manifest_status="COMMITTED",
        manifest_tick=0,
        manifest_pending_tick=0,
        manifest_refs=frozenset([genesis_ref]),
        temp_manifest=None,
        journal_committed_len=1,
        journal_file_len=1,
        owner_blobs=frozenset([genesis_ref]),
        recovery_receipts=frozenset(),
        proc_phase="IDLE",
        candidate_tick=0,
        candidate_refs=frozenset(),
    )


def check_invariants(state: State) -> None:
    # I1: No unconfirmed cuts accepted as committed
    if state.manifest_status == "COMMITTED":
        for ref in state.manifest_refs:
            assert ref.tick == state.manifest_tick, f"I1 violation: ref.tick {ref.tick} != {state.manifest_tick}"
        assert state.journal_committed_len >= 1, "I1 violation: journal length < 1"

    # I2: Reference coherence
    for ref in state.manifest_refs:
        assert ref in state.owner_blobs, f"I2 violation: ref {ref} not in owner_blobs"

    # I3: Journal consistency
    assert state.journal_committed_len <= state.journal_file_len, "I3 violation: committed length > file length"

    # I4: Fail closed
    if state.proc_phase == "IDLE":
        assert state.manifest_status != "PREPARING", "I4 violation: idle process observed PREPARING manifest"

    # I5: Receipt integrity
    for receipt in state.recovery_receipts:
        assert receipt.restored_tick <= state.manifest_tick + 1, "I5 violation: invalid receipt restored_tick"


def next_states(state: State, max_tick: int = 2) -> list[State]:
    succs: list[State] = []

    # ProcessCrash
    if state.proc_phase not in ("IDLE", "CRASHED"):
        succs.append(
            State(
                manifest_status=state.manifest_status,
                manifest_tick=state.manifest_tick,
                manifest_pending_tick=state.manifest_pending_tick,
                manifest_refs=state.manifest_refs,
                temp_manifest=None,
                journal_committed_len=state.journal_committed_len,
                journal_file_len=state.journal_file_len,
                owner_blobs=state.owner_blobs,
                recovery_receipts=state.recovery_receipts,
                proc_phase="CRASHED",
                candidate_tick=0,
                candidate_refs=frozenset(),
            )
        )

    # StartupReopen
    if state.proc_phase == "CRASHED":
        new_status = (
            "RECOVERY_REQUIRED" if state.manifest_status == "PREPARING" else state.manifest_status
        )
        succs.append(
            State(
                manifest_status=new_status,
                manifest_tick=state.manifest_tick,
                manifest_pending_tick=state.manifest_pending_tick,
                manifest_refs=state.manifest_refs,
                temp_manifest=None,
                journal_committed_len=state.journal_committed_len,
                journal_file_len=state.journal_file_len,
                owner_blobs=state.owner_blobs,
                recovery_receipts=state.recovery_receipts,
                proc_phase="IDLE",
                candidate_tick=0,
                candidate_refs=frozenset(),
            )
        )

    # TickBegin
    if state.proc_phase == "IDLE" and state.manifest_status == "COMMITTED" and state.manifest_tick < max_tick:
        cand_tick = state.manifest_tick + 1
        succs.append(
            State(
                manifest_status=state.manifest_status,
                manifest_tick=state.manifest_tick,
                manifest_pending_tick=state.manifest_pending_tick,
                manifest_refs=state.manifest_refs,
                temp_manifest=TempManifest("PREPARING", state.manifest_tick, cand_tick, state.manifest_refs),
                journal_committed_len=state.journal_committed_len,
                journal_file_len=state.journal_file_len,
                owner_blobs=state.owner_blobs,
                recovery_receipts=state.recovery_receipts,
                proc_phase="PREPARING_MANIFEST",
                candidate_tick=cand_tick,
                candidate_refs=frozenset(),
            )
        )

    # PublishPreparingManifest
    if state.proc_phase == "PREPARING_MANIFEST" and state.temp_manifest is not None:
        succs.append(
            State(
                manifest_status="PREPARING",
                manifest_tick=state.manifest_tick,
                manifest_pending_tick=state.candidate_tick,
                manifest_refs=state.manifest_refs,
                temp_manifest=None,
                journal_committed_len=state.journal_committed_len,
                journal_file_len=state.journal_file_len,
                owner_blobs=state.owner_blobs,
                recovery_receipts=state.recovery_receipts,
                proc_phase="WRITING_OWNERS",
                candidate_tick=state.candidate_tick,
                candidate_refs=state.candidate_refs,
            )
        )

    # WriteOwnerCheckpoints
    if state.proc_phase == "WRITING_OWNERS":
        new_ref = CheckpointRef("organism", state.candidate_tick)
        succs.append(
            State(
                manifest_status=state.manifest_status,
                manifest_tick=state.manifest_tick,
                manifest_pending_tick=state.manifest_pending_tick,
                manifest_refs=state.manifest_refs,
                temp_manifest=None,
                journal_committed_len=state.journal_committed_len,
                journal_file_len=state.journal_file_len,
                owner_blobs=state.owner_blobs | frozenset([new_ref]),
                recovery_receipts=state.recovery_receipts,
                proc_phase="WRITING_JOURNAL",
                candidate_tick=state.candidate_tick,
                candidate_refs=frozenset([new_ref]),
            )
        )

    # WriteJournalDelta
    if state.proc_phase == "WRITING_JOURNAL":
        succs.append(
            State(
                manifest_status=state.manifest_status,
                manifest_tick=state.manifest_tick,
                manifest_pending_tick=state.manifest_pending_tick,
                manifest_refs=state.manifest_refs,
                temp_manifest=TempManifest("COMMITTED", state.candidate_tick, 0, state.candidate_refs),
                journal_committed_len=state.journal_committed_len,
                journal_file_len=state.journal_committed_len + 1,
                owner_blobs=state.owner_blobs,
                recovery_receipts=state.recovery_receipts,
                proc_phase="COMMITTING_MANIFEST",
                candidate_tick=state.candidate_tick,
                candidate_refs=state.candidate_refs,
            )
        )

    # CommitTransition
    if state.proc_phase == "COMMITTING_MANIFEST" and state.temp_manifest is not None:
        succs.append(
            State(
                manifest_status="COMMITTED",
                manifest_tick=state.candidate_tick,
                manifest_pending_tick=0,
                manifest_refs=state.candidate_refs,
                temp_manifest=None,
                journal_committed_len=state.journal_file_len,
                journal_file_len=state.journal_file_len,
                owner_blobs=state.owner_blobs,
                recovery_receipts=state.recovery_receipts,
                proc_phase="IDLE",
                candidate_tick=0,
                candidate_refs=frozenset(),
            )
        )

    # RecoveryBegin
    if state.proc_phase == "IDLE" and state.manifest_status == "RECOVERY_REQUIRED":
        succs.append(
            State(
                manifest_status=state.manifest_status,
                manifest_tick=state.manifest_tick,
                manifest_pending_tick=state.manifest_pending_tick,
                manifest_refs=state.manifest_refs,
                temp_manifest=None,
                journal_committed_len=state.journal_committed_len,
                journal_file_len=state.journal_file_len,
                owner_blobs=state.owner_blobs,
                recovery_receipts=state.recovery_receipts,
                proc_phase="RECOVERING_TAIL",
                candidate_tick=0,
                candidate_refs=frozenset(),
            )
        )

    # RecoveryDiscardTail
    if state.proc_phase == "RECOVERING_TAIL":
        succs.append(
            State(
                manifest_status=state.manifest_status,
                manifest_tick=state.manifest_tick,
                manifest_pending_tick=state.manifest_pending_tick,
                manifest_refs=state.manifest_refs,
                temp_manifest=None,
                journal_committed_len=state.journal_committed_len,
                journal_file_len=state.journal_committed_len,
                owner_blobs=state.owner_blobs,
                recovery_receipts=state.recovery_receipts,
                proc_phase="RECOVERING_RECEIPT",
                candidate_tick=0,
                candidate_refs=frozenset(),
            )
        )

    # RecoveryWriteReceipt
    if state.proc_phase == "RECOVERING_RECEIPT":
        receipt = RecoveryReceipt(state.manifest_pending_tick, state.manifest_tick)
        succs.append(
            State(
                manifest_status=state.manifest_status,
                manifest_tick=state.manifest_tick,
                manifest_pending_tick=state.manifest_pending_tick,
                manifest_refs=state.manifest_refs,
                temp_manifest=TempManifest("COMMITTED", state.manifest_tick, 0, state.manifest_refs),
                journal_committed_len=state.journal_committed_len,
                journal_file_len=state.journal_file_len,
                owner_blobs=state.owner_blobs,
                recovery_receipts=state.recovery_receipts | frozenset([receipt]),
                proc_phase="RECOVERING_MANIFEST",
                candidate_tick=0,
                candidate_refs=frozenset(),
            )
        )

    # RecoveryCommitManifest
    if state.proc_phase == "RECOVERING_MANIFEST" and state.temp_manifest is not None:
        succs.append(
            State(
                manifest_status="COMMITTED",
                manifest_tick=state.manifest_tick,
                manifest_pending_tick=0,
                manifest_refs=state.manifest_refs,
                temp_manifest=None,
                journal_committed_len=state.journal_committed_len,
                journal_file_len=state.journal_file_len,
                owner_blobs=state.owner_blobs,
                recovery_receipts=state.recovery_receipts,
                proc_phase="IDLE",
                candidate_tick=0,
                candidate_refs=frozenset(),
            )
        )

    return succs


def run_model_checker(max_tick: int = 2) -> None:
    initial = get_initial_state()
    queue = collections.deque([initial])
    visited = {initial}

    total_transitions = 0

    while queue:
        current = queue.popleft()
        check_invariants(current)

        for succ in next_states(current, max_tick=max_tick):
            total_transitions += 1
            if succ not in visited:
                visited.add(succ)
                queue.append(succ)

    print(f"Model checking passed successfully:")
    print(f"  Max tick: {max_tick}")
    print(f"  Distinct reachable states explored: {len(visited)}")
    print(f"  State transitions explored: {total_transitions}")
    print(f"  Invariant violations: 0")


if __name__ == "__main__":
    run_model_checker(max_tick=2)
