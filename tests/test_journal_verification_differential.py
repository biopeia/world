"""Differential checks for the cached journal-prefix verifier (V-06).

These tests compare the process-local optimization with the full verifier. They
exercise deterministic generated sequences, not a proof over all journals or
filesystem behaviors.
"""

from __future__ import annotations

from pathlib import Path

from world import CausalRecord, DurableJournalSnapshotStore, EventJournal, WorldId


def _record(world_id: str, index: int, value: int) -> CausalRecord:
    return CausalRecord(
        f"record-{index}",
        world_id,
        index,
        "test.generated",
        payload={"index": index, "value": value},
    )


def test_cached_and_full_journal_verifiers_agree_for_seeded_prefixes(tmp_path: Path):
    """Exercise reproducible, varied prefixes with cache hits and cold checks."""
    world_id = WorldId("world-verify-cache-differential")
    store = DurableJournalSnapshotStore(tmp_path / "journal", world_id)
    journal = EventJournal()
    reference = None
    references = []

    # A fixed arithmetic sequence avoids a random dependency while spanning
    # empty/small and larger payload values reproducibly.
    for index in range(64):
        journal.append(_record(world_id.value, index, (index * 7919 + 17) % 65521))
        reference = store.write(
            journal,
            causal_tick=index,
            previous_reference=reference,
        )
        references.append(reference)

    for reference in references:
        full = DurableJournalSnapshotStore.verify(reference)
        cached = store.verify_cached(reference)
        cold = DurableJournalSnapshotStore(tmp_path / "journal", world_id).verify_cached(reference)
        assert cached == full
        assert cold == full
        assert full


def test_cached_verifier_rechecks_when_committed_file_changes(tmp_path: Path):
    world_id = WorldId("world-verify-cache-tamper")
    store = DurableJournalSnapshotStore(tmp_path / "journal", world_id)
    journal = EventJournal()
    journal.append(_record(world_id.value, 0, 1))
    reference = store.write(journal, causal_tick=0)
    assert store.verify_cached(reference)
    assert DurableJournalSnapshotStore.verify(reference)

    path = Path(reference.locator)
    data = bytearray(path.read_bytes())
    data[-3] ^= 1
    path.write_bytes(data)

    assert not DurableJournalSnapshotStore.verify(reference)
    assert store.verify_cached(reference) == DurableJournalSnapshotStore.verify(reference)
    assert not store.verify_cached(reference)
