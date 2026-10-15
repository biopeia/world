#!/usr/bin/env python3
"""Reproducible synthetic journal scale diagnostic for World issue #32.

This is a diagnostic, not an acceptance test. By default it includes the
platform fsync calls used by the production journal store. `--algorithm-only`
disables fsync and must not be presented as durable commit latency.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import tempfile
import time
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

from world import CausalRecord, DurableJournalSnapshotStore, EventJournal, WorldId


def run(cuts: int, *, algorithm_only: bool) -> dict[str, object]:
    world_id = WorldId("world-issue-32-scale")
    journal = EventJournal()
    latencies: list[float] = []
    marginal_bytes: list[int] = []
    reference = None
    with tempfile.TemporaryDirectory(prefix="world-32-journal-") as directory:
        store = DurableJournalSnapshotStore(Path(directory), world_id)
        fsync_control = patch("os.fsync", lambda _fd: None) if algorithm_only else nullcontext()
        with fsync_control:
            for index in range(cuts):
                tick = index + 1
                journal.append(
                    CausalRecord(
                        record_id=f"synthetic-cut-{tick:010d}",
                        world_id=world_id.value,
                        causal_tick=tick,
                        record_type="world.synthetic.cut",
                        payload={"payload": "x" * 32},
                    )
                )
                prior_bytes = Path(reference.locator).stat().st_size if reference else 0
                started = time.perf_counter()
                reference = store.write(
                    journal,
                    causal_tick=tick,
                    previous_reference=reference,
                )
                latencies.append(time.perf_counter() - started)
                marginal_bytes.append(Path(reference.locator).stat().st_size - prior_bytes)

        assert reference is not None
        started = time.perf_counter()
        recovered = store.read(reference)
        recovery_seconds = time.perf_counter() - started
        page, cursor, has_more = recovered.page_after(
            f"synthetic-cut-{max(1, cuts // 2):010d}", limit=10
        )
        integrity_started = time.perf_counter()
        integrity = store.verify(reference)
        integrity_seconds = time.perf_counter() - integrity_started
        journal_bytes = Path(reference.locator).stat().st_size
        return {
            "cuts": cuts,
            "mode": "algorithm-only-no-fsync" if algorithm_only else "fsync-enabled-local-filesystem",
            "journal_bytes": journal_bytes,
            "marginal_bytes_min": min(marginal_bytes),
            "marginal_bytes_max": max(marginal_bytes),
            "commit_latency_ms_p50": round(statistics.median(latencies) * 1000, 4),
            "commit_latency_ms_p95_sampled": round(
                sorted(latencies)[int((len(latencies) - 1) * 0.95)] * 1000, 4
            ),
            "read_recovery_seconds": round(recovery_seconds, 4),
            "historical_page_count": len(page),
            "historical_page_cursor": cursor,
            "historical_page_has_more": has_more,
            "integrity_valid": integrity,
            "integrity_seconds": round(integrity_seconds, 4),
            "recovered_records": len(recovered),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuts", nargs="+", type=int, default=[10_000, 100_000, 1_000_000])
    parser.add_argument(
        "--algorithm-only",
        action="store_true",
        help="disable fsync; output is not durable-commit latency evidence",
    )
    args = parser.parse_args()
    if any(cuts < 1 for cuts in args.cuts):
        parser.error("each cut count must be positive")
    for cuts in args.cuts:
        print(json.dumps(run(cuts, algorithm_only=args.algorithm_only), sort_keys=True))


if __name__ == "__main__":
    main()
