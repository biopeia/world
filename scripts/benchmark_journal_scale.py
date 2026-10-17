"""Reproducible synthetic journal scale diagnostic for World issue #172.

This is a diagnostic, not an acceptance test. By default it includes the
platform fsync calls used by the production journal store. `--algorithm-only`
disables fsync and must not be presented as durable commit latency.
Run from the World repository root with ``uv run python
scripts/benchmark_journal_scale.py --cuts 100 1000 10000``. Results describe
only this host, local filesystem and synthetic fixed-size records. The script
does not make an optimization recommendation or set a performance threshold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import resource
import shutil
import subprocess
import tempfile
import time
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

from world import (
    CausalRecord,
    CheckpointReference,
    DurableJournalSnapshotStore,
    DurableWorldRuntime,
    EventJournal,
    PresenceId,
    PresenceOwnerLink,
    WorldCausalManifestStore,
    WorldGenesisSpec,
    WorldId,
)


class _SyntheticOwnerContract:
    """No-op synthetic owner codec for exercising World restore/recovery."""

    def validate_checkpoint(self, encoded: bytes, owner_identity: str, schema_version: str) -> bool:
        return bool(encoded) and bool(owner_identity) and bool(schema_version)

    def verify_restoration(self, reference: CheckpointReference) -> bool:
        try:
            return self.validate_checkpoint(
                Path(reference.locator).read_bytes(),
                reference.owner_identity,
                reference.schema_version,
            )
        except OSError:
            return False


def _percentile_ms(samples: list[float], percentile: float) -> float:
    ordered = sorted(samples)
    index = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * percentile) - 1))
    return round(ordered[index] * 1000, 4)


def _io_counters() -> dict[str, int]:
    try:
        values = {}
        for line in Path("/proc/self/io").read_text().splitlines():
            key, value = line.split(":", 1)
            if key.strip() in {"rchar", "wchar", "read_bytes", "write_bytes", "syscr", "syscw"}:
                values[key.strip()] = int(value.strip())
        return values
    except (OSError, ValueError):
        return {}


def _environment() -> dict[str, object]:
    cpu_model = "unknown"
    memory_total_kib = None
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                memory_total_kib = int(line.split()[1])
                break
    except (OSError, ValueError):
        pass
    try:
        mount = subprocess.run(
            ["findmnt", "-T", tempfile.gettempdir(), "-n", "-o", "SOURCE,FSTYPE,OPTIONS"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        mount = "unavailable"
    disk = shutil.disk_usage(tempfile.gettempdir())
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_model": cpu_model,
        "cpu_count": os.cpu_count(),
        "memory_total_kib": memory_total_kib,
        "filesystem_target": tempfile.gettempdir(),
        "filesystem_mount": mount,
        "filesystem_free_bytes": disk.free,
    }


def run(cuts: int, *, algorithm_only: bool) -> dict[str, object]:
    world_id = WorldId("world-issue-32-scale")
    journal = EventJournal()
    latencies: list[float] = []
    marginal_bytes: list[int] = []
    reference = None
    append_started = time.perf_counter()
    append_cpu_started = time.process_time()
    io_before = _io_counters()
    with tempfile.TemporaryDirectory(prefix="world-172-journal-") as directory:
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

        append_seconds = time.perf_counter() - append_started
        append_cpu_seconds = time.process_time() - append_cpu_started
        io_after_append = _io_counters()
        assert reference is not None
        retrieval_started = time.perf_counter()
        recovered = store.read(reference)
        retrieval_seconds = time.perf_counter() - retrieval_started
        page_started = time.perf_counter()
        page, cursor, has_more = recovered.page_after(
            f"synthetic-cut-{max(1, cuts // 2):010d}", limit=10
        )
        page_seconds = time.perf_counter() - page_started
        integrity_samples = []
        for _ in range(10):
            sample_started = time.perf_counter()
            integrity = store.verify(reference)
            integrity_samples.append(time.perf_counter() - sample_started)
        io_after_verify = _io_counters()
        journal_bytes = Path(reference.locator).stat().st_size
        original_bytes = Path(reference.locator).read_bytes()
        marker = original_bytes.find(b'"payload":"')
        if marker < 0:
            raise RuntimeError("synthetic payload marker missing")
        corrupted = bytearray(original_bytes)
        payload_byte = marker + len(b'"payload":"')
        corrupted[payload_byte] = ord("y") if corrupted[payload_byte] != ord("y") else ord("z")
        Path(reference.locator).write_bytes(corrupted)
        tamper_rejected = not DurableJournalSnapshotStore.verify(reference)
        Path(reference.locator).write_bytes(original_bytes)
        Path(reference.locator).write_bytes(original_bytes[:-1])
        truncation_rejected = not DurableJournalSnapshotStore.verify(reference)
        Path(reference.locator).write_bytes(original_bytes)
        return {
            "cuts": cuts,
            "mode": "algorithm-only-no-fsync" if algorithm_only else "fsync-enabled-local-filesystem",
            "journal_bytes": journal_bytes,
            "append_wall_seconds": round(append_seconds, 4),
            "append_process_cpu_seconds": round(append_cpu_seconds, 4),
            "append_write_latency_ms_p50": _percentile_ms(latencies, 0.50),
            "append_write_latency_ms_p95": _percentile_ms(latencies, 0.95),
            "append_write_latency_ms_p99": _percentile_ms(latencies, 0.99),
            "peak_rss_kib_process": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "append_io_delta": {
                key: io_after_append[key] - io_before[key]
                for key in io_before.keys() & io_after_append.keys()
            },
            "verification_io_delta": {
                key: io_after_verify[key] - io_after_append[key]
                for key in io_after_append.keys() & io_after_verify.keys()
            },
            "marginal_bytes_min": min(marginal_bytes),
            "marginal_bytes_max": max(marginal_bytes),
            "committed_prefix_read_seconds": round(retrieval_seconds, 4),
            "historical_page_lookup_seconds": round(page_seconds, 6),
            "historical_page_count": len(page),
            "historical_page_cursor": cursor,
            "historical_page_has_more": has_more,
            "integrity_valid": integrity,
            "integrity_verify_latency_ms_p50": _percentile_ms(integrity_samples, 0.50),
            "integrity_verify_latency_ms_p95": _percentile_ms(integrity_samples, 0.95),
            "integrity_verify_latency_ms_p99": _percentile_ms(integrity_samples, 0.99),
            "integrity_verify_sample_count": len(integrity_samples),
            "negative_tests": {
                "same_length_record_tamper_rejected": tamper_rejected,
                "truncated_committed_prefix_rejected": truncation_rejected,
            },
            "recovered_records": len(recovered),
        }


def run_runtime_restore_recovery(cuts: int) -> dict[str, object]:
    """Measure actual DurableWorldRuntime restore and explicit recovery once."""
    world_id = WorldId("world-issue-172-runtime")
    owner_identity = "synthetic-owner"
    contract = _SyntheticOwnerContract()
    contracts = {("organism", owner_identity): contract}
    with tempfile.TemporaryDirectory(prefix="world-172-runtime-") as directory:
        root = Path(directory)
        owner_path = root / "owner.json"
        owner_bytes = b'{"synthetic":true}'
        owner_path.write_bytes(owner_bytes)
        owner_ref = CheckpointReference(
            owner="organism",
            owner_identity=owner_identity,
            checkpoint_id="synthetic-owner-state",
            schema_version="synthetic-v1",
            locator=str(owner_path),
            sha256=hashlib.sha256(owner_bytes).hexdigest(),
        )
        manifest = WorldCausalManifestStore(root / "manifest.json")
        journal_store = DurableJournalSnapshotStore(root / "journal", world_id)
        genesis = WorldGenesisSpec(
            world_id=world_id,
            initial_presence_ids=("synthetic-presence",),
            registered_owner_ids=(("organism", owner_identity),),
            initial_owner_references=(owner_ref,),
            owner_checkpoint_contracts=contracts,
            presence_owner_links=(
                PresenceOwnerLink(PresenceId("synthetic-presence"), "organism", owner_identity),
            ),
        )
        runtime = DurableWorldRuntime.initialize(
            genesis,
            "synthetic-presence",
            manifest,
            journal_store,
            owner_checkpoint_verifier=lambda refs: all(
                Path(ref.locator).is_file()
                and hashlib.sha256(Path(ref.locator).read_bytes()).hexdigest() == ref.sha256
                for ref in refs
            ),
        )
        append_samples = []
        runtime_append_started = time.perf_counter()
        runtime_append_cpu_started = time.process_time()
        runtime_io_before = _io_counters()
        for tick in range(1, cuts + 1):
            started = time.perf_counter()
            runtime.tick(f"synthetic-transition-{tick}", lambda _tick: (owner_ref,))
            append_samples.append(time.perf_counter() - started)
        runtime_append_seconds = time.perf_counter() - runtime_append_started
        runtime_append_cpu_seconds = time.process_time() - runtime_append_cpu_started
        runtime_io_after_append = _io_counters()
        verifier = lambda refs: all(
            Path(ref.locator).is_file()
            and hashlib.sha256(Path(ref.locator).read_bytes()).hexdigest() == ref.sha256
            for ref in refs
        )
        contracts = {("organism", owner_identity): contract}
        started = time.perf_counter()
        restore_cpu_started = time.process_time()
        restore_io_before = _io_counters()
        restored = DurableWorldRuntime.restore(
            world_id,
            "synthetic-presence",
            WorldCausalManifestStore(root / "manifest.json"),
            DurableJournalSnapshotStore(root / "journal", world_id),
            owner_checkpoint_verifier=verifier,
            owner_checkpoint_contracts=contracts,
            owner_checkpoint_restorer=lambda _refs: None,
        )
        restore_seconds = time.perf_counter() - started
        restore_cpu_seconds = time.process_time() - restore_cpu_started
        restore_io_after = _io_counters()
        try:
            restored.tick(
                "synthetic-interrupted-transition",
                lambda _tick: (_ for _ in ()).throw(RuntimeError("synthetic interruption")),
            )
        except RuntimeError as exc:
            if str(exc) != "synthetic interruption":
                raise
        started = time.perf_counter()
        recovery_cpu_started = time.process_time()
        recovery_io_before = _io_counters()
        recovered = DurableWorldRuntime.recover_previous_cut(
            world_id,
            "synthetic-presence",
            WorldCausalManifestStore(root / "manifest.json"),
            DurableJournalSnapshotStore(root / "journal", world_id),
            owner_checkpoint_verifier=verifier,
            owner_checkpoint_contracts=contracts,
            owner_checkpoint_restorer=lambda _refs: None,
        )
        recovery_seconds = time.perf_counter() - started
        recovery_cpu_seconds = time.process_time() - recovery_cpu_started
        recovery_io_after = _io_counters()
        total_storage_bytes = sum(
            path.stat().st_size for path in root.rglob("*") if path.is_file()
        )
        return {
            "cuts": cuts,
            "durable_tick_batch_seconds": round(runtime_append_seconds, 4),
            "durable_tick_batch_process_cpu_seconds": round(runtime_append_cpu_seconds, 4),
            "tick_latency_ms_p50": _percentile_ms(append_samples, 0.50),
            "tick_latency_ms_p95": _percentile_ms(append_samples, 0.95),
            "tick_latency_ms_p99": _percentile_ms(append_samples, 0.99),
            "restore_runtime_seconds": round(restore_seconds, 4),
            "restore_process_cpu_seconds": round(restore_cpu_seconds, 4),
            "restore_io_delta": {
                key: restore_io_after[key] - restore_io_before[key]
                for key in restore_io_before.keys() & restore_io_after.keys()
            },
            "explicit_recovery_seconds": round(recovery_seconds, 4),
            "explicit_recovery_process_cpu_seconds": round(recovery_cpu_seconds, 4),
            "explicit_recovery_io_delta": {
                key: recovery_io_after[key] - recovery_io_before[key]
                for key in recovery_io_before.keys() & recovery_io_after.keys()
            },
            "durable_tick_io_delta": {
                key: runtime_io_after_append[key] - runtime_io_before[key]
                for key in runtime_io_before.keys() & runtime_io_after_append.keys()
            },
            "peak_rss_kib_process": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "runtime_storage_bytes": total_storage_bytes,
            "recovered_causal_tick": recovered.causal_tick,
            "recovery_status_committed": recovered.status.value == "COMMITTED",
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuts", nargs="+", type=int, default=[100, 1000, 10000])
    parser.add_argument("--runtime-cuts", nargs="+", type=int, default=[100, 1000])
    parser.add_argument(
        "--algorithm-only",
        action="store_true",
        help="disable fsync; output is not durable-commit latency evidence",
    )
    args = parser.parse_args()
    if any(cuts < 1 for cuts in args.cuts):
        parser.error("each cut count must be positive")
    print(json.dumps({"environment": _environment() | {
        "fsync_enabled_for_journal_store": not args.algorithm_only,
        "fsync_enabled_for_durable_runtime": True,
    }}, sort_keys=True))
    for cuts in args.cuts:
        print(json.dumps({"journal_store": run(cuts, algorithm_only=args.algorithm_only)}, sort_keys=True))
    for cuts in args.runtime_cuts:
        print(json.dumps({"durable_runtime": run_runtime_restore_recovery(cuts)}, sort_keys=True))


if __name__ == "__main__":
    main()
