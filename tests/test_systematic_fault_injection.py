"""Systematic fault-injection harness and invariant verification for World (V-03).

This suite exhaustively verifies the process-termination fault boundaries (G1–G7)
and safety invariants (I1–I6) for World causal commit and recovery, as
formulated in Docs #234, #236, ADR-0083, and ADR-0093.

Failure model: process termination (os._exit). Completed filesystem system
calls remain durable on disk; process memory and in-flight buffers evaporate.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import pytest

from world import (
    CheckpointReference,
    DurableJournalSnapshotStore,
    DurableWorldRuntime,
    ManifestError,
    PresenceId,
    PresenceOwnerLink,
    WorldCausalManifestStore,
    WorldGenesisSpec,
    WorldId,
)
from world.manifest import ManifestStatus, RecoveryRequiredError


class _MockOwnerContract:
    """Minimal opaque owner contract for testing."""

    def validate_checkpoint(
        self, encoded: bytes, owner_identity: str, schema_version: str
    ) -> bool:
        return bool(encoded) and owner_identity == "organism-identity" and schema_version == "v1"

    def verify_restoration(self, reference: CheckpointReference) -> bool:
        path = Path(reference.locator)
        if not path.is_file():
            return False
        return self.validate_checkpoint(path.read_bytes(), reference.owner_identity, reference.schema_version)


_CHILD_RUNNER_CODE = r"""
import hashlib
import json
import os
import sys
from pathlib import Path

mode = sys.argv[1]  # "trace", "init", "prep_fail", or integer abort index
root = Path(sys.argv[2])
op = sys.argv[3]    # "init", "tick", "prep_fail", "recover"

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" if "__file__" in globals() else "src"))
from world import (
    CheckpointReference,
    DurableJournalSnapshotStore,
    DurableWorldRuntime,
    PresenceId,
    PresenceOwnerLink,
    WorldCausalManifestStore,
    WorldGenesisSpec,
    WorldId,
)
import world.manifest as manifest_module
import world.owner_checkpoint_store as owner_module

world_id = WorldId("runner-world")
mstore = WorldCausalManifestStore(root / "manifest.json")
jstore = DurableJournalSnapshotStore(root / "journal", world_id)

class Contract:
    def validate_checkpoint(self, e, o, s):
        return bool(e) and o == "organism-identity" and s == "v1"
    def verify_restoration(self, r):
        p = Path(r.locator)
        return p.is_file() and self.validate_checkpoint(p.read_bytes(), r.owner_identity, r.schema_version)

contracts = {("organism", "organism-identity"): Contract()}

def _verifier(refs):
    return all(
        Path(ref.locator).is_file()
        and hashlib.sha256(Path(ref.locator).read_bytes()).hexdigest() == ref.sha256
        for ref in refs
    )

if op == "init":
    owner_zero = root / "owner-0.json"
    owner_zero.write_bytes(b"owner state at tick 0")
    digest0 = hashlib.sha256(b"owner state at tick 0").hexdigest()
    ref0 = CheckpointReference("organism", "organism-identity", "c0", "v1", str(owner_zero), digest0)
    genesis = WorldGenesisSpec(
        world_id=world_id,
        initial_presence_ids=("presence-a",),
        registered_owner_ids=(("organism", "organism-identity"),),
        initial_owner_references=(ref0,),
        owner_checkpoint_contracts=contracts,
        presence_owner_links=(PresenceOwnerLink(PresenceId("presence-a"), "organism", "organism-identity"),),
    )
    DurableWorldRuntime.initialize(
        genesis, "presence-a", mstore, jstore,
        owner_checkpoint_verifier=_verifier,
    )
    sys.exit(0)

if op == "prep_fail":
    rt = DurableWorldRuntime.restore(
        world_id, "presence-a", mstore, jstore,
        owner_checkpoint_verifier=_verifier,
        owner_checkpoint_contracts=contracts,
        owner_checkpoint_restorer=lambda refs: None,
    )
    try:
        rt.tick("failed-transition", lambda t: (_ for _ in ()).throw(RuntimeError("injected transition failure")))
    except RuntimeError:
        pass
    # Simulate uncommitted journal garbage from interrupted writer
    jfile = root / "journal" / "world-causal-journal-v3.jsonl"
    with jfile.open("ab") as stream:
        stream.write(b"uncommitted-junk-tail\n")
    sys.exit(0)

effects = []
orig_replace = os.replace
orig_link = os.link
orig_unlink = os.unlink
orig_fsync = os.fsync
orig_manifest_fsync_dir = manifest_module._fsync_directory
orig_owner_fsync_dir = owner_module._fsync_directory

abort_index = int(mode) if mode not in ("trace", "init", "prep_fail") else None
effect_counter = 0

def record_effect(kind, *args):
    global effect_counter
    effect_counter += 1
    effects.append((kind, str(args)))
    if abort_index is not None and effect_counter == abort_index:
        os._exit(73)

def traced_replace(src, dst, *args, **kwargs):
    record_effect("replace", src, dst)
    return orig_replace(src, dst, *args, **kwargs)

def traced_link(src, dst, *args, **kwargs):
    record_effect("link", src, dst)
    return orig_link(src, dst, *args, **kwargs)

def traced_unlink(path, *args, **kwargs):
    record_effect("unlink", path)
    return orig_unlink(path, *args, **kwargs)

def traced_fsync(fd):
    record_effect("fsync", fd)
    return orig_fsync(fd)

def traced_manifest_fsync_dir(path):
    record_effect("manifest_fsync_dir", str(path))
    return orig_manifest_fsync_dir(path)

def traced_owner_fsync_dir(path):
    record_effect("owner_fsync_dir", str(path))
    return orig_owner_fsync_dir(path)

os.replace = traced_replace
os.link = traced_link
os.unlink = traced_unlink
os.fsync = traced_fsync
manifest_module._fsync_directory = traced_manifest_fsync_dir
owner_module._fsync_directory = traced_owner_fsync_dir

if op == "tick":
    rt = DurableWorldRuntime.restore(
        world_id, "presence-a", mstore, jstore,
        owner_checkpoint_verifier=_verifier,
        owner_checkpoint_contracts=contracts,
        owner_checkpoint_restorer=lambda refs: None,
    )
    def write_owner(tick):
        p = root / f"owner-{tick}.json"
        p.write_bytes(f"owner state at tick {tick}".encode())
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        return (CheckpointReference("organism", "organism-identity", f"c-{tick}", "v1", str(p), digest),)

    rt.tick("transition-1", write_owner)
    if mode == "trace":
        print(len(effects))
    sys.exit(0)

if op == "recover":
    DurableWorldRuntime.recover_previous_cut(
        world_id, "presence-a", mstore, jstore,
        owner_checkpoint_verifier=_verifier,
        owner_checkpoint_contracts=contracts,
        owner_checkpoint_restorer=lambda refs: None,
    )
    if mode == "trace":
        print(len(effects))
    sys.exit(0)
"""


def _run_child(cmd_args: list[str], root_dir: Path) -> subprocess.CompletedProcess[str]:
    """Execute child process in the World test environment."""
    src_dir = Path(__file__).resolve().parents[1] / "src"
    env = {**os.environ, "PYTHONPATH": str(src_dir)}
    return subprocess.run(
        [sys.executable, "-c", _CHILD_RUNNER_CODE, *cmd_args],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )


def _init_world(root: Path) -> None:
    res = _run_child(["init", str(root), "init"], root)
    assert res.returncode == 0, res.stderr


def _verifier(references: tuple[CheckpointReference, ...]) -> bool:
    return all(
        Path(ref.locator).is_file()
        and hashlib.sha256(Path(ref.locator).read_bytes()).hexdigest() == ref.sha256
        for ref in references
    )


def test_systematic_n_th_effect_tick_fault_injection(tmp_path: Path) -> None:
    """Systematically terminate process at every filesystem effect of tick().

    Verifies Invariants I1 (no unconfirmed cut acceptance), I2 (reference coherence),
    I3 (causal continuity), and I5 (fail-closed) across the full spectrum
    of commit-phase interruptions.
    """
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    _init_world(trace_dir)

    res = _run_child(["trace", str(trace_dir), "tick"], trace_dir)
    assert res.returncode == 0, res.stderr
    total_effects = int(res.stdout.strip())
    assert total_effects >= 8, f"Expected at least 8 effects during tick, got {total_effects}"

    world_id = WorldId("runner-world")
    contracts = {("organism", "organism-identity"): _MockOwnerContract()}

    commit_point_k = None

    for k in range(1, total_effects + 1):
        test_dir = tmp_path / f"tick_k_{k}"
        test_dir.mkdir()
        _init_world(test_dir)

        # Inject process crash at effect k
        res = _run_child([str(k), str(test_dir), "tick"], test_dir)
        assert res.returncode == 73, f"k={k} expected exit code 73, got {res.returncode}: {res.stderr}"

        mstore = WorldCausalManifestStore(test_dir / "manifest.json")
        jstore = DurableJournalSnapshotStore(test_dir / "journal", world_id)
        manifest_on_disk = mstore.read()

        if manifest_on_disk.status is ManifestStatus.COMMITTED and manifest_on_disk.causal_tick == 1:
            if commit_point_k is None:
                commit_point_k = k
            # Invariant I1 & I3: Manifest has crossed commit point; must restore tick 1 safely.
            rt = DurableWorldRuntime.restore(
                world_id,
                "presence-a",
                mstore,
                jstore,
                owner_checkpoint_verifier=_verifier,
                owner_checkpoint_contracts=contracts,
                owner_checkpoint_restorer=lambda refs: None,
            )
            assert rt.causal_tick == 1
            # Next tick advances to tick 2 (Invariant I3)
            p = test_dir / "owner-2.json"
            p.write_bytes(b"owner state at tick 2")
            ref2 = CheckpointReference("organism", "organism-identity", "c-2", "v1", str(p), hashlib.sha256(b"owner state at tick 2").hexdigest())
            rt.tick("transition-2", lambda t: (ref2,))
            assert rt.causal_tick == 2
        else:
            # Pre-commit points: Manifest is either COMMITTED at 0 or PREPARING at 0.
            # Invariant I1: The unconfirmed tick 1 candidate MUST NOT be treated as committed.
            assert manifest_on_disk.causal_tick == 0
            if manifest_on_disk.status is ManifestStatus.PREPARING:
                # Invariant I5 (fail-closed): Startup restore must refuse unrecovered PREPARING
                with pytest.raises(RecoveryRequiredError):
                    DurableWorldRuntime.restore(
                        world_id,
                        "presence-a",
                        mstore,
                        jstore,
                        owner_checkpoint_verifier=_verifier,
                        owner_checkpoint_contracts=contracts,
                        owner_checkpoint_restorer=lambda refs: None,
                    )
                # Recovery must cleanly restore tick 0
                recovered_rt = DurableWorldRuntime.recover_previous_cut(
                    world_id,
                    "presence-a",
                    mstore,
                    jstore,
                    owner_checkpoint_verifier=_verifier,
                    owner_checkpoint_contracts=contracts,
                    owner_checkpoint_restorer=lambda refs: None,
                )
                assert recovered_rt.causal_tick == 0
                # Invariant I3: Subsequent transition from recovered state produces valid tick 1
                p = test_dir / "owner-1.json"
                p.write_bytes(b"owner state at tick 1")
                ref1 = CheckpointReference("organism", "organism-identity", "c-1", "v1", str(p), hashlib.sha256(b"owner state at tick 1").hexdigest())
                recovered_rt.tick("transition-1-retry", lambda t: (ref1,))
                assert recovered_rt.causal_tick == 1
            else:
                assert manifest_on_disk.status is ManifestStatus.COMMITTED
                rt = DurableWorldRuntime.restore(
                    world_id,
                    "presence-a",
                    mstore,
                    jstore,
                    owner_checkpoint_verifier=_verifier,
                    owner_checkpoint_contracts=contracts,
                    owner_checkpoint_restorer=lambda refs: None,
                )
                assert rt.causal_tick == 0

    assert commit_point_k is not None, "Commit point was never crossed in systematic trace"


def test_systematic_n_th_effect_recovery_fault_injection(tmp_path: Path) -> None:
    """Systematically terminate process at every filesystem effect of recover_previous_cut().

    Verifies Invariant I4 (recovery idempotence and re-tryability) and
    Invariant I5 (fail-closed protection) covering all sub-cases of G5.
    """
    trace_dir = tmp_path / "trace_rec"
    trace_dir.mkdir()
    _init_world(trace_dir)
    res = _run_child(["prep_fail", str(trace_dir), "prep_fail"], trace_dir)
    assert res.returncode == 0, res.stderr

    res = _run_child(["trace", str(trace_dir), "recover"], trace_dir)
    assert res.returncode == 0, res.stderr
    total_effects = int(res.stdout.strip())
    assert total_effects >= 10, f"Expected at least 10 effects during recovery, got {total_effects}"

    world_id = WorldId("runner-world")
    contracts = {("organism", "organism-identity"): _MockOwnerContract()}

    recovery_commit_point = None

    for k in range(1, total_effects + 1):
        test_dir = tmp_path / f"rec_k_{k}"
        test_dir.mkdir()
        _init_world(test_dir)
        res = _run_child(["prep_fail", str(test_dir), "prep_fail"], test_dir)
        assert res.returncode == 0, res.stderr

        # Inject termination during recovery at effect k
        res = _run_child([str(k), str(test_dir), "recover"], test_dir)
        assert res.returncode == 73, f"k={k} expected exit code 73, got {res.returncode}: {res.stderr}"

        mstore = WorldCausalManifestStore(test_dir / "manifest.json")
        jstore = DurableJournalSnapshotStore(test_dir / "journal", world_id)
        manifest_on_disk = mstore.read()

        if manifest_on_disk.status is ManifestStatus.COMMITTED:
            if recovery_commit_point is None:
                recovery_commit_point = k
            # Recovery completed its atomic replace
            rt = DurableWorldRuntime.restore(
                world_id,
                "presence-a",
                mstore,
                jstore,
                owner_checkpoint_verifier=_verifier,
                owner_checkpoint_contracts=contracts,
                owner_checkpoint_restorer=lambda refs: None,
            )
            assert rt.causal_tick == 0
        else:
            # Recovery interrupted before completing publication
            assert manifest_on_disk.status is ManifestStatus.RECOVERY_REQUIRED
            # Invariant I5: Normal restore must fail closed
            with pytest.raises(RecoveryRequiredError):
                DurableWorldRuntime.restore(
                    world_id,
                    "presence-a",
                    mstore,
                    jstore,
                    owner_checkpoint_verifier=_verifier,
                    owner_checkpoint_contracts=contracts,
                    owner_checkpoint_restorer=lambda refs: None,
                )
            # Invariant I4: Re-executing recovery must succeed idempotently
            rt = DurableWorldRuntime.recover_previous_cut(
                world_id,
                "presence-a",
                mstore,
                jstore,
                owner_checkpoint_verifier=_verifier,
                owner_checkpoint_contracts=contracts,
                owner_checkpoint_restorer=lambda refs: None,
            )
            assert rt.causal_tick == 0

        # In all cases, subsequent tick must succeed (Invariant I3)
        p = test_dir / "owner-1.json"
        p.write_bytes(b"owner state at tick 1")
        ref1 = CheckpointReference("organism", "organism-identity", "c-1", "v1", str(p), hashlib.sha256(b"owner state at tick 1").hexdigest())
        rt.tick("transition-post-recovery", lambda t: (ref1,))
        assert rt.causal_tick == 1

    assert recovery_commit_point is not None, "Recovery commit point was never reached"


def test_invariant_i1_no_unconfirmed_cut_acceptance(tmp_path: Path) -> None:
    """I1: Incomplete or unconfirmed candidate cuts are never accepted on restore."""
    _init_world(tmp_path)
    world_id = WorldId("runner-world")
    mstore = WorldCausalManifestStore(tmp_path / "manifest.json")
    jstore = DurableJournalSnapshotStore(tmp_path / "journal", world_id)
    contracts = {("organism", "organism-identity"): _MockOwnerContract()}

    runtime = DurableWorldRuntime.restore(
        world_id, "presence-a", mstore, jstore,
        owner_checkpoint_verifier=_verifier,
        owner_checkpoint_contracts=contracts,
        owner_checkpoint_restorer=lambda refs: None,
    )

    # Intercept write_owner to write a candidate checkpoint but crash before commit
    candidate_blob = tmp_path / "owner-candidate.json"
    candidate_blob.write_bytes(b"candidate unconfirmed state")
    candidate_ref = CheckpointReference(
        "organism", "organism-identity", "candidate-c", "v1",
        str(candidate_blob), hashlib.sha256(b"candidate unconfirmed state").hexdigest(),
    )

    with pytest.raises(RuntimeError, match="deliberate failure before commit"):
        runtime.tick("unconfirmed-transition", lambda t: (_ for _ in ()).throw(RuntimeError("deliberate failure before commit")))

    # Manifest is RECOVERY_REQUIRED. restore() must refuse to accept candidate cut
    with pytest.raises(RecoveryRequiredError):
        DurableWorldRuntime.restore(
            world_id, "presence-a", mstore, jstore,
            owner_checkpoint_verifier=_verifier,
            owner_checkpoint_contracts=contracts,
            owner_checkpoint_restorer=lambda refs: None,
        )

    # Recovery brings it back to tick 0, NOT candidate tick 1
    recovered = DurableWorldRuntime.recover_previous_cut(
        world_id, "presence-a", mstore, jstore,
        owner_checkpoint_verifier=_verifier,
        owner_checkpoint_contracts=contracts,
        owner_checkpoint_restorer=lambda refs: None,
    )
    assert recovered.causal_tick == 0
    ref_locators = [ref.locator for ref in recovered.manifest.checkpoint_references]
    assert str(candidate_blob) not in ref_locators


def test_invariant_i2_reference_coherence_across_transitions(tmp_path: Path) -> None:
    """I2: All owner checkpoints in any committed cut belong to the exact same transition."""
    _init_world(tmp_path)
    world_id = WorldId("runner-world")
    mstore = WorldCausalManifestStore(tmp_path / "manifest.json")
    jstore = DurableJournalSnapshotStore(tmp_path / "journal", world_id)
    contracts = {("organism", "organism-identity"): _MockOwnerContract()}

    runtime = DurableWorldRuntime.restore(
        world_id, "presence-a", mstore, jstore,
        owner_checkpoint_verifier=_verifier,
        owner_checkpoint_contracts=contracts,
        owner_checkpoint_restorer=lambda refs: None,
    )

    # Tick 1
    p1 = tmp_path / "owner-1.json"
    p1.write_bytes(b"state 1")
    ref1 = CheckpointReference("organism", "organism-identity", "c-1", "v1", str(p1), hashlib.sha256(b"state 1").hexdigest())
    runtime.tick("t1", lambda t: (ref1,))

    # Tick 2
    p2 = tmp_path / "owner-2.json"
    p2.write_bytes(b"state 2")
    ref2 = CheckpointReference("organism", "organism-identity", "c-2", "v1", str(p2), hashlib.sha256(b"state 2").hexdigest())
    runtime.tick("t2", lambda t: (ref2,))

    manifest = mstore.read()
    assert manifest.causal_tick == 2
    owner_refs = [ref for ref in manifest.checkpoint_references if ref.owner != "world-journal"]
    assert len(owner_refs) == 1
    assert owner_refs[0].checkpoint_id == "c-2"
    assert owner_refs[0].locator == str(p2)


def test_invariant_i3_causal_continuity_multi_cycle(tmp_path: Path) -> None:
    """I3: Multi-cycle causal continuity across alternating commits and recoveries."""
    _init_world(tmp_path)
    world_id = WorldId("runner-world")
    mstore = WorldCausalManifestStore(tmp_path / "manifest.json")
    jstore = DurableJournalSnapshotStore(tmp_path / "journal", world_id)
    contracts = {("organism", "organism-identity"): _MockOwnerContract()}

    runtime = DurableWorldRuntime.restore(
        world_id, "presence-a", mstore, jstore,
        owner_checkpoint_verifier=_verifier,
        owner_checkpoint_contracts=contracts,
        owner_checkpoint_restorer=lambda refs: None,
    )
    assert runtime.causal_tick == 0

    # Cycle 1: Failed transition 1 -> recover back to tick 0
    with pytest.raises(RuntimeError):
        runtime.tick("t1-fail", lambda t: (_ for _ in ()).throw(RuntimeError("fail 1")))
    runtime = DurableWorldRuntime.recover_previous_cut(
        world_id, "presence-a", mstore, jstore,
        owner_checkpoint_verifier=_verifier,
        owner_checkpoint_contracts=contracts,
        owner_checkpoint_restorer=lambda refs: None,
    )
    assert runtime.causal_tick == 0

    # Cycle 2: Successful transition 1
    p1 = tmp_path / "owner-1.json"
    p1.write_bytes(b"state 1")
    ref1 = CheckpointReference("organism", "organism-identity", "c-1", "v1", str(p1), hashlib.sha256(b"state 1").hexdigest())
    runtime.tick("t1-ok", lambda t: (ref1,))
    assert runtime.causal_tick == 1

    # Cycle 3: Successful transition 2
    p2 = tmp_path / "owner-2.json"
    p2.write_bytes(b"state 2")
    ref2 = CheckpointReference("organism", "organism-identity", "c-2", "v1", str(p2), hashlib.sha256(b"state 2").hexdigest())
    runtime.tick("t2-ok", lambda t: (ref2,))
    assert runtime.causal_tick == 2

    # Cycle 4: Failed transition 3 -> recover back to tick 2
    with pytest.raises(RuntimeError):
        runtime.tick("t3-fail", lambda t: (_ for _ in ()).throw(RuntimeError("fail 3")))
    runtime = DurableWorldRuntime.recover_previous_cut(
        world_id, "presence-a", mstore, jstore,
        owner_checkpoint_verifier=_verifier,
        owner_checkpoint_contracts=contracts,
        owner_checkpoint_restorer=lambda refs: None,
    )
    assert runtime.causal_tick == 2

    # Cycle 5: Successful transition 3
    p3 = tmp_path / "owner-3.json"
    p3.write_bytes(b"state 3")
    ref3 = CheckpointReference("organism", "organism-identity", "c-3", "v1", str(p3), hashlib.sha256(b"state 3").hexdigest())
    runtime.tick("t3-ok", lambda t: (ref3,))
    assert runtime.causal_tick == 3

    # Verify journal integrity and continuity across entire history
    reloaded = DurableWorldRuntime.restore(
        world_id, "presence-a", mstore, jstore,
        owner_checkpoint_verifier=_verifier,
        owner_checkpoint_contracts=contracts,
        owner_checkpoint_restorer=lambda refs: None,
    )
    assert reloaded.causal_tick == 3
    # Check that journal records contain tick 0, 1, 2, 3 in strict order
    jref = next(ref for ref in reloaded.manifest.checkpoint_references if ref.owner == "world-journal")
    journal = jstore.read(jref)
    ticks = [rec.causal_tick for rec in journal if rec.record_type == "world.causal_tick.committed"]
    assert ticks == [1, 2, 3]


def test_invariant_i4_recovery_idempotence(tmp_path: Path) -> None:
    """I4: Repeated recovery calls without intermediate ticks are safe and idempotent."""
    _init_world(tmp_path)
    world_id = WorldId("runner-world")
    mstore = WorldCausalManifestStore(tmp_path / "manifest.json")
    jstore = DurableJournalSnapshotStore(tmp_path / "journal", world_id)
    contracts = {("organism", "organism-identity"): _MockOwnerContract()}

    runtime = DurableWorldRuntime.restore(
        world_id, "presence-a", mstore, jstore,
        owner_checkpoint_verifier=_verifier,
        owner_checkpoint_contracts=contracts,
        owner_checkpoint_restorer=lambda refs: None,
    )

    with pytest.raises(RuntimeError):
        runtime.tick("fail-trans", lambda t: (_ for _ in ()).throw(RuntimeError("fail")))

    assert mstore.read().status is ManifestStatus.RECOVERY_REQUIRED

    # First recovery
    rt1 = DurableWorldRuntime.recover_previous_cut(
        world_id, "presence-a", mstore, jstore,
        owner_checkpoint_verifier=_verifier,
        owner_checkpoint_contracts=contracts,
        owner_checkpoint_restorer=lambda refs: None,
    )
    assert rt1.causal_tick == 0
    assert mstore.read().status is ManifestStatus.COMMITTED

    # Second recovery on already-committed cut raises ManifestError without corrupting state
    with pytest.raises(ManifestError, match="explicit recovery requires RECOVERY_REQUIRED state"):
        DurableWorldRuntime.recover_previous_cut(
            world_id, "presence-a", mstore, jstore,
            owner_checkpoint_verifier=_verifier,
            owner_checkpoint_contracts=contracts,
            owner_checkpoint_restorer=lambda refs: None,
        )

    # State is still valid committed tick 0
    rt2 = DurableWorldRuntime.restore(
        world_id, "presence-a", mstore, jstore,
        owner_checkpoint_verifier=_verifier,
        owner_checkpoint_contracts=contracts,
        owner_checkpoint_restorer=lambda refs: None,
    )
    assert rt2.causal_tick == 0


def test_invariant_i5_fail_closed_on_corruptions(tmp_path: Path) -> None:
    """I5: World fails closed on corrupt manifest, missing checkpoints, or invalid hashes."""
    _init_world(tmp_path)
    world_id = WorldId("runner-world")
    manifest_path = tmp_path / "manifest.json"
    mstore = WorldCausalManifestStore(manifest_path)
    jstore = DurableJournalSnapshotStore(tmp_path / "journal", world_id)
    contracts = {("organism", "organism-identity"): _MockOwnerContract()}

    # Case A: Corrupted manifest JSON bytes
    manifest_bytes = manifest_path.read_bytes()
    manifest_path.write_bytes(b"INVALID-JSON-CORRUPTED")
    with pytest.raises(ManifestError, match="cannot read valid World causal manifest"):
        DurableWorldRuntime.restore(
            world_id, "presence-a", mstore, jstore,
            owner_checkpoint_verifier=_verifier,
            owner_checkpoint_contracts=contracts,
            owner_checkpoint_restorer=lambda refs: None,
        )
    # Restore valid manifest
    manifest_path.write_bytes(manifest_bytes)

    # Case B: Missing owner checkpoint file
    owner_file = tmp_path / "owner-0.json"
    owner_backup = owner_file.read_bytes()
    owner_file.unlink()
    with pytest.raises(ManifestError, match="checkpoint dependency closure is not verified"):
        DurableWorldRuntime.restore(
            world_id, "presence-a", mstore, jstore,
            owner_checkpoint_verifier=_verifier,
            owner_checkpoint_contracts=contracts,
            owner_checkpoint_restorer=lambda refs: None,
        )
    owner_file.write_bytes(owner_backup)

    # Case C: Tampered owner checkpoint bytes (hash mismatch)
    owner_file.write_bytes(b"TAMPERED-OWNER-DATA")
    with pytest.raises(ManifestError, match="checkpoint dependency closure is not verified"):
        DurableWorldRuntime.restore(
            world_id, "presence-a", mstore, jstore,
            owner_checkpoint_verifier=_verifier,
            owner_checkpoint_contracts=contracts,
            owner_checkpoint_restorer=lambda refs: None,
        )
    owner_file.write_bytes(owner_backup)

    # Case D: Tampered journal snapshot file
    journal_file = tmp_path / "journal" / "world-causal-journal-v3.jsonl"
    journal_data = bytearray(journal_file.read_bytes())
    journal_data[-2] ^= 1  # flip bit in last record
    journal_file.write_bytes(journal_data)
    with pytest.raises(ManifestError, match="cannot read valid World journal snapshot"):
        DurableWorldRuntime.restore(
            world_id, "presence-a", mstore, jstore,
            owner_checkpoint_verifier=_verifier,
            owner_checkpoint_contracts=contracts,
            owner_checkpoint_restorer=lambda refs: None,
        )


def test_invariant_i6_domain_authority_opaque_owner_payloads(tmp_path: Path) -> None:
    """I6: World treats owner checkpoints opaquely without decoding private payloads."""
    _init_world(tmp_path)
    world_id = WorldId("runner-world")
    mstore = WorldCausalManifestStore(tmp_path / "manifest.json")
    jstore = DurableJournalSnapshotStore(tmp_path / "journal", world_id)

    class OpaqueBinaryContract:
        def validate_checkpoint(self, e: bytes, o: str, s: str) -> bool:
            # Accepts arbitrary non-empty binary bytes (e.g. gzip, pickle, raw C struct)
            return len(e) > 0 and o == "organism-identity" and s == "v1"

        def verify_restoration(self, r: CheckpointReference) -> bool:
            p = Path(r.locator)
            return p.is_file() and self.validate_checkpoint(p.read_bytes(), r.owner_identity, r.schema_version)

    contracts = {("organism", "organism-identity"): OpaqueBinaryContract()}

    runtime = DurableWorldRuntime.restore(
        world_id, "presence-a", mstore, jstore,
        owner_checkpoint_verifier=_verifier,
        owner_checkpoint_contracts=contracts,
        owner_checkpoint_restorer=lambda refs: None,
    )

    # Supply completely non-JSON, arbitrary raw binary data with null bytes and invalid UTF-8
    opaque_binary = b"\x00\xff\xfe\x01\x10\x80\x99\xaa\xbb\xcc\xdd\xee"
    owner_blob = tmp_path / "opaque_owner_state.bin"
    owner_blob.write_bytes(opaque_binary)
    digest = hashlib.sha256(opaque_binary).hexdigest()
    opaque_ref = CheckpointReference("organism", "organism-identity", "opaque-1", "v1", str(owner_blob), digest)

    # World commits the opaque cut without error
    cut_record = runtime.tick("opaque-transition", lambda t: (opaque_ref,))
    assert runtime.causal_tick == 1
    assert cut_record.causal_tick == 1

    # World restores the opaque cut without attempting to interpret the bytes
    restored = DurableWorldRuntime.restore(
        world_id, "presence-a", mstore, jstore,
        owner_checkpoint_verifier=_verifier,
        owner_checkpoint_contracts=contracts,
        owner_checkpoint_restorer=lambda refs: None,
    )
    assert restored.causal_tick == 1
    stored_ref = [ref for ref in restored.manifest.checkpoint_references if ref.owner != "world-journal"][0]
    assert stored_ref.sha256 == digest
    assert Path(stored_ref.locator).read_bytes() == opaque_binary
