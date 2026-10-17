from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess
import sys

import pytest

from world.owner_checkpoint_store import (
    FileOwnerCheckpointStore,
    OwnerCheckpointPayload,
)
from world.physical_step import PhysicalStepContext


def _context() -> PhysicalStepContext:
    return PhysicalStepContext(
        world_id="world.store-test",
        source_causal_tick=4,
        target_causal_tick=5,
        physical_step_id="step.store-test.5",
        body_id="body.store-test",
        embodiment_episode_id="episode.store-test",
        morphology_revision="morphology.store-test.r1",
        environment_revision="environment.store-test.r1",
        solver_config_revision="solver.store-test.r1",
    )


def test_file_store_returns_exact_immutable_owner_checkpoint_reference(
    tmp_path,
) -> None:
    store = FileOwnerCheckpointStore(tmp_path / "new" / "checkpoint-root")
    payload = OwnerCheckpointPayload(
        owner="embodiment",
        owner_identity="body.store-test",
        schema_version="embodiment-test-v1",
        encoded=b"owner checkpoint bytes",
    )

    reference, = store.write(_context(), (payload,))

    assert reference.owner == payload.owner
    assert reference.owner_identity == payload.owner_identity
    assert reference.schema_version == payload.schema_version
    assert reference.checkpoint_id.startswith("5:")
    assert reference.sha256 == hashlib.sha256(payload.encoded).hexdigest()
    assert reference.locator.endswith(f"{reference.sha256}.blob")
    assert Path(reference.locator).read_bytes() == payload.encoded

    # Repeated content may reuse the same immutable blob, never overwrite it.
    second, = store.write(_context(), (payload,))
    assert second == reference


@pytest.mark.parametrize(
    "payloads",
    [
        (),
        (
            OwnerCheckpointPayload("organism", "same", "v1", b"one"),
            OwnerCheckpointPayload("organism", "same", "v2", b"two"),
        ),
    ],
)
def test_file_store_rejects_empty_or_ambiguous_checkpoint_sets(tmp_path, payloads) -> None:
    store = FileOwnerCheckpointStore(tmp_path)
    with pytest.raises(ValueError):
        store.write(_context(), payloads)


def test_file_store_writes_owner_authored_genesis_checkpoint_without_step_identity(
    tmp_path: Path,
) -> None:
    store = FileOwnerCheckpointStore(tmp_path / "genesis")
    payload = OwnerCheckpointPayload("organism", "organism.genesis", "v1", b"initial owner bytes")

    reference, = store.write_genesis((payload,))

    assert reference.owner == payload.owner
    assert reference.owner_identity == payload.owner_identity
    assert reference.schema_version == payload.schema_version
    assert reference.checkpoint_id.startswith("genesis:")
    assert reference.sha256 == hashlib.sha256(payload.encoded).hexdigest()
    assert Path(reference.locator).read_bytes() == payload.encoded


def test_file_store_rejects_ambiguous_genesis_checkpoint_sets(tmp_path: Path) -> None:
    store = FileOwnerCheckpointStore(tmp_path)
    duplicate = OwnerCheckpointPayload("organism", "same", "v1", b"one")
    with pytest.raises(ValueError):
        store.write_genesis((duplicate, duplicate))


@pytest.mark.parametrize("termination_point", ["after-link", "after-directory-fsync"])
def test_process_termination_after_owner_blob_link_is_retryable(tmp_path: Path, termination_point: str) -> None:
    """A linked immutable blob remains valid if its writer dies before temp cleanup."""
    encoded = b"owner-authored checkpoint interrupted during publication"
    digest = hashlib.sha256(encoded).hexdigest()
    target = tmp_path / digest[:2] / f"{digest}.blob"
    child = r"""
import hashlib, os, sys
from pathlib import Path
import world.owner_checkpoint_store as module
from world.owner_checkpoint_store import FileOwnerCheckpointStore, OwnerCheckpointPayload
root = Path(sys.argv[1])
point = sys.argv[2]
encoded = b"owner-authored checkpoint interrupted during publication"
digest = hashlib.sha256(encoded).hexdigest()
target = root / digest[:2] / f"{digest}.blob"
if point == "after-link":
    original_link = os.link
    def link_then_exit(source, destination, *args, **kwargs):
        original_link(source, destination, *args, **kwargs)
        if Path(destination) == target:
            os._exit(73)
    os.link = link_then_exit
else:
    original_fsync = module._fsync_directory
    def fsync_then_exit(path):
        original_fsync(path)
        if Path(path) == target.parent and target.is_file():
            os._exit(73)
    module._fsync_directory = fsync_then_exit
FileOwnerCheckpointStore(root).write_genesis((
    OwnerCheckpointPayload("organism", "organism.genesis", "v1", encoded),
))
os._exit(74)
"""
    result = subprocess.run(
        [sys.executable, "-c", child, str(tmp_path), termination_point],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")},
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 73, result.stderr
    assert target.read_bytes() == encoded
    # Process death bypasses finally cleanup. The orphan temp file is expected
    # and must not prevent a later identical content-addressed retry.
    orphan_temps = tuple(target.parent.glob(".checkpoint-*"))
    assert len(orphan_temps) == 1

    reference, = FileOwnerCheckpointStore(tmp_path).write_genesis((
        OwnerCheckpointPayload("organism", "organism.genesis", "v1", encoded),
    ))
    assert reference.sha256 == digest
    assert Path(reference.locator) == target
    assert target.read_bytes() == encoded
