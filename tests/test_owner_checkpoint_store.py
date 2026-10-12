from __future__ import annotations

import hashlib
from pathlib import Path

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
