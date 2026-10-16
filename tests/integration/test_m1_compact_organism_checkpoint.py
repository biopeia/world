"""M1 production composition with the compact Organism checkpoint (organism#183).

Opt-in ``organism_checkpoint="v5"``: real MuJoCo cycles commit v5 Organism
checkpoints whose action history is a World-verified dependency; the default
stays v4.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("mujoco")

from organism.actuation import (  # noqa: E402
    ORGANISM_EXPLORATION_CHECKPOINT_SCHEMA,
    ORGANISM_EXPLORATION_CHECKPOINT_V5_SCHEMA,
)
from organism.actuation.action_history import exploration_v5_to_v4  # noqa: E402

from world.causal_cycle import run_scalar_sensorimotor_session  # noqa: E402
from world.checkpoint_dependency import resolve_dependency  # noqa: E402
from world.m1_profile import construct_fresh_runtime, load_operator_profile  # noqa: E402

PROFILE = Path(__file__).parents[2] / "profiles/m1/scalar-sensorimotor-mujoco-3.15-v2.json"


def _run(state_root: Path, organism_checkpoint: str, cycles: int):
    profile, digest = load_operator_profile(PROFILE)
    runtime = construct_fresh_runtime(profile, digest, state_root, {"world": "test"}, observation_port=0,
                                      organism_checkpoint=organism_checkpoint)
    results = []
    try:
        run_scalar_sensorimotor_session(runtime.world, runtime.observation_plane, run_id="m3", max_cycles=cycles,
                                        cycle_inputs=runtime.cycle_inputs, on_cycle_result=results.append,
                                        source_revisions=runtime.source_revisions)
    finally:
        runtime.observation_bridge.close()
    return [next(r for r in res.committed_cut.owner_checkpoint_references if r.owner == "organism") for res in results]


def test_m1_v5_commits_compact_checkpoints_equivalent_to_v4(tmp_path: Path) -> None:
    cycles = 8
    v4 = _run(tmp_path / "v4", "v4", cycles)
    v5 = _run(tmp_path / "v5", "v5", cycles)
    assert {r.schema_version for r in v4} == {ORGANISM_EXPLORATION_CHECKPOINT_SCHEMA}
    assert {r.schema_version for r in v5} == {ORGANISM_EXPLORATION_CHECKPOINT_V5_SCHEMA}
    assert any(r.dependency_sha256 for r in v5)
    resolve = lambda digest: resolve_dependency(tmp_path / "v5" / "owners", digest)  # noqa: E731
    for r4, r5 in zip(v4, v5):
        assert exploration_v5_to_v4(Path(r5.locator).read_bytes(), resolve=resolve) == Path(r4.locator).read_bytes()


def test_m1_rejects_unknown_organism_checkpoint_option(tmp_path: Path) -> None:
    from world.m1_profile import ProfileError
    profile, digest = load_operator_profile(PROFILE)
    with pytest.raises(ProfileError, match="organism_checkpoint"):
        construct_fresh_runtime(profile, digest, tmp_path / "x", {"world": "test"}, observation_port=0,
                                organism_checkpoint="v6")
