"""Test that the formal transition system model satisfies all safety invariants."""

from __future__ import annotations

import sys
from pathlib import Path

# Add specs directory to path
specs_dir = Path(__file__).resolve().parents[1] / "specs"
if str(specs_dir) not in sys.path:
    sys.path.insert(0, str(specs_dir))

from verify_model import run_model_checker


def test_formal_specification_satisfies_all_invariants() -> None:
    """Run the exhaustive finite-state model checker over WorldCommitRecovery.tla."""
    # Runs BFS state exploration across ticks and crashes; asserts 0 invariant violations
    run_model_checker(max_tick=2)
