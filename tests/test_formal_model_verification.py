"""Test that the formal transition system model satisfies all safety invariants."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

# Add specs directory to path
specs_dir = Path(__file__).resolve().parents[1] / "specs"
if str(specs_dir) not in sys.path:
    sys.path.insert(0, str(specs_dir))

from verify_model import run_model_checker

PINNED_TLA2TOOLS_VERSION = "v1.8.0"
PINNED_TLA2TOOLS_SHA256 = "15d103b7e5617780bcc904c0af12f1ad242e2b345c0301136c5be4588f17edac"


def test_python_differential_model_satisfies_all_invariants() -> None:
    """Run the exhaustive finite-state model checker over the abstract transition system."""
    # Runs BFS state exploration across ticks and crashes; asserts 0 invariant violations
    run_model_checker(max_tick=2)


def test_tlc_model_checker_execution() -> None:
    """Execute TLC directly on specs/WorldCommitRecovery.tla with pinned digest verification.

    Never silently passes if Java or the TLC artifact is missing. Verifies the exact
    artifact hash and checks that TLC reports 0 invariant violations.
    """
    java_bin = shutil.which("java")
    if not java_bin:
        pytest.skip("Java is required to run TLC model checker (not found in PATH)")

    jar_path = Path(os.environ.get("TLA2TOOLS_JAR", "/tmp/tla2tools.jar"))
    if not jar_path.is_file():
        pytest.skip(
            f"TLC jar not found at {jar_path}. Run specs/run_tlc.sh to fetch or set TLA2TOOLS_JAR."
        )

    # Verify cryptographic digest of TLC artifact
    actual_sha = hashlib.sha256(jar_path.read_bytes()).hexdigest()
    assert actual_sha == PINNED_TLA2TOOLS_SHA256, (
        f"TLC jar {jar_path} sha256 mismatch: expected {PINNED_TLA2TOOLS_SHA256}, got {actual_sha}"
    )

    tla_spec = specs_dir / "WorldCommitRecovery.tla"
    result = subprocess.run(
        [java_bin, "-XX:+UseParallelGC", "-cp", str(jar_path), "tlc2.TLC", "-workers", "1", str(tla_spec)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, f"TLC failed with code {result.returncode}:\n{result.stdout}\n{result.stderr}"
    assert "Model checking completed. No error has been found." in result.stdout
    assert "205 states generated, 142 distinct states found" in result.stdout
    assert "0 states left on queue." in result.stdout
