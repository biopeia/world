"""Test that the formal transition system model satisfies all safety invariants."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

# Add specs directory to path
specs_dir = Path(__file__).resolve().parents[1] / "specs"
if str(specs_dir) not in sys.path:
    sys.path.insert(0, str(specs_dir))

from verify_model import run_model_checker


def test_python_differential_model_satisfies_all_invariants() -> None:
    """Run the exhaustive finite-state model checker over the abstract transition system."""
    # Runs BFS state exploration across ticks and crashes; asserts 0 invariant violations
    run_model_checker(max_tick=2)


def test_tlc_model_checker_execution() -> None:
    """Execute TLC directly on specs/WorldCommitRecovery.tla if Java is available."""
    java_bin = shutil.which("java")
    if not java_bin:
        return

    jar_path = Path(os.environ.get("TLA2TOOLS_JAR", "/tmp/tla2tools.jar"))
    if not jar_path.is_file():
        # Attempt download if not present
        try:
            import urllib.request
            urllib.request.urlretrieve(
                "https://github.com/tlaplus/tlaplus/releases/download/v1.8.0/tla2tools.jar",
                str(jar_path),
            )
        except Exception:
            return

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
    assert "0 states left on queue." in result.stdout
