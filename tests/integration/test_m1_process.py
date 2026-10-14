from __future__ import annotations

import json
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

PROFILE = Path(__file__).parents[2] / "profiles/m1/scalar-sensorimotor-mujoco-3.15-v1.json"


def _records(output: str) -> list[dict]:
    return [json.loads(line) for line in output.splitlines()]


def test_fresh_world_process_runs_real_cycles_and_serves_postcommit_observations(tmp_path: Path) -> None:
    state = tmp_path / "state"
    process = subprocess.Popen(
        [sys.executable, "-m", "world.m1_process", "--profile", str(PROFILE), "--state-root", str(state), "--cycles", "20"],
        cwd=Path(__file__).parents[2],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert process.stdout is not None
    lines: list[str] = []
    bridge_payload = None
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        line = process.stdout.readline()
        if line:
            lines.append(line)
            event = json.loads(line)
            if event["event"] == "ready":
                base = event["observation_bridge"]
                request = urllib.request.Request(
                    f"{base}/v1/current", headers={"Origin": "http://127.0.0.1:4173"}
                )
                bridge_payload = json.loads(urllib.request.urlopen(request, timeout=2).read())
            elif event["event"] == "cycle_committed" and bridge_payload is not None and not bridge_payload.get("messages"):
                base = next(record for record in map(json.loads, lines) if record["event"] == "ready")["observation_bridge"]
                request = urllib.request.Request(
                    f"{base}/v1/current", headers={"Origin": "http://127.0.0.1:4173"}
                )
                bridge_payload = json.loads(urllib.request.urlopen(request, timeout=2).read())
        if process.poll() is not None:
            break
    remaining, stderr = process.communicate(timeout=5)
    lines.extend(remaining.splitlines(keepends=True))
    assert process.returncode == 0, stderr
    records = _records("".join(lines))
    assert [record["event"] for record in records[:3]] == ["starting", "initializing", "ready"]
    committed = [record for record in records if record["event"] == "cycle_committed"]
    assert len(committed) == 20
    assert [record["last_verified_committed_tick"] for record in committed] == list(range(1, 21))
    terminal = [record for record in records if record["event"] in {"completed", "stopped", "failed"}]
    assert len(terminal) == 1 and terminal[0]["event"] == "completed"
    assert terminal[0]["last_verified_committed_tick"] == 20
    assert all(len(record["source_revisions"]["world"]) == 40 for record in records)
    assert bridge_payload is not None
    assert bridge_payload["schema"] == "biopeia.observation.current.v1"
    assert bridge_payload["availability"] == "available"
    qualified_families = {
        ".".join((message["observation"]["claim"]["family"]["authority"],
                  *message["observation"]["claim"]["family"]["path"]))
        for message in bridge_payload["messages"]
    }
    assert qualified_families <= {
        "embodiment.body_pose", "environment.finite_scalar_field"
    }


def test_preflight_failure_does_not_create_or_mutate_state_root(tmp_path: Path) -> None:
    state = tmp_path / "not-created"
    bad_profile = tmp_path / "bad.json"
    bad_profile.write_text('{"schema": "invalid"}', encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-m", "world.m1_process", "--profile", str(bad_profile), "--state-root", str(state), "--cycles", "2"],
        cwd=Path(__file__).parents[2], capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 2
    records = _records(result.stdout)
    assert len(records) == 1 and records[0]["event"] == "failed"
    assert records[0]["outcome"] == "preflight_failed"
    assert not state.exists()


def test_sigterm_stops_after_a_verified_cycle_boundary(tmp_path: Path) -> None:
    state = tmp_path / "state"
    process = subprocess.Popen(
        [sys.executable, "-m", "world.m1_process", "--profile", str(PROFILE), "--state-root", str(state), "--cycles", "10000"],
        cwd=Path(__file__).parents[2], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    assert process.stdout is not None
    lines: list[str] = []
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        line = process.stdout.readline()
        if not line:
            if process.poll() is not None:
                break
            continue
        lines.append(line)
        event = json.loads(line)
        if event["event"] == "cycle_committed":
            process.send_signal(signal.SIGTERM)
            break
    remaining, stderr = process.communicate(timeout=30)
    lines.extend(remaining.splitlines(keepends=True))
    assert process.returncode == 0, stderr
    records = _records("".join(lines))
    terminal = [record for record in records if record["event"] in {"completed", "stopped", "failed"}]
    committed = [record for record in records if record["event"] == "cycle_committed"]
    assert len(terminal) == 1 and terminal[0]["event"] == "stopped"
    assert committed
    assert terminal[0]["last_verified_committed_tick"] == committed[-1]["last_verified_committed_tick"]
    assert terminal[0]["commit_id"] == committed[-1]["commit_id"]


def test_existing_state_target_is_byte_for_byte_unchanged_on_preflight_failure(tmp_path: Path) -> None:
    state = tmp_path / "existing"
    state.mkdir()
    sentinel = state / "keep.bin"
    sentinel.write_bytes(b"\x00existing state\xff")
    before = {path.relative_to(state): path.read_bytes() for path in state.rglob("*") if path.is_file()}
    result = subprocess.run(
        [sys.executable, "-m", "world.m1_process", "--profile", str(PROFILE), "--state-root", str(state), "--cycles", "2"],
        cwd=Path(__file__).parents[2], capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 2
    records = _records(result.stdout)
    assert len(records) == 1 and records[0]["event"] == "failed"
    assert records[0]["outcome"] == "preflight_failed"
    after = {path.relative_to(state): path.read_bytes() for path in state.rglob("*") if path.is_file()}
    assert after == before


def test_incompatible_dependency_revision_fails_preflight_without_mutation(tmp_path: Path) -> None:
    state = tmp_path / "not-created"
    wrapper = "\n".join((
        "import sys",
        "import world.m1_process as process",
        "def incompatible_revisions():",
        "    raise RuntimeError('fixture dependency revision mismatch')",
        "process.verified_revisions = incompatible_revisions",
        "raise SystemExit(process.main(['--profile', sys.argv[1], '--state-root', sys.argv[2], '--cycles', '2']))",
    ))
    result = subprocess.run(
        [sys.executable, "-c", wrapper, str(PROFILE), str(state)],
        cwd=Path(__file__).parents[2], capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 2
    records = _records(result.stdout)
    assert len(records) == 1 and records[0]["event"] == "failed"
    assert records[0]["outcome"] == "preflight_failed"
    assert "revision mismatch" in records[0]["error"]
    assert not state.exists()


def test_runtime_failure_reports_last_committed_cut_and_nonzero_exit(tmp_path: Path) -> None:
    state = tmp_path / "state"
    wrapper = "\n".join((
        "import sys",
        "import world.causal_cycle as cycle",
        "real = cycle.run_scalar_sensorimotor_session",
        "def fail_after_commit(*args, **kwargs):",
        "    kwargs['max_cycles'] = 1",
        "    real(*args, **kwargs)",
        "    raise RuntimeError('injected failure after committed cycle')",
        "cycle.run_scalar_sensorimotor_session = fail_after_commit",
        "from world.m1_process import main",
        "raise SystemExit(main(['--profile', sys.argv[1], '--state-root', sys.argv[2], '--cycles', '20']))",
    ))
    result = subprocess.run(
        [sys.executable, "-c", wrapper, str(PROFILE), str(state)],
        cwd=Path(__file__).parents[2], capture_output=True, text=True, timeout=30,
    )
    assert result.returncode != 0
    records = _records(result.stdout)
    committed = [record for record in records if record["event"] == "cycle_committed"]
    terminal = [record for record in records if record["event"] in {"completed", "stopped", "failed"}]
    assert len(committed) == 1
    assert len(terminal) == 1 and terminal[0]["event"] == "failed"
    assert terminal[0]["outcome"] == "recovery_required"
    assert terminal[0]["last_verified_committed_tick"] == committed[0]["last_verified_committed_tick"] == 1
    assert terminal[0]["commit_id"] == committed[0]["commit_id"]
    assert "injected failure after committed cycle" in result.stderr
