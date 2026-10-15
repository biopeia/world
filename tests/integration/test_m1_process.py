from __future__ import annotations

import json
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

PROFILE = Path(__file__).parents[2] / "profiles/m1/scalar-sensorimotor-mujoco-3.15-v2.json"


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
    for record in committed:
        regulation = record["velocity_regulation"]
        assert regulation["policy"] == "m1-fixed-single-hinge-grid-v1"
        assert abs(regulation["applied_torque_nm"]) <= regulation["effort_limit_nm"]
        assert abs(regulation["candidate_velocity_rad_per_s"]) <= regulation["velocity_cap_rad_per_s"]
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
        [sys.executable, "-m", "world.m1_process", "--profile", str(PROFILE), "--state-root", str(state)],
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
    assert terminal[0]["completed_cycle_count"] == len(committed)


def test_signal_during_observation_bridge_teardown_does_not_interrupt_close(tmp_path: Path) -> None:
    state = tmp_path / "state"
    wrapper = "\n".join((
        "import os, signal, sys",
        "import world.m1_process as process",
        "real_construct = process.construct_fresh_runtime",
        "def construct(*args, **kwargs):",
        "    runtime = real_construct(*args, **kwargs)",
        "    real_close = runtime.observation_bridge.close",
        "    def close():",
        "        os.kill(os.getpid(), signal.SIGINT)",
        "        real_close()",
        "    runtime.observation_bridge.close = close",
        "    return runtime",
        "process.construct_fresh_runtime = construct",
        "raise SystemExit(process.main(['--profile', sys.argv[1], '--state-root', sys.argv[2], '--cycles', '10000']))",
    ))
    process = subprocess.Popen(
        [sys.executable, "-c", wrapper, str(PROFILE), str(state)],
        cwd=Path(__file__).parents[2], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, bufsize=1,
    )
    lines = []
    assert process.stdout is not None
    for line in process.stdout:
        lines.append(line)
        event = json.loads(line)
        if event["event"] == "cycle_committed":
            process.send_signal(signal.SIGTERM)
            break
    remaining, stderr = process.communicate(timeout=30)
    lines.extend(remaining.splitlines(keepends=True))
    assert process.returncode == 0, stderr
    assert "KeyboardInterrupt" not in stderr
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


def test_incompatible_installed_world_version_fails_preflight_without_mutation(tmp_path: Path) -> None:
    state = tmp_path / "not-created"
    wrapper = "\n".join((
        "import sys",
        "import world.m1_process as process",
        "import world.m1_process as entrypoint",
        "real_version = entrypoint.metadata.version",
        "entrypoint.metadata.version = lambda name: '0.0.0' if name == 'world' else real_version(name)",
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
    assert "installed World distribution version does not match" in records[0]["error"]
    assert not state.exists()


def test_runtime_failure_reports_last_committed_cut_and_nonzero_exit(tmp_path: Path) -> None:
    state = tmp_path / "state"
    wrapper = "\n".join((
        "import sys",
        "import world.causal_cycle as cycle",
        "import world.observation_publication as publication",
        "def fail_publication(world, plane, *, committed_cut, owner_observations):",
        "    raise publication.ObservationPublicationFailure(committed_cut=committed_cut, delivery_sequences=(), failed_index=0, failure='injected publication failure')",
        "cycle.publish_committed_owner_observations = fail_publication",
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
    assert len(committed) == 0  # No cycle is claimed fully published.
    assert len(terminal) == 1 and terminal[0]["event"] == "failed"
    assert terminal[0]["outcome"] == "failed"
    assert terminal[0]["last_verified_committed_tick"] == 1
    assert terminal[0]["commit_id"] is not None
    assert terminal[0]["publication_uncertain"] is True
    assert "ObservationPublicationFailure" in terminal[0]["error"]
