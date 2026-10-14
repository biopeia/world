"""Bounded, fresh-session M1 World process and JSONL protocol."""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import signal
import subprocess
import sys
import uuid
from pathlib import Path

from .m1_profile import construct_fresh_runtime, load_operator_profile

PROTOCOL_VERSION = 1
MAX_CYCLES = 10_000


def verified_revisions() -> dict[str, str]:
    """Resolve installed cross-domain VCS commits and the clean World source SHA."""
    import tomllib

    root = Path(__file__).resolve().parents[2]
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    try:
        world_revision = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--verify", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"World source revision cannot be verified: {exc}") from exc
    if len(world_revision) != 40 or dirty:
        raise RuntimeError("World source checkout must have an exact commit and clean worktree")

    world_version = project["version"]
    if metadata.version("world") != world_version:
        raise RuntimeError("installed World distribution version does not match the source checkout")
    revisions = {"world": world_revision, "world_package_version": world_version}
    for requirement in project["optional-dependencies"]["runtime"]:
        name, uri = requirement.split(" @ ", 1)
        expected_revision = uri.rsplit("@", 1)[1]
        distribution = metadata.distribution(name)
        direct_url = json.loads(distribution.read_text("direct_url.json") or "{}")
        actual_revision = direct_url.get("vcs_info", {}).get("commit_id")
        if actual_revision != expected_revision:
            raise RuntimeError(
                f"{name} installed VCS revision cannot be verified: "
                f"expected {expected_revision}, got {actual_revision}"
            )
        revisions[name] = actual_revision
        revisions[f"{name}_package_version"] = metadata.version(name)

    import mujoco

    if mujoco.__version__ != "3.15.0":
        raise RuntimeError("MuJoCo must be exactly 3.15.0")
    revisions["mujoco"] = mujoco.__version__
    return revisions


def emit(
    event: str,
    world_id: str | None,
    run_id: str,
    source_revisions: dict[str, str],
    *,
    tick: int | None = None,
    commit_id: str | None = None,
    outcome: str | None = None,
    profile_id: str | None = None,
    profile_sha256: str | None = None,
    **extra: object,
) -> None:
    record = {
        "protocol_version": PROTOCOL_VERSION,
        "event": event,
        "world_id": world_id,
        "run_id": run_id,
        "profile_id": profile_id,
        "profile_sha256": profile_sha256,
        "source_revisions": source_revisions,
        "last_verified_committed_tick": tick,
        "commit_id": commit_id,
        "outcome": outcome,
        **extra,
    }
    print(json.dumps(record, separators=(",", ":")), flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, help="explicit operator profile JSON path")
    parser.add_argument("--state-root", required=True, help="new, non-existing durable state directory")
    parser.add_argument("--cycles", required=True, type=int, help="positive finite cycle count")
    args = parser.parse_args(argv)
    if not 1 <= args.cycles <= MAX_CYCLES:
        parser.error(f"--cycles must be between 1 and {MAX_CYCLES}")

    run_id = str(uuid.uuid4())
    world_id: str | None = None
    profile_id: str | None = None
    profile_sha256: str | None = None
    source_revisions: dict[str, str] = {}
    state_root = Path(args.state_root).expanduser().absolute()

    # This entire block must remain read-only. The durable initialization
    # boundary is the explicit `initializing` event emitted after it succeeds.
    try:
        profile, profile_sha256 = load_operator_profile(args.profile)
        world_id = profile["world"]["id"]
        profile_id = profile["profile_id"]
        source_revisions = verified_revisions()
        if state_root.exists() or state_root.is_symlink():
            raise ValueError("state root already exists; fresh runs only")
        if state_root.parent.resolve(strict=True) != state_root.parent:
            raise ValueError("state path contains a symlink or unresolved parent")
    except Exception as exc:
        emit(
            "failed", world_id, run_id, source_revisions,
            outcome="preflight_failed", profile_id=profile_id,
            profile_sha256=profile_sha256, error=str(exc),
        )
        return 2

    emit("starting", world_id, run_id, source_revisions, outcome="starting",
         profile_id=profile_id, profile_sha256=profile_sha256)
    emit("initializing", world_id, run_id, source_revisions, outcome="initializing",
         profile_id=profile_id, profile_sha256=profile_sha256)
    stop_requested = {"value": False}
    original_handlers = {
        signum: signal.signal(signum, lambda *_: stop_requested.__setitem__("value", True))
        for signum in (signal.SIGINT, signal.SIGTERM)
    }
    runtime = None
    try:
        runtime = construct_fresh_runtime(
            profile, profile_sha256, state_root, source_revisions, observation_port=0
        )
        runtime.observation_bridge.start()
        bridge_host, bridge_port = runtime.observation_bridge.address
        emit(
            "ready", world_id, run_id, source_revisions, outcome="ready",
            profile_id=profile_id, profile_sha256=profile_sha256,
            observation_bridge=f"http://{bridge_host}:{bridge_port}",
        )

        from .causal_cycle import run_scalar_sensorimotor_session

        def on_cycle(result: object) -> None:
            del result  # The independently reverified current World cut is authoritative.
            manifest = runtime.world.manifest
            cut = runtime.world.committed_cut_for(
                world_id=manifest.world_id,
                causal_tick=manifest.causal_tick,
                commit_id=manifest.commit_id,
                expected_owner_references=tuple(
                    reference for reference in manifest.checkpoint_references
                    if reference.owner != "world-journal"
                ),
            )
            emit(
                "cycle_committed", world_id, run_id, source_revisions,
                tick=cut.causal_tick, commit_id=cut.commit_id,
                outcome="committed_and_published", profile_id=profile_id,
                profile_sha256=profile_sha256,
            )

        result = run_scalar_sensorimotor_session(
            runtime.world,
            runtime.observation_plane,
            run_id=run_id,
            cycle_inputs=runtime.cycle_inputs,
            source_revisions=source_revisions,
            max_cycles=args.cycles,
            stop_requested=lambda: stop_requested["value"],
            on_cycle_result=on_cycle,
        )
        cut = result.last_committed_cut
        event = result.status.value
        emit(
            event, world_id, run_id, source_revisions,
            tick=cut.causal_tick if cut else None,
            commit_id=cut.commit_id if cut else None,
            outcome=event, profile_id=profile_id, profile_sha256=profile_sha256,
            error=result.error,
            publication_uncertain=result.publication_delivery_status_uncertain,
        )
        return 1 if event == "failed" else 0
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        tick = commit_id = None
        try:
            if runtime is not None:
                manifest = runtime.world.manifest
                cut = runtime.world.committed_cut_for(
                    world_id=manifest.world_id,
                    causal_tick=manifest.causal_tick,
                    commit_id=manifest.commit_id,
                    expected_owner_references=tuple(
                        reference for reference in manifest.checkpoint_references
                        if reference.owner != "world-journal"
                    ),
                )
                tick, commit_id = cut.causal_tick, cut.commit_id
        except Exception:
            pass
        emit(
            "failed", world_id, run_id, source_revisions,
            tick=tick, commit_id=commit_id, outcome="recovery_required",
            profile_id=profile_id, profile_sha256=profile_sha256, error=str(exc),
        )
        return 1
    finally:
        for signum, handler in original_handlers.items():
            signal.signal(signum, handler)
        if runtime is not None:
            try:
                runtime.observation_bridge.close()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
