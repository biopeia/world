"""Fresh-session M1 World process and JSONL lifecycle protocol."""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import hashlib
import signal
import subprocess
import sys
import uuid
from pathlib import Path

from .m1_profile import construct_fresh_runtime, load_operator_profile

PROTOCOL_VERSION = 1
MAX_CYCLES = 10_000


def verified_revisions() -> dict[str, str]:
    """Record current source/package revisions as provenance, not admission pins."""
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
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"World source revision cannot be verified: {exc}") from exc
    if len(world_revision) != 40:
        raise RuntimeError("World source revision must be a full Git commit id")

    world_version = project["version"]
    if metadata.version("world") != world_version:
        raise RuntimeError("installed World distribution version does not match the source checkout")
    status = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"],
        check=True, capture_output=True, text=True,
    ).stdout
    digest = hashlib.sha256()
    if status:
        diff = subprocess.run(["git", "-C", str(root), "diff", "HEAD", "--binary"],
                              check=True, capture_output=True).stdout
        digest.update(diff)
        paths = subprocess.run(["git", "-C", str(root), "ls-files", "--others", "--exclude-standard", "-z"],
                               check=True, capture_output=True).stdout.split(b"\0")
        for raw_path in sorted(path for path in paths if path):
            path = root / raw_path.decode("utf-8", errors="surrogateescape")
            if path.is_file() and not path.is_symlink():
                digest.update(raw_path + b"\0" + path.read_bytes())
    revisions = {"world": world_revision, "world_package_version": world_version,
                 "world_worktree": "dirty" if status else "clean"}
    if status:
        revisions["world_worktree_sha256"] = digest.hexdigest()
    for requirement in project["optional-dependencies"]["runtime"]:
        name, uri = requirement.split(" @ ", 1)
        distribution = metadata.distribution(name)
        direct_url = json.loads(distribution.read_text("direct_url.json") or "{}")
        actual_revision = direct_url.get("vcs_info", {}).get("commit_id")
        revisions[name] = actual_revision or "unavailable"
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
    parser.add_argument(
        "--cycles", type=int,
        help=f"optional finite cycle count (1-{MAX_CYCLES}); omitted means continuous until stopped",
    )
    parser.add_argument(
        "--organism-checkpoint",
        choices=("v4", "v5"),
        default="v4",
        help="Organism checkpoint schema version: 'v4' (default) or 'v5' (compact with action-history dependencies)",
    )
    args = parser.parse_args(argv)
    if args.cycles is not None and not 1 <= args.cycles <= MAX_CYCLES:
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
            profile, profile_sha256, state_root, source_revisions, observation_port=0,
            organism_checkpoint=args.organism_checkpoint,
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
                velocity_regulation=result.physical_outcome.solver_evidence,
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
            retain_cycle_results=args.cycles is not None,
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
            completed_cycle_count=result.completed_cycle_count,
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
        try:
            if runtime is not None:
                try:
                    runtime.observation_bridge.close()
                except Exception:
                    pass
        finally:
            for signum, handler in original_handlers.items():
                signal.signal(signum, handler)


if __name__ == "__main__":
    raise SystemExit(main())
