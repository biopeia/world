# Runtime M1 World Process Contract

**Status:** Implemented candidate for World Issue #150. This document is not a claim of scientific validation, domain-wide production readiness, migration completion, or longitudinal continuity.

## Decision and profile provenance

The operator profile for this bounded M1 execution is derived from the scalar sensorimotor scenario documented in `biopeia/docs@586dcbb8376c2800fc7ea8e04ba0b486db3e87db`, `technical/migration/MIGRATION_END_TO_END_ACCEPTANCE_REPORT.md`, scenario configuration (lines 48–61). Its adoption was authorized by the workspace user on 2026-10-15 and recorded on World #150. The values are explicit profile/configuration inputs, not measured facts and not runtime observations. This is a new operator-profile adoption; it must not be described as prior approval of a production bootstrap.

The acceptance report records a bounded two-run cycle and an independent review, while explicitly excluding proof of complete bootstrap/genesis recovery, network Observation transport, Frontend completeness, and performance. The process in this contract is fresh-session-only. Restore and continuity remain unsupported here.

## Responsibility boundary

- World owns construction of the World runtime, the scalar session, causal commit, lifecycle status, and post-commit Observation hand-off.
- The operator supplies a versioned profile path and may supply an optional finite cycle count. If omitted, the World process runs continuously until an orderly stop or a legitimate domain termination. The profile contains the named scenario identifiers and initial configuration values from the cited Docs revision.
- The profile does not claim external/physical-world truth. Receptor sample/config values are provenance-bearing scenario inputs. Only values produced by the production cycle and published after a verified World commit may be labeled runtime Observation.
- Workspace may supervise the executable and consume its documented process protocol. It must not construct domain objects, duplicate session semantics, or synthesize Observation.
- No new claim family, history, external goal/reward/cognition, public listener, or persistent Observation history is introduced.

## Input and compatibility policy

1. Accept exactly schema version `1`; reject unknown fields, missing fields, malformed values, non-finite numbers, invalid identifiers, unsupported cycle counts, and any symlink/path ambiguity. The caller supplies one explicit profile file path; no ambient profile root or implicit lookup exists.
2. Make the profile file an explicit command-line input. Do not include an implicit default profile or import any test module/helper at runtime. Keep the shipped example profile separate from tests and label it `operator-supplied, Docs acceptance scenario`.
3. Require exact compatibility with the installed World package and declared dependency pins. Report actual installed distribution versions and verifiable VCS revision metadata. Fail preflight if required revisions cannot be verified; do not treat caller-supplied `source_revisions` as verification.
4. Complete schema, dependency, output-path, and existing-store checks before creating directories, manifests, journals, or checkpoints. A pre-existing target path is refused unless an explicit future restore contract is added. Preflight failures must leave all target paths byte-for-byte unchanged. Runtime initialization starts only after an `initializing` lifecycle record; a failure after that boundary is not mislabeled as a no-mutation preflight failure and must report the durable state as known, unknown, or recovery-required.
5. Initial implementation supports fresh runs only. It does not claim restore, restart continuation, or cross-process identity continuity.

The Docs report describes its historical acceptance run with MuJoCo `3.14.0`; World `main` at this contract's base selects `mujoco==3.15.0` for current runtime/integration. Therefore the old report is provenance for scenario intent and bounded-cycle evidence, not exact-candidate validation for this executable. The implementation uses a distinct, explicitly named M1 profile revision with the same declared physical parameters and an updated solver revision for MuJoCo `3.15.0`; it does not mutate or claim exact reproduction of the historical profile. The process acceptance must run this profile against compatible runtime dependencies and record their actual resolved revisions.

The M1 solver profile revision `operator-r3` enables `m1-fixed-single-hinge-grid-v1`: World tests a deterministic bounded set of applied torques from a fresh MuJoCo state and commits only a candidate whose joint velocity respects the Embodiment cap. It never clips `qvel`. This is an M1 fixed-base single-hinge policy, not a general multi-joint controller or proof of feasibility; failure to find a compliant tested candidate fails the cycle. Each `cycle_committed` JSONL record includes the requested/applied torque, effort limit, velocity cap, and candidate velocity. This evidence is process output, not an extension to the durable CausalRecord or checkpoint journal.

## Process behavior

- One invocation runs serial cycles through `run_scalar_sensorimotor_session` and the existing production cycle/commit path. An explicit cycle count is finite and positive; omission selects continuous mode, which requires a stop predicate. Continuous mode streams cycle results and does not retain an unbounded in-memory result history.
- Standard output is newline-delimited JSON lifecycle records. It emits `starting`, `initializing`, `ready`, one `cycle_committed` record per verified committed cut, and exactly one terminal record (`completed`, `stopped`, or `failed`). Diagnostics go to standard error; no unstructured log lines go to standard output.
- Every status record carries protocol version, World ID, process/run ID where assigned, exact source/package revisions actually resolved, last verified committed tick and commit ID (or explicit null before the first verified cut), and outcome. Never report an in-flight or unverified candidate as committed.
- Each `cycle_committed` record also carries the World-owned `velocity_regulation` evidence for the M1 solver profile; this field is emitted only for the configured M1 regulator and does not claim durable journal persistence.
- `ready` is emitted only after all preflight checks and fresh runtime initialization have succeeded. Preflight failures occur before durable state mutation; failures after `initializing` report the durable state as known, unknown, or recovery-required rather than claiming no mutation. Runtime failures produce a non-zero exit and a terminal `failed` record with the last independently reverified committed cut; preserve uncertainty if commit outcome cannot be verified.
- SIGINT/SIGTERM request cooperative stop at the next cycle boundary. They do not interrupt an in-flight solver, commit, or publication. A stop during an in-flight cycle is reported only after that cycle returns. Signal-driven `stopped` is distinct from finite `completed`.
- Separate commit outcome from publication outcome. A committed cut with incomplete post-commit publication must be represented as such and must not be reported as a fully successful published cycle.
- The current session result's `cycles` contains only cycles whose production method returned full publication results. If a runtime error occurs after World commit but before publication completes, terminal state must not erase the committed cut or imply rollback; surface a non-zero failed/incomplete outcome with the independently verified last cut and explicit publication uncertainty.
- The local Observation bridge binds only to loopback and exposes only the approved post-commit current Observation projections. It is not a public network listener or durable Observation store.

## Invocation

The candidate command is `world-m1 --profile <operator-profile.json> --state-root <new-directory> [--cycles <positive-count>]`. No implicit profile is selected. The state root and its parent must already be unambiguous, and the target root must not exist. An explicit `--cycles` is bounded to 1–10,000; omission runs continuously until signalled or a legitimate domain termination. The process reports its active loopback bridge URL in `ready` and closes the bridge on exit.

## Validation requirements

- Unit tests cover schema validation, unknown keys/version, explicit profile provenance, exact compatibility, and all preflight failures.
- Process tests verify that invalid input changes no bytes under the configured state root; a fresh process emits valid lifecycle records, executes multiple actual cycles, reports verified commit identifiers, cooperatively stops at a cycle boundary, and exits cleanly.
- Failure tests cover startup failure before mutation and runtime/publication failure with correct last-cut and non-zero exit semantics.
- At least one test asserts no runtime import/use of test fixture builders and no Observation publication before successful commit.
- Run the full pinned integration suite against the exact candidate dependency revisions; record exact command, source SHAs, result counts, CI, independent adversarial review, and World owner acceptance.
- Workspace #42 may consume the executable only after this interface is merged and verified. Frontend #16 remains a separately gated passive browser acceptance.

## Non-goals and limits

This contract does not establish empirical validity of the scenario, arbitrary restore, durable run status, cancellation of blocked operations, a network Observation transport, browser rendering completeness, external deployments, universal crash safety, performance guarantees, or migration closure. Those claims require separate owner decisions and evidence.
