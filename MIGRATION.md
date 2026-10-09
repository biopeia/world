# World migration record

## Source and provenance

- Source repository: `symbiont-lab` at `/home/alessbarb/workspace/repos/active/symbiont-lab`
- Source SHA: `80d372a186513035f85a769db886e505cabba52d`
- Architecture audit baseline: `cef998a55aae32ed8b2f7373079c11667b01e915`
- Relevant source file: `environment/src/environment/events.py`
- Relevant source history: `13720935595fedabec2c376dc933629b0deae5aa` introduced the package path; earlier World journal history includes `c300b7bb8` (age-independent journal indexes/digests) and `be763b6dc` (staged event commit within integrated tick transactions).
- This extraction is copy-first. No source files or scientific assets were modified or moved. No synthetic commit history is claimed.

## Included slice

| Target path | Source provenance | Disposition | Notes |
| --- | --- | --- | --- |
| `src/world/journal.py` | Split/reconstruction from `environment/src/environment/events.py` (`EventJournal`, event snapshot/index/digest/pagination behavior) | SPLIT | Retains append ordering, staged append, replay, per-tick indexing, bounded pages, snapshots, and prefix digests. The source-specific event-kind allowlist and `WorldEvent` schema were intentionally not copied; target records carry opaque `record_type` values so this repository does not pre-empt domain event contracts. |
| `src/world/__init__.py` | New narrow public surface over the selected journal | REWRITE | Exposes only `CausalRecord` and `EventJournal`. |
| `tests/test_journal.py` | Focused tests adapted from `environment/tests/unit/world/test_events.py` plus journal API edge coverage | SPLIT | Covers selected behavior, not the full source World runtime or cross-authority causal commit. |
| `pyproject.toml`, `.github/workflows/ci.yml` | New repository packaging/CI | REWRITE | Minimal standard-library runtime; pytest and build are CI-only tools. |

## Incremental causal coordinator slice

| Target path | Source provenance | Disposition | Notes |
| --- | --- | --- | --- |
| `src/world/runtime.py` | New minimal implementation derived from accepted Biopeia TARGET decisions AD-0005, AD-0006, AD-0009, AD-0019 and the first-tick integration task; no CURRENT runtime code copied | REWRITE | Owns the prototype World ID, one active presence identifier, committed integer tick, synchronous callback/record publication order, in-memory `RECOVERY_REQUIRED` latch, and nonblocking single-tick guard. It imports only the local journal. |
| `tests/test_runtime.py` | New tests for the bounded target-runtime contract | REWRITE | Checks deterministic tick numbering, publish-after-callback ordering, failure latch/no-retry behavior, and fail-fast concurrent/reentrant call rejection. |
| `src/world/__init__.py`, `README.md` | Updated local public exports and scope documentation | REWRITE | Exposes the prototype coordinator and `TickInProgressError` without claiming persistence or participant-state ownership. |

This coordinator deliberately runs the opaque participant callback before journal publication. A callback failure can leave participant mutations in place; a journal append failure can occur after the callback has succeeded. Either failure latches `RECOVERY_REQUIRED`, and no later tick is allowed. Concurrent or reentrant calls fail fast with `TickInProgressError` before invoking their callbacks; this rejection does not itself latch recovery. No rollback or recovery operation is implemented. Both the latch and journal are in-memory only, so this prototype is not durable and does not satisfy AD-0019's durable post-failure contract.

### Opaque callback event payload follow-up

| Target path | Provenance | Disposition | Notes |
| --- | --- | --- | --- |
| `src/world/runtime.py` | Target-only extension of the coordinator above, retaining accepted AD-0005/AD-0006/AD-0009/AD-0019 boundaries; no source implementation copied | REWRITE | A transition callback may return `None` (unchanged behavior) or a JSON object. A returned object is included under `event_payload` in the committed `CausalRecord`; World does not interpret its keys or values. Strict JSON serialization rejects unsupported values and non-finite numbers before record publication. |
| `tests/test_runtime.py` | New target contract tests | REWRITE | Covers payload inclusion, invalid-payload failure/no record, and `None` compatibility. These tests do not establish rollback of callback-owned mutations. |

The payload is optional opaque data, not a new domain event taxonomy or a cross-owner transaction contract. A callback may already have mutated participant state before returning an invalid payload; as with callback or journal failure, World latches `RECOVERY_REQUIRED` and publishes no failed record, but does not roll back those mutations. The payload is in-memory journal data and inherits the durability/recovery limits below. This is a target-only additive rewrite; it does not change or copy the source monorepo.

## Durable causal-cut manifest slice

| Target path | Provenance | Disposition | Notes |
| --- | --- | --- | --- |
| `src/world/manifest.py` | New target contract implementation from MIG-003 and accepted AD-0006, AD-0009, AD-0018, AD-0019, AD-0032 | REWRITE | Stores World ID, committed causal tick, exact opaque owner checkpoint references and hashes, inclusive durable journal `last_sequence`/prefix digest, schema version, and `COMPLETE`/`RECOVERY_REQUIRED` status. It never decodes owner-private state. |
| `tests/test_manifest.py` | New bounded tests for MIG-003 | REWRITE | Covers schema validation, inclusive empty/non-empty journal boundary, persisted pending status, dependency-verification rejection, and successful manifest publication. These are API-level tests, not crash experiments. |

## Exclusive tick-zero manifest initialization follow-up

| Target path | Provenance | Disposition | Notes |
| --- | --- | --- | --- |
| `src/world/manifest.py` | Target-only hardening of the tick-zero manifest initializer, constrained by AD-0010's single-writer/explicit-coordination invariant; no CURRENT code copied | REWRITE | Replaces check-then-replace initialization with same-directory temporary serialization and atomic no-overwrite hard-link publication. This prevents competing initializers from silently replacing the first tick-zero manifest on filesystems supporting same-filesystem hard links. |
| `tests/test_manifest.py` | New concurrent initializer test | REWRITE | Two threads race to initialize one path; verifies one succeeds and the other receives `FileExistsError`, leaving one valid manifest. This is a process-local concurrency test, not multi-process or crash-durability evidence. |

The increment is intentionally limited to first initialization. It does **not** coordinate later manifest updates across processes, implement a lock/lease, establish a deployment multiplicity contract, or resolve the outstanding identity/checkpoint-retention/zero-presence/tick-ordering questions tracked by MIG-003. Hard-link support and durability behavior are platform/filesystem dependent and were not broadly validated. No source-monorepo code was changed.

## Integrated durable tick publication slice

| Target path | Provenance | Disposition | Notes |
| --- | --- | --- | --- |
| `src/world/durable_runtime.py` | New composition over local `runtime.py`, `journal.py`, and `manifest.py`; implements accepted AD-0006, AD-0009, AD-0018, AD-0019, AD-0012 | REWRITE | Writes a durable `RECOVERY_REQUIRED` marker before invoking owner checkpoint writers, persists an immutable journal prefix, verifies the opaque owner dependency closure, and publishes a `COMPLETE` manifest only after those references are available. World does not decode owner checkpoints. |
| `src/world/manifest.py` | Schema v2 amendment in this migration slice | REWRITE | Persists the active presence IDs for the causal cut and rejects restore through a caller-supplied presence absent from that cut. Schema v2 is new, with no historical-format compatibility obligation. |
| `tests/test_durable_runtime.py` | New bounded synthetic filesystem fixtures | REWRITE | Verifies successful publication, validated restore delegation, and refusal to publish a cut with an unavailable owner reference. These are contract tests, not fault injection or filesystem crash-consistency evidence. |
| `src/world/__init__.py` | Updated package exports | REWRITE | Exposes the durable coordinator and journal snapshot store. |

## Explicit bounded recovery slice

| Target path | Provenance | Disposition | Notes |
| --- | --- | --- | --- |
| `src/world/durable_runtime.py` | New implementation over the target manifest/journal APIs, constrained by accepted AD-0009, AD-0018 and AD-0019; no CURRENT recovery implementation copied | REWRITE | Adds `DurableWorldRuntime.recover_previous_cut`: validates the prior committed cut, delegates opaque owner restoration, re-verifies references, then explicitly republishes that cut. |
| `src/world/manifest.py` | New target implementation constrained by accepted AD-0009/AD-0019; no CURRENT source copied | REWRITE | Adds a private recovery publication helper and a content-checked immutable recovery receipt. The parent directory is synced after ensuring the receipt directory exists, and the receipt directory is synced even when an identical receipt preexists, before clearing the pending status. |
| `tests/test_durable_runtime.py` | New synthetic filesystem/API contract tests | REWRITE | Verifies successful explicit restoration/receipt, owner-restore failure, and retry after simulated publication interruption with directory-sync ordering. This is not process/power-loss fault injection or platform durability validation. |
| `README.md` | Updated to describe the bounded recovery contract | REWRITE | Claims only explicit return to the last verified committed cut; does not claim general crash recovery or AD-0019 conformance. |

## Restore dependency revalidation

| Target path | Provenance | Disposition | Notes |
| --- | --- | --- | --- |
| `src/world/durable_runtime.py` | Follow-up to the target-only restore/recovery contract above; no additional source-monorepo code copied | REWRITE | Revalidates opaque checkpoint dependency references after delegated owner restoration and refuses to return an active runtime if their availability/content changed during that call. |
| `tests/test_durable_runtime.py` | Synthetic owner-checkpoint fixture for the Biopeia restore boundary | REWRITE | Covers refusal when a restore callback changes a referenced checkpoint. This is an API boundary test, not a crash/fault-injection test and not a cross-owner consistency proof. |

The additional check narrows a validation gap between pre-restore verification and returning a usable runtime. It does not establish that an owner restorer is permitted to write checkpoint storage, set the manifest to `RECOVERY_REQUIRED`, or provide a general recovery response when references change. A failed restore returns no runtime; callers must handle that failure explicitly. No claim is made about filesystem races after the final verification.

This slice integrates World tick publication with owner-specific checkpoint references and durable journal snapshots. The manifest binds the active presence ID(s) for the cut; the current coordinator still supports only the single presence supplied to its runtime instance and does not provide presence registration/termination orchestration. The owner callback is responsible for persisting each authority's checkpoint through that authority's own contract and returning the exact dependency closure for the proposed cut. `owner_checkpoint_verifier` is the boundary that verifies availability/integrity; it does not prove semantic completeness of the callback's roster. The initial slice had no explicit recovery operation; the bounded follow-up below permits explicit restoration only to the last committed cut. It does not resolve ambiguous publication outcomes, implement roll-forward, provide concurrent-process writer coordination, or prove power-loss/platform guarantees. No fault injection was run.

The store uses same-directory temporary files, file `fsync`, atomic replacement and directory `fsync` on the local filesystem. `begin_transition` must return successfully before causal mutation starts; it publishes durable `RECOVERY_REQUIRED` first. A candidate `COMPLETE` manifest is accepted only for the pending next tick and only after a caller-supplied dependency verifier accepts the owner checkpoint references. If verification fails, the pending recovery record remains.

This slice still does not prove that all references describe a semantically complete cross-owner causal cut or guarantee cross-file/process crash consistency. It delegates owner restoration and does not inspect or interpret private Organism state. The explicit recovery operation is limited to restoring the exact previous committed cut; it does not roll forward an interrupted transition or resolve an ambiguous commit outcome. Owner restoration must be retry-safe if it partially restores state and leaves `RECOVERY_REQUIRED` in place. Filesystem/platform durability behavior has not been experimentally validated; no process/power-loss fault injection was run. The earlier in-memory runtime's transition callback ordering remains unchanged and is not covered by this store's guarantees.

## Excluded and deferred source responsibilities

| Source path/family | Disposition | Reason |
| --- | --- | --- |
| `environment/src/environment/state.py`, `topology.py`, `movement.py`, `checkpoint.py`, `deferred.py` | DEFER | State currently couples World coordination to body placement/occupancy and environmental state. A separate World contract slice requires explicit cross-domain contracts; copying these files would preserve mixed ownership. |
| `environment/src/environment/contracts.py`, `observation.py` | DEFER | Mixed World-facing contracts and passive projection. Observation remains a separate passive capability; only the journal record envelope is included here. |
| `environment/src/environment/genesis*.py`, `constitution.py`, `rng.py` | DEFER | Scenario composition, environment construction, authority fingerprints, and RNG ownership require a broader World/Environment composition boundary. |
| `lab/src/lab/world/{runtime,population,persistence,transaction,cli_view}.py` | DEFER | These compose Environment, Organism, Embodiment, Observation, Lab storage, and application concerns. No wholesale `lab/world` copy was made. |
| `lab/src/lab/integration/world/adapter.py` | DEFER | Adapter composes Organism and Embodiment state, somatic/interoceptive signals, and experimental-clean routes. It crosses several ratified boundaries. |
| Observation, UI, launcher and study callers | DEFER | Transport/presentation, operator control, and experimental apparatus do not belong in this causal journal package. |

## Known rewrites and limits

1. The source `WorldEvent` catalog included event-kind semantics (including unqualified `DEATH`) that should not be installed as a new target event taxonomy. Target `CausalRecord.record_type` is therefore opaque.
2. `CausalRecord`/`EventJournal` are in-memory only. They do not provide durable publication, multi-owner atomicity, recovery, or the `RECOVERY_REQUIRED` state required by accepted TARGET contracts.
3. Staging is an in-memory coordination aid. It is not a transaction across World, Organism, Embodiment, or Environment authorities.
4. Prefix digests establish a deterministic digest over this journal's serialized record representation only; they are not signatures, durable checksums, or proof of scientific validity.
5. The extraction has no runtime dependency on other Biopeia repositories and does not define a final World API or storage format.

## Validation scope

CI builds a wheel, installs it without dependencies, and runs the focused unit tests against Python 3.12. Passing this suite validates only this in-memory journal slice; it does not validate World lifecycle, physical integration, environmental law, organism continuity, persistence, or scientific behavior.
