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
