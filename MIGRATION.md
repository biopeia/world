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
| `src/world/manifest.py` | New target contract implementation from MIG-003 and accepted AD-0006, AD-0009, AD-0018, AD-0019, AD-0032 | REWRITE | Stores World ID, committed causal tick, exact opaque owner checkpoint references and hashes, inclusive durable journal `last_sequence`/prefix digest, schema version, and `COMMITTED`/`RECOVERY_REQUIRED` status. It never decodes owner-private state. |
| `tests/test_manifest.py` | New bounded tests for MIG-003 | REWRITE | Covers schema validation, inclusive empty/non-empty journal boundary, persisted pending status, dependency-verification rejection, and successful manifest publication. These are API-level tests, not crash experiments. |

## Exclusive tick-zero manifest initialization follow-up

| Target path | Provenance | Disposition | Notes |
| --- | --- | --- | --- |
| `src/world/manifest.py` | Target-only hardening of the tick-zero manifest initializer, constrained by AD-0010's single-writer/explicit-coordination invariant; no CURRENT code copied | REWRITE | Replaces check-then-replace initialization with same-directory temporary serialization and atomic no-overwrite hard-link publication. This prevents competing initializers from silently replacing the first tick-zero manifest on filesystems supporting same-filesystem hard links. |
| `tests/test_manifest.py` | New concurrent initializer test | REWRITE | Two threads race to initialize one path; verifies one succeeds and the other receives `FileExistsError`, leaving one valid manifest. This is a process-local concurrency test, not multi-process or crash-durability evidence. |

At the time of this historical increment, only initialization was serialized. The later schema-v3 follow-up below adds POSIX advisory locking for store operations; this still does not provide distributed locking, consensus, or general platform durability proof. No source-monorepo code was changed.

## Integrated durable tick publication slice

| Target path | Provenance | Disposition | Notes |
| --- | --- | --- | --- |
| `src/world/durable_runtime.py` | New composition over local `runtime.py`, `journal.py`, and `manifest.py`; implements accepted AD-0006, AD-0009, AD-0018, AD-0019, AD-0012 | REWRITE | Writes a durable `RECOVERY_REQUIRED` marker before invoking owner checkpoint writers, persists an immutable journal prefix, verifies the opaque owner dependency closure, and publishes a `COMMITTED` manifest only after those references are available. World does not decode owner checkpoints. |
| `src/world/manifest.py` | Schema v3 amendment in this migration slice | REWRITE | Persists the active presence IDs for the causal cut and rejects restore through a caller-supplied presence absent from that cut. Schema v3 is new, with no historical-format compatibility obligation. |
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

Historical pre-schema-v3 state: World bound the active presence ID(s), but had no presence lifecycle or owner roster API. The owner callback supplied the dependency closure and an aggregate verifier checked availability/integrity only. The initial slice had no explicit recovery operation; the bounded follow-up below added restoration of the prior committed cut. Those earlier limitations are superseded in part by the schema-v3 follow-up below; cross-owner semantics, roll-forward, distributed writer coordination, and crash/power-loss proof remain out of scope.

The store uses same-directory temporary files, file `fsync`, atomic replacement and directory `fsync` on the local filesystem. `begin_transition` must return successfully before causal mutation starts; it publishes durable `PREPARING` first while retaining the previous committed cut. Runtime failure or startup finding a stranded prepare yields `RECOVERY_REQUIRED`. Historically, a candidate manifest was accepted after a caller-supplied dependency verifier accepted the references. Schema v3 adds exact pending commit/participant validation and a registered-owner codec validation gate; if validation fails, the pending recovery record remains.

This historical slice did not prove that all references describe a semantically complete cross-owner causal cut or guarantee cross-file/process crash consistency. Schema v3 adds an explicit registered-owner codec validation gate, but does not claim adversarial validation or empirical crash consistency. It delegates owner restoration and does not inspect or interpret private Organism state. The explicit recovery operation is limited to restoring the exact previous committed cut; it does not roll forward an interrupted transition or resolve an ambiguous commit outcome. Owner restoration must be retry-safe if it partially restores state and leaves `RECOVERY_REQUIRED` in place. Filesystem/platform durability behavior has not been generally validated. The bounded process-termination/reopen evidence added for World issue #33 is recorded below; it is not power-loss fault injection or a general crash-consistency proof. The earlier in-memory runtime's transition callback ordering remains unchanged and is not covered by this store's guarantees.

## Superseded preliminary source-family inventory

The coarse source-family `DEFER` table formerly recorded in this section has
been superseded by the pinned, symbol-level source, test and consumer inventory
below. It is not a second active disposition list. Use that later inventory for
current source dispositions; `DEFER` there remains provisional only where its
row says so.

## Known rewrites and limits

1. The source `WorldEvent` catalog included event-kind semantics (including unqualified `DEATH`) that should not be installed as a new target event taxonomy. Target `CausalRecord.record_type` is therefore opaque.
2. `CausalRecord`/`EventJournal` are in-memory only. They do not provide durable publication, multi-owner atomicity, recovery, or the `RECOVERY_REQUIRED` state required by accepted TARGET contracts.
3. Staging is an in-memory coordination aid. It is not a transaction across World, Organism, Embodiment, or Environment authorities.
4. Prefix digests establish a deterministic digest over this journal's serialized record representation only; they are not signatures, durable checksums, or proof of scientific validity.
5. The extraction has no runtime dependency on other Biopeia repositories and does not define a final World API or storage format.

## Validation scope

CI builds a wheel, installs it without dependencies, and runs the focused unit tests against Python 3.12. Passing this suite validates only this in-memory journal slice; it does not validate World lifecycle, physical integration, environmental law, organism continuity, persistence, or scientific behavior.

# Durable cut participant closure

Owner roster changes after creation are explicit World transitions. Tick-zero genesis is defined by a `WorldGenesisSpec` accepted as authoritative input to World creation: it binds the World ID, initial presences, registered owner identities, initial owner checkpoint references, and owner checkpoint contracts. World verifies that the tick-zero cut matches this accepted specification; it does not introspect the system to discover authorities omitted from it. Therefore the guarantee is completeness relative to the accepted genesis input, not proof that the input enumerates every real authority.


## Ratified participant and commit identity follow-up

Historical schema v3 recorded `world_id`, `causal_tick`, `commit_id`, active `presence_ids`, registered owner identities, step-only participants, opaque checkpoint references, journal sequence/digest, and `PREPARING` / `COMMITTED` / `RECOVERY_REQUIRED` status. Its absence of durable presence-owner association is superseded by the schema v4 migration slice below. The current schema v4 retains those fields and adds the committed and pending `presence_owner_links`. `DurableWorldRuntime` internally asks the manifest store to durably publish `PREPARING`; startup changes an interrupted prepare to `RECOVERY_REQUIRED`, preserving the previous committed cut and proposed roster. A committed transition is accepted only for the pending transition ID/tick and exact World-prepared presence, owner, and presence-owner rosters. The private manifest persistence store serializes operations with a POSIX advisory lock; this is single-logical-writer coordination on supported local filesystems, not distributed locking or consensus. Its mutators are underscored implementation details, not an independent supported commit API.

The runtime exposes explicit presence and owner register/unregister inputs and step-only participants. This records World's prepared roster; it does not discover external owners by introspection. Each registered owner supplies an owner-codec capability via the accepted genesis spec or explicit lifecycle transition. World verifies the reference bytes and digest and asks that owner codec to validate identity and schema; a literal `True` result is an in-process gate and is not materialized as a per-owner receipt. The durable receipt written by explicit recovery describes the rollback operation, not owner validation. The same owner validation is required after restore. The genesis spec is the authoritative declared input to creation and tick-zero closure is checked against it; omitted external authorities remain undiscoverable. `PresenceId` is distinct from owner identity. The former schema-v3 implementation validated a genesis association but did not persist it; that limitation is superseded by schema v4 below.

## Presence-owner roster durability (schema v4)

The Architecture Owner ratified exactly one registered causal owner per active presence. A registered owner may be linked to multiple presences; a presence may not have zero or multiple owners. The relation is explicit and typed as `(PresenceId, owner_id, owner_identity)`; World does not infer ownership from list positions or assume `OwnerId → one PresenceId`.

Schema v4 durably stores the committed `presence_owner_links` and the proposed `pending_presence_owner_links` alongside the corresponding committed and pending presence/owner rosters. `WorldGenesisSpec` requires exactly one explicit link per initial presence, and links must target registered owners. Presence creation requires an owner registration or existing registered owner plus an explicit link in the same causal transition. A presence cannot be committed without exactly one registered owner. Owner unregistration is rejected unless every linked presence is removed or explicitly reassigned in that same transition. A committed manifest must match the exact World-prepared presence-owner roster. Failed/interrupted transitions retain the prior committed association and the pending proposal; rollback recovery restores the prior association. Removing a presence removes its link in the same committed cut. No ownership structure beyond this one-owner-per-presence rule is implied.

Schema v4 is a new strict TARGET format. The decoder rejects schema v3 rather than silently supplying missing links or guessing ownership. This does not provide an upgrade path for existing data; no backward-compatibility behavior is intended. The acceptance evidence covers unit and integration tests for explicit genesis, persistence/reopen, lifecycle registration/removal/reassignment rules, and recovery of the prior association. It does not prove power-loss durability or discover authorities omitted from the accepted genesis input.

**Manifest persistence API boundary:** `WorldCausalManifestStore` is exported as a construction/persistence handle, but its mutators are private-by-convention underscored implementation details. `DurableWorldRuntime` is the supported World commit authority; callers must not publish commits through the store directly. This prevents accidental bypass through the supported public API, not arbitrary in-process code or direct filesystem modification.

**BAW-001 / World issue #31 disposition:** source finding F-0004 was recorded against `biopeia/world@cb4bd6b`, at `src/world/durable_runtime.py:DurableWorldRuntime._verify_owner_receipts` and `tests/test_durable_runtime.py::test_world_issues_receipt_only_after_owner_checkpoint_validation`; current behavior was rechecked on base `f079de0`. The per-owner `OwnerRecoveryReceipt` was removed because it was derived from the same reference it purported to verify, was neither persisted nor returned, and therefore had no effect. Per-owner validation remains the digest check plus the registered codec returning literal `True`. The separate immutable receipt written by `recover_previous_cut` is retained because it records a completed rollback publication. No signed, durable per-owner attestation or malicious-adapter resistance is claimed. Source review: [World issue #31](https://github.com/biopeia/world/issues/31); source documentation finding: [durable runtime explanation](https://github.com/biopeia/docs/blob/main/explanation/world/durable-runtime.md).

**Trust boundary and unproven evidence:** `OwnerCheckpointContract` is the explicit owner registration interface and owner-codec validation gate. The owner validation implementation is trusted in-process code; the Boolean validation result is not signed or independently verifiable against a malicious adapter. Thus World enforces declared-roster closure and fail-closed owner-byte validation, but cannot discover omitted authorities from outside the supplied initial roster or prove storage survival through actual power/process loss. No adversarial sandbox claim or empirical crash-durability claim is made. This remains a bounded migration slice, not proof of full operational durability.

## Historical World and Environment source inventory (pinned audit)

This is a source-disposition inventory, not authorization to copy or preserve
legacy code. The architectural owner has ratified the TARGET split: **World owns
causal runtime, identity, presence/participant coordination, causal ordering,
commit and recovery coordination; Environment owns external environmental
state, spatial geometry/topology, resources, laws, dynamics, genesis and ground
truth.** The historical `environment` package and `lab.world` implementation
predate that split and are mixed CURRENT evidence, not a TARGET authority map.
All outcomes below are migration candidates; a `MOVE`/`SPLIT` identifies
semantic responsibility only. Any implementation must be independently
rewritten to the target contract; no legacy implementation, old API, Hex
compatibility, or historical behavior is being migrated by this record.

- Source repository: `https://github.com/alessbarb/symbiont-lab`
- Pinned source commit: `80d372a186513035f85a769db886e505cabba52d`
- Pinned source tree: `915146b807a2f3b4b362b485d0fd4010bb4c7cde`
- Inventory scope: every Python module under `environment/src/environment/`;
  source World composition under `lab/src/lab/world/` and
  `lab/src/lab/integration/world/`; Environment package metadata; directly
  related Environment/World tests; and Lab callers that directly import
  Environment or Lab World symbols or depend on the listed passive
  World/Environment projections.
- Pin verification: source paths, blobs and path-changing commits were resolved
  against the pinned commit in a read-only source checkout. The source checkout
  was not modified. The same source pin underlies the earlier bounded journal
  slice recorded above.
- Tests cited here establish only historical implementation behavior at this
  pin. They are not target requirements, scientific validation, or permission to
  preserve behavior.

### Source module and top-level symbol dispositions

`DEFER` is provisional and not final. The row states the trigger for resolution;
if the trigger finds no independently justified TARGET capability, close it as
`DROP`. If it finds one, record a target-owned contract and `REWRITE`/`SPLIT` the
required semantics without copying the historical code. The ratified ownership
split is fixed; file-level disposition does not reopen that split.

| Source file: material top-level symbols | Candidate disposition | Target responsibility / reason and target evidence | Source tests, consumers and caveats |
|---|---|---|---|
| `environment/pyproject.toml`: package name/version/description/build metadata | `REWRITE` | Environment owns its independent package metadata and build configuration. Recreate metadata from the target repository's package contract; historical version and description are not compatibility requirements. | No runtime behavior; source blob and path provenance are in the appendix. |
| `environment/__init__.py`: `__version__`, `__all__`, all package re-exports | `REWRITE` | Keep explicit domain-owned public surfaces; do not recreate an omnibus Environment/World facade. World's target public surface is local to `world`; environmental public surface belongs to `environment`; passive projections belong to Observation. | Consumers import many re-exports; see consumer inventory below. No compatibility re-export is required. |
| `checkpoint.py`: `WorldCheckpoint`, `take_checkpoint`, `restore` | `SPLIT` | World owns its causal manifest/commit references; Environment owns its private environment state/checkpoint; Embodiment owns body state. The source's combined snapshot of `WorldState`, `BodyPlacement`, `HexCoord` and `OccupancyGrid` is not a target owner checkpoint. Target World `src/world/manifest.py` and durable runtime are owner-reference coordination only; they do not replace owner codecs. | `environment/tests/unit/world/test_checkpoint.py`, `environment/tests/integration/test_world_checkpoint_continuation.py`. Preserve no combined source snapshot format or API. |
| `constitution.py`: `CONSTITUTION_SCHEMA_VERSION`, `WorldConstitution` | `SPLIT` | Split World identity/genesis/causal-roster assertions from Environment genesis parameters/laws. The new World genesis input is authoritative only for its declared initial World roster; it does not own Environment truth. | `test_constitution.py`, Lab `test_genesis_v1.py`. Historical fingerprint/schema and fixed map config are not TARGET semantics. |
| `contracts.py`: `ContactEvidence`, `ReceivedEmission`, `WorldObservation`, `WorldAction` | `SPLIT` | Separate Environment-facing physical/causal inputs, Organism/Modality-owned action semantics, and passive Observation projection. World coordinates a causal step but must not own participant-private action or environment state. | `test_contracts.py`, Lab actuation/adapter/observation tests. Source records contain mixed authority and implicit hex-world coupling; no old data shape is guaranteed. |
| `deferred.py`: `MAX_QUEUE_SIZE`, `DeferredEffect`, `DeferredEffectQueue` | `DROP` | No generic actor-addressed delayed-effect capability is established by the ratified World contract or the minimal physical-step direction. The source queue's owner, durability and failure semantics are unresolved, and no historical behavior is retained for compatibility. A future need requires a fresh owner-specific contract, not migration of this queue. | `test_deferred.py`, `lab.world.population` and transaction paths. This closes the historical queue artifact only; it does not forbid a separately justified future capability. |
| `events.py`: `EVENT_KINDS`, `WorldEvent`, `EventJournal` | `SPLIT` | `EventJournal` has already been extracted as target `src/world/journal.py` with opaque record types. World may own causal journal ordering; domain event meaning belongs to the relevant authority and passive delivery to Observation. Historical event-kind taxonomy is not to be copied. | `test_events.py`, consumers include Lab population/runtime/transaction and viewer paths. The journal slice is bounded and in-memory unless using separate durable World contract; it does not prove a domain event taxonomy. |
| `genesis.py`: `FieldId`, `ResourceId`, `HazardId`, `RegionId`, `_freeze_nested`, `GroundTruth`, `WorldEnvironment` | `SPLIT` | Environment owns `GroundTruth`, environmental identity/value and Environment genesis. World owns distinct World identity and accepts a `WorldGenesisSpec` for initial presence/owner roster and causal setup. No generic `WorldEnvironment` aggregate should cross-own those states. | `test_genesis.py`, `test_genesis_regional.py`; Lab World population/runtime, integration adapter, observation and studies. Opaque identifier technique may be independently useful, but names/types here are not target contracts. |
| `genesis_v1.py`: fixed `SCHEMA`, `WIDTH`, `HEIGHT`, `SEED`, `FOUNDERS`, `CAPACITY`, `CHECKPOINT_INTERVAL`; `_FIELD_LABELS`, `_RESOURCE_LABELS`, `_HAZARD_LABELS`; `FIELD_IDS`, `RESOURCE_IDS`, `HAZARD_IDS`, `GENESIS_V1_METADATA`; `_fields`, `_resources`, `_hazards`, `_hazard_region`, `_regional_hazards`, `build_ground_truth`, `_laws_hash`, `SMOKE_WIDTH`, `SMOKE_HEIGHT`, `build_constitution`, `GenesisV1`, `build_genesis_v1`, `build_genesis_smoke_v1` | `DROP` | This is a historical fixed hex genesis/preset plus human-readable fixture labels and a fixed population. Target Environment may support explicit and deterministic procedural genesis, but must define its own input contract and generator revision; World GenesisSpec must separately define its roster. No fixed historical map or founder count is canonical. | Lab `test_genesis_v1.py`, source genesis and viability experiments. Preserve provenance/results in historical source; do not transplant the preset, constants, hashes or labels as target truth. |
| `genesis_v2.py`: `REGION_NORTH`, `REGION_SOUTH`, `REGION_METADATA`, `region_of`, `build_ground_truth_v2` | `DROP` | Region override preset is tied to the old hex map and inherits the v1 generator; procedural generation remains a capability, not this implementation or its regional semantics. | Lab `test_genesis_v2.py`, genesis v2 studies. Historical regional results remain evidence; no TARGET region schema is authorized. |
| `laws.py`: `PeriodicFieldLaw`, `ResourceLaw`, `HazardLaw` | `SPLIT` | Environmental laws, resource dynamics and environmental ground truth belong to Environment, never World. Any needed law must be justified and rewritten as an Environment-owned contract with explicit physical units/semantics where relevant; do not migrate these formulas merely because names fit. | `test_laws.py`, Environment genesis, Lab population and environment studies. Tests establish deterministic/bounded properties for these particular historical formulas only. |
| `movement.py`: `resolve_movement` | `DROP` | The old discrete hex movement resolver is not the canonical physical runtime and is explicitly not to be mechanically upgraded to 3D. A justified physical transition is a new World-coordinated step using Environment + Embodiment projections and a solver, then owner adoption. | `test_movement.py`, Lab population and transaction tests. Do not preserve old movement semantics or compatibility as a fallback. |
| `observation.py`: `ResourcePoolItem`, `ResourcePoolObservation`, `opaque_signal_id`, `LOCAL_OCCUPANCY_SIGNAL`, `_LOCAL_OCCUPANCY_SIGNAL`, `local_observation`, `resource_pool_observation` | `SPLIT` | Environment owns resource truth; Observation owns passive projections and semantic claims. World may correlate a committed causal cut but does not become observation producer/authority by hosting the old query. | `test_observation.py`, Lab adapter, World CLI/view, environment-resource and scene projections. The opaque hash and fields are not an approved Observation family/schema. |
| `physics3d/__init__.py` | `DROP` | Empty namespace file; no behavior or standalone contract. | No test/consumer obligation. |
| `physics3d/environments.py`: `ENVIRONMENT_NAMES`, `_METABOLIC_SUPPORT`, `_RECIPES`, `environment_recipe`, `nursery_support_amount`, `resolve_environment`, `motion_parameters`, `fixture_position`, `_fixture_name`, `update_environment`, `build_environment` | `SPLIT` | Separate Environment-owned external geometry/material/field inputs from Lab scientific fixture selection and Embodiment/Organism metabolic support. Scene-building callbacks coupled to a solver are not Environment semantics; a target projection must be engine-independent and owners adopt results. | Lab Physics3D runtime, CLI/effective-model audit, genesis/learning studies and integration tests. The named recipes, nursery semantics, fixture positions and engine-specific scene mutation are not target contracts. |
| `physics3d/resource.py`: `ContactMaterial`, `RESOURCE_MATERIAL`, `PhysicalResource`, `__all__` | `SPLIT` | External material/resource state belongs to Environment; solver contact projections are physical substrate capability; resource interpretation or utility must not be attached by World. Rebuild only the necessary semantic Environment descriptor separately from engine-facing material compilation. | Lab resource tests, Physics3D runtime and resource/observation paths. No physical calibration/equivalence claim follows from these constants. |
| `rng.py`: `derive_world_seed`, `derive_world_rng` | `SPLIT` | Deterministic Environment genesis/random fields belong to Environment input/config; causal stochastic scheduling, if required, belongs to the causal runtime contract; neither may rely on host wall clock or hidden RNG state. Define owner/scope/revision before creating a target API. | `test_rng.py`, genesis, movement, population and physics consumers. Namespace hash behavior is a historical implementation, not an agreed cross-repo RNG protocol. |
| `state.py`: `TickAborted`, `WorldState`, `TickTransaction` | `SPLIT` | `WorldState` combines causal tick/registry with external truth and body placement; `TickTransaction` attempts local rollback across those mixed values. World TARGET owns causal cut coordination, each semantic owner retains its state, and failed transitions must use the accepted recovery protocol rather than assume local rollback. | `test_state.py`, `test_movement.py`, Lab population/transaction/runtime. Historical atomicity is process-local to copied in-memory structures, not a multi-owner durable commit guarantee. |
| `terrain.py`: `CellPhenotype`, `SubstrateImpulseResult`, `DynamicGeography` | `DROP` | The symbols implement historical hex geography/substrate impulse behavior and should not be migrated as a pseudo-3D compatibility layer. If experiments establish a still-required environmental phenomenon, Environment must define and implement a fresh contract from evidence. | Lab terrain/movement and world-ecology tests and historical studies. The disposition is not a claim that these phenomena are scientifically irrelevant; historical evidence stays in source. |
| `topology.py`: `_DIRECTIONS`, `HexCoord`, `BodyPlacement`, `HexTopology`, `OccupancyGrid` | `SPLIT` | `_DIRECTIONS`, `HexCoord`, `HexTopology`: `DROP` as canonical topology (not `HexCoord`→`Vec3`). `BodyPlacement`: body identity/pose belongs to Embodiment; occupancy/presence linkage is World coordination over Environment geometry, so independently specify and rewrite only if required. `OccupancyGrid`: `SPLIT` across World presence/overlap policy and Environment spatial indexing; do not preserve a discrete hex grid for compatibility. | `test_topology.py`, checkpoint, movement, population, adapter, scene and Lab view tests. No top-level array ordering or hex coordinate identity survives as a target guarantee. |
| `lab/world/__init__.py`: package re-exports | `REWRITE` | Replace monolithic Lab facade with independent World contracts and Lab apparatus/application imports. | Lab callers import runtime/genesis/storage/population bundle. No façade compatibility requirement. |
| `lab/world/cli_view.py`: `_label`, `_calculate_density`, `world_snapshot`, `render_world` | `DROP` | CLI visualization is Lab/application or Observation presentation, not World causal authority. Do not migrate this hex renderer as a TARGET view. | `lab/tests/unit/lab/world/test_cli_view.py`, Lab CLI and Workbench callers. Historical view can remain in source. |
| `lab/world/persistence.py`: `PERSISTENCE_SCHEMA_VERSION`, `_restore_rng_state`, `PersistentWorldCheckpoint`, `capture_checkpoint`, `restore_population_from_checkpoint`, `WorldStorage` | `SPLIT` | World stores causal manifest and owner refs; owners checkpoint their own authoritative state; Lab may own experiment/run storage and evidence. The historical aggregate checkpoint is not a target format. Target World store/manifest already exists independently; its bounded guarantees and limits are documented earlier in this file. | `lab/tests/unit/lab/world/test_persistence.py`, integration continuation, CLI/runtime/studies. Do not migrate schema v5, whole-world serialization or legacy restore behavior. |
| `lab/world/population.py`: `founder_placement`, `PopulationTickRecord`, `PopulationGenesisRuntime` | `SPLIT` | World owns presence/lifecycle and causal coordination; Environment owns genesis; Organism owns identity/longitudinal state; Lab owns controlled population-generation study design. Candidate boundary must be explicit before any runtime is implemented. | Lab population, actuation, ecology and genesis tests. Historical population API couples owner creation to hex placement and WorldEnvironment. |
| `lab/world/runtime.py`: `WorldRuntimeState` | `SPLIT` | World may compose causal steps; Lab `WorldRuntimeState` also creates fixed Environment genesis, renders views, and uses Lab storage. No wholesale class moves; retain only an independently justified World composition contract. | Lab runtime, CLI, studies and integration tests. Existing Biopeia World runtime is a separate target implementation and this source is not provenance for it. |
| `lab/world/transaction.py`: `IntegratedWorldTickTransaction` | `REWRITE` | World owns cross-owner causal commit/recovery boundary, but legacy transaction is an in-memory multi-object mutation/rollback coordinator. Replace only after proving owner checkpoint/adoption and durable cut protocol; do not copy implementation or assert durable atomicity from tests. | `lab/tests/unit/lab/world/test_transaction_integrity.py`, population/runtime, actuation integration. Tests are local transaction assertions, not crash recovery or multi-process evidence. |
| `lab/integration/world/adapter.py`: `LONGITUDINAL_CONTRACT`, `_OCCUPANCY_SIGNAL`, `_RECEPTION_PRESENT_SIGNAL`, `_RECEPTION_SYMBOL_SIGNAL`, `_RECEPTION_INTENSITY_SIGNAL`, `_LOCAL_SURFACE_WATER_SIGNAL`, `_LOCAL_DETRITUS_SIGNAL`, `_LOCAL_DISTURBANCE_SIGNAL`, `_PHYSICAL_RECEPTOR_IDS`; `physical_receptor_ids`, `WorldDiscoveryProvider`, `WorldReadingProvider`, `local_substrate_signals`, `_capabilities_for`, `_unit_interval`, `_mix_weight`, `physical_receptor_signals`, `clean_world_observation`, `_load_base_genome`, `ActuationBinding`, `ActuationBindingConstitution`, `ActuationAdapter`, `default_world_actuation_binding`, `ActionExecutionResult`, `WorldTickRecord`, `_OrganismRig`, `_construct_organism`, `_act` | `SPLIT` | This Lab integration mixes Organism construction/actions, Environment ground truth/topology, modality-like signal adaptation, Embodiment actuators, and World tick. Split by semantic owner/capability; rebuild only the minimal justified integration after target contracts exist. | Lab adapter/actuation tests, end-to-end tests, embodiment and reproduction integrations. `clean_world_observation` and its physical receptor signals are not automatically approved Observation semantics; do not keep old test route as compatibility. |
| `lab/integration/world/__init__.py`: legacy package docstring | `DROP` | The file only documents coupling between the historical hex World and Organism; it has no runtime behavior or independent TARGET contract. Drop it from TARGET migration without deleting or modifying the source artifact. | Pinned source blob: `366e8f126df46bb6eda3cf0bff057c9d06c671cc` (54 bytes); source remains preserved at the pinned commit. |
| `lab/observation/world_scene.py`: `SCENE_CONTRACT`, `project_world_scene`, `WorldScenePublisher`, `apply_world_event` | `SPLIT` | Passive visualization/projection belongs to Observation or Lab presentation; World may provide committed causal data only. Presentation event path is not World event authority. | Lab scene, PhysicsWorldObserver and Observatory integration tests. Existing scene contract/version is not target wire-format authorization. |
| `lab/physics3d/world_observation.py`: `PhysicsWorldObserver` | `SPLIT` | Passive projection belongs to Observation/Lab visualization; physical solver output and Environment/Embodiment adoption remain separate. World only correlates with the committed causal cut. | `lab/tests/unit/lab/observation/test_physics_world_observer.py`, Physics3D runtime and observation consumers. Do not infer semantics from the observer class name. |

### Source test-file inventory and disposition

All listed source tests were checked at source commit
`80d372a186513035f85a769db886e505cabba52d` (tree
`915146b807a2f3b4b362b485d0fd4010bb4c7cde`). `SPLIT` means that only
independently justified semantics are assigned to their Biopeia owners; it
does not migrate the source test implementation. `DROP` means the historical
test is not a TARGET test. Tests remain preserved in the read-only source
repository, and any future TARGET tests must be written against owner-specific
Biopeia contracts rather than ported for compatibility.

| Pinned source test file | Disposition | What it historically checks / target caveat |
|---|---|---|
| `environment/tests/test_self_contained.py` | `SPLIT` | Its standalone-install/domain-isolation check may be replaced by target package-boundary validation; historical fixture-recipe and `derive_world_rng` assertions do not transfer. No source test implementation or old API is retained. |
| `environment/tests/unit/world/test_checkpoint.py` | `SPLIT` | Historical aggregate hex/body/RNG snapshot is not a target checkpoint; World cut references and Environment/Embodiment owner checkpoints are separate contracts. |
| `environment/tests/unit/world/test_constitution.py` | `SPLIT` | Historical fingerprint bundled fixed Environment configuration with World identity; target WorldGenesisSpec and EnvironmentGenesisSpec have separate semantics. |
| `environment/tests/unit/world/test_contracts.py` | `SPLIT` | Historical action, observation and contact records combine authorities; use independently specified owner contracts, not the old shapes. |
| `environment/tests/unit/world/test_deferred.py` | `DROP` | Tests only the historical actor-addressed bounded queue, which is not a ratified generic target capability. Any future delayed-effect contract needs fresh owner-specific tests. |
| `environment/tests/unit/world/test_events.py` | `SPLIT` | Historical WorldEvent taxonomy plus EventJournal behavior; journal behavior is already covered by target `tests/test_journal.py`, but event-kind taxonomy is not a target commitment. |
| `environment/tests/unit/world/test_genesis.py` | `SPLIT` | Fixed hex Environment fields/resources/hazards are not World genesis; World roster genesis and any justified Environment genesis require separate contracts and tests. |
| `environment/tests/unit/world/test_genesis_regional.py` | `DROP` | Specific v1 hex regional overrides and legacy fallback behavior; no target compatibility obligation. |
| `environment/tests/unit/world/test_laws.py` | `SPLIT` | External laws/dynamics are Environment-owned, not World behavior; no historical formula or numeric bound transfers without independent justification. |
| `environment/tests/unit/world/test_movement.py` | `DROP` | Hex move/competition/rollback behavior is not the target 3D physical step. |
| `environment/tests/unit/world/test_observation.py` | `SPLIT` | Historical Environment queries and occupancy observations; Environment truth vs passive Observation claims must be tested separately. |
| `environment/tests/unit/world/test_rng.py` | `DROP` | Tests the historical World seed-derivation implementation only. No universal TARGET RNG derivation contract is established; a future owner-specific randomness contract needs fresh tests. |
| `environment/tests/unit/world/test_state.py` | `DROP` | In-memory aggregate WorldState rollback semantics conflict with owner-owned state and durable cut contract; target World tests are separate. |
| `environment/tests/unit/world/test_topology.py` | `DROP` | Hex topology, occupancy and hex body placement. Do not translate assertions mechanically to 3D. |
| `environment/tests/unit/lab/world/test_genesis_v1.py` | `DROP` | Historical fixed preset counts, labels/IDs and fingerprint. Retain historical experiment evidence only. |
| `environment/tests/unit/lab/world/test_genesis_v2.py` | `DROP` | Specific regional preset and v1 identity/fallback semantics. |
| `environment/tests/integration/test_world_checkpoint_continuation.py` | `SPLIT` | Historical aggregate restore/trajectory behavior is not retained. Target continuity must be tested through owner checkpoints and World causal cuts, without preserving this aggregate format. |
| `lab/tests/unit/lab/world/test_actuation_adapter.py` | `SPLIT` | Lab composition of action/Embodiment/World adapter; revalidate per target owner contracts. |
| `lab/tests/unit/lab/world/test_actuation_end_to_end.py` | `SPLIT` | Historical Organism→actuation→hex World loop; not evidence of target 3D physical causal step. |
| `lab/tests/unit/lab/world/test_adapter.py` | `SPLIT` | Cross-domain signal and action adapter; no monolithic adapter survives by default. |
| `lab/tests/unit/lab/world/test_cli_view.py` | `DROP` | Hex display rendering only. |
| `lab/tests/unit/lab/world/test_hazard_ecology_v2.py` | `DROP` | Legacy hazard/ecology labels and regional preset. |
| `lab/tests/unit/lab/world/test_persistence.py` | `SPLIT` | Whole-world source checkpoint and Lab storage; target owner-ref manifest tests are not equivalent to whole source serialization. |
| `lab/tests/unit/lab/world/test_population.py` | `SPLIT` | Founder generation, placement, Environment and Organism runtime coupling; ownership must be separated. |
| `lab/tests/unit/lab/world/test_terrain_and_movement.py` | `DROP` | Hex terrain/movement result tests. |
| `lab/tests/unit/lab/world/test_transaction_integrity.py` | `REWRITE` | Historical transaction rollback invariants; target protocol is owner adoption + durable World commit/recovery, not copied rollback internals. |
| `lab/tests/unit/lab/world/test_world_ecology_v2.py` | `DROP` | Historical hex ecology and regional labels; preserve as source evidence, not target contract. |
| `lab/tests/unit/lab/observation/test_physics_world_observer.py` | `SPLIT` | Physics/Environment observation; owner claims and passive projection must be separately tested. |
| `lab/tests/unit/lab/observation/test_world_environments.py` | `SPLIT` | Legacy Environment recipes and Lab/world query integration; no target API equivalence. |
| `lab/tests/unit/lab/observation/test_world_scene.py` | `SPLIT` | Passive world scene projection and presentation event application; no transport/schema authority. |
| `lab/tests/unit/lab/observation/test_observability_integrity.py` | `SPLIT` | This broad Observation/Lab test file includes a `world_scene` overflow/rebase assertion; that passive transport behavior is not World authority and must be tested at the owning Observation/Lab boundary if retained. The rest of the file is outside this World inventory. |
| `lab/tests/unit/observatory/test_world_integration.py` | `SPLIT` | Observatory visualization integration; Observation/Lab presentation not causal World. |

### Pinned source consumers and cross-domain boundaries

The following source files directly import Environment modules, Lab World
composition, or target-relevant World/Environment projection. Consumers are
listed to show coupling and migration impact, not as target requirements. Their
material responsibilities must be re-audited when a Biopeia caller is
implemented; update callers to the owning repository instead of retaining a
legacy façade. Each path's source blob and latest path-changing commit are
included in the provenance appendix below.

| Consumer family and source paths | Candidate disposition | Why this is cross-domain / what remains |
|---|---|---|
| Lab runtime/CLI/API: `lab/src/lab/cli/world.py`; `lab/src/lab/app/physics3d/runs.py`; `lab/src/lab/physics3d/cli.py`; `lab/src/lab/physics3d/runtime.py`; `lab/src/lab/server/api.py` | `SPLIT` | Lab application/run orchestration, Environment projection and World causal orchestration are mixed. Keep app/run controls and scientific apparatus in Lab; route authoritative changes through target owners. |
| Lab physics/model auditing: `lab/src/lab/physics3d/effective_model_audit.py`; `lab/src/lab/physics3d/effective_model_audit_cli.py`; `lab/src/lab/physics3d/world_observation.py` | `SPLIT` | Solver execution/audit and passive physical projection are capabilities/Observation; Environment/Embodiment adopt authoritative results; World coordinates/commits. |
| Lab domain integration and studies: `lab/src/lab/integration/world/adapter.py`; `lab/src/lab/studies/world/genesis_viability.py`; `lab/src/lab/studies/embodiment/integrity_gates.py`; `lab/src/lab/studies/embodiment/label_invariance.py` | `SPLIT` | Integrations and protocols remain Lab; state semantics and environmental truth belong to their authorities. Preserve experiment definitions/results/provenance; do not move Lab controls into World. |
| Lab Observation/projection: `lab/src/lab/observation/environment_resource.py`; `lab/src/lab/observation/world_scene.py`; downstream consumers `lab/src/lab/observation/bus.py`, `lab/src/lab/observation/physics3d.py` | `SPLIT` | Environment state projection vs Observation transport/presentation/evidence admission; the bus and Physics3D bridge consume passive `world_scene` projections and are not World authorities. No causal writeback from observer. |
| Lab test callers: `lab/tests/experiments/protocols/test_visual_acquisition_protocol.py`; `lab/tests/integration/test_physics3d_effective_model_audit.py`; `lab/tests/integration/test_physics3d_vision.py`; `lab/tests/unit/lab/observation/test_remaining_semantic_families.py`; `lab/tests/unit/lab/physics3d/test_effective_model_audit.py`; `lab/tests/unit/lab/physics3d/test_resource.py`; `tests/experimental_integrity/test_embodiment_inheritance_integrity.py` | `DEFER` | Tests are protocol/solver/Observation/Embodiment consumers, not World unit tests. Reassess against the target owner contract and scientific protocol; do not relabel them World tests solely because Environment imports appear. |
| Historical benchmark: `scripts/bench_world_journal_index.py` | `DROP` | Source-specific benchmark implementation is not a TARGET artifact. Its provenance remains in the source repository; benchmark the target journal anew only if an explicit workload/performance requirement warrants it. |

### Source provenance appendix

Blob IDs and last path-changing commits below resolve at the pinned source
commit. A path-changing commit is provenance, not proof that its behavior is
correct or was current elsewhere. Commit hashes are abbreviated to 12 hex
characters in this table; source SHA above pins unambiguous resolution.

| Source path | Blob | Last path-changing commit |
|---|---|---|
| `environment/pyproject.toml` | `74c8e12d6bac767014c32bb8558e840d95b31bee` | `792d2dbd3` |
| `environment/src/environment/__init__.py` | `61b117880cd104aa90bffaad39ed2a16ca238911` | `7d090d58e214` |
| `environment/src/environment/checkpoint.py` | `3d586a6df303bf5dfdff49860bbde19b8fd1e556` | `13720935595f` |
| `environment/src/environment/constitution.py` | `288733dbd05c697f6f538b39d99c5a9399c7dacf` | `13720935595f` |
| `environment/src/environment/contracts.py` | `72122f3da2f6683449b1cf248eba3cd147c6f6dd` | `13720935595f` |
| `environment/src/environment/deferred.py` | `afc63f1ccd1bb72ec4204689b1ede61f1267b982` | `fa7178685aca` |
| `environment/src/environment/events.py` | `0d53017ade217269d1f602dfd9710b60e27e0254` | `13720935595f` |
| `environment/src/environment/genesis.py` | `34d90b1b33c1434d6b21164c9ceced08603d2271` | `b61eefbe3a09` |
| `environment/src/environment/genesis_v1.py` | `fed73bde2b6ba03bacd5396b00f71805ff82b5aa` | `43a65f0ab220` |
| `environment/src/environment/genesis_v2.py` | `d4b842fc73c3872a16ec79c5f9ee20ef32a3499f` | `43a65f0ab220` |
| `environment/src/environment/laws.py` | `5b2c1133fe7a93a42da59c72eb9d94899cf02329` | `13720935595f` |
| `environment/src/environment/movement.py` | `12c88ad1b925478851743e4554fae26e15ca4bfb` | `13720935595f` |
| `environment/src/environment/observation.py` | `86c19ea60ba17f58d5a44408d19bd5208730878b` | `08932ad3af80` |
| `environment/src/environment/physics3d/__init__.py` | `e69de29bb2d1d6434b8b29ae775ad8c2e48c5391` | `cb0e54016dbe` |
| `environment/src/environment/physics3d/environments.py` | `14a1bcd5c12c1855bf1e637292010c0c52e723c7` | `0f75c709bba8` |
| `environment/src/environment/physics3d/resource.py` | `f793e53c5183ea7219a34ac98d045f4e4b41b569` | `0f75c709bba8` |
| `environment/src/environment/rng.py` | `9e55b43f8d77c1d3155972f0a28d12016b16787c` | `13720935595f` |
| `environment/src/environment/state.py` | `0fc5120235da9264486513c493963c9b88fe47bc` | `13720935595f` |
| `environment/src/environment/terrain.py` | `7b241ec9e6cd1f854a0052428800bcd9b83dd7b2` | `43a65f0ab220` |
| `environment/src/environment/topology.py` | `2bdd5eb8e99b48fb70611c78cc46cdc51a3a2415` | `13720935595f` |
| `lab/src/lab/world/__init__.py` | `73fb4a3b7f9c3380b6aa94f11f1bb62bbe494d02` | `06f33acca0b0` |
| `lab/src/lab/world/cli_view.py` | `0f0ed5fcb41253a05b37413de3f6cdd08f1297ab` | `13720935595f` |
| `lab/src/lab/world/persistence.py` | `0871079c908d2efb0329c3f0571a1d22671f12d7` | `f7407d9b320c` |
| `lab/src/lab/world/population.py` | `bcb046937b036fad9264599a7832afd95e810d29` | `0550aede453d` |
| `lab/src/lab/world/runtime.py` | `de62c2b2e56cb5c6a08fb431441ff3ecb8db9844` | `17f56e5fa85d` |
| `lab/src/lab/world/transaction.py` | `0229a2b970246163e3ead08b5a8026ffd2066dc7` | `fa7178685aca` |
| `lab/src/lab/integration/world/__init__.py` | `366e8f126df46bb6eda3cf0bff057c9d06c671cc` | `9891c6a02871` |
| `lab/src/lab/integration/world/adapter.py` | `1e8c7b50d658ac2fc06db98868c713a517bc27b7` | `06f33acca0b0` |
| `lab/src/lab/observation/bus.py` | `f270ded3e9963456eea3d18c94ba0b1d2bfb3e81` | `2981f219c` |
| `lab/src/lab/observation/physics3d.py` | `116d19599724ddfb785e7e8ea4eaacfeb0c27a4a` | `966c32db1` |
| `lab/src/lab/observation/world_scene.py` | `fdf54be9b1dce8c0347c0ea4c40abf2cb6299699` | `13720935595f` |
| `lab/src/lab/physics3d/world_observation.py` | `76b380080757ad1a9819b5eb2664539d3febe818` | `40adf60c4232` |
| `environment/tests/test_self_contained.py` | `f21847b3c9e5d2e18e7624b5326771a9dc47ce54` | `792d2dbd3` |
| `environment/tests/unit/world/test_checkpoint.py` | `12e8a8be6953271b522c4bb26f90482aeb551294` | `282c44eea894` |
| `environment/tests/unit/world/test_constitution.py` | `7d9ad3ae334084808fccef0a897fdc13e47168a2` | `282c44eea894` |
| `environment/tests/unit/world/test_contracts.py` | `6474069260fa072f999c471390af750b5a92ad6b` | `282c44eea894` |
| `environment/tests/unit/world/test_deferred.py` | `970173ac29ca8b8c8a9b3ba6514557e2ac1654ae` | `fa7178685aca` |
| `environment/tests/unit/world/test_events.py` | `c6dc800e3620f06b1562cf63a97e531cb91136f2` | `282c44eea894` |
| `environment/tests/unit/world/test_genesis.py` | `5a9b24c2afa57006119d183a9f1a4c6f2578f71c` | `282c44eea894` |
| `environment/tests/unit/world/test_genesis_regional.py` | `cd265b072eb28e998621c3fe7a8074f51a3701dc` | `282c44eea894` |
| `environment/tests/unit/world/test_laws.py` | `6f3c1651f0f36edafa8f9c794ae7bb1dca1ac78c` | `282c44eea894` |
| `environment/tests/unit/world/test_movement.py` | `1591d4ea42a7d8f4b310a7780be271adeb0ca6f1` | `282c44eea894` |
| `environment/tests/unit/world/test_observation.py` | `bfd8729c5e77befc873a002043255ffb2ce2fb2d` | `12165be0fb80` |
| `environment/tests/unit/world/test_rng.py` | `f7f0f1343fa5cf928b8b1d30b01f1b7e6f094a00` | `282c44eea894` |
| `environment/tests/unit/world/test_state.py` | `1ff410be5b1df1250b44d783de490ceb8ac4d77f` | `282c44eea894` |
| `environment/tests/unit/world/test_topology.py` | `3d5a1901b80edd8a5313ad65452e1dd56677ccbd` | `282c44eea894` |
| `environment/tests/unit/lab/world/test_genesis_v1.py` | `ce19d980464b01c799b3ed7411ba7d538db9ed1b` | `282c44eea894` |
| `environment/tests/unit/lab/world/test_genesis_v2.py` | `b48a19f1e4ba4e22d9b51e840a2786309a5305de` | `282c44eea894` |
| `environment/tests/integration/test_world_checkpoint_continuation.py` | `d2ed447f08b2a5a44a3cd2c39042123d0737d2b1` | `282c44eea894` |
| `lab/tests/unit/lab/world/test_actuation_adapter.py` | `2e455a391b7c88cff4ae7188ad25bf34143d5df8` | `b2fd85c0c18d` |
| `lab/tests/unit/lab/world/test_actuation_end_to_end.py` | `a48e5d466bbbadd1e0d41d11876af2d357334a1a` | `282c44eea894` |
| `lab/tests/unit/lab/world/test_adapter.py` | `a988d7b25f3c46a8c3c3e8c1ce5888909b7915d9` | `06f33acca0b0` |
| `lab/tests/unit/lab/world/test_cli_view.py` | `823bf47dbe838670d69023170fc139066fb50e72` | `282c44eea894` |
| `lab/tests/unit/lab/world/test_hazard_ecology_v2.py` | `2778086788cd3858f078731c224fc368dc4e556c` | `282c44eea894` |
| `lab/tests/unit/lab/world/test_persistence.py` | `bf8cffa68919adad493d89abf2378db19d18d4fc` | `f7407d9b320c` |
| `lab/tests/unit/lab/world/test_population.py` | `3a97bfca5c5a2071129fef73ab17bc3dff3acffd` | `282c44eea894` |
| `lab/tests/unit/lab/world/test_terrain_and_movement.py` | `7f72c92bdd27522de8de0e8b0f322daf0a5a646a` | `282c44eea894` |
| `lab/tests/unit/lab/world/test_transaction_integrity.py` | `04c54e80fbcd846573fdabb334d52fb77cd64999` | `fa7178685aca` |
| `lab/tests/unit/lab/world/test_world_ecology_v2.py` | `ae2ec41f3f34d7c0808bb251a3195872bee7d098` | `282c44eea894` |
| `lab/tests/unit/lab/observation/test_atlas.py` | `0fa5f707fe46f78d3dda50c652856e72b3eb9198` | `282c44eea894` |
| `lab/tests/unit/lab/observation/test_cadence.py` | `650fb7b1708067f63351d3ae35cdf916fd461be0` | `282c44eea894` |
| `lab/tests/unit/lab/observation/test_development_history.py` | `135aa96560e5551990cd98f2c779953e2988d771` | `c34824713034` |
| `lab/tests/unit/lab/observation/test_live_delta.py` | `b356437f98859b69f36025a36d3a443555a4bd42` | `a8ab93276083` |
| `lab/tests/unit/lab/observation/test_observability_integrity.py` | `13245d94df94d4ed60a487632ecb28422d766f35` | `4648916b2b0a` |
| `lab/tests/unit/lab/observation/test_physics_world_observer.py` | `a21b7fa92a71605731470671c8ba27e4db831aa1` | `40adf60c4232` |
| `lab/tests/unit/lab/observation/test_remaining_semantic_families.py` | `bf0cfc06fbbe01b85308bff344a95bf77523c4c3` | `baca8d35eaff` |
| `lab/tests/unit/lab/observation/test_run_lifecycle.py` | `d6ac30c60af46bbad906e24d7fb507f7b2420978` | `8a7479e35ce7` |
| `lab/tests/unit/lab/observation/test_run_lifecycle_reconnect.py` | `8bb49e2711340881507841caea28c8c56d461927` | `ac4b2a17b418` |
| `lab/tests/unit/lab/observation/test_semantic_body_pose_vertical.py` | `04b573eb01e40ccf5405bc06fe84d54e27a321ac` | `9896d4a7ff4c` |
| `lab/tests/unit/lab/observation/test_semantic_delivery.py` | `033b33588512fa26e1ba3e4ef033f232398b11d1` | `f83895ddbd10` |
| `lab/tests/unit/lab/observation/test_semantic_model.py` | `332580349d8fd41693b1830e2cbd5893c6879c4f` | `d3b21610df2a` |
| `lab/tests/unit/lab/observation/test_world_environments.py` | `bc68c49fd1547dc78521a5fb6cd8d6b6ec1b2c40` | `54acefa003f4` |
| `lab/tests/unit/lab/observation/test_world_scene.py` | `198859f9fcfc1af6547d7657dbb42680fc5d9636` | `282c44eea894` |
| `lab/tests/unit/observatory/test_world_integration.py` | `130f5b03f4e8a187387a54a7b7b5ea604e83b195` | `30ebc07c9c08` |
| `lab/src/lab/app/physics3d/runs.py` | `dfad9b2c0e05602a38b33aff5d85c38de38b0e7a` | `55753a097d8a` |
| `lab/src/lab/cli/world.py` | `03e6317e0516acaac36aa4852b9d5ce6ed3c5b10` | `43a65f0ab220` |
| `lab/src/lab/observation/environment_resource.py` | `c849ed5be32744799949d0529a65e37965ace9e2` | `041e3cb30c06` |
| `lab/src/lab/physics3d/cli.py` | `e0e4a390f25db1b6f6bfe807222b3153b5eb3d6e` | `792d2dbd3e50` |
| `lab/src/lab/physics3d/effective_model_audit.py` | `33d20da18a82ea2d1e5013619eae606a8c30509f` | `9eafaaa8eab0` |
| `lab/src/lab/physics3d/effective_model_audit_cli.py` | `79efc42e6ab1d42f7aa12de8066b712d0e5dbcc7` | `6aac46a46b6e` |
| `lab/src/lab/physics3d/runtime.py` | `d2a70e298d05547c1aadb75195ab6c0f17496a43` | `5e9a250f1033` |
| `lab/src/lab/server/api.py` | `aebb3305f42fc7daa10293ef07096d55e19aa2bd` | `7078e649262e` |
| `lab/src/lab/studies/embodiment/integrity_gates.py` | `3ee7088151a3b9f8f955c40338438ba685f12a0f` | `9891c6a02871` |
| `lab/src/lab/studies/embodiment/label_invariance.py` | `fcedfe90794aeba598a5e3185c6511e9f4dceaf1` | `13720935595f` |
| `lab/src/lab/studies/world/genesis_viability.py` | `9169a76066e0e62077662cee1d342cd6cae72bee` | `43a65f0ab220` |
| `lab/tests/experiments/protocols/test_visual_acquisition_protocol.py` | `29b86aef93d8d9afb49ba81fc8b1b1c94749cffb` | `282c44eea894` |
| `lab/tests/integration/test_physics3d_effective_model_audit.py` | `1b4d41170c4c3411c806a33e8b027730d5c9b69d` | `3a75ab94a0a4` |
| `lab/tests/integration/test_physics3d_vision.py` | `855806550cccea55d21c2fac89df230a00700ffc` | `40adf60c4232` |
| `lab/tests/unit/lab/physics3d/test_effective_model_audit.py` | `5388b1e2e2d871b8fd5607bb681adf1057c0ba78` | `905322b926d8` |
| `lab/tests/unit/lab/physics3d/test_resource.py` | `69cb74a1b790b109bbc5fd69d93ef23e6da07929` | `55fafbb1488e` |
| `tests/experimental_integrity/test_embodiment_inheritance_integrity.py` | `a708ab8021d4962d773c379009fb7eb7406bc7b1` | `c077496ba8bb` |
| `scripts/bench_world_journal_index.py` | `47e66ac469626339ee156d4ea00a4dcdc8330f4a` | `13720935595f` |

## Integration tests moved from the integration workspace

Three cross-repository tests were copied unchanged from the unversioned local integration workspace, where they had no history, by Architecture Owner decision of 2026-10-09: the versioned source of evidence for ordinary causal integration belongs with World, the causal runtime and composition authority.

| Workspace path | Target path |
|---|---|
| `tests/integration/test_first_biopeia_causal_tick.py` | `tests/integration/test_first_biopeia_causal_tick.py` |
| `tests/integration/test_durable_checkpoint_restore_slice.py` | `tests/integration/test_durable_checkpoint_restore_slice.py` |
| `tests/integration/test_first_biopeia_physical_causal_step.py` | `tests/integration/physical_step/test_first_biopeia_physical_causal_step.py` |

They are new Biopeia tests, not migrated source tests; no path of the historical source repository corresponds to them. The fourth workspace test, the operator material transfer, imports Lab and was placed in the Lab repository so that World does not depend on Lab.

Verified on 2026-10-09 in a clean environment built from these revisions, 9 passed: `organism@f15c4c5`, `embodiment@3fe3cda`, `environment@f2ef8c2`, `modality@9a0656a`, `observation@cb07ace`, with this repository at `826aef5` plus this change. The `integration` extra pins the same revisions.

**Update, same day.** Modality removed its unratified transduction operators and catalogue (`modality@709996a`). `test_first_biopeia_causal_tick.py` used the identity transduction from that module; it now passes the environmental signal through unchanged, as the copy in the integration workspace had already been updated to do. No integration test imports `modality` any longer, so its pin was removed from the `integration` extra. Verified: 9 passed with `organism@f15c4c5`, `embodiment@3fe3cda`, `environment@f2ef8c2`, `observation@cb07ace`. With the previous test and Modality at `main`, collection failed on the removed import.

## Biopeia physical-step coordinator (World #29)

| Target path | Source/revision | Disposition | Notes |
| --- | --- | --- | --- |
| `src/world/physical_step.py` | Rewrite of the bounded orchestration sequence in `tests/integration/physical_step/test_first_biopeia_physical_causal_step.py` at World base `c162b67e698de9b47450264402677b0165e6e20e` | REWRITE | Adds a production-importable World coordinator with injected owner/solver ports. It issues one explicit source/target causal-tick context inside the durable `PREPARING` transition, orders projections → solver → owner adoption → checkpoint refs, and delegates final cut validation/publication to `DurableWorldRuntime`. Solver remains a numerical capability, not an authority. |
| `tests/unit/test_physical_step.py` | New target contract tests | REWRITE | Checks phase ordering, shared correlation context, source/target tick relationship, invalid identities and solver-failure propagation without a committed next tick. |
| `tests/integration/physical_step/test_first_biopeia_physical_causal_step.py` | Existing Biopeia integration fixture at the source revision above | REWRITE | Reuses the same deterministic MuJoCo fixture through the production coordinator instead of directly running the World checkpoint callback. This demonstrates the coordinator against the existing articulated/free-root integration slice, not a production physics adapter or validated environment dynamics. |

Failures propagate into the existing World recovery protocol; the coordinator does not undo owner in-memory mutations and does not make solver output durable. Owner adapters remain responsible for validating candidate identity/revisions and creating valid checkpoints. Static Environment adoption remains the only covered Environment path in this fixture.

The free-root integration fixture derives Environment expected correlation from the `PhysicalStepContext` during projection, before solver execution, and supplies that independent value at owner adoption. A wrong-World candidate is rejected, owner checkpoint writing is not reached, and the next causal cut remains uncommitted (`RECOVERY_REQUIRED`). This is World-side integration evidence; it does not add a World dependency to Environment or establish cryptographic authentication, production adapter behavior, or crash durability.

## Embodiment actuator binding integration follow-up (World #44)

At World base `e2098a9523446b6a32feda5f05140225f4d2ab86`, Embodiment PR #34 made `project_motor_command` require the concrete, Embodiment-owned `EmbodimentActuatorBinding`. The World integration extra was pinned to the exact merged Embodiment revision `6f5fd9edb46c3a4557733df08149453f16bddb9f`; both integration fixtures now build their bindings from explicit actuator/effector identifiers and declared channel contracts, never from collection position. The fixtures retain reversed/permuted channel ordering as a check that identity—not order—carries the association.

This is a downstream integration adaptation for [World issue #44](https://github.com/biopeia/world/issues/44), related to but not closing the broader [Embodiment issue #28](https://github.com/biopeia/embodiment/issues/28). The contract and its semantics remain owned by Embodiment. This change does not add a compatibility shim, alter step correlation or authorize it, modify World/Environment/Observation authority, or prove a production physical adapter, scientific efficacy, full causal-step durability, or crash recovery. The binding is fixture input and is not evidence that World authenticates its issuer.

Validation on the World work branch: `uv sync --extra integration` resolved Embodiment at the exact pin above; `uv run pytest -q` passed (65 tests); `uv run pytest -q tests/integration` passed (16 tests). Workspace-wide integration validation was not run from the isolated World worktree; the package's complete pinned integration suite was run instead.

## Opaque owner-contributed records in durable World cuts

| Target path | Source/provenance | Disposition | Notes |
| --- | --- | --- | --- |
| `src/world/durable_runtime.py::OwnerCausalRecord` and `DurableWorldRuntime.tick(write_owner_records=...)` | Biopeia TARGET follow-up to ratified World commit and Lab intervention/evidence contracts; requested by [World issue #46](https://github.com/biopeia/world/issues/46), related to [Lab #31](https://github.com/biopeia/lab/issues/31) and [Lab #32](https://github.com/biopeia/lab/issues/32). No historical implementation or API copied. | REWRITE | Lets a participating owner contribute a stable-ID, strict-JSON opaque record. World validates participant binding and uniqueness within the World, then writes the record into the same immutable journal prefix as the causal tick before committing its manifest reference. Domain meaning and scientific custody remain with the owner. |
| `tests/test_durable_runtime.py` | New tests from the accepted target contract and World issue #46 | REWRITE | Covers persistence across later ticks and restore, detached payload ownership, unique/bound identities, invalid payload keys, and failed-tick non-publication. It does not prove power-loss durability or scientific validity of owner payloads. |
| `src/world/__init__.py`, `README.md` | New public export and contract documentation | REWRITE | Exposes the target capability and states its bounded durability/trust claims. |

This capability is not a Lab evidence registry. A Lab record may preserve an intervention action/evidence reference in a committed World cut, but Lab remains responsible for its semantics, evidence custody, immutable artifact retention, and provenance. World does not infer `run_id`, authorize operators, validate scientific claims, or interpret the owner payload.

## Process termination and reopen evidence (World issue #33)

| Target path | Provenance | Disposition | Notes |
| --- | --- | --- | --- |
| `tests/test_durable_runtime.py` | New Biopeia TARGET test, based on the ratified single-logical-writer and explicit-recovery contracts; no historical implementation or API copied | REWRITE | A child process starts from an existing committed cut, reaches the owner callback only after the store reports `PREPARING`, then terminates via `os._exit` before candidate owner state, journal prefix, or `COMMITTED` manifest publication. A fresh runtime restore detects the stranded prepare, retains the prior tick/commit/references as `RECOVERY_REQUIRED`, refuses ordinary restore, and succeeds only after explicit `recover_previous_cut`; a subsequent tick then commits. |

This is evidence for one controlled process-termination boundary on the CI/local filesystem, not power-loss testing, filesystem/platform durability validation, multi-writer coordination, or a general crash-consistency guarantee. The first test does not terminate during an individual filesystem write and does not claim every publication boundary is covered.

## Partial manifest temporary-write interruption (World #33)

| Target path | Source/provenance | Disposition | Notes |
| --- | --- | --- | --- |
| `tests/test_durable_runtime.py` | Follow-up Biopeia TARGET test for [World issue #33](https://github.com/biopeia/world/issues/33), based on World `cb6bfb7c9bccd16c2f85f0f24deba1a12bfc80c8` and the ratified single-logical-writer / durable-cut contract; no historical implementation or API copied | REWRITE | A child restores a committed tick-zero cut, begins a transition, and exits while only a prefix of the `PREPARING` manifest has been written to its temporary file, before file `fsync` or atomic replacement. The test verifies the old `COMMITTED` manifest remains intact and a fresh runtime restores that prior cut; the incomplete temporary file remains. |

This adds evidence for one process-termination point during one manifest temporary-file write on the test filesystem. It does not exercise journal or owner-checkpoint writes, a second process, power loss, storage-device failure, every publication boundary, or general crash consistency. World issue #33 remains open for its other evidence gaps.

## Partial journal temporary-write interruption (World #33)

| Target path | Provenance | Disposition | Notes |
| --- | --- | --- | --- |
| `tests/test_durable_runtime.py` | Follow-up Biopeia TARGET test for [World issue #33](https://github.com/biopeia/world/issues/33), based on World `16414d7f5fa141251c756c8f3ae3dd28ef1a2b9e` and the ratified single-logical-writer / durable-cut contract; no historical implementation or API copied | REWRITE | A child publishes `PREPARING`, writes a candidate owner checkpoint file, and exits after writing/flushing a prefix of the candidate immutable journal snapshot temp file, before file `fsync` or hard-link publication. The previous committed journal and cut remain the only published versions. Reopen marks the transition `RECOVERY_REQUIRED`; explicit recovery restores the previous cut. The partial temp and unreferenced candidate owner file remain. The test does not fsync or claim durable publication of the candidate owner checkpoint. |

This is one process-termination test for one journal temporary-file write on the exercised local filesystem. It does not cover owner-checkpoint write interruption, termination during journal fsync/link/directory fsync, second-process interaction, power loss/storage-device behavior, automatic orphan cleanup, or general crash consistency. World issue #33 remains open.

## Journal temporary-file complete before file fsync (World #33)

| Target path | Provenance | Disposition | Notes |
| --- | --- | --- | --- |
| `tests/test_durable_runtime.py` | Follow-up Biopeia TARGET test for [World issue #33](https://github.com/biopeia/world/issues/33), based on World `a7cf15b1e898ee30f56694a13c90adefcd97eac5` and the ratified single-logical-writer / durable-cut contract; no historical implementation or API copied | REWRITE | A child publishes `PREPARING`, writes an owner candidate, then exits when `fsync` is called for the candidate journal temporary file. The full JSON payload has already been written and flushed to the OS, but that file `fsync` has not run and the immutable journal snapshot has not been hard-link published. The prior committed manifest and published journal remain unchanged; startup requires recovery and explicit recovery returns the prior cut. The temporary file and candidate owner file remain; candidate owner durability is not claimed. |

This adds evidence for process termination immediately before the candidate journal temporary-file `fsync` call on the exercised local filesystem. It does not prove the completed write survives power loss, that `fsync` itself succeeds or is durable on other filesystems, or behavior after successful file `fsync` and during hard-link/directory `fsync`. It does not cover owner-checkpoint write interruption, second-process interaction, power loss/storage-device behavior, orphan cleanup, or general crash consistency. World issue #33 remains open.

## Journal file fsync returned before hard-link publication (World #33)

| Target path | Provenance | Disposition | Notes |
| --- | --- | --- | --- |
| `tests/test_durable_runtime.py` | Follow-up Biopeia TARGET test for [World issue #33](https://github.com/biopeia/world/issues/33), based on World `98e3bf4299e1da95d3688f56bbbc067e13240d33` and the ratified single-logical-writer / durable-cut contract; no historical implementation or API copied | REWRITE | A parameterized subprocess test covers both termination at the candidate journal temporary-file `fsync` call and termination after that call returns successfully but before hard-link publication. For the post-fsync case, the child invokes the real `fsync` on the candidate journal file, confirms its return, then exits at the attempted hard link. In both cases the previously published manifest and journal set remain unchanged; reopening requires explicit recovery, which restores the previous cut. The candidate journal temporary file and owner file may remain; owner-checkpoint durability is not claimed. |

At the time of this slice, this added process-termination evidence for one post-successful-file-`fsync`, pre-hard-link boundary on the exercised local filesystem. That slice did not test termination after hard-link publication or during directory `fsync`, power loss/storage-device persistence, owner-checkpoint write interruption or durability, second-process interaction, orphan cleanup, other filesystem/platform behavior, or general crash consistency. The later section below adds a separate post-link/pre-directory-fsync test; all other gaps remain, and World issue #33 remains open.

## Journal hard-link before directory fsync (World #33)

| Target path | Provenance | Disposition | Notes |
| --- | --- | --- | --- |
| `tests/test_durable_runtime.py` | Follow-up Biopeia TARGET test for [World issue #33](https://github.com/biopeia/world/issues/33), based on World `89a50b19e0fad8b4bb3d96610c0267227fcd9ed5` and the ratified single-logical-writer / durable-cut contract; no historical implementation or API copied | REWRITE | The parameterized subprocess test exits after the candidate journal file has been successfully fsynced and hard-linked to its immutable target, but before the journal directory fsync. On the exercised local filesystem, the candidate pathname is visible while the manifest remains `PREPARING`; reopening requires recovery and explicit recovery returns the previous committed cut. The candidate remains unreferenced; no cleanup policy is introduced. |

This is process-termination evidence for one post-link/pre-directory-fsync boundary on the exercised filesystem. It does not emulate loss of directory metadata, establish power-loss/storage-device durability, test termination during directory `fsync`, owner-checkpoint write interruption/durability, second-process interaction, automatic orphan cleanup, other filesystems/platforms, or general crash consistency. World issue #33 remains open.

## Embodiment physical projection consumer update (Embodiment #32)

| Target path | Provenance | Disposition | Notes |
| --- | --- | --- | --- |
| `tests/integration/physical_step/test_first_biopeia_physical_causal_step.py` | World base `a99ebf9353d39da12bdc205962343aee21a346ae`; adapts the consumer to the Embodiment state-derived projection merged at `embodiment@97cb09a82b0bc9bf4ad11f6497cce071b85efedd` for F-0115 / [Embodiment issue #32](https://github.com/biopeia/embodiment/issues/32) | REWRITE | Root state is bound to its active morphology before projection. The articulated solver fixture initializes root and joint inputs from the projected `BodyPhysicalInput`, keyed by stable joint identity, including after checkpoint restore; it no longer reads a joint directly from owner state outside the projection. |
| `tests/integration/physical_step/test_procedural_sphere_world_step.py` | Same World and Embodiment revisions | REWRITE | Free-root fixtures build an explicit zero-joint articulated owner state and use the state-derived projection. The physical geometry and Environment owner contracts are unchanged. |
| `pyproject.toml` integration extra | Exact merged Embodiment revision above | REWRITE | Pins the integration consumer to the owner API revision it exercises. |

This is World-owned consumer integration evidence only. It does not add or authenticate World authorization, implement a production solver adapter, change owner-adoption semantics, establish material coefficient semantics, or prove scientific validity or full crash/power-loss recovery. It contributes to Embodiment issue #32 acceptance; the issue remains open until this cross-repository consumer slice is merged and verified.

## Embodiment checkpoint v2 consumer contract

World integration fixtures and the pinned integration dependency now use the
Embodiment owner schema `embodiment-checkpoint-v2` introduced at
`biopeia/embodiment@028e2f912eccbe554a9d7f8543fe9dfe2bb94218`. Prior v1
checkpoint references are not accepted as current TARGET test data. This
updates only World-owned test fixtures and the explicit integration pin; World
continues to treat owner checkpoint payloads as opaque. It does not add
checkpoint migration/fallback behavior or expand World recovery guarantees.

## Second-process read during an open transition (World #33)

| Target path | Provenance | Disposition | Notes |
| --- | --- | --- | --- |
| `tests/test_durable_runtime.py` | Follow-up Biopeia TARGET test for [World issue #33](https://github.com/biopeia/world/issues/33), based on World `ab33cc3ee5183f7999d3a7253251b9b1c5b392af` and the ratified single-logical-writer contract; no historical implementation or API copied | REWRITE | While the writer is paused in its owner-checkpoint callback after publishing `PREPARING`, a child process reads the manifest and observes the old committed tick plus pending transition metadata. The child does not mutate the manifest or attempt recovery; after it exits, the original writer completes the commit. |

This demonstrates read-only cross-process visibility of one open transition on the exercised local filesystem. It does not exercise a second writer, concurrent recovery, process termination, owner-checkpoint interruption/durability, directory-`fsync` failure, power loss/storage-device behavior, other filesystems/platforms, or general crash consistency. The single-logical-writer precondition remains in force, and World issue #33 remains open for the other evidence gaps.

## Owner checkpoint callback interrupted during a partial write (World #33)

| Target path | Provenance | Disposition | Notes |
| --- | --- | --- | --- |
| `tests/test_durable_runtime.py` | Follow-up Biopeia TARGET test for [World issue #33](https://github.com/biopeia/world/issues/33), based on World `8f8b2446e1328b4c7446d3538e56204139abf1b4` and the existing opaque owner-checkpoint callback contract; no historical implementation or API copied | REWRITE | After `PREPARING` is published, the subprocess owner callback writes only a prefix of a candidate checkpoint and exits without returning a checkpoint reference. No candidate journal or committed cut is published; restart requires explicit recovery to the prior cut. The test does not `fsync` the candidate and does not claim owner-byte durability. |

This is a bounded process-termination test on the exercised local filesystem. It does not establish the behavior of arbitrary owner implementations, actual interrupted system calls, power loss, storage-device persistence, other filesystems/platforms, or general crash consistency. The partial candidate file remains outside the committed cut; orphan cleanup is not tested.

## Injected journal directory-fsync failure (World #33)

| Target path | Provenance | Disposition | Notes |
| --- | --- | --- | --- |
| `tests/test_durable_runtime.py` | Follow-up Biopeia TARGET test for [World issue #33](https://github.com/biopeia/world/issues/33), based on World `f2334b9030ffbf7fc7d91a1da03ad8c83fd2ef7f` and the existing durable-journal contract; no historical implementation or API copied | REWRITE | Injects an `OSError` at the journal-directory-fsync boundary after the candidate journal link is visible, before calling the underlying OS `fsync`. World records `RECOVERY_REQUIRED` with the prior committed tick/refs; explicit recovery restores the prior cut. The candidate journal path remains visible but unreferenced. |

This verifies World failure handling for an injected directory-fsync-boundary error. It does not exercise a real OS-level `fsync` failure, directory-entry durability after power loss, arbitrary filesystems/platforms, owner checkpoint durability, concurrent second-process mutation/recovery, or general crash consistency. World issue #33 remains open.
