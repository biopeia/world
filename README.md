# World

Incremental Biopeia World extraction. This repository currently contains a bounded, transport-neutral causal journal and a minimal synchronous causal coordinator; it is not yet a complete World runtime.

`WorldRuntime` coordinates one active presence and an opaque transition callback. It invokes the callback for the proposed tick, then appends one committed causal record. Only one tick may be in progress: concurrent or reentrant calls fail fast with `TickInProgressError` before invoking their callback and do not change runtime status. If the callback or append fails, the runtime enters `RECOVERY_REQUIRED` and refuses further ticks. The callback runs before publication; it may already have mutated participant state when it fails, and this prototype performs no rollback or recovery.

The journal/runtime coordinate World-scoped causal records without owning Organism-private state, Embodiment somatic state, or Environment state. Event types are opaque here: this package does not define which authority owns their semantics. IDs and tick numbering are deterministic; there is no RNG or host-clock dependency.

`WorldCausalManifestStore` adds a durable, versioned manifest for exact owner checkpoint references, World causal tick, inclusive durable journal boundary, and publication status. It persists `RECOVERY_REQUIRED` before a transition is allowed to proceed; ordinary execution remains blocked until a complete causal cut with verified checkpoint dependencies is published.

This is a storage primitive, not yet wired into `WorldRuntime` or owner checkpoint writers. It does not atomically capture owner state, prove process-crash consistency across arbitrary filesystems, or implement recovery. A caller must ensure referenced owner checkpoints and journal prefix are already durable before publishing `COMPLETE`. Do not infer full AD-0019 compliance from manifest round-trip tests.

See [`MIGRATION.md`](MIGRATION.md) for exact source provenance, dispositions, validation scope, and deferred responsibilities.
