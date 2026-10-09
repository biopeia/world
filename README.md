# World

Incremental Biopeia World extraction. This repository currently contains a bounded, transport-neutral causal journal and a minimal synchronous causal coordinator; it is not yet a complete World runtime.

`WorldRuntime` coordinates one active presence and an opaque transition callback. It invokes the callback for the proposed tick, then appends one committed causal record. If the callback or append fails, the runtime enters `RECOVERY_REQUIRED` and refuses further ticks. The callback runs before publication; it may already have mutated participant state when it fails, and this prototype performs no rollback or recovery.

The journal/runtime coordinate World-scoped causal records without owning Organism-private state, Embodiment somatic state, or Environment state. Event types are opaque here: this package does not define which authority owns their semantics. IDs and tick numbering are deterministic; there is no RNG or host-clock dependency.

`RECOVERY_REQUIRED` and journal contents are in-memory only. This prototype is not durable and does not establish compliance with AD-0019.

See [`MIGRATION.md`](MIGRATION.md) for exact source provenance, dispositions, validation scope, and deferred responsibilities.
