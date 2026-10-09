# World

Incremental Biopeia World extraction. This repository currently contains a bounded, transport-neutral causal journal slice; it is not yet a complete World runtime.

The journal coordinates World-scoped causal records without owning Organism-private state, Embodiment somatic state, or Environment state. Event types are opaque here: this package does not define which authority owns their semantics.

See [`MIGRATION.md`](MIGRATION.md) for exact source provenance, dispositions, validation scope, and deferred responsibilities.
