"""World-owned invocation boundary for an Organism causal transition.

World supplies physical-step correlation and invokes the Organism public
transition as an opaque participant operation. Cognition and checkpoint
semantics remain owned by Organism; this module only rejects a transition
whose supplied sensory provenance does not describe the active World step.
"""

from __future__ import annotations

from organism.cognition import (
    OpaqueSenseSignal,
    SenseChannelBinding,
    SenseProvenance,
    advance_organism_from_sense_signals,
)
from organism.cognition.limits import KernelLimits
from organism.transition import OrganismTransitionState, WorldCausalCorrelation

from .physical_step import PhysicalStepContext


def advance_organism_for_physical_step(
    context: PhysicalStepContext,
    current: OrganismTransitionState,
    bindings: tuple[SenseChannelBinding, ...],
    signals: tuple[OpaqueSenseSignal, ...],
    *,
    expected_provenance: SenseProvenance,
    kernel_limits: KernelLimits,
) -> OrganismTransitionState:
    """Invoke the Organism-owned transition for a World-issued physical step.

    ``expected_provenance`` is supplied by the caller from the owner
    projections. This boundary verifies only that its causal correlation
    agrees with the World context; it does not authenticate its producer.
    Organism remains responsible for checking signal bindings, lifecycle,
    embodiment applicability, cognition, and accepted owner state.
    """
    if not isinstance(context, PhysicalStepContext):
        raise TypeError("context must be a World-issued PhysicalStepContext")
    if not isinstance(expected_provenance, SenseProvenance):
        raise TypeError("expected_provenance must be SenseProvenance")
    if (
        expected_provenance.world_id != context.world_id
        or expected_provenance.causal_tick != context.source_causal_tick
        or expected_provenance.physical_step_id != context.physical_step_id
        or expected_provenance.environment_revision != context.environment_revision
        or expected_provenance.body_id != context.body_id
        or expected_provenance.embodiment_episode_id != context.embodiment_episode_id
        or expected_provenance.morphology_revision != context.morphology_revision
    ):
        raise ValueError("Organism input provenance does not match the World physical step")

    accepted = advance_organism_from_sense_signals(
        current,
        bindings,
        signals,
        expected_provenance=expected_provenance,
        kernel_limits=kernel_limits,
    )
    if accepted.state.world_correlation != WorldCausalCorrelation(
        context.world_id, context.source_causal_tick
    ):
        raise RuntimeError("Organism returned a transition with unexpected World correlation")
    return accepted.state


__all__ = ["advance_organism_for_physical_step"]
