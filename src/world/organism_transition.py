"""World-owned invocation boundary for an Organism causal transition.

World supplies physical-step correlation and invokes the Organism public
transition as an opaque participant operation. Cognition and checkpoint
semantics remain owned by Organism; this module only rejects a transition
whose supplied sensory provenance does not describe the active World step.
"""

from __future__ import annotations

from dataclasses import dataclass

from embodiment.physics3d.physical_transition import EmbodimentPhysicalState
from embodiment.physics3d.receptor import (
    EmbodimentReceptor,
    ReceptorSamplePoint,
    project_receptor_sample_point,
)
from environment import (
    FiniteScalarFieldSample,
    FrameDescriptor,
    IdentifiedFiniteScalarField,
    PhysicalStepCorrelation as EnvironmentPhysicalStepCorrelation,
)
from modality import DimensionlessScalarFieldIdentityV1
from organism.cognition import (
    OpaqueSenseSignal,
    SenseChannelBinding,
    SenseProvenance,
    advance_organism_from_sense_signals,
)
from organism.cognition.limits import KernelLimits
from organism.transition import OrganismTransitionState, WorldCausalCorrelation

from .physical_step import PhysicalStepContext


@dataclass(frozen=True, slots=True)
class ScalarFieldOrganismInput:
    """Owner-issued sample and opaque signal accepted for one World step."""

    receptor_point: ReceptorSamplePoint
    sample: FiniteScalarFieldSample
    signal: OpaqueSenseSignal
    state: OrganismTransitionState


def advance_organism_from_scalar_field_for_physical_step(
    context: PhysicalStepContext,
    current: OrganismTransitionState,
    *,
    receptor: EmbodimentReceptor,
    embodiment_state: EmbodimentPhysicalState,
    environment_field: IdentifiedFiniteScalarField,
    sampling_frame: FrameDescriptor,
    transduction: DimensionlessScalarFieldIdentityV1,
    binding: SenseChannelBinding,
    kernel_limits: KernelLimits,
) -> ScalarFieldOrganismInput:
    """Compose the approved scalar-field sensor path for one World source cut.

    World supplies the causal step and composes public owner contracts only:
    Embodiment projects receptor location, Environment samples its field,
    Modality performs the declared transduction, and Organism validates and
    adopts the opaque input. This function assigns no environmental meaning
    and does not own any participant state.
    """
    if not isinstance(context, PhysicalStepContext):
        raise TypeError("context must be a World-issued PhysicalStepContext")
    if not isinstance(current, OrganismTransitionState):
        raise TypeError("current must be an OrganismTransitionState")
    if not isinstance(receptor, EmbodimentReceptor):
        raise TypeError("receptor must be an EmbodimentReceptor")
    if not isinstance(embodiment_state, EmbodimentPhysicalState):
        raise TypeError("embodiment_state must be an EmbodimentPhysicalState")
    if not isinstance(environment_field, IdentifiedFiniteScalarField):
        raise TypeError("environment_field must be an IdentifiedFiniteScalarField")
    if not isinstance(sampling_frame, FrameDescriptor):
        raise TypeError("sampling_frame must be a FrameDescriptor")
    if not isinstance(transduction, DimensionlessScalarFieldIdentityV1):
        raise TypeError("transduction must be DimensionlessScalarFieldIdentityV1")
    if not isinstance(binding, SenseChannelBinding):
        raise TypeError("binding must be a SenseChannelBinding")
    if binding.receptor_id != receptor.receptor_id:
        raise ValueError("Organism channel binding does not match Embodiment receptor identity")
    if environment_field.environment_revision != context.environment_revision:
        raise ValueError("Environment field revision does not match the World physical step")

    receptor_point = project_receptor_sample_point(
        receptor, embodiment_state, context
    )
    if receptor_point.frame_id != sampling_frame.frame_id:
        raise ValueError(
            "receptor sample frame does not match the Environment sampling frame"
        )
    correlation = EnvironmentPhysicalStepCorrelation(
        world_id=context.world_id,
        causal_tick=context.source_causal_tick,
        physical_step_id=context.physical_step_id,
        environment_id=environment_field.environment_id,
        environment_revision=environment_field.environment_revision,
    )
    sample = environment_field.sample_field(
        receptor_point.position_m,
        frame=sampling_frame,
        correlation=correlation,
    )
    provenance = SenseProvenance(
        world_id=sample.world_id,
        causal_tick=sample.causal_tick,
        physical_step_id=sample.physical_step_id,
        environment_id=sample.environment_id,
        environment_revision=sample.environment_revision,
        sample_frame_id=sample.frame_id,
        sample_position_m=sample.point_m,
        body_id=receptor_point.body_id,
        embodiment_episode_id=receptor_point.embodiment_episode_id,
        morphology_revision=receptor_point.morphology_revision,
    )
    signal = OpaqueSenseSignal(
        channel_id=binding.channel_id,
        receptor_id=receptor_point.receptor_id,
        value=transduction.transduce(sample.value),
        provenance=provenance,
        contract_id=transduction.CONTRACT_ID,
        contract_revision=transduction.REVISION,
        quantity_kind=transduction.QUANTITY_KIND,
        unit=transduction.UNIT,
    )
    accepted = advance_organism_for_physical_step(
        context,
        current,
        (binding,),
        (signal,),
        expected_provenance=provenance,
        kernel_limits=kernel_limits,
    )
    return ScalarFieldOrganismInput(
        receptor_point=receptor_point,
        sample=sample,
        signal=signal,
        state=accepted,
    )


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


__all__ = [
    "ScalarFieldOrganismInput",
    "advance_organism_for_physical_step",
    "advance_organism_from_scalar_field_for_physical_step",
]
