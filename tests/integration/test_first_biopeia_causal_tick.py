"""Deterministic integration fixture with an explicit external signal input.

This verifies a bounded discrete causal path through Organism command
production, Embodiment candidate projection, World publication and passive
Observation. It does not claim solver integration, somatic result adoption,
scientific efficacy, or durable recovery. Those cross-owner physical APIs are
not yet integrated. The cognitive graph produces consultative readouts; it
does not select or execute body actions.
"""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from embodiment.checkpoint import EmbodimentCheckpoint
from embodiment.metabolism import MetabolicLedger
from embodiment.observation import BodyPoseSample
from embodiment.physiology import LivingBodyState
from embodiment.physics3d.actuation_projection import project_motor_command
from observation.projection import runtime_tick_events
from organism.actuation import (
    ActionCommitment,
    ActuatorSurface,
    MotorIntent,
    SequenceController,
    motor_command_from_frame,
)
from organism.cognition.graph import (
    CognitiveGraph,
    PlasticEdge,
    PlasticNode,
)
from organism.cognition.limits import KernelLimits
from organism.cognition.transition import CognitiveTransitionState, advance_cognition
from organism.cognition.types import EdgeKind, NodeKind
from organism.identity import OrganismIdentity
from world.runtime import WorldId, WorldRuntime


@dataclass(frozen=True)
class EffectorDescriptor:
    effector_id: str
    quantity: str = "normalized_activation"
    unit: str = "1"
    minimum: float = 0.0
    neutral: float = 0.0
    maximum: float = 1.0
    saturation: str = "reject"


@dataclass(frozen=True)
class PhysicalChannel:
    actuator_id: str
    effector_id: str
    command_kind: str = "normalized_activation"
    quantity: str = "normalized_activation"
    unit: str = "1"
    minimum: float = 0.0
    neutral: float = 0.0
    maximum: float = 1.0
    saturation: str = "reject"
    available: bool = True


@dataclass(frozen=True)
class PhysicalSurface:
    contract_fingerprint: str
    morphology_revision: str
    channels: tuple[PhysicalChannel, ...]


@dataclass(frozen=True)
class ActiveMorphology:
    body_id: str
    embodiment_episode_id: str
    morphology_revision: str
    effectors: tuple[EffectorDescriptor, ...]


@dataclass(frozen=True)
class TickSnapshot:
    tick: int
    environment_signal: float
    transduced_signal: float
    graph_readout: float
    body_age_ticks: int
    body_energy: float
    motor_command_id: str
    world_record_id: str
    observations: tuple[dict[str, object], ...]


def _run_three_ticks() -> tuple[OrganismIdentity, tuple[TickSnapshot, ...]]:
    organism_id = OrganismIdentity.parse("12345678-1234-5678-1234-567812345678")
    body = LivingBodyState(energy_reserve=1.0, max_energy=2.0)
    body_id = "integration-body"
    episode_id = "integration-episode"
    morphology_revision = "integration-morphology-r1"
    embodiment_checkpoint = EmbodimentCheckpoint(
        body_id,
        episode_id,
        body,
        MetabolicLedger(body_state=body),
    )
    active_morphology = ActiveMorphology(
        body_id,
        episode_id,
        morphology_revision,
        (EffectorDescriptor("joint.elbow"), EffectorDescriptor("joint.wrist")),
    )
    body_pose = BodyPoseSample(
        body_id,
        "integration.world",
        "body.root",
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
        "fixture",
        0,
    )
    actuator_surface = ActuatorSurface.from_effector_ids(
        ("joint.elbow", "joint.wrist"),
        fingerprint_material="integration-command-surface-v1",
    )
    # This is explicit embodiment-supplied identity mapping, not positional
    # authority: reverse the order to ensure IDs carry the association.
    physical_surface = PhysicalSurface(
        actuator_surface.contract_fingerprint,
        morphology_revision,
        (
            PhysicalChannel(actuator_surface.actuator_ids[1], "joint.wrist"),
            PhysicalChannel(actuator_surface.actuator_ids[0], "joint.elbow"),
        ),
    )
    graph = CognitiveGraph(
        nodes=(
            PlasticNode("opaque-signal", NodeKind.SENSE),
            PlasticNode("consultative-readout", NodeKind.READOUT),
        ),
        edges=(
            PlasticEdge(
                source_id="opaque-signal",
                target_id="consultative-readout",
                kind=EdgeKind.EXCITATORY,
                weight=1.0,
                plasticity=0.0,
                delay_ticks=0,
            ),
        ),
        kernel_limits=KernelLimits(),
    )
    cognitive_state = CognitiveTransitionState.initial(organism_id)
    commitment = ActionCommitment(
        commitment_id="commitment.integration.forward",
        proposal_id="proposal.integration.forward",
        effect_target_id="effect.forward",
        competence_id="competence.forward",
        started_tick=1,
        controller_id="controller.integration",
        surface_fingerprint=actuator_surface.contract_fingerprint,
        embodiment_id="integration-episode",
    )
    controller = SequenceController(
        "controller.integration",
        ((MotorIntent(actuator_surface.actuator_ids[0], 0.5),),),
    )
    world = WorldRuntime(WorldId("integration-world"), presence_id=str(organism_id))
    snapshots: list[TickSnapshot] = []

    for _ in range(3):
        staged: dict[str, object] = {}
        pending_body: LivingBodyState | None = None
        pending_cognition: CognitiveTransitionState | None = None
        pending_controller: SequenceController | None = None

        def transition(causal_tick: int) -> dict[str, object]:
            nonlocal pending_body, pending_cognition, pending_controller

            # Compute against private candidates. The World callback may fail
            # before journal publication; authoritative fixture state is only
            # replaced after WorldRuntime.tick returns a committed record.
            candidate_body = LivingBodyState.from_checkpoint(body.checkpoint())
            candidate_embodiment = EmbodimentCheckpoint(
                body_id,
                episode_id,
                candidate_body,
                MetabolicLedger(body_state=candidate_body),
            )

            environmental_signal = 0.5  # Explicit deterministic fixture input; no Environment law is under test.
            opaque_signal = environmental_signal
            candidate_cognition, frame = advance_cognition(
                cognitive_state,
                graph,
                {"opaque-signal": opaque_signal},
                causal_tick=causal_tick,
            )

            # Organism emits a provenance-bearing logical command. Embodiment
            # projects it to a sparse physical candidate using explicit stable
            # effector identity and active morphology contracts. This does not
            # invoke a solver or adopt body-state output.
            candidate_controller = SequenceController(
                controller.controller_id,
                controller.sequence,
                step=controller.step,
            )
            control_frame = candidate_controller.frame(commitment, tick=causal_tick)
            command = motor_command_from_frame(
                commitment,
                control_frame,
                command_id=f"command.integration.{causal_tick}",
                issued_at_tick=causal_tick,
            )
            physical_actuation = project_motor_command(
                command,
                physical_surface,
                active_morphology,
                body_pose=body_pose,
                embodiment_checkpoint=candidate_embodiment,
                command_kind="normalized_activation",
            )
            # Cognition consumes body energy. This fixture does not infer an
            # action from the consultative readout.
            candidate_body.consume_energy(0.01)
            candidate_body.advance_age()

            staged.update(
                tick=causal_tick,
                environmental_signal=environmental_signal,
                transduced_signal=opaque_signal,
                graph_readout=frame.readouts["consultative-readout"],
                motor_command_id=command.command_id,
                physical_channels=physical_actuation.channels,
            )
            pending_body = candidate_body
            pending_cognition = candidate_cognition
            pending_controller = candidate_controller
            return {
                "organism_motor_command": {
                    "command_id": command.command_id,
                    "commitment_id": command.commitment_id,
                    "controller_id": command.controller_id,
                    "surface_fingerprint": command.surface_fingerprint,
                    "issued_at_tick": command.issued_at_tick,
                    "channels": dict(command.channels),
                },
                "embodiment_physical_actuation_candidate": [
                    {
                        "body_id": channel.body_id,
                        "embodiment_episode_id": channel.embodiment_episode_id,
                        "morphology_revision": channel.morphology_revision,
                        "effector_id": channel.effector_id,
                        "command_kind": channel.command_kind,
                        "value": channel.value,
                        "quantity": channel.quantity,
                        "unit": channel.unit,
                        "valid_range": [channel.minimum, channel.maximum],
                        "neutral": channel.neutral,
                    }
                    for channel in physical_actuation.channels
                ],
            }

        record = world.tick(transition)
        assert pending_body is not None and pending_cognition is not None
        assert pending_controller is not None
        body = pending_body
        cognitive_state = pending_cognition
        controller = pending_controller
        # Observation is a passive post-commit projection, never part of the
        # callback that decides whether World publishes the causal tick.
        observations = runtime_tick_events(
            {
                "source": "biopeia.integration",
                "instance_id": str(organism_id),
                "run_id": "deterministic-fixture",
                "sequence": record.causal_tick,
                "tick": record.causal_tick,
                "world_record_id": record.record_id,
                "body_kind": "synthetic-fixture",
                "body_age_ticks": body.age_ticks,
                "metabolic_reserve_ratio": body.energy_reserve / body.max_energy,
                "cognitive_readouts": 1,
            }
        )
        committed_command = record.payload["event_payload"]["organism_motor_command"]
        assert committed_command["command_id"] == staged["motor_command_id"]
        committed_physical = record.payload["event_payload"][
            "embodiment_physical_actuation_candidate"
        ]
        assert committed_physical == (
            {
                "body_id": body_id,
                "embodiment_episode_id": episode_id,
                "morphology_revision": morphology_revision,
                "effector_id": "joint.elbow",
                "command_kind": "normalized_activation",
                "value": 0.5,
                "quantity": "normalized_activation",
                "unit": "1",
                "valid_range": (0.0, 1.0),
                "neutral": 0.0,
            },
        )
        snapshots.append(
            TickSnapshot(
                tick=record.causal_tick,
                environment_signal=float(staged["environmental_signal"]),
                transduced_signal=float(staged["transduced_signal"]),
                graph_readout=float(staged["graph_readout"]),
                body_age_ticks=body.age_ticks,
                body_energy=body.energy_reserve,
                motor_command_id=str(staged["motor_command_id"]),
                world_record_id=record.record_id,
                observations=observations,
            )
        )

    assert world.causal_tick == 3
    assert len(world.records) == 3
    return organism_id, tuple(snapshots)


def test_first_biopeia_causal_ticks_are_deterministic() -> None:
    first_identity, first = _run_three_ticks()
    second_identity, second = _run_three_ticks()

    assert first_identity == second_identity
    assert first == second
    assert [tick.tick for tick in first] == [1, 2, 3]
    assert [tick.body_age_ticks for tick in first] == [1, 2, 3]
    assert [tick.body_energy for tick in first] == [0.99, 0.98, 0.97]
    assert [tick.motor_command_id for tick in first] == [
        "command.integration.1",
        "command.integration.2",
        "command.integration.3",
    ]
    assert [tick.world_record_id for tick in first] == [
        "integration-world:tick:1",
        "integration-world:tick:2",
        "integration-world:tick:3",
    ]
    for tick in first:
        assert tick.environment_signal == tick.transduced_signal
        assert tick.graph_readout != tick.transduced_signal
        assert [event["type"] for event in tick.observations] == [
            "body",
            "cognition",
            "vitals",
        ]
        assert tick.observations[1]["cognitive_readouts"] == 1
        assert tick.observations[2]["body_age_ticks"] == tick.tick
        # Observation deliberately projects only its declared passive view
        # contract; causal record identity stays with World, not the body view.
        assert tick.observations[0]["sequence"] == tick.tick


def test_fixture_identity_is_explicit_not_host_generated() -> None:
    identity = OrganismIdentity.create(UUID("12345678-1234-5678-1234-567812345678"))
    assert str(identity) == "12345678-1234-5678-1234-567812345678"
