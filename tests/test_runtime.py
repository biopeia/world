from threading import Event, Thread

import pytest

from world import (
    RecoveryRequiredError,
    RuntimeStatus,
    TickInProgressError,
    WorldId,
    WorldRuntime,
)


def test_successful_ticks_publish_one_record_after_transition_in_tick_order():
    runtime = WorldRuntime(WorldId("world-test"), presence_id="presence-a")
    observed_ticks = []

    first = runtime.tick(observed_ticks.append)
    second = runtime.tick(observed_ticks.append)

    assert observed_ticks == [1, 2]
    assert runtime.causal_tick == 2
    assert runtime.status is RuntimeStatus.ACTIVE
    assert [record.causal_tick for record in runtime.records] == [1, 2]
    assert [record.record_id for record in runtime.records] == [
        "world-test:tick:1",
        "world-test:tick:2",
    ]
    assert first.actor_id == second.actor_id == "presence-a"
    assert first.payload == {"presence_id": "presence-a"}


def test_transition_json_payload_is_included_opaquely_in_committed_record():
    runtime = WorldRuntime(WorldId("world-test"), presence_id="presence-a")
    supplied = {"opaque": {"quantity": 1.25}, "labels": ["x", True, None]}

    record = runtime.tick(lambda tick: supplied)

    assert record.payload["presence_id"] == "presence-a"
    assert record.payload["event_payload"]["opaque"]["quantity"] == 1.25
    assert record.payload["event_payload"]["labels"] == ("x", True, None)
    assert runtime.records == (record,)


@pytest.mark.parametrize("payload", [[], "not-an-object", {"value": float("nan")}])
def test_invalid_transition_payload_fails_closed_without_publishing_record(payload):
    runtime = WorldRuntime(WorldId("world-test"), presence_id="presence-a")

    with pytest.raises((TypeError, ValueError)):
        runtime.tick(lambda tick: payload)

    assert runtime.status is RuntimeStatus.RECOVERY_REQUIRED
    assert runtime.failed_tick == 1
    assert runtime.causal_tick == 0
    assert runtime.records == ()


def test_transition_failure_requires_recovery_and_publishes_no_failed_record():
    runtime = WorldRuntime(WorldId("world-test"), presence_id="presence-a")
    callback_calls = []

    def fail(tick):
        callback_calls.append(tick)
        raise RuntimeError("synthetic transition failure")

    with pytest.raises(RuntimeError, match="synthetic transition failure"):
        runtime.tick(fail)

    assert runtime.status is RuntimeStatus.RECOVERY_REQUIRED
    assert runtime.failed_tick == 1
    assert runtime.causal_tick == 0
    assert runtime.records == ()
    assert callback_calls == [1]

    with pytest.raises(RecoveryRequiredError):
        runtime.tick(lambda tick: callback_calls.append(tick))

    assert callback_calls == [1]
    assert runtime.records == ()


def test_keyboard_interrupt_after_transition_starts_marks_runtime_unusable():
    runtime = WorldRuntime(WorldId("world-test"), presence_id="presence-a")
    callback_calls = []

    def interrupt(tick):
        callback_calls.append(tick)
        raise KeyboardInterrupt("synthetic interruption")

    with pytest.raises(KeyboardInterrupt, match="synthetic interruption"):
        runtime.tick(interrupt)

    assert runtime.status is RuntimeStatus.RECOVERY_REQUIRED
    assert runtime.failed_tick == 1
    assert runtime.failure == "KeyboardInterrupt: synthetic interruption"
    assert runtime.causal_tick == 0
    assert runtime.records == ()

    with pytest.raises(RecoveryRequiredError):
        runtime.tick(lambda tick: callback_calls.append(tick))

    assert callback_calls == [1]
    assert runtime.records == ()


def test_invalid_world_and_presence_ids_are_rejected():
    with pytest.raises(ValueError):
        WorldId(" ")
    with pytest.raises(ValueError):
        WorldRuntime(WorldId("world-test"), presence_id="")


def test_concurrent_tick_fails_fast_without_starting_or_poisoning_runtime():
    runtime = WorldRuntime(WorldId("world-test"), presence_id="presence-a")
    callback_entered = Event()
    release_callback = Event()
    worker_errors = []

    def blocking_transition(tick):
        assert tick == 1
        callback_entered.set()
        assert release_callback.wait(timeout=2)

    def run_first_tick():
        try:
            runtime.tick(blocking_transition)
        except Exception as exc:  # surfaced in the main test thread below
            worker_errors.append(exc)

    worker = Thread(target=run_first_tick)
    worker.start()
    assert callback_entered.wait(timeout=2)

    rejected_callback_calls = []
    with pytest.raises(TickInProgressError):
        runtime.tick(rejected_callback_calls.append)

    assert runtime.status is RuntimeStatus.ACTIVE
    assert runtime.causal_tick == 0
    assert rejected_callback_calls == []
    release_callback.set()
    worker.join(timeout=2)

    assert not worker.is_alive()
    assert worker_errors == []
    assert runtime.causal_tick == 1
    assert [record.causal_tick for record in runtime.records] == [1]


def test_reentrant_tick_fails_fast_without_preventing_outer_commit():
    runtime = WorldRuntime(WorldId("world-test"), presence_id="presence-a")
    nested_calls = []

    def outer_transition(tick):
        assert tick == 1
        with pytest.raises(TickInProgressError):
            runtime.tick(nested_calls.append)

    runtime.tick(outer_transition)

    assert nested_calls == []
    assert runtime.status is RuntimeStatus.ACTIVE
    assert runtime.causal_tick == 1
    assert [record.causal_tick for record in runtime.records] == [1]
