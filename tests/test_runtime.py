import pytest

from world import RecoveryRequiredError, RuntimeStatus, WorldId, WorldRuntime


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


def test_invalid_world_and_presence_ids_are_rejected():
    with pytest.raises(ValueError):
        WorldId(" ")
    with pytest.raises(ValueError):
        WorldRuntime(WorldId("world-test"), presence_id="")
