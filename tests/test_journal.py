import pytest

from world import CausalRecord, EventJournal


def record(**overrides):
    values = {
        "record_id": "r1",
        "world_id": "world-a",
        "causal_tick": 0,
        "record_type": "opaque.example",
        "actor_id": "organism-a",
        "payload": {"value": 1},
    }
    values.update(overrides)
    return CausalRecord(**values)


def test_record_requires_identity_type_and_nonnegative_tick():
    with pytest.raises(ValueError):
        record(record_id="")
    with pytest.raises(ValueError):
        record(world_id="")
    with pytest.raises(ValueError):
        record(record_type="")
    with pytest.raises(ValueError):
        record(causal_tick=-1)


def test_causal_and_contributing_ids_are_distinct_tuples():
    item = record(causal_parent_ids=["p1"], contributing_record_ids=["c1", "c2"])
    assert item.causal_parent_ids == ("p1",)
    assert item.contributing_record_ids == ("c1", "c2")


def test_journal_is_append_only_and_replays_in_order():
    journal = EventJournal()
    first = record(record_id="r1", causal_tick=0)
    second = record(record_id="r2", causal_tick=1)
    journal.append(first)
    journal.append(second)
    assert len(journal) == 2
    assert journal.replay() == (first, second)
    assert not hasattr(journal, "remove")
    assert not hasattr(journal, "update")


def test_staged_records_commit_or_abort_as_a_batch():
    journal = EventJournal()
    first = record(record_id="r1")
    second = record(record_id="r2")
    journal.stage(first)
    journal.stage(second)
    assert len(journal) == 0
    assert journal.commit_staged() == (first, second)
    assert journal.replay() == (first, second)

    journal.stage(record(record_id="r3"))
    journal.abort_staged()
    assert len(journal) == 2


def test_indexed_queries_match_append_order():
    journal = EventJournal()
    records = [
        record(record_id="r1", causal_tick=0),
        record(record_id="r2", causal_tick=0),
        record(record_id="r3", causal_tick=1),
        record(record_id="r4", causal_tick=2),
    ]
    for item in records:
        journal.append(item)
    assert journal.records_at_tick(0) == tuple(records[:2])
    assert journal.records_at_tick(9) == ()
    assert journal.tail(2) == tuple(records[-2:])
    assert journal.snapshot_range(1, 3) == journal.snapshot()[1:3]
    assert journal.tail(0) == ()
    with pytest.raises(ValueError):
        journal.tail(-1)


def test_pagination_and_unknown_cursor():
    journal = EventJournal()
    for index in range(4):
        journal.append(record(record_id=f"r{index}", causal_tick=index))
    page, cursor, has_more = journal.page_after("r0", limit=2)
    assert tuple(item.record_id for item in page) == ("r1", "r2")
    assert cursor == "r2"
    assert has_more is True
    with pytest.raises(ValueError):
        journal.page_after("absent")
    with pytest.raises(ValueError):
        journal.page_after(limit=0)


def test_prefix_digests_survive_snapshot_restore_and_detect_changes():
    journal = EventJournal()
    for index in range(5):
        journal.append(record(record_id=f"r{index}", causal_tick=index // 2))
    restored = EventJournal.from_snapshot(journal.snapshot())
    assert restored.prefix_digest() == journal.prefix_digest()
    for count in range(6):
        assert restored.prefix_digest(count) == journal.prefix_digest(count)
    changed = journal.snapshot()
    changed[0] = {**changed[0], "record_id": "changed"}
    assert EventJournal.from_snapshot(changed).prefix_digest(1) != journal.prefix_digest(1)
    with pytest.raises(ValueError):
        journal.prefix_digest(6)


def test_payload_is_deeply_read_only_and_json_compatible():
    source_payload = {"nested": {"items": [1, 2]}}
    item = record(payload=source_payload)
    source_payload["nested"]["items"].append(3)
    assert item.payload["nested"]["items"] == (1, 2)
    with pytest.raises(TypeError):
        item.payload["nested"]["new"] = True


def test_encoding_failure_does_not_publish_record_or_partial_staged_batch(monkeypatch):
    journal = EventJournal()
    first = record(record_id="r1")
    second = record(record_id="r2")
    with pytest.raises(TypeError):
        record(record_id="invalid", payload={"value": object()})

    journal.stage(first)
    journal.stage(second)
    real_encode = EventJournal._encode
    calls = 0

    def fail_second(record_item):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise TypeError("synthetic encoding failure")
        return real_encode(record_item)

    monkeypatch.setattr(EventJournal, "_encode", staticmethod(fail_second))
    with pytest.raises(TypeError):
        journal.commit_staged()
    assert len(journal) == 0
    assert journal.prefix_digest() == ""
    monkeypatch.undo()

    invalid_float = record(record_id="nan", payload={"value": float("nan")})
    with pytest.raises(ValueError):
        journal.append(invalid_float)
    assert len(journal) == 0
