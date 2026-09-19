import sqlite3
from dataclasses import replace

import pytest

from odin_vision.memory import Memory
from odin_vision.vision import Detection


def event(index, kind="sighting"):
    return (1000.+index, "session", index, "person", "A", .95, {"room": "lab"}, kind)


def test_batch_retention_order_and_restart(settings):
    settings = replace(settings, event_limit=3)
    memory = Memory(settings, "test")
    memory.event_many([event(0), event(1)])
    memory.event_many([event(i) for i in range(2, 8)])
    memory.close()
    memory = Memory(settings, "test")
    try:
        rows = memory.events(label="A", since=1006.)
        assert [r["track_id"] for r in rows] == [6, 7]
        assert [r["timestamp"] for r in rows] == [1006., 1007.]
        assert all(r["location"] == {"room": "lab"} for r in rows)
        assert [r["track_id"] for r in memory.events()] == [5, 6, 7]
    finally:
        memory.close()


def test_failed_batch_is_atomic_and_retryable(settings):
    memory = Memory(settings, "test")
    try:
        memory.event_many([event(0)])
        with pytest.raises(sqlite3.IntegrityError):
            memory.event_many([event(1), event(2, kind=None)])
        assert [r["track_id"] for r in memory.events()] == [0]
        memory.event_many([event(1), event(2)])
        assert [r["track_id"] for r in memory.events()] == [0, 1, 2]
    finally:
        memory.close()


def test_empty_batch_does_not_open_transaction(settings):
    memory = Memory(settings, "test")
    try:
        statements = []
        memory.db.set_trace_callback(statements.append)
        memory.event_many([])
        assert not statements
    finally:
        memory.close()


def test_frame_events_marked_only_after_commit(engine, scene, monkeypatch):
    sid = engine.create().id
    persist = engine.memory.event_many

    def fail(events):
        raise sqlite3.OperationalError("simulated write failure")

    monkeypatch.setattr(engine.memory, "event_many", fail)
    with pytest.raises(sqlite3.OperationalError):
        engine.process(sid, scene)
    track = next(iter(engine.session(sid).tracker.tracks.values()))
    assert not track.announced and track.last_event == 0
    monkeypatch.setattr(engine.memory, "event_many", persist)
    engine.process(sid, scene)
    assert track.announced and track.last_event > 0
    rows = engine.memory.events()
    assert len(rows) == 1 and rows[0]["kind"] == "new_object"
    engine.process(sid, scene)
    assert len(engine.memory.events()) == 1


def test_multiple_frame_events_share_one_commit(engine, scene, monkeypatch):
    monkeypatch.setattr(engine.vision, "detect", lambda image, classes: [
        Detection((0, 0, .4, .4), "person", 1.), Detection((.5, .5, .9, .9), "person", 1.)])
    statements = []
    engine.memory.db.set_trace_callback(statements.append)
    result = engine.process(engine.create().id, scene)
    rows = engine.memory.events()
    assert [row["track_id"] for row in rows] == [obj["id"] for obj in result["objects"]]
    assert len(rows) == 2 and all(row["kind"] == "new_object" for row in rows)
    assert sum(statement == "COMMIT" for statement in statements) == 1
