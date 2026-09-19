import itertools

from odin_vision.memory import Memory


def test_event_filters_preserve_order_pagination_and_literal_values(settings):
    memory = Memory(settings, "test")
    try:
        labels = [None, "", "common", "rare", "x' OR 1=1 --"]
        memory.event_many([
            (float((i * 17) % 41), "s", i, "person", labels[i % len(labels)], .9,
             {"index": i}, "learned" if i % 3 == 0 else "sighting")
            for i in range(100)
        ])
        all_events = memory.events(limit=1000)
        for label, kind, since, after, limit in itertools.product(
            labels + ["missing"], [None, "learned", ""], [0, 20, 41], [0, 51, 101], [1, 13],
        ):
            expected = [row for row in all_events if row["id"] > after
                        and row["timestamp"] >= since
                        and (label is None or row["label"] == label)
                        and (kind is None or row["kind"] == kind)][:limit]
            assert memory.events(after, limit, label, since, kind) == expected
    finally:
        memory.close()


def test_existing_database_gets_event_pagination_index(settings):
    memory = Memory(settings, "test")
    memory.event("s", 1, "person", "A", .9, None)
    expected = memory.events()
    memory.db.execute("DROP INDEX event_label_id")
    memory.db.execute("CREATE INDEX event_label_time ON events(label, timestamp)")
    memory.db.commit()
    memory.close()
    memory = Memory(settings, "test")
    try:
        indexes = {row[1] for row in memory.db.execute("PRAGMA index_list(events)")}
        assert "event_label_id" in indexes
        assert "event_label_time" not in indexes
        assert memory.events(label="A") == expected
    finally:
        memory.close()
