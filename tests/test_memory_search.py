import numpy as np
import pytest

from odin_vision.memory import Memory
from odin_vision.vision import normalized


def scalar_match(memory, vector, category, threshold, margin):
    """Referencia independiente: búsqueda exhaustiva anterior, sin caché ni preselección."""
    vector = normalized(vector)
    scores = {}
    for name, raw in memory.db.execute(
        "SELECT l.name,e.vector FROM examples e JOIN labels l ON e.label_id=l.id "
        "WHERE l.category=? ORDER BY e.id",
        (category,),
    ):
        value = float(vector @ np.frombuffer(raw, dtype=np.float32))
        scores[name] = max(scores.get(name, -1), value)
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    if not ranked:
        return None, None
    name, score = ranked[0]
    score = min(1., max(-1., score))
    if score < threshold or (len(ranked) > 1 and score-ranked[1][1] < margin):
        return None, score
    return name, score


@pytest.mark.parametrize("dimension", [48, 768])
def test_batch_matches_scalar_with_chunking_and_boundaries(settings, dimension):
    rng = np.random.default_rng(932)
    memory = Memory(settings, "test")
    try:
        for index in range(64):
            memory.learn(str(index), "person", rng.normal(size=(16, dimension)))
        probes = rng.normal(size=(65, dimension))
        score = scalar_match(memory, probes[0], "person", -1., 0.)[1]
        for threshold, margin in [(-1., 0.), (score, 0.), (np.nextafter(score, np.inf), .02), (.91, .02)]:
            assert memory.match_many(probes, "person", threshold, margin) == [
                scalar_match(memory, vector, "person", threshold, margin) for vector in probes]
            assert memory.match_many(probes[:16], "person", threshold, margin) == [
                scalar_match(memory, vector, "person", threshold, margin) for vector in probes[:16]]
        assert memory.match_many([], "person", .9, .02) == []
        assert memory.match_many(probes[:1], "person", .91, .02) == [
            scalar_match(memory, probes[0], "person", .91, .02)]
        assert memory.match_many(probes[:2], "absent", .9, .02) == [(None, None)] * 2
        with pytest.raises(ValueError, match="Dimensión"):
            memory.match_many([probes[0], np.ones(dimension+1)], "person", .9, .02)
        memory.learn("tie-a", "tie", [probes[0]])
        memory.learn("tie-b", "tie", [probes[0]])
        assert memory.match_many(probes[:2], "tie", -1., 0.) == [
            scalar_match(memory, vector, "tie", -1., 0.) for vector in probes[:2]]
    finally:
        memory.close()


@pytest.mark.parametrize("dimension", [2, 48, 768])
def test_search_matches_scalar_at_boundaries(settings, dimension):
    rng = np.random.default_rng(1984)
    memory = Memory(settings, "test")
    try:
        for index in range(25):
            memory.learn(str(index), "person", rng.normal(size=(4, dimension)))
        probes = rng.normal(size=(25, dimension))
        for vector in probes:
            _, score = scalar_match(memory, vector, "person", 0., 0.)
            for threshold in [0., .91, score, np.nextafter(score, np.inf)]:
                for margin in [0., .02, .1]:
                    assert memory.match(vector, "person", threshold, margin) == scalar_match(
                        memory, vector, "person", threshold, margin)
        with pytest.raises(ValueError, match="Dimensión"):
            memory.match(np.ones(dimension+1), "person", .85, .05)
    finally:
        memory.close()


def test_cache_tracks_local_and_external_commits(settings):
    first, second = Memory(settings, "test"), Memory(settings, "test")
    try:
        assert first.match([1, 0], "person", .8, .05) == (None, None)
        label = second.learn("A", "person", [[1, 0]])
        assert first.match([1, 0], "person", .8, .05) == ("A", 1.)
        first.learn("B", "person", [[0, 1]])
        assert first.match([0, 1], "person", .8, .05) == ("B", 1.)
        second.delete(label["id"])
        assert first.match([1, 0], "person", .8, .05) == (None, 0.)
        second.learn("B", "person", [[1, 0]])
        assert first.match([1, 0], "person", .8, .05) == ("B", 1.)
        first.delete(first.labels()[0]["id"])
        assert first.match([1, 0], "person", .8, .05) == (None, None)
    finally:
        first.close()
        second.close()


def test_warm_search_does_not_reload_vectors_after_sighting(settings):
    memory = Memory(settings, "test")
    try:
        memory.learn("A", "person", [[1, 0]])
        memory.match([1, 0], "person", .8, .05)
        statements = []
        memory.db.set_trace_callback(statements.append)
        memory.event("session", 1, "person", "A", 1., None)
        for _ in range(5):
            assert memory.match([1, 0], "person", .8, .05) == ("A", 1.)
        assert not any("e.vector" in statement for statement in statements)
        for index in range(100):
            memory.match([1, 0], f"absent-{index}", .8, .05)
        assert len(memory._galleries) == 1
    finally:
        memory.close()


def test_tied_labels_keep_original_order_and_margin(settings):
    memory = Memory(settings, "test")
    try:
        memory.learn("Z", "person", [[1, 0]])
        memory.learn("A", "person", [[0, 1], [1, 0]])
        memory.learn("Z", "person", [[0, 1]])
        assert memory.match([1, 0], "person", .8, 0.) == ("Z", 1.)
        assert memory.match([1, 0], "person", .8, .01) == (None, 1.)
        assert memory.match([-1, -1], "person", 0., 0.) == scalar_match(
            memory, [-1, -1], "person", 0., 0.)
    finally:
        memory.close()


def test_multiple_best_examples_share_label_with_interleaved_rows(settings):
    memory = Memory(settings, "test")
    try:
        memory.learn("winner", "person", [[.98, np.sqrt(1-.98**2)]])
        memory.learn("runner-up", "person", [[.95, np.sqrt(1-.95**2)]])
        memory.learn("negative", "person", [[-1, 0]])
        memory.learn("winner", "person", [[.979, -np.sqrt(1-.979**2)]])
        query = [1, 0]
        assert memory.match(query, "person", .9, .02)[0] == "winner"
        score = scalar_match(memory, query, "person", .9, 0.)[1]
        for threshold in [.9, score, np.nextafter(score, np.inf)]:
            for margin in [0., .02, .04]:
                expected = scalar_match(memory, query, "person", threshold, margin)
                assert memory.match(query, "person", threshold, margin) == expected
                assert memory.match_many([query]*33, "person", threshold, margin) == [expected]*33
    finally:
        memory.close()


def test_sparse_ranking_preserves_late_example_ties(settings):
    memory = Memory(settings, "test")
    try:
        for name in ["unused-0", "unused-1", "earlier"]:
            memory.learn(name, "person", [[0, 1]])
        memory.learn("later", "person", [[1, 0]])
        memory.learn("earlier", "person", [[1, 0]])
        assert memory.match_many([[1, 0]], "person", .9, 0.) == [("earlier", 1.)]
        assert memory.match_many([[1, 0]], "person", .9, .01) == [(None, 1.)]
    finally:
        memory.close()


def test_sparse_ranking_preserves_negative_sentinel_and_margin(settings):
    memory = Memory(settings, "test")
    try:
        for name in ["first", "second", "third"]:
            memory.learn(name, "person", [[1, 0]])
        assert memory.match_many([[-1, 0]], "person", -1., 0.) == [("first", -1.)]
        assert memory.match_many([[-1, 0]], "person", -1., .01) == [(None, -1.)]
    finally:
        memory.close()


def test_single_label_ranking_preserves_threshold_and_ignores_margin(settings):
    memory = Memory(settings, "test")
    try:
        memory.learn("only", "person", [[1, 0]])
        for vector in [[1, 0], [0, 1], [-1, 0], [.3, .7]]:
            _, score = scalar_match(memory, vector, "person", -1., 0.)
            for threshold in [score, np.nextafter(score, np.inf)]:
                expected = scalar_match(memory, vector, "person", threshold, 1.)
                assert memory.match(vector, "person", threshold, 1.) == expected
                assert memory.match_many([vector], "person", threshold, 1.) == [expected]
    finally:
        memory.close()


def test_rounding_cannot_change_margin_decision(settings):
    rng = np.random.default_rng(17)
    memory = Memory(settings, "test")
    try:
        probe = normalized(rng.normal(size=768))
        own = normalized(probe + rng.normal(scale=.01, size=768))
        rival = normalized(probe + rng.normal(scale=.011, size=768))
        memory.learn("own", "person", [own])
        memory.learn("rival", "person", [rival])
        raw = [np.frombuffer(row[0], dtype=np.float32)
               for row in memory.db.execute("SELECT vector FROM examples")]
        scores = sorted([float(normalized(probe) @ v) for v in raw], reverse=True)
        gap = min(1., scores[0])-scores[1]
        for margin in (gap, np.nextafter(gap, np.inf), np.nextafter(gap, -np.inf)):
            assert memory.match(probe, "person", .8, margin) == scalar_match(memory, probe, "person", .8, margin)
    finally:
        memory.close()


def test_legacy_database_indexes_preserve_example_order(settings):
    import sqlite3

    memory = Memory(settings, "test")
    memory.close()
    # Simular un archivo anterior sin los índices, con orden de ejemplos distinto al de etiquetas.
    with sqlite3.connect(settings.data_dir / "memory.sqlite3") as db:
        db.execute("DROP INDEX examples_label")
        db.execute("DROP INDEX labels_category")
        db.executemany("INSERT INTO labels(id,name,category) VALUES (?,?,?)",
                       [(1, "later-example", "person"), (2, "first-example", "person")])
        db.executemany("INSERT INTO examples(id,label_id,vector) VALUES (?,?,?)",
                       [(1, 2, normalized([1, 0]).tobytes()), (2, 1, normalized([1, 0]).tobytes())])
    db.close()
    memory = Memory(settings, "test")
    try:
        assert memory.match([1, 0], "person", .8, 0.) == ("first-example", 1.)
        assert memory.match([1, 0], "person", .8, .01) == (None, 1.)
        assert len(memory.labels()) == 2
        assert memory.delete(2)
        assert memory.match([1, 0], "person", .8, 0.) == ("later-example", 1.)
        assert memory.db.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        memory.close()


def test_interleaved_gallery_rebuild_keeps_scalar_decisions(settings):
    rng = np.random.default_rng(41)
    memory = Memory(settings, "test")
    try:
        for _ in range(4):
            for label in range(20):
                memory.learn(str(label), "person", [rng.normal(size=48)])
        probes = rng.normal(size=(10, 48))
        score = scalar_match(memory, probes[0], "person", -1., 0.)[1]
        for threshold in [-1., score, np.nextafter(score, np.inf), .91]:
            for margin in [0., .02]:
                memory._galleries.clear()
                assert memory.match_many(probes, "person", threshold, margin) == [
                    scalar_match(memory, probe, "person", threshold, margin) for probe in probes]
        assert memory._gallery("person")[0] == tuple(str(label) for label in range(20))
    finally:
        memory.close()
