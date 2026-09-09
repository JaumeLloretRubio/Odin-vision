from dataclasses import replace

import numpy as np
import pytest

from odin_vision.memory import Memory


def test_persistence_multiple_views_and_delete(settings):
    memory = Memory(settings, "test")
    row = memory.learn("Falcon", "drone", [[1, 0, 0], [0, 1, 0], [1, 0, 0]])
    assert row["examples"] == 2
    memory.close()
    memory = Memory(settings, "test")
    assert memory.match([1, 0, 0], "drone", .85, .05) == ("Falcon", 1.0)
    assert memory.match([1, 0, 0], "person", .85, .05) == (None, None)
    assert memory.delete(row["id"])
    assert not memory.labels()
    assert memory.db.execute("SELECT count(*) FROM examples").fetchone()[0] == 0
    memory.close()


def test_ambiguity_threshold_and_validation(settings):
    memory = Memory(settings, "test")
    memory.learn("A", "drone", [[1, 0]])
    memory.learn("B", "drone", [[1, .01]])
    assert memory.match([1, 0], "drone", .8, .05)[0] is None
    assert memory.match([0, 1], "drone", .8, 0)[0] is None
    for bad in ([0, 0], [np.nan, 1], [np.inf, 0], [1, 0, 0]):
        with pytest.raises(ValueError):
            memory.learn("bad", "drone", [bad])
    assert len(memory.labels()) == 2
    with pytest.raises(ValueError):
        memory.learn("A", "car", [[1, 0]])
    memory.close()


def test_encoder_change_rejected(settings):
    Memory(settings, "first").close()
    with pytest.raises(ValueError, match="encoder"):
        Memory(settings, "different")


def test_limits_pagination_and_event_retention(settings):
    memory = Memory(replace(settings, max_labels=2, max_examples=1, event_limit=2), "test")
    assert memory.learn("A", "drone", [[1, 0], [0, 1]])["examples"] == 1
    memory.learn("B", "drone", [[0, 1]])
    with pytest.raises(ValueError):
        memory.learn("C", "drone", [[1, 0]])
    first = memory.labels(limit=1)[0]
    assert memory.labels(after=first["id"])[0]["label"] == "B"
    for i in range(3):
        memory.event("session", i, "drone", "A", .9, {"latitude": 40, "longitude": -3})
    events = memory.events(label="A")
    assert [e["track_id"] for e in events] == [1, 2]
    assert events[0]["location"]["latitude"] == 40
    assert not memory.events(label="B")
    memory.close()


def test_calibrated_margin_still_labels_similar_instances(settings):
    """Regresión: con dos instancias parecidas en una categoría, un margen alto anulaba todo reconocimiento.

    Números tomados del laboratorio: la sonda se parece 0.96 a su propia instancia y 0.90 a la otra persona
    presente en la escena. Con el margen calibrado se etiqueta; con 0.10 se perdía el reconocimiento entero.
    """
    memory = Memory(settings, "test")
    own = np.array([1.0, 0.0], dtype=np.float32)
    other = np.array([np.cos(0.42), np.sin(0.42)], dtype=np.float32)  # ~0.914 de coseno con la primera
    memory.learn("sujeto", "person", [own])
    memory.learn("otro", "person", [other])
    probe = np.array([np.cos(0.05), np.sin(0.05)], dtype=np.float32)
    threshold, tight = 0.91, 0.10
    calibrated = 0.02
    assert memory.match(probe, "person", threshold, calibrated)[0] == "sujeto"
    assert memory.match(probe, "person", threshold, tight)[0] is None
    memory.close()
