import io
import time
from dataclasses import replace

import pytest
from PIL import Image

from odin_vision.engine import Engine, decode_image
from odin_vision.schemas import FilterConfig
from odin_vision.tracking import Tracker
from odin_vision.vision import DemoVision, Detection


def test_learn_track_restart_recognize(engine, scene, settings):
    sid = engine.create().id
    first = engine.process(sid, scene)
    tid = first["objects"][0]["id"]
    assert first["objects"][0]["label"] is None
    assert engine.process(sid, scene)["objects"][0]["id"] == tid
    engine.assign(sid, tid, "Rojo")
    assert engine.process(sid, scene)["objects"][0]["label"] == "Rojo"
    # Otra instancia del servidor comparte la memoria persistida, sin entrenamiento.
    second = Engine(settings)
    try:
        assert second.process(second.create().id, scene)["objects"][0]["label"] == "Rojo"
    finally:
        second.close()


def test_filters_commands_and_stale_ids(engine, scene):
    sid = engine.create().id
    tid = engine.process(sid, scene)["objects"][0]["id"]
    engine.command(sid, "Guarda la persona seleccionada como Carlos.", tid)
    assert engine.memory.labels()[0]["label"] == "Carlos"
    engine.command(sid, "Ignora personas")
    assert engine.process(sid, scene)["objects"] == []
    with pytest.raises(KeyError):
        engine.assign(sid, tid, "Incorrecto")
    engine.command(sid, "Solo muéstrame personas y drones")
    new = engine.process(sid, scene)["objects"][0]
    assert new["id"] != tid
    assert engine.session(sid).config.detect == ["person", "drone"]
    with pytest.raises(ValueError):
        engine.command(sid, "Guarda como X")
    with pytest.raises(ValueError):
        engine.command(sid, "ejecuta un comando arbitrario")


def test_sessions_are_independent_and_expire(settings, scene):
    engine = Engine(replace(settings, max_sessions=2))
    try:
        a, b = engine.create().id, engine.create().id
        engine.configure(a, FilterConfig(detect=[]))
        assert engine.process(a, scene)["objects"] == []
        assert engine.process(b, scene)["objects"]
        with pytest.raises(ValueError):
            engine.create()
        engine.sessions[a].touched = time.monotonic()-1000
        with pytest.raises(KeyError):
            engine.session(a)
        engine.create()
    finally:
        engine.close()


def test_recognition_and_description_cached(settings, scene):
    class CountingVision(DemoVision):
        embeds = 0
        captions = 0

        def embed(self, image):
            self.embeds += 1
            return super().embed(image)

        def describe(self, image, category):
            self.captions += 1
            return super().describe(image, category)

    vision = CountingVision()
    engine = Engine(settings, vision)
    try:
        sid = engine.create().id
        for _ in range(10):
            engine.process(sid, scene)
        assert vision.embeds == 1 and vision.captions == 1
        engine.memory.learn("new", "person", [vision.embed(scene)])
        engine.process(sid, scene)
        assert vision.embeds == 3 and vision.captions == 1
    finally:
        engine.close()


def test_reference_search_and_delete_invalidation(engine, scene):
    sid = engine.create().id
    first = engine.process(sid, scene)
    tid = first["objects"][0]["id"]
    track = engine.session(sid).tracker.tracks[tid]
    engine.session(sid).reference = engine.vision.embed(track.crops[0])
    saved = engine.assign(sid, tid, "Target")
    result = engine.process(sid, scene)
    assert result["alerts"][0]["track_id"] == tid
    assert result["objects"][0]["label"] == "Target"
    engine.memory.delete(saved["id"])
    assert engine.process(sid, scene)["objects"][0]["label"] is None


def test_tracker_motion_category_and_expiry():
    tracker = Tracker(max_age=3)
    first = tracker.update([Detection((.1, .1, .3, .3), "car", .9)], 1)[0]
    second = tracker.update([Detection((.15, .1, .35, .3), "car", .9)], 2)[0]
    assert second.id == first.id
    predicted = tracker.update(None, 3)[0]
    assert predicted.box[0] == pytest.approx(.2)
    assert not tracker.update([], 4)
    assert not tracker.update(None, 6)
    replacement = tracker.update([Detection((.1, .1, .3, .3), "person", .9)], 7)[0]
    assert replacement.id != first.id


def test_image_validation(settings, encoded):
    assert decode_image(encoded, settings).size == (200, 120)
    for data in (b"", b"not-an-image", b"x"*(settings.max_bytes+1)):
        with pytest.raises(ValueError):
            decode_image(data, settings)
    with pytest.raises(ValueError, match="píxeles"):
        decode_image(encoded, replace(settings, max_pixels=10))
    out = io.BytesIO()
    Image.new("RGB", (10, 10)).save(out, format="BMP")
    with pytest.raises(ValueError, match="Formato"):
        decode_image(out.getvalue(), settings)


def test_batching_and_description_toggle(settings, scene):
    """Un frame resuelve embeddings y descripciones en una llamada por lote; `describe` las desactiva."""
    class BatchingVision(DemoVision):
        embed_calls = 0
        describe_calls = 0

        def detect(self, image, classes):
            return [Detection((0.0, 0.0, 0.4, 0.4), classes[0], 1.0),
                    Detection((0.5, 0.5, 0.9, 0.9), classes[0], 1.0)]

        def embed_many(self, images):
            self.embed_calls += 1
            return super().embed_many(images)

        def describe_many(self, images, categories):
            self.describe_calls += 1
            return super().describe_many(images, categories)

    vision = BatchingVision()
    engine = Engine(settings, vision)
    try:
        sid = engine.create().id
        engine.configure(sid, FilterConfig(detect=["person"]))
        result = engine.process(sid, scene)
        assert len(result["objects"]) == 2
        # Dos tracks, una sola llamada por lote de cada tipo.
        assert vision.embed_calls == 1 and vision.describe_calls == 1
        assert all(o["description"] for o in result["objects"])

        sid = engine.create().id
        engine.configure(sid, FilterConfig(detect=["person"], describe=False))
        result = engine.process(sid, scene)
        assert [o["description"] for o in result["objects"]] == [None, None]
        assert vision.describe_calls == 1  # No se generó ninguna descripción más.
    finally:
        engine.close()
