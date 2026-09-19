import io
import time
from dataclasses import replace

import pytest
from PIL import Image

from odin_vision.engine import Engine, decode_image
from odin_vision.schemas import FilterConfig
from odin_vision.tracking import Tracker
from odin_vision.vision import DemoVision, Detection


@pytest.mark.parametrize("format,mode", [("JPEG", "RGB"), ("JPEG", "L"), ("JPEG", "CMYK"),
                                         ("PNG", "RGB"), ("PNG", "RGBA"), ("WEBP", "RGB")])
def test_decoded_pixels_survive_source_close(settings, format, mode):
    original = Image.new(mode, (47, 31))
    original.paste(Image.new(mode, (17, 13), "white"), (9, 7))
    buffer = io.BytesIO()
    original.save(buffer, format=format)
    encoded = buffer.getvalue()
    with Image.open(io.BytesIO(encoded)) as source:
        expected = source.convert("RGB")
    decoded = decode_image(encoded, settings)
    assert decoded.mode == "RGB"
    assert decoded.tobytes() == expected.tobytes()
    assert decoded.crop((3, 4, 29, 24)).resize((19, 11)).tobytes() == expected.crop((3, 4, 29, 24)).resize((19, 11)).tobytes()
    decoded.putpixel((0, 0), (12, 34, 56))
    assert decoded.getpixel((0, 0)) == (12, 34, 56)


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


def test_repeated_config_keeps_tracks_crops_and_cached_descriptions(engine, scene, monkeypatch):
    session = engine.create()
    first = engine.process(session.id, scene)
    tid = first["objects"][0]["id"]
    engine.assign(session.id, tid, "Rojo")
    track = session.tracker.tracks[tid]
    crops = list(track.crops)

    def unexpected_caption(*args):
        pytest.fail("La misma configuración no debe regenerar descripciones")

    monkeypatch.setattr(engine.vision, "describe_many", unexpected_caption)
    # None y los valores calibrados explícitos tienen el mismo efecto.
    for config in [FilterConfig(), FilterConfig(**session.config.model_dump())]:
        assert engine.configure(session.id, config) == session.config.model_dump()
        assert session.tracker.tracks[tid] is track
        assert list(track.crops) == crops
        assert session.last is first
    second = engine.process(session.id, scene)
    assert second["frame"] == first["frame"] + 1
    assert second["objects"][0]["id"] == tid
    assert second["objects"][0]["label"] == "Rojo"
    assert second["objects"][0]["description"] == first["objects"][0]["description"]


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
        assert vision.embeds == 0 and vision.captions == 1
        engine.memory.learn("new", "person", [vision.embed(scene)])
        engine.process(sid, scene)
        assert vision.embeds == 2 and vision.captions == 1
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


def test_reference_without_gallery_still_embeds(engine, scene):
    sid = engine.create().id
    first = engine.process(sid, scene)
    tid = first["objects"][0]["id"]
    track = engine.session(sid).tracker.tracks[tid]
    engine.session(sid).reference = engine.vision.embed(track.crops[0])
    result = engine.process(sid, scene)
    assert result["alerts"][0]["track_id"] == tid
    assert result["objects"][0]["label"] is None
    assert not engine.memory.labels()


def test_embedding_batch_only_contains_categories_with_examples(settings, scene):
    class MixedVision(DemoVision):
        batch_sizes = None

        def detect(self, image, classes):
            return [Detection((0, 0, .4, .4), "person", 1.),
                    Detection((.5, .5, .9, .9), "car", 1.)]

        def embed_many(self, images):
            self.batch_sizes.append(len(images))
            return super().embed_many(images)

    vision = MixedVision()
    vision.batch_sizes = []
    engine = Engine(settings, vision)
    try:
        engine.memory.learn("car", "car", [vision.embed(scene)])
        result = engine.process(engine.create().id, scene)
        assert len(result["objects"]) == 2
        assert vision.batch_sizes == [1]
        assert all(obj["description"] for obj in result["objects"])
    finally:
        engine.close()


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


def test_predicted_frame_does_not_extract_crops(engine, scene, monkeypatch):
    sid = engine.create().id
    engine.configure(sid, FilterConfig(detector_interval=3))
    first = engine.process(sid, scene)
    track = engine.session(sid).tracker.tracks[first["objects"][0]["id"]]
    saved = len(track.crops)

    def unexpected_crop(*args):
        raise AssertionError("Un frame predicho no necesita recortes")

    monkeypatch.setattr("odin_vision.engine.crop", unexpected_crop)
    result = engine.process(sid, scene)
    assert result["objects"][0]["predicted"]
    assert result["objects"][0]["id"] == track.id
    assert len(track.crops) == saved


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
        engine.memory.learn("known", "person", [vision.embed(scene)])
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


def test_recognition_batches_categories_and_respects_interval(engine, scene, monkeypatch):
    vector = engine.vision.embed(scene)
    engine.memory.learn("Person", "person", [vector])
    engine.memory.learn("Car", "car", [vector])
    monkeypatch.setattr(engine.vision, "detect", lambda image, classes: [
        Detection((0., 0., .2, .2), "person", 1.),
        Detection((.3, .3, .5, .5), "car", 1.),
        Detection((.6, .6, .8, .8), "person", 1.),
    ])
    monkeypatch.setattr(engine.vision, "embed_many", lambda images: [vector] * len(images))
    calls = []
    original = engine.memory.match_many

    def match_many(vectors, category, threshold, margin):
        calls.append((category, len(vectors)))
        return original(vectors, category, threshold, margin)

    monkeypatch.setattr(engine.memory, "match_many", match_many)
    sid = engine.create().id
    engine.configure(sid, FilterConfig(detect=["person", "car"], recognition_interval=10))
    engine.session(sid).reference = vector
    first = engine.process(sid, scene)
    assert sorted(calls) == [("car", 1), ("person", 2)]
    assert [obj["label"] for obj in first["objects"]] == ["Person", "Car", "Person"]
    calls.clear()
    second = engine.process(sid, scene)
    assert not calls  # La referencia requiere embeddings, pero no otra búsqueda de etiquetas.
    assert len(second["alerts"]) == 3
    assert [obj["label"] for obj in second["objects"]] == ["Person", "Car", "Person"]
