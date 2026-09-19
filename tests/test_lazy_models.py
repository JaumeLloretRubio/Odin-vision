import sys
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from odin_vision.vision import ModelVision


@pytest.fixture
def lazy_vision(monkeypatch, settings):
    loads, fail = [], set()

    class Inputs(dict):
        def to(self, device):
            return self

    class Processor:
        def __call__(self, images, **kwargs):
            return Inputs(count=len(images))

        def batch_decode(self, rows, **kwargs):
            return ["caption" for row in rows]

    class Model:
        def to(self, device):
            return self

        def eval(self):
            return self

        def get_image_features(self, count):
            return SimpleNamespace(ndim=2, cpu=lambda: SimpleNamespace(
                numpy=lambda: np.tile([3., 4.], (count, 1))))

        def generate(self, count, **kwargs):
            return SimpleNamespace(cpu=lambda: [[42] for _ in range(count)])

    def loader(name, factory):
        def load(model_name):
            loads.append(name)
            if name in fail:
                fail.remove(name)
                raise OSError("simulated model load failure")
            return factory()
        return SimpleNamespace(from_pretrained=load)

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(inference_mode=nullcontext))
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLOWorld=lambda name: object()))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoImageProcessor=loader("encoder_processor", Processor), AutoModel=loader("encoder", Model),
        BlipProcessor=loader("caption_processor", Processor),
        BlipForConditionalGeneration=loader("captioner", Model)))
    return ModelVision(settings), loads, fail


def test_unused_and_empty_models_are_not_loaded(lazy_vision):
    vision, loads, _ = lazy_vision
    assert vision.encoder is None and vision.captioner is None
    assert vision.embed_many([]) == []
    assert vision.describe_many([], []) == []
    assert loads == []


def test_first_use_loads_only_needed_model_and_reuses_it(lazy_vision, scene):
    vision, loads, _ = lazy_vision
    assert len(vision.embed_many([scene, scene])) == 2
    assert np.allclose(vision.embed(scene), [.6, .8])
    assert loads == ["encoder_processor", "encoder"]
    assert vision.captioner is None
    assert vision.describe_many([scene, scene], ["person", "car"]) == ["caption", "caption"]
    assert vision.describe(scene, "person") == "caption"
    assert loads == ["encoder_processor", "encoder", "caption_processor", "captioner"]


@pytest.mark.parametrize("model", ["encoder", "captioner"])
def test_failed_load_does_not_publish_partial_model_and_can_retry(lazy_vision, scene, model):
    vision, loads, fail = lazy_vision
    fail.add(model)
    call = (lambda: vision.embed(scene)) if model == "encoder" else (lambda: vision.describe(scene, "person"))
    with pytest.raises(OSError):
        call()
    processor = "encoder_processor" if model == "encoder" else "caption_processor"
    assert getattr(vision, model) is None and getattr(vision, processor) is None
    call()
    call()
    assert loads.count(model) == 2 and loads.count(processor) == 2


@pytest.mark.parametrize("model_type", ["siglip", "other"])
def test_only_siglip_releases_unused_text_before_device_transfer(lazy_vision, scene, monkeypatch, model_type):
    vision, _, _ = lazy_vision
    loader = sys.modules["transformers"].AutoModel
    original = loader.from_pretrained
    text_tower = object()

    def load(name):
        model = original(name)
        model.config = SimpleNamespace(model_type=model_type)
        model.text_model = text_tower

        def move(device):
            assert model.text_model is (None if model_type == "siglip" else text_tower)
            return model

        model.to = move
        return model

    monkeypatch.setattr(loader, "from_pretrained", load)
    assert np.allclose(vision.embed(scene), [.6, .8])


@pytest.mark.parametrize("mode", ["RGB", "L"])
def test_siglip_lookup_preserves_channel_values_and_falls_back(lazy_vision, monkeypatch, mode):
    vision, _, _ = lazy_vision
    calls = []

    class Inputs(dict):
        def to(self, device):
            return self

    class SiglipImageProcessor:
        def __call__(self, images, do_resize=True, do_rescale=True, do_normalize=True, return_tensors="np"):
            calls.append((do_resize, do_rescale, do_normalize, return_tensors))
            pixels = np.stack([np.asarray(image.convert("RGB")).transpose(2, 0, 1) for image in images])
            if do_rescale:
                pixels = pixels.astype(np.float32) / np.float32(255)
            if do_normalize:
                pixels = (pixels - np.array([.1, .2, .3], dtype=np.float32)[None, :, None, None]) / np.float32(.5)
            return Inputs(pixel_values=pixels)

    processor = SiglipImageProcessor()
    monkeypatch.setattr(sys.modules["transformers"].AutoImageProcessor, "from_pretrained", lambda _: processor)
    monkeypatch.setattr(vision.torch, "from_numpy", lambda value: value, raising=False)
    vision._load_encoder()
    image = Image.fromarray(np.random.default_rng(12).integers(0, 256, (19, 27, 3), dtype=np.uint8)).convert(mode)
    expected = processor([image])["pixel_values"]
    calls.clear()

    def features(pixel_values):
        np.testing.assert_array_equal(pixel_values, expected)
        return SimpleNamespace(ndim=2, cpu=lambda: SimpleNamespace(numpy=lambda: np.array([[3., 4.]])))

    vision.encoder.get_image_features = features
    assert np.allclose(vision.embed(image), [.6, .8])
    assert calls == [(True, mode != "RGB", mode != "RGB", "np" if mode == "RGB" else "pt")]
