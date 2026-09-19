"""Prueba del adaptador con dobles; no sustituye la aceptación con pesos reales."""
from types import SimpleNamespace

import numpy as np
import pytest

from odin_vision.vision import ModelVision


def test_yoloworld_adapter_updates_vocabulary_and_boxes(scene):
    class Tensor:
        def __init__(self, value):
            self.value = value

        def cpu(self):
            return self

        def tolist(self):
            return self.value

    class Detector:
        classes: list = []  # noqa: RUF012 - doble de prueba, estado compartido intencional
        updates = 0

        def set_classes(self, classes):
            self.classes = classes
            self.updates += 1

        def predict(self, image, **kwargs):
            assert image is scene
            return [SimpleNamespace(names={0: self.classes[0]}, boxes=SimpleNamespace(
                xyxyn=Tensor([[.1, .2, .3, .4]]), cls=Tensor([0]), conf=Tensor([.9])))]

    vision = ModelVision.__new__(ModelVision)
    vision.device, vision.classes, vision.detector = "cpu", None, Detector()
    assert vision.detect(scene, ["drone"])[0].category == "drone"
    assert vision.detect(scene, ["drone"])[0].box == (.1, .2, .3, .4)
    assert vision.detector.updates == 1
    assert vision.detect(scene, ["backpack"])[0].category == "backpack"
    assert vision.detector.updates == 2
    assert vision.detect(scene, []) == []


@pytest.mark.parametrize("fail_first", [False, True])
def test_cached_text_encoder_is_restored_after_prediction(scene, fail_first):
    cached = object()
    model = SimpleNamespace(clip_model=cached)
    empty = SimpleNamespace(cpu=lambda: SimpleNamespace(tolist=list))
    result = SimpleNamespace(names={}, boxes=SimpleNamespace(xyxyn=empty, cls=empty, conf=empty))
    failures = [fail_first]

    def set_classes(classes):
        assert model.clip_model is cached

    def predict(*args, **kwargs):
        assert model.clip_model is None
        if failures and failures.pop():
            raise RuntimeError("prediction failed")
        return [result]

    vision = ModelVision.__new__(ModelVision)
    vision.device, vision.classes = "cpu", None
    vision.detector = SimpleNamespace(model=model, set_classes=set_classes, predict=predict)
    if fail_first:
        with pytest.raises(RuntimeError, match="prediction failed"):
            vision.detect(scene, ["person"])
        assert model.clip_model is cached
    for classes in [["person"], ["person", "bus"], ["person"]]:
        assert vision.detect(scene, classes) == []
        assert model.clip_model is cached


@pytest.mark.parametrize("count", [0, 100])
def test_detector_bulk_results_keep_classes_confidence_and_order(scene, count):
    transfers = []

    class Tensor:
        def __init__(self, values):
            self.values = values

        def cpu(self):
            transfers.append(1)
            return self

        def tolist(self):
            return self.values

    boxes = [[i/200, .1, i/200+.2, .8] for i in range(count)]
    classes = [i % 2 for i in range(count)]
    confidence = [.5+i/200 for i in range(count)]
    result = SimpleNamespace(names={0: "person", 1: "car"}, boxes=SimpleNamespace(
        xyxyn=Tensor(boxes), cls=Tensor(classes), conf=Tensor(confidence)))
    vision = ModelVision.__new__(ModelVision)
    vision.device, vision.classes = "cuda:0", ["person", "car"]
    vision.detector = SimpleNamespace(predict=lambda *args, **kwargs: [result])
    detections = vision.detect(scene, ["person", "car"])
    assert len(detections) == count
    assert [d.box for d in detections] == [tuple(box) for box in boxes]
    assert [d.category for d in detections] == [result.names[c] for c in classes]
    assert [d.confidence for d in detections] == confidence
    assert len(transfers) == 3


def test_quality_rejects_flat_and_small_images():
    from PIL import Image

    from odin_vision.vision import quality
    assert not quality(Image.new("RGB", (100, 100), "gray"))
    assert not quality(Image.new("RGB", (5, 5)))
    checker = (np.indices((100, 100)).sum(axis=0) % 2 * 255).astype(np.uint8)
    assert quality(Image.fromarray(checker).convert("RGB"))


def test_pooled_extraction_handles_both_transformers_shapes():
    """El encoder debe entregar un vector por imagen, venga como tensor o como salida con `pooler_output`."""
    import numpy as np
    import pytest

    from odin_vision.vision import pooled

    class Output:
        def __init__(self, pooler_output, last_hidden_state):
            self.pooler_output = pooler_output
            self.last_hidden_state = last_hidden_state

    tensor = np.zeros((2, 768), dtype=np.float32)
    grid = np.zeros((2, 196, 768), dtype=np.float32)
    assert pooled(tensor).shape == (2, 768)
    assert pooled(Output(tensor, grid)).shape == (2, 768)
    with pytest.raises(ValueError, match="vector por imagen"):
        pooled(grid)
