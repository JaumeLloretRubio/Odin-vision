import numpy as np
from PIL import Image

from odin_vision.vision import DemoVision


def test_components_preserve_connectivity_area_and_order():
    pixels = np.full((80, 100, 3), 255, dtype=np.uint8)
    pixels[0:5, 0:8] = (255, 0, 0)  # Exactamente 40; toca el borde.
    pixels[5:10, 8:16] = (0, 255, 0)  # Solo contacto diagonal: otra instancia.
    pixels[20:23, 0:13] = (0, 0, 255)  # 39: descartar.
    detections = DemoVision().detect(Image.fromarray(pixels), ["person"])
    assert [d.box for d in detections] == [(0, 0, .08, .0625), (.08, .0625, .16, .125)]
    assert all(d.category == "person" and d.confidence == 1 for d in detections)


def test_empty_and_full_masks():
    vision = DemoVision()
    assert vision.detect(Image.new("RGB", (320, 180), "white"), ["person"]) == []
    assert vision.detect(Image.new("RGB", (320, 180), "red"), ["person"])[0].box == (0, 0, 1, 1)
