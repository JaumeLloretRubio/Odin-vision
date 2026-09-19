import numpy as np
import pytest
from PIL import Image

from odin_vision.vision import quality


def reference_variance(image):
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    lap = gray[1:-1, :-2] + gray[1:-1, 2:] + gray[:-2, 1:-1] + gray[2:, 1:-1] - 4*gray[1:-1, 1:-1]
    return float(lap.var())


@pytest.mark.parametrize("size", [(11, 3000), (12, 12), (127, 258), (128, 256), (224, 224), (1920, 1080)])
def test_quality_matches_original_across_sizes(size):
    rng = np.random.default_rng(414)
    width, height = size
    for levels in [2, 3, 8, 256]:
        array = rng.integers(0, levels, (height, width, 3), dtype=np.uint8)
        image = Image.fromarray(array)
        assert quality(image) == (min(size) >= 12 and reference_variance(image) >= 8)


def test_large_image_exact_quality_threshold():
    y, x = np.indices((130, 258))
    pattern = ((x // 2) % 2 + (y // 2) % 2).astype(np.uint8)
    for scale, expected_variance in [(1, 2), (2, 8), (3, 18)]:
        image = Image.fromarray(pattern * scale)
        assert reference_variance(image) == expected_variance
        assert quality(image) == (expected_variance >= 8)
