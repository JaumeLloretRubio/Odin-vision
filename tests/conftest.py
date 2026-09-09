import io

import pytest
from PIL import Image, ImageDraw

from odin_vision.config import Settings
from odin_vision.engine import Engine


@pytest.fixture
def settings(tmp_path):
    return Settings(data_dir=tmp_path)


@pytest.fixture
def engine(settings):
    instance = Engine(settings)
    yield instance
    instance.close()


@pytest.fixture
def scene():
    image = Image.new("RGB", (200, 120), "white")
    ImageDraw.Draw(image).rectangle((30, 20, 80, 90), fill="red")
    return image


@pytest.fixture
def encoded(scene):
    output = io.BytesIO()
    scene.save(output, format="PNG")
    return output.getvalue()
