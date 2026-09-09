import itertools
import json
from types import SimpleNamespace

import numpy as np
import pytest

from odin_vision import camera


def test_headless_backpressure_and_json(monkeypatch, capsys):
    monkeypatch.setitem(__import__("sys").modules, "cv2", SimpleNamespace(
        imencode=lambda *args: (True, np.array([1, 2, 3], dtype=np.uint8)), IMWRITE_JPEG_QUALITY=1))
    clock = itertools.count()
    monkeypatch.setattr(camera.time, "monotonic", lambda: next(clock))
    frames = iter([np.zeros((20, 20, 3), dtype=np.uint8) for _ in range(3)])

    class Capture:
        def isOpened(self):
            return True

        def read(self):
            frame = next(frames, None)
            return frame is not None, frame

    class Client:
        calls = 0

        def post(self, path, **kwargs):
            self.calls += 1
            assert path == "/api/v1/sessions/test/frames"
            assert kwargs["content"] == bytes([1, 2, 3])
            return SimpleNamespace(status_code=429 if self.calls == 1 else 200,
                                   raise_for_status=lambda: None, json=lambda: {"frame": self.calls})

    client = Client()
    assert camera.transmit(Capture(), client, "test", 5, 960, max_frames=1) == 1
    assert client.calls == 2
    assert json.loads(capsys.readouterr().out) == {"frame": 2}


def test_camera_bad_args_fail_before_capture(monkeypatch):
    monkeypatch.setattr("sys.argv", ["odin-camera", "--fps", "0"])
    with pytest.raises(SystemExit) as exc:
        camera.main()
    assert exc.value.code == 2


def test_auth_header_omitted_without_token(monkeypatch):
    monkeypatch.delenv("ODIN_TOKEN", raising=False)
    assert camera.auth_headers() == {}
    monkeypatch.setenv("ODIN_TOKEN", "  ")
    assert camera.auth_headers() == {}
    monkeypatch.setenv("ODIN_TOKEN", "secreto")
    assert camera.auth_headers() == {"Authorization": "Bearer secreto"}
