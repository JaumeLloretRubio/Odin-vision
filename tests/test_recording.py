import json

import pytest

from odin_vision.recording import Recorder

cv2 = pytest.importorskip("cv2", reason="Las pruebas de clips requieren OpenCV real")


def test_record_and_finalize(tmp_path, scene):
    recorder = Recorder(tmp_path)
    frame = {"frame": 1, "timestamp": 10, "objects": [{"label": "Falcon", "predicted": False}]}
    recorder.add(scene, frame, ["Falcon"])
    recorder.add(scene, {**frame, "frame": 2}, ["Falcon"])
    recorder.close()
    path = next(tmp_path.glob("*.avi"))
    capture = cv2.VideoCapture(str(path))
    try:
        assert capture.read()[0]
        assert capture.read()[0]
        assert not capture.read()[0]
    finally:
        capture.release()
    metadata = json.loads(path.with_suffix(".json").read_text())
    assert metadata["labels"] == ["Falcon"]
    assert len(metadata["frames"]) == 2


def test_no_recording_without_match_and_retention(tmp_path, scene):
    recorder = Recorder(tmp_path, max_bytes=1)
    recorder.add(scene, {"objects": []}, ["Falcon"])
    assert not list(tmp_path.iterdir())
    recorder.add(scene, {"frame": 1, "timestamp": 0, "objects": [{"label": "Falcon", "predicted": False}]}, ["Falcon"])
    recorder.close()
    assert not list(tmp_path.iterdir())
