"""Retención de archivos; independiente de la aceptación del códec real."""
import os
from types import SimpleNamespace

from odin_vision.recording import Recorder


def finishing_recorder(directory, budget):
    recorder = Recorder(directory, max_bytes=budget)
    recorder.path = directory / "finished.avi"
    recorder.path.write_bytes(b"finished")
    recorder.writer = SimpleNamespace(release=lambda: None)
    return recorder


def test_retention_never_deletes_another_sessions_active_clip(tmp_path):
    active = tmp_path / "active.avi"
    active.write_bytes(b"active-clip")
    os.utime(active, (1, 1))
    recorder = finishing_recorder(tmp_path, 1)
    recorder.close()
    assert active.exists(), "Retención no debe tocar AVI sin manifiesto de finalización"
    assert not recorder.path.exists()


def test_retention_defers_locked_finalized_clip(tmp_path, monkeypatch):
    recorder = finishing_recorder(tmp_path, 1)
    unlink = type(tmp_path).unlink

    def locked(path, *args, **kwargs):
        if path == recorder.path:
            raise PermissionError("archivo descargándose")
        return unlink(path, *args, **kwargs)

    monkeypatch.setattr(type(tmp_path), "unlink", locked)
    recorder.close()
    assert recorder.path.exists()
    assert recorder.path.with_suffix(".json").exists()
