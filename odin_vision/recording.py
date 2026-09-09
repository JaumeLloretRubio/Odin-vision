"""Clips locales AVI/MJPEG con tiempos originales en un fichero lateral JSON."""
import json
import logging
import time
import uuid
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class Recorder:
    def __init__(self, directory: Path, max_bytes=100_000_000):
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_bytes
        self.writer = None
        self.frames = []
        self.started = 0
        self.path = None
        self.labels = set()

    def add(self, image, result, wanted):
        import cv2
        matches = {o["label"] for o in result["objects"] if o["label"] in wanted and not o["predicted"]}
        if self.writer and (not matches or time.time()-self.started >= 10):
            self.close()
        if not matches:
            return
        if self.writer is None:
            self.path = self.directory / f"{uuid.uuid4().hex}.avi"
            self.size = (min(960, image.width)//2*2, min(540, image.height)//2*2)
            if min(self.size) < 2:
                return
            self.writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"MJPG"), 10, self.size)
            if not self.writer.isOpened():
                self.writer.release()
                self.writer = None
                raise ValueError("No se pudo abrir el codificador de clips")
            self.started = time.time()
            self.frames = []
            self.labels = set()
        self.writer.write(cv2.cvtColor(np.asarray(image.resize(self.size)), cv2.COLOR_RGB2BGR))
        self.labels.update(matches)
        self.frames.append({"frame": result["frame"], "timestamp": result["timestamp"]})
        if len(self.frames) >= 300:
            self.close()

    def close(self):
        if self.writer is None:
            return
        self.writer.release()
        self.writer = None
        manifest = self.path.with_suffix(".json")
        temporary = manifest.with_suffix(".json.tmp")
        temporary.write_text(json.dumps({
            "id": self.path.stem, "started": self.started, "labels": sorted(self.labels),
            "fps": 10, "frames": self.frames,
        }), encoding="utf-8")
        temporary.replace(manifest)
        # El manifiesto solo se publica después de cerrar el codificador.
        clips = sorted((p for p in self.directory.glob("*.avi") if p.with_suffix(".json").exists()),
                       key=lambda p: p.stat().st_mtime, reverse=True)
        total = 0
        for path in clips:
            total += path.stat().st_size
            if total > self.max_bytes:
                try:
                    path.unlink()
                    path.with_suffix(".json").unlink(missing_ok=True)
                except PermissionError:
                    # Windows bloquea archivos abiertos por una descarga: reintentar al próximo cierre.
                    logger.warning("Retención aplazada para el clip %s", path.stem)
