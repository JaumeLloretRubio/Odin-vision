import io
import re
import time
import uuid
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, UnidentifiedImageError

from .config import Settings
from .memory import Memory
from .recording import Recorder
from .schemas import FilterConfig
from .tracking import Tracker
from .vision import DemoVision, ModelVision, crop, quality


@dataclass
class Session:
    id: str
    config: FilterConfig = field(default_factory=FilterConfig)
    tracker: Tracker = field(default_factory=Tracker)
    frame: int = 0
    touched: float = field(default_factory=time.monotonic)
    location: dict | None = None
    reference: np.ndarray | None = None
    visible: set = field(default_factory=set)
    last: dict | None = None
    recorder: Recorder | None = None


def decode_image(data, settings):
    if len(data) > settings.max_bytes:
        raise ValueError("La imagen supera el límite de bytes")
    try:
        with io.BytesIO(data) as source, Image.open(source) as image:
            if image.format not in {"JPEG", "PNG", "WEBP"}:
                raise ValueError("Formato de imagen no permitido")
            if image.width * image.height > settings.max_pixels:
                raise ValueError("La imagen supera el límite de píxeles")
            image.load()
            # JPEG ya cargado no necesita otra copia RGB; PNG/WebP retienen estado del decodificador.
            if image.format == "JPEG" and image.mode == "RGB":
                return image
            return image.convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("Imagen inválida") from exc


class Engine:
    def __init__(self, settings: Settings, vision=None):
        if settings.backend not in {"demo", "models"}:
            raise ValueError("ODIN_BACKEND debe ser demo o models")
        self.settings = settings
        self.vision = vision or (DemoVision() if settings.backend == "demo" else ModelVision(settings))
        self.memory = Memory(settings, self.vision.fingerprint)
        self.sessions = {}

    def resolve(self, config):
        """Rellena umbral y margen no especificados con el valor calibrado del backend activo."""
        threshold, margin = self.settings.recognition_defaults()
        if config.threshold is not None and config.margin is not None:
            return config
        return config.model_copy(update={
            "threshold": threshold if config.threshold is None else config.threshold,
            "margin": margin if config.margin is None else config.margin})

    def create(self):
        now = time.monotonic()
        for session in self.sessions.values():
            if now-session.touched >= self.settings.session_ttl and session.recorder:
                session.recorder.close()
        self.sessions = {sid: s for sid, s in self.sessions.items() if now-s.touched < self.settings.session_ttl}
        if len(self.sessions) >= self.settings.max_sessions:
            raise ValueError("Límite de sesiones alcanzado")
        session = Session(uuid.uuid4().hex, config=self.resolve(FilterConfig()))
        self.sessions[session.id] = session
        return session

    def delete(self, sid):
        session = self.sessions.pop(sid, None)
        if session and session.recorder:
            session.recorder.close()

    def close(self):
        for sid in list(self.sessions):
            self.delete(sid)
        self.memory.close()

    def session(self, sid):
        session = self.sessions.get(sid)
        if session is None or time.monotonic()-session.touched > self.settings.session_ttl:
            self.delete(sid)
            raise KeyError("Sesión inexistente o caducada")
        session.touched = time.monotonic()
        return session

    def configure(self, sid, config):
        session = self.session(sid)
        config = self.resolve(config)
        if config == session.config:
            return config.model_dump()
        session.config = config
        next_id = session.tracker.next_id
        session.tracker = Tracker(max_age=max(15, config.detector_interval*3))
        session.tracker.next_id = next_id
        if session.recorder:
            session.recorder.close()
        session.visible.clear()
        session.last = None
        session.frame = 0
        return config.model_dump()

    def process(self, sid, image):
        started = time.perf_counter()
        session = self.session(sid)
        session.frame += 1
        frame, config = session.frame, session.config
        classes = config.detect
        run_detector = (frame-1) % config.detector_interval == 0
        detections = self.vision.detect(image, classes) if run_detector and classes else ([] if run_detector else None)
        if detections is not None:
            detections = [d for d in detections if d.category in classes and d.category not in config.ignore and
                          0 <= d.box[0] < d.box[2] <= 1 and 0 <= d.box[1] < d.box[3] <= 1]
        tracks = session.tracker.update(detections, frame)
        session.visible = {t.id for t in tracks}
        objects, alerts = [], []
        # Una pasada por lote para todo el frame: en CPU el coste por track baja frente a una llamada por track.
        patches = {track.id: crop(image, track.box) for track in tracks if track.last_seen == frame}
        pending = [track for track in tracks if track.last_seen == frame and
                   (frame-track.recognized_at >= config.recognition_interval or
                    track.revision != self.memory.revision or
                    (session.reference is not None))]
        galleries = {category: self.memory.has_examples(category) for category in {t.category for t in pending}}
        embedded = [t for t in pending if session.reference is not None or galleries[t.category]]
        vectors = {}
        if embedded:
            vectors = dict(zip([t.id for t in embedded],
                               self.vision.embed_many([patches[t.id] for t in embedded])))
        recognition = {}
        groups = {}
        for track in embedded:
            if frame-track.recognized_at >= config.recognition_interval or track.revision != self.memory.revision:
                groups.setdefault(track.category, []).append(track)
        for category, group in groups.items():
            matches = self.memory.match_many([vectors[t.id] for t in group], category,
                                             config.threshold, config.margin)
            recognition.update(zip([t.id for t in group], matches))
        described = [track for track in pending if config.describe and track.description is None]
        if described:
            for track, text in zip(described, self.vision.describe_many([patches[t.id] for t in described],
                                                                        [t.category for t in described])):
                track.description = text
        events, event_tracks = [], []
        for track in tracks:
            observed = track.last_seen == frame
            if observed:
                patch = patches[track.id]
                if self.settings.backend == "demo" or quality(patch):
                    track.crops.append(patch.resize((min(patch.width, 224), min(patch.height, 224))))
            vector = vectors.get(track.id)
            if observed and (frame-track.recognized_at >= config.recognition_interval or
                             track.revision != self.memory.revision):
                track.label, track.similarity = recognition.get(track.id, (None, None))
                track.recognized_at, track.revision = frame, self.memory.revision
            now = time.time()
            if observed and (not track.announced or now-track.last_event >= 5):
                events.append((now, sid, track.id, track.category, track.label, track.similarity, session.location,
                               "new_object" if not track.announced else "sighting"))
                event_tracks.append((track, now))
            match = None
            if observed and session.reference is not None:
                match = float(np.clip(vector @ session.reference, -1, 1))
                if match >= config.threshold:
                    alerts.append({"track_id": track.id, "similarity": match})
            objects.append({"id": track.id, "box": list(track.box), "class": track.category,
                            "label": track.label, "description": track.description, "confidence": track.confidence,
                            "similarity": track.similarity, "reference_similarity": match, "predicted": not observed})
        self.memory.event_many(events)
        for track, timestamp in event_tracks:
            track.announced = True
            track.last_event = timestamp
        session.last = {"frame": frame, "timestamp": time.time(), "width": image.width, "height": image.height,
                        "objects": objects, "alerts": alerts, "backend": self.settings.backend,
                        "processing_ms": round((time.perf_counter()-started)*1000, 2)}
        if config.record_labels:
            if session.recorder is None:
                session.recorder = Recorder(self.settings.data_dir / "clips")
            session.recorder.add(image, session.last, config.record_labels)
        return session.last

    def assign(self, sid, tid, label):
        session = self.session(sid)
        track = session.tracker.tracks.get(tid)
        if track is None or tid not in session.visible:
            raise KeyError("El objeto ya no está visible")
        vectors = self.vision.embed_many(list(track.crops))
        result = self.memory.learn(label, track.category, vectors)
        track.label = label
        track.recognized_at = -10000
        self.memory.event(sid, tid, track.category, label, 1.0, session.location, "assigned")
        return result

    def command(self, sid, text, tid=None):
        session = self.session(sid)
        text = text.strip().rstrip(".!?")
        match = re.fullmatch(r"(?:guarda|llama)(?: (?:el|la))?(?: objeto| persona)?(?: seleccionad[oa])? como (.+)",
                             text, re.IGNORECASE)
        if match:
            if tid is None:
                raise ValueError("Seleccione un objeto antes de asignarle una etiqueta")
            label = match[1].strip()
            if not 1 <= len(label) <= 100:
                raise ValueError("La etiqueta debe tener entre 1 y 100 caracteres")
            return self.assign(sid, tid, label)
        aliases = {"personas": "person", "persona": "person", "coches": "car", "coche": "car",
                   "drones": "drone", "dron": "drone", "mochilas": "backpack", "mochila": "backpack"}
        match = re.fullmatch(r"(?:no me marques|ignora) (.+)", text, re.IGNORECASE)
        if match:
            category = aliases.get(match[1].lower(), match[1].lower())
            config = session.config.model_dump()
            config["ignore"] = list(dict.fromkeys([*config["ignore"], category]))
            return self.configure(sid, FilterConfig(**config))
        match = re.fullmatch(r"solo muéstrame (.+)", text, re.IGNORECASE)
        if match:
            config = session.config.model_dump()
            config["detect"] = [aliases.get(c.strip().lower(), c.strip().lower()) for c in re.split(r",| y ", match[1])]
            config["ignore"] = []
            return self.configure(sid, FilterConfig(**config))
        raise ValueError("Comando no reconocido. Use: guarda como X; ignora coches; solo muéstrame personas y drones")
