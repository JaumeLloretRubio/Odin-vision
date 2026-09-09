import os
from dataclasses import dataclass
from pathlib import Path

# Calibrado el 9 de septiembre de 2026 sobre vídeo de laboratorio (curva ROC, 4000 positivos y
# 3166 negativos): 0.91 es el umbral más bajo sin falsos positivos, con 87% de aciertos. El margen
# sale de la identificación en conjunto cerrado: 0.02 da 98.8% de precisión y 69% de aceptación,
# mientras que 0.10 bloquea dos tercios de los aciertos para evitar solo seis confusiones más.
# demo usa histogramas de color, con una escala de similitud distinta a la de un encoder aprendido.
BACKEND_DEFAULTS = {"demo": (0.85, 0.05), "models": (0.91, 0.02)}


def optional_ratio(name):
    raw = os.getenv(name, "").strip()
    if not raw:
        return None
    value = float(raw)
    if not 0 <= value <= 1:
        raise ValueError(f"{name} debe estar entre 0 y 1")
    return value


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path("data")
    backend: str = "demo"
    token: str = ""
    detector_model: str = "yolov8s-worldv2.pt"
    encoder_model: str = "google/siglip-base-patch16-224"
    caption_model: str = "Salesforce/blip-image-captioning-base"
    device: str = "cpu"
    max_sessions: int = 8
    max_bytes: int = 2_000_000
    max_pixels: int = 1920 * 1080
    session_ttl: float = 300
    max_labels: int = 1000
    max_examples: int = 16
    event_limit: int = 10000
    threshold: float | None = None
    margin: float | None = None

    def recognition_defaults(self):
        threshold, margin = BACKEND_DEFAULTS[self.backend]
        return (self.threshold if self.threshold is not None else threshold,
                self.margin if self.margin is not None else margin)

    @classmethod
    def from_env(cls):
        return cls(
            data_dir=Path(os.getenv("ODIN_DATA_DIR", "data")),
            backend=os.getenv("ODIN_BACKEND", "demo"),
            token=os.getenv("ODIN_TOKEN", ""),
            detector_model=os.getenv("ODIN_DETECTOR", "yolov8s-worldv2.pt"),
            encoder_model=os.getenv("ODIN_ENCODER", "google/siglip-base-patch16-224"),
            caption_model=os.getenv("ODIN_CAPTIONER", "Salesforce/blip-image-captioning-base"),
            device=os.getenv("ODIN_DEVICE", "cpu"),
            threshold=optional_ratio("ODIN_THRESHOLD"),
            margin=optional_ratio("ODIN_MARGIN"),
        )
