"""Backends intercambiables; el modo demo nunca se presenta como reconocimiento real."""
from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass
class Detection:
    box: tuple[float, float, float, float]
    category: str
    confidence: float


def normalized(vector):
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(vector)
    if not vector.size or not np.all(np.isfinite(vector)) or norm < 1e-8:
        raise ValueError("Embedding vacío o no finito")
    return vector / norm


class DemoVision:
    fingerprint = "demo-histogram-v1"

    def detect(self, image, classes):
        # Objetos de color saturado sobre fondo neutro, para probar sin pesos.
        sample = image.copy()
        sample.thumbnail((320, 180))
        array = np.asarray(sample).astype(np.int16)
        maximum, minimum = array.max(axis=2), array.min(axis=2)
        mask = (maximum >= 40) & ((maximum-minimum) > maximum*.32)
        h, w = mask.shape
        category = classes[0] if classes else "object"
        results = []
        for y, x in zip(*np.where(mask)):
            if not mask[y, x]:
                continue
            stack = [(int(x), int(y))]
            mask[y, x] = False
            x1 = x2 = int(x)
            y1 = y2 = int(y)
            count = 0
            while stack:
                px, py = stack.pop()
                count += 1
                x1, x2, y1, y2 = min(x1, px), max(x2, px), min(y1, py), max(y2, py)
                for nx, ny in ((px-1, py), (px+1, py), (px, py-1), (px, py+1)):
                    if 0 <= nx < w and 0 <= ny < h and mask[ny, nx]:
                        mask[ny, nx] = False
                        stack.append((nx, ny))
            if count >= 40:
                results.append(Detection((x1/w, y1/h, (x2+1)/w, (y2+1)/h), category, 1.0))
        return sorted(results, key=lambda d: d.box)[:100]

    def embed(self, image):
        arr = np.asarray(image.resize((32, 32)))
        hist = np.concatenate([np.histogram(arr[:, :, i], bins=16, range=(0, 256))[0] for i in range(3)])
        return normalized(hist)

    def describe(self, image, category):
        color = np.asarray(image.resize((1, 1)))[0, 0]
        return f"{category} {['rojo', 'verde', 'azul'][int(np.argmax(color))]} (demo)"

    def embed_many(self, images):
        return [self.embed(image) for image in images]

    def describe_many(self, images, categories):
        return [self.describe(image, category) for image, category in zip(images, categories)]


class ModelVision:
    def __init__(self, settings):
        import torch
        from transformers import AutoModel, AutoProcessor, BlipForConditionalGeneration, BlipProcessor
        from ultralytics import YOLOWorld

        self.torch = torch
        self.device = settings.device
        self.detector = YOLOWorld(settings.detector_model)
        self.encoder_processor = AutoProcessor.from_pretrained(settings.encoder_model)
        self.encoder = AutoModel.from_pretrained(settings.encoder_model).to(self.device).eval()
        self.caption_processor = BlipProcessor.from_pretrained(settings.caption_model)
        self.captioner = BlipForConditionalGeneration.from_pretrained(settings.caption_model).to(self.device).eval()
        # El sufijo distingue la extracción: transformers>=5 devuelve un objeto y hay que tomar el
        # vector agrupado; una memoria escrita con otra extracción no es comparable.
        self.fingerprint = f"siglip-pooled:{settings.encoder_model}"
        self.classes = None

    def detect(self, image, classes):
        if not classes:
            return []
        if classes != self.classes:
            self.detector.set_classes(classes)
            self.classes = list(classes)
        result = self.detector.predict(image, device=self.device, verbose=False, max_det=100)[0]
        return [Detection(tuple(b.xyxyn[0].tolist()), result.names[int(b.cls.item())], float(b.conf.item()))
                for b in result.boxes]

    def embed(self, image):
        return self.embed_many([image])[0]

    def embed_many(self, images):
        if not images:
            return []
        inputs = self.encoder_processor(images=list(images), return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            features = self.encoder.get_image_features(**inputs)
        return [normalized(row) for row in pooled(features).cpu().numpy()]

    def describe(self, image, category):
        return self.describe_many([image], [category])[0]

    def describe_many(self, images, categories):
        if not images:
            return []
        inputs = self.caption_processor(images=list(images), return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            output = self.captioner.generate(**inputs, max_new_tokens=30)
        return [self.caption_processor.decode(row, skip_special_tokens=True) for row in output]


def pooled(features):
    """transformers>=5 devuelve BaseModelOutputWithPooling; versiones anteriores, el tensor ya agrupado."""
    vectors = getattr(features, "pooler_output", features)
    if vectors.ndim != 2:
        raise ValueError("El encoder no devolvió un vector por imagen; revise el modelo configurado")
    return vectors


def crop(image: Image.Image, box):
    w, h = image.size
    x1, y1, x2, y2 = box
    return image.crop((max(0, int(x1*w)), max(0, int(y1*h)), min(w, max(int(x1*w)+1, int(x2*w))),
                       min(h, max(int(y1*h)+1, int(y2*h)))))


def quality(image):
    if min(image.size) < 12:
        return False
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    laplacian = gray[1:-1, :-2] + gray[1:-1, 2:] + gray[:-2, 1:-1] + gray[2:, 1:-1] - 4*gray[1:-1, 1:-1]
    return float(laplacian.var()) >= 8
