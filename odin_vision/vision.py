"""Backends intercambiables; el modo demo nunca se presenta como reconocimiento real."""
from dataclasses import dataclass

import cv2
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
        if not mask.any():
            return []
        h, w = mask.shape
        category = classes[0] if classes else "object"
        # Mantener cuatro vecinos: componentes que solo se tocan en diagonal siguen separados.
        _, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=4)
        results = [Detection((int(x)/w, int(y)/h, int(x+width)/w, int(y+height)/h), category, 1.0)
                   for x, y, width, height, area in stats[1:] if area >= 40]
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
        from ultralytics import YOLOWorld

        self.torch = torch
        self.device = settings.device
        self.settings = settings
        self.detector = YOLOWorld(settings.detector_model)
        self.encoder_processor = self.encoder = None
        self.encoder_lookup = None
        self.caption_processor = self.captioner = None
        # El sufijo distingue la extracción: transformers>=5 devuelve un objeto y hay que tomar el
        # vector agrupado; una memoria escrita con otra extracción no es comparable.
        self.fingerprint = f"siglip-pooled:{settings.encoder_model}"
        self.classes = None

    def _load_encoder(self):
        if self.encoder is None:
            from transformers import AutoImageProcessor, AutoModel

            processor = AutoImageProcessor.from_pretrained(self.settings.encoder_model)
            encoder = AutoModel.from_pretrained(self.settings.encoder_model)
            if getattr(getattr(encoder, "config", None), "model_type", None) == "siglip":
                # get_image_features solo usa vision_model; liberar texto antes de copiar a la GPU.
                encoder.text_model = None
            encoder = encoder.to(self.device).eval()
            lookup = None
            if type(processor).__name__ == "SiglipImageProcessor":
                # RGB uint8 tiene 256 valores: conservar el redondeo exacto del procesador por canal.
                palette = np.repeat(np.arange(256, dtype=np.uint8).reshape(16, 16, 1), 3, axis=2)
                values = processor(images=[Image.fromarray(palette)], do_resize=False,
                                   return_tensors="np")["pixel_values"]
                if values.shape == (1, 3, 16, 16) and values.dtype == np.float32:
                    lookup = values.reshape(3, 256)
            self.encoder_lookup = lookup
            self.encoder_processor, self.encoder = processor, encoder

    def _load_captioner(self):
        if self.captioner is None:
            from transformers import BlipForConditionalGeneration, BlipProcessor

            processor = BlipProcessor.from_pretrained(self.settings.caption_model)
            captioner = BlipForConditionalGeneration.from_pretrained(self.settings.caption_model).to(self.device).eval()
            self.caption_processor, self.captioner = processor, captioner

    def detect(self, image, classes):
        if not classes:
            return []
        if classes != self.classes:
            self.detector.set_classes(classes)
            self.classes = list(classes)
        model = getattr(self.detector, "model", None)
        cached_clip = getattr(model, "clip_model", None)
        # El predictor usa txt_feats, no CLIP. Evitar copiar sus pesos al backend de imágenes.
        if cached_clip is not None:
            model.clip_model = None
        try:
            result = self.detector.predict(image, device=self.device, verbose=False, max_det=100)[0]
        finally:
            if cached_clip is not None:
                model.clip_model = cached_clip
        boxes = result.boxes
        # Transferir cada tensor una vez evita sincronizaciones GPU por cada detección.
        return [Detection(tuple(box), result.names[int(category)], float(confidence))
                for box, category, confidence in zip(boxes.xyxyn.cpu().tolist(), boxes.cls.cpu().tolist(),
                                                      boxes.conf.cpu().tolist())]

    def embed(self, image):
        return self.embed_many([image])[0]

    def embed_many(self, images):
        if not images:
            return []
        self._load_encoder()
        if self.encoder_lookup is not None and all(isinstance(image, Image.Image) and image.mode == "RGB"
                                                   for image in images):
            inputs = self.encoder_processor(images=list(images), do_rescale=False, do_normalize=False,
                                            return_tensors="np")
            pixels = inputs["pixel_values"]
            channels = np.arange(3)[None, :, None, None]
            inputs["pixel_values"] = self.torch.from_numpy(self.encoder_lookup[channels, pixels])
            inputs = inputs.to(self.device)
        else:
            inputs = self.encoder_processor(images=list(images), return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            features = self.encoder.get_image_features(**inputs)
        return [normalized(row) for row in pooled(features).cpu().numpy()]

    def describe(self, image, category):
        return self.describe_many([image], [category])[0]

    def describe_many(self, images, categories):
        if not images:
            return []
        self._load_captioner()
        inputs = self.caption_processor(images=list(images), return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            output = self.captioner.generate(**inputs, max_new_tokens=30)
        # Una transferencia del lote evita sincronizar la GPU por cada descripción.
        return self.caption_processor.batch_decode(output.cpu(), skip_special_tokens=True)


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
    if image.width * image.height < 32768:
        gray = np.asarray(image.convert("L"), dtype=np.float32)
        laplacian = gray[1:-1, :-2] + gray[1:-1, 2:] + gray[:-2, 1:-1] + gray[2:, 1:-1] - 4*gray[1:-1, 1:-1]
    else:
        gray = np.asarray(image.convert("L"))
        # Igual kernel de cuatro vecinos; excluir el borde y conservar la reducción contigua float32.
        laplacian = cv2.Laplacian(gray, cv2.CV_32F, ksize=1)[1:-1, 1:-1].copy()
    return float(laplacian.var()) >= 8
