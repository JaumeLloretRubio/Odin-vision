"""Aceptación manual con imágenes propias: ninguna descarga de imágenes implícita."""
import argparse
import json
from pathlib import Path

from odin_vision.config import Settings
from odin_vision.engine import Engine, decode_image
from odin_vision.schemas import FilterConfig


def main():
    parser = argparse.ArgumentParser(description="Verifica detección, memoria y persistencia con modelos reales")
    parser.add_argument("--reference", type=Path, required=True, help="Recorte de un objeto distintivo")
    parser.add_argument("--scene", type=Path, required=True, help="Foto del mismo objeto en una escena")
    parser.add_argument("--category", required=True, help="Vocabulario del detector, p. ej. backpack")
    parser.add_argument("--label", default="acceptance-object")
    parser.add_argument("--data-dir", type=Path, required=True, help="Directorio aislado para esta aceptación")
    parser.add_argument("--threshold", type=float, default=.85)
    args = parser.parse_args()
    base = Settings.from_env()
    from dataclasses import replace
    settings = replace(base, backend="models", data_dir=args.data_dir)
    reference = decode_image(args.reference.read_bytes(), settings)
    scene = decode_image(args.scene.read_bytes(), settings)
    engine = Engine(settings)
    try:
        engine.memory.learn(args.label, args.category, [engine.vision.embed(reference)])
    finally:
        engine.close()
    engine = Engine(settings)
    try:
        sid = engine.create().id
        engine.configure(sid, FilterConfig(detect=[args.category], threshold=args.threshold))
        result = engine.process(sid, scene)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if not any(obj["label"] == args.label and obj["description"] for obj in result["objects"]):
            raise SystemExit("ACEPTACIÓN FALLIDA: falta detección, descripción o reconocimiento después del reinicio")
    finally:
        engine.close()


if __name__ == "__main__":
    main()
