"""Renderiza un vídeo con las cajas, los IDs y las etiquetas dibujadas encima, para revisión visual.

Usa el motor real contra una memoria existente, así que muestra exactamente lo que el sistema decide.
"""
import argparse
import time
from dataclasses import replace
from pathlib import Path

from odin_vision.config import Settings
from odin_vision.engine import Engine
from odin_vision.schemas import FilterConfig

RECONOCIDO = (80, 220, 120)
ANONIMO = (200, 200, 200)
PREDICHO = (90, 160, 235)


def draw(frame, result, font_scale=0.5):
    import cv2

    height, width = frame.shape[:2]
    for obj in result["objects"]:
        x1, y1, x2, y2 = (int(obj["box"][0]*width), int(obj["box"][1]*height),
                          int(obj["box"][2]*width), int(obj["box"][3]*height))
        color = RECONOCIDO if obj["label"] else (PREDICHO if obj["predicted"] else ANONIMO)
        thickness = 3 if obj["label"] else 1
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        parts = [f"#{obj['id']} {obj['class']}"]
        if obj["label"]:
            parts.append(f"{obj['label']} {obj['similarity']:.2f}")
        elif obj["similarity"] is not None:
            parts.append(f"{obj['similarity']:.2f}")
        text = " | ".join(parts)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        top = max(th+4, y1)
        cv2.rectangle(frame, (x1, top-th-4), (x1+tw+4, top), color, -1)
        cv2.putText(frame, text, (x1+2, top-3), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (20, 20, 20), 1,
                    cv2.LINE_AA)
    banner = (f"frame {result['frame']}  |  {len(result['objects'])} objetos  |  "
              f"{result['processing_ms']:.0f} ms  |  backend {result['backend']}")
    cv2.rectangle(frame, (0, 0), (width, 24), (25, 25, 25), -1)
    cv2.putText(frame, banner, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1, cv2.LINE_AA)
    return frame


def main():
    parser = argparse.ArgumentParser(description="Vídeo anotado con las decisiones reales del motor")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True, help="Memoria con las etiquetas ya aprendidas")
    parser.add_argument("--classes", default="person")
    parser.add_argument("--stride", type=int, default=3, help="Procesar uno de cada N frames del original")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--describe", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--contact-sheet", type=Path, default=None, help="PNG con una rejilla de muestras")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    import cv2
    import numpy as np
    from PIL import Image

    settings = replace(Settings.from_env(), backend="models", data_dir=args.data_dir)
    engine = Engine(settings)
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise SystemExit(f"No se pudo abrir {args.video}")
    source_fps = capture.get(cv2.CAP_PROP_FPS) or 25
    writer, samples, processed = None, [], 0
    started = time.perf_counter()
    try:
        sid = engine.create().id
        engine.configure(sid, FilterConfig(detect=[c.strip() for c in args.classes.split(",")],
                                           describe=args.describe))
        index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            index += 1
            if (index-1) % args.stride:
                continue
            height, original = frame.shape[:2]
            factor = min(1.0, args.width/original)
            if factor < 1:
                frame = cv2.resize(frame, (round(original*factor), round(height*factor)))
            result = engine.process(sid, Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            processed += 1
            annotated = draw(frame, result)
            if writer is None:
                size = (annotated.shape[1]//2*2, annotated.shape[0]//2*2)
                args.out.parent.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(str(args.out), cv2.VideoWriter_fourcc(*"mp4v"),
                                         source_fps/args.stride, size)
                if not writer.isOpened():
                    raise SystemExit("No se pudo abrir el codificador de vídeo")
            writer.write(cv2.resize(annotated, (annotated.shape[1]//2*2, annotated.shape[0]//2*2)))
            if any(o["label"] for o in result["objects"]) or len(samples) < 2:
                samples.append(annotated.copy())
            if processed % 25 == 0:
                print(f"{processed} frames procesados, {time.perf_counter()-started:.0f} s", flush=True)
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        engine.close()

    print(f"Vídeo anotado: {args.out} ({processed} frames, "
          f"{(time.perf_counter()-started)/max(1, processed)*1000:.0f} ms por frame)")
    if args.contact_sheet and samples:
        picks = np.linspace(0, len(samples)-1, min(6, len(samples))).astype(int)
        columns = 2
        tiles = [cv2.resize(samples[i], (480, 270)) for i in picks]
        rows = [np.hstack(tiles[i:i+columns]) for i in range(0, len(tiles)-len(tiles) % columns, columns)]
        cv2.imwrite(str(args.contact_sheet), np.vstack(rows))
        print("Muestras:", args.contact_sheet)


if __name__ == "__main__":
    main()
