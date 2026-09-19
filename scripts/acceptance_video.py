"""Aceptación con vídeo, modelos reales y HTTP/WebSocket sobre TCP local.

Reutiliza el streaming de la v1. ODIN_DETECTOR, ODIN_ENCODER y ODIN_CAPTIONER
permiten usar pesos locales. No accede a una cámara física ni cambia umbrales.
"""
import argparse
import asyncio
import json
import socket
import threading
import time
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

from scripts.lab_stream import digest, encode, run_stream


@contextmanager
def local_server(app):
    import uvicorn

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        server = uvicorn.Server(uvicorn.Config(app, log_level="warning"))
        thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
        thread.start()
        try:
            deadline = time.monotonic() + 120
            while not server.started:
                if not thread.is_alive() or time.monotonic() > deadline:
                    raise RuntimeError("El servidor local no arrancó")
                time.sleep(.01)
            yield f"http://127.0.0.1:{port}"
        finally:
            server.should_exit = True
            thread.join(timeout=30)
            if thread.is_alive():
                raise RuntimeError("El servidor local no terminó")


async def measure_stream(server, settings, args, teach):
    started = time.perf_counter()
    report = await run_stream(
        server, settings.token, args.video, args.width, 10, args.seconds,
        label="video-acceptance", teach_after=5 if teach else None,
        categories=args.classes.split(","),
        tuning={"describe": True, "record_labels": ["video-acceptance"]},
    )
    elapsed = time.perf_counter() - started
    summary = digest(report, elapsed)
    summary.update(elapsed_seconds=elapsed, config=report["config"],
                   frame_samples_ms=report["latencies_ms"])
    if report["errors"] or not report["recognized_frames"]:
        raise RuntimeError(f"Streaming sin reconocimiento o con errores: {summary}")
    if teach and (not report["assignment"] or report["assignment"]["status"] != 201):
        raise RuntimeError(f"No se pudo enseñar: {report['assignment']}")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path, help="Directorio nuevo y aislado")
    parser.add_argument("--classes", default="car")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--seconds", type=float, default=20)
    args = parser.parse_args()
    if args.seconds <= 0 or not 1 <= args.width <= 1920:
        parser.error("--seconds debe ser positivo; --width debe estar entre 1 y 1920")
    args.video = args.video.resolve()
    args.out_dir = args.out_dir.resolve()
    if not args.video.is_file():
        parser.error("--video debe ser un fichero existente")
    if args.out_dir.exists():
        parser.error("--out-dir debe ser nuevo para no reutilizar memoria")

    import cv2
    import httpx
    import torch
    import transformers
    from PIL import Image

    from odin_vision.config import Settings
    from odin_vision.server import create_app

    args.out_dir.mkdir(parents=True)
    settings = replace(Settings.from_env(), backend="models", data_dir=args.out_dir / "memory")
    capture = cv2.VideoCapture(str(args.video))
    try:
        source = {"frames": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
                  "fps": capture.get(cv2.CAP_PROP_FPS),
                  "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))}
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError("No se pudo decodificar el vídeo")
        first = encode(frame, args.width)
    finally:
        capture.release()
    results = {"scope": "Vídeo real reproducido por TCP local, sin cámara física ni red remota",
               "video": args.video.name, "source": source,
               "torch": torch.__version__, "transformers": transformers.__version__,
               "device": settings.device,
               "float32_matmul_precision": torch.get_float32_matmul_precision(),
               "phases": {}, "passed": False}
    headers = {"Authorization": f"Bearer {settings.token}"} if settings.token else {}
    app = create_app(settings)
    try:
        with (local_server(app) as server,
              httpx.Client(base_url=server, headers=headers, timeout=180, trust_env=False) as client):
            # Cargar los modelos antes de iniciar el reloj del vídeo en vivo.
            started = time.perf_counter()
            response = client.post("/api/v1/sessions")
            response.raise_for_status()
            sid = response.json()["id"]
            response = client.put(f"/api/v1/sessions/{sid}/config",
                                  json={"detect": args.classes.split(","), "describe": True})
            response.raise_for_status()
            response = client.post(f"/api/v1/sessions/{sid}/frames", content=first)
            response.raise_for_status()
            objects = response.json()["objects"]
            if not objects or not all(obj["description"] for obj in objects):
                raise RuntimeError("Faltan detecciones o descripciones en el calentamiento")
            app.state.engine.vision.embed_many([Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))])
            client.delete(f"/api/v1/sessions/{sid}").raise_for_status()
            results["warmup_seconds"] = time.perf_counter() - started
            print("Modelos cargados; empieza el streaming", flush=True)
            results["phases"]["live"] = asyncio.run(measure_stream(server, settings, args, True))
            print("Etiquetado en caliente verificado; empieza una sesión nueva", flush=True)
            results["phases"]["new_session"] = asyncio.run(measure_stream(server, settings, args, False))
            response = client.get("/api/v1/clips")
            response.raise_for_status()
            clips = response.json()["items"]
            if not clips:
                raise RuntimeError("No se generaron clips")
            downloaded = args.out_dir / "clips"
            downloaded.mkdir()
            decoded = 0
            for manifest in clips:
                response = client.get(f"/api/v1/clips/{manifest['id']}")
                response.raise_for_status()
                path = downloaded / f"{manifest['id']}.avi"
                path.write_bytes(response.content)
                decoder = cv2.VideoCapture(str(path))
                count = 0
                try:
                    while decoder.read()[0]:
                        count += 1
                finally:
                    decoder.release()
                if not count or count != len(manifest["frames"]):
                    raise RuntimeError("El clip descargado no coincide con su manifiesto")
                decoded += count
            response = client.get("/api/v1/events?label=video-acceptance&limit=200")
            response.raise_for_status()
            events = response.json()["items"]
            if not events:
                raise RuntimeError("No hay eventos de la etiqueta aprendida")
            results.update(clips=len(clips), decoded_clip_frames=decoded, queried_label_events=len(events))
            vision = app.state.engine.vision
        # Nueva aplicación y conexión SQLite; reutilizar pesos no reutiliza la memoria.
        with local_server(create_app(settings, vision=vision)) as server:
            print("Conexión SQLite reabierta; comprobando persistencia", flush=True)
            results["phases"]["reopened_memory"] = asyncio.run(measure_stream(server, settings, args, False))
        results["passed"] = True
    finally:
        (args.out_dir / "report.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Aceptación superada. Informe: {args.out_dir / 'report.json'}")


if __name__ == "__main__":
    main()
