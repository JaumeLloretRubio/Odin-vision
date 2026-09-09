"""Simula una cámara en directo a partir de un vídeo y valida el etiquetado en tiempo real.

No accede a ninguna cámara física: reproduce el fichero a su cadencia real, descarta frames atrasados como
haría una fuente en vivo y usa el mismo WebSocket que el cliente web.

Fases:
1. Calentamiento y streaming en vivo.
2. Etiquetado en caliente: se elige el track más estable visible y se le asigna una etiqueta sin cortar el stream.
3. Verificación en el mismo stream: cuántos frames posteriores reconocen la etiqueta.
4. Verificación en una sesión nueva: la memoria debe reconocer la instancia sin volver a enseñarla.
"""
import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path


def open_source(path, width):
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise SystemExit(f"No se pudo abrir {path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 25
    return capture, fps, width


def encode(frame, width):
    import cv2

    height, original = frame.shape[:2]
    factor = min(1.0, width/original)
    if factor < 1:
        frame = cv2.resize(frame, (round(original*factor), round(height*factor)))
    ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        raise RuntimeError("No se pudo codificar el frame")
    return buffer.tobytes()


class Stream:
    """Fuente en vivo simulada: el reloj manda y los frames atrasados se descartan."""

    def __init__(self, path, width, send_fps):
        self.capture, self.fps, self.width = open_source(path, width)
        self.period = 1/send_fps
        self.started = time.monotonic()
        self.read = self.dropped = 0

    def next(self):
        target = self.started + self.read*(1/self.fps)
        while True:
            ok, frame = self.capture.read()
            if not ok:
                return None
            self.read += 1
            target = self.started + self.read*(1/self.fps)
            behind = time.monotonic() - target
            if behind > self.period:
                self.dropped += 1  # La cámara no espera: el frame atrasado se pierde.
                continue
            if behind < 0:
                time.sleep(-behind)
            return encode(frame, self.width)

    def close(self):
        self.capture.release()


async def run_stream(server, token, path, width, send_fps, seconds, label=None, teach_after=None,
                     categories=("person",), tuning=None):
    import httpx
    import websockets

    report = {"frames": 0, "latencies_ms": [], "recognized_frames": 0, "labelled_track": None,
              "assignment": None, "errors": []}
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    async with httpx.AsyncClient(base_url=server, timeout=60, headers=headers) as http:
        response = await http.post("/api/v1/sessions")
        response.raise_for_status()
        sid = response.json()["id"]
        config = await http.put(f"/api/v1/sessions/{sid}/config",
                                json={"detect": list(categories), **(tuning or {})})
        config.raise_for_status()
        report["config"] = config.json()
        stream = Stream(path, width, send_fps)
        url = server.replace("http://", "ws://").replace("https://", "wss://")
        try:
            async with websockets.connect(f"{url}/api/v1/sessions/{sid}/stream",
                                          max_size=8_000_000, origin=None) as socket:
                await socket.send(token)
                ready = json.loads(await socket.recv())
                if not ready.get("ready"):
                    raise SystemExit(f"El servidor no aceptó el stream: {ready}")
                deadline = time.monotonic() + seconds
                while time.monotonic() < deadline:
                    frame = stream.next()
                    if frame is None:
                        break
                    sent = time.perf_counter()
                    await socket.send(frame)
                    result = json.loads(await socket.recv())
                    report["latencies_ms"].append((time.perf_counter()-sent)*1000)
                    if "error" in result:
                        report["errors"].append(result["error"])
                        continue
                    report["frames"] += 1
                    objects = result.get("objects", [])
                    if label and any(o["label"] == label for o in objects):
                        report["recognized_frames"] += 1
                    if teach_after and report["labelled_track"] is None and report["frames"] >= teach_after:
                        visible = [o for o in objects if not o["predicted"]]
                        if not visible:
                            continue
                        target = max(visible, key=lambda o: o["confidence"])
                        assignment = await http.post(
                            f"/api/v1/sessions/{sid}/tracks/{target['id']}/labels", json={"label": label})
                        report["assignment"] = {"status": assignment.status_code, "body": assignment.json(),
                                                "track": target["id"], "description": target["description"],
                                                "frame": result["frame"]}
                        report["labelled_track"] = target["id"]
                        report["recognized_frames"] = 0  # Solo cuentan los frames posteriores a enseñar.
                        report["taught_at_frame"] = result["frame"]
        finally:
            stream.close()
            await http.delete(f"/api/v1/sessions/{sid}")
        report["read_frames"] = stream.read
        report["dropped_frames"] = stream.dropped
        report["session"] = sid
    return report


def latency(values):
    if not values:
        return {}
    ordered = sorted(values)
    return {"mean_ms": round(statistics.fmean(ordered), 1), "p50_ms": round(ordered[len(ordered)//2], 1),
            "p95_ms": round(ordered[min(len(ordered)-1, int(len(ordered)*0.95))], 1),
            "max_ms": round(ordered[-1], 1)}


def digest(report, seconds):
    return {"frames_processed": report["frames"], "frames_read": report["read_frames"],
            "frames_dropped_by_pacing": report["dropped_frames"],
            "effective_fps": round(report["frames"]/seconds, 2), "latency": latency(report["latencies_ms"]),
            "errors": report["errors"][:3], "assignment": report.get("assignment"),
            "recognized_frames_after_labelling": report["recognized_frames"]}


async def main_async(args):
    label = args.label
    tuning = {"describe": args.describe, "recognition_interval": args.recognition_interval,
              "detector_interval": args.detector_interval}
    if args.threshold is not None:
        tuning["threshold"] = args.threshold
    live = await run_stream(args.server, args.token, args.video, args.width, args.fps, args.seconds,
                            label=label, teach_after=args.teach_after, categories=args.classes.split(","),
                            tuning=tuning)
    print("== Fase 1-3: stream en vivo con etiquetado en caliente ==")
    print(json.dumps(digest(live, args.seconds), indent=2, ensure_ascii=False))
    if live["labelled_track"] is None:
        raise SystemExit("No se pudo etiquetar ningún track visible; sube --seconds o baja --teach-after")

    recall = await run_stream(args.server, args.token, args.video, args.width, args.fps, args.seconds,
                              label=label, categories=args.classes.split(","), tuning=tuning)
    print("== Fase 4: sesión nueva, memoria persistente sin reenseñar ==")
    summary = digest(recall, args.seconds)
    summary["recognized_frames_after_labelling"] = recall["recognized_frames"]
    summary["recognition_rate"] = round(recall["recognized_frames"]/max(1, recall["frames"]), 3)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"live": digest(live, args.seconds), "new_session": summary,
                                        "label": label, "video": str(args.video)}, indent=2, ensure_ascii=False),
                            encoding="utf-8")
        print("Informe:", args.out)
    if not recall["recognized_frames"]:
        raise SystemExit("FALLO: la etiqueta aprendida no se reconoció en una sesión nueva")


def main():
    parser = argparse.ArgumentParser(description="Streaming simulado y etiquetado en tiempo real")
    parser.add_argument("--server", default="http://127.0.0.1:8000")
    parser.add_argument("--video", type=Path, required=True, help="Fichero local que hace de cámara")
    parser.add_argument("--token", default="")
    parser.add_argument("--classes", default="person")
    parser.add_argument("--label", default="sujeto-lab")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--fps", type=float, default=10, help="Cadencia máxima de envío")
    parser.add_argument("--seconds", type=float, default=20)
    parser.add_argument("--teach-after", type=int, default=5, help="Etiquetar tras N frames procesados")
    parser.add_argument("--describe", action=argparse.BooleanOptionalAction, default=True,
                        help="Generar descripciones con el VLM; desactivarlo es la palanca de tiempo real")
    parser.add_argument("--recognition-interval", type=int, default=15)
    parser.add_argument("--detector-interval", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=None, help="Por defecto, el calibrado del backend")
    parser.add_argument("--out", type=Path, default=Path("data/lab/stream.json"))
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
