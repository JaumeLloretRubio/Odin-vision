"""Cliente sin modelos: cámara/archivo/RTSP configurado localmente -> JPEG HTTP."""
import argparse
import json
import os
import sys
import time


def parser():
    result = argparse.ArgumentParser(description="Captura vídeo y envía frames a Odin Vision; resultados JSONL")
    result.add_argument("--server", default="http://127.0.0.1:8000")
    result.add_argument("--source", default="0", help="Índice de cámara, archivo o URL de vídeo accesible al cliente")
    result.add_argument("--fps", type=float, default=5, help="Máximo de frames enviados por segundo")
    result.add_argument("--width", type=int, default=960)
    result.add_argument("--classes", default="person,car,backpack,drone")
    result.add_argument("--max-frames", type=int, default=0, help="Finalizar tras N frames; 0 hasta fin/Ctrl+C")
    return result


def transmit(capture, client, sid, fps, width, max_frames=0, live=True):
    import cv2
    sent, due = 0, 0.0
    while capture.isOpened():
        ok, frame = capture.read()
        if not ok:
            break
        now = time.monotonic()
        if now < due:
            if live:
                continue  # Drenar la captura para no acumular vídeo atrasado.
            time.sleep(due-now)
            now = time.monotonic()
        due = now + 1/fps
        height, original_width = frame.shape[:2]
        factor = min(1.0, width/original_width, 1080/height)
        if factor < 1:
            frame = cv2.resize(frame, (max(1, round(original_width*factor)), max(1, round(height*factor))))
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if not ok:
            raise RuntimeError("No se pudo codificar el frame")
        response = client.post(f"/api/v1/sessions/{sid}/frames", content=encoded.tobytes(),
                               headers={"Content-Type": "image/jpeg"})
        if response.status_code == 429:
            continue
        response.raise_for_status()
        print(json.dumps(response.json(), ensure_ascii=False), flush=True)
        sent += 1
        if max_frames and sent >= max_frames:
            break
    return sent


def auth_headers():
    # Sin token no se envía cabecera: "Bearer " con espacio final es un valor de cabecera ilegal.
    token = os.getenv("ODIN_TOKEN", "").strip()
    return {"Authorization": f"Bearer {token}"} if token else {}


def main():
    arg_parser = parser()
    args = arg_parser.parse_args()
    if not 0 < args.fps <= 60 or not 16 <= args.width <= 1920 or args.max_frames < 0:
        arg_parser.error("FPS debe estar entre 0 y 60, ancho entre 16 y 1920 y max-frames no puede ser negativo")
    try:
        import cv2
        import httpx
    except ImportError:
        arg_parser.exit(2, "Instale odin-vision[client] para usar el cliente de cámara\n")
    source = int(args.source) if args.source.isdecimal() else args.source
    capture = cv2.VideoCapture(source)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not capture.isOpened():
        capture.release()
        arg_parser.exit(1, "No se pudo abrir la fuente de vídeo\n")
    sid = None
    try:
        with httpx.Client(base_url=args.server.rstrip("/"), timeout=60, headers=auth_headers()) as client:
            try:
                response = client.post("/api/v1/sessions")
                response.raise_for_status()
                sid = response.json()["id"]
                print(f"Sesión: {sid}", file=sys.stderr)
                response = client.put(f"/api/v1/sessions/{sid}/config", json={
                    "detect": [c.strip() for c in args.classes.split(",") if c.strip()]})
                response.raise_for_status()
                transmit(capture, client, sid, args.fps, args.width, args.max_frames,
                         live=isinstance(source, int) or "://" in str(source))
            finally:
                if sid:
                    try:
                        client.delete(f"/api/v1/sessions/{sid}")
                    except httpx.HTTPError:
                        print("No se pudo cerrar la sesión; caducará por inactividad", file=sys.stderr)
    except KeyboardInterrupt:
        pass
    except (httpx.HTTPError, RuntimeError) as exc:
        arg_parser.exit(1, f"Captura detenida: {type(exc).__name__}\n")
    finally:
        capture.release()


if __name__ == "__main__":
    main()
