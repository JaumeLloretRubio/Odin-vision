"""API monousuario: un token compartido y memoria común; sesiones separan streams."""
import asyncio
import hmac
import json
import logging
import re
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import Settings
from .engine import Engine, decode_image
from .schemas import Assignment, Command, FilterConfig, Location, Offer

logger = logging.getLogger(__name__)


def create_app(settings=None, vision=None):
    settings = settings or Settings.from_env()
    lock = asyncio.Lock()
    peers = {}
    media_lock = asyncio.Lock()

    @asynccontextmanager
    async def lifespan(app):
        app.state.engine = await asyncio.to_thread(Engine, settings, vision)
        try:
            yield
        finally:
            await asyncio.gather(*(peer.close() for peer in list(peers.values())), return_exceptions=True)
            async with lock:
                await asyncio.to_thread(app.state.engine.close)

    app = FastAPI(title="Odin Vision", version="0.1.0", lifespan=lifespan)

    async def run(fn, *args, **kwargs):
        # Evita colas de inferencia ilimitadas y serializa modelos/SQLite.
        if lock.locked():
            raise HTTPException(429, "Servidor ocupado; vuelva a intentarlo")
        async with lock:
            task = asyncio.create_task(asyncio.to_thread(partial(fn, *args, **kwargs)))
            try:
                return await asyncio.shield(task)
            except asyncio.CancelledError:
                await task
                raise

    def engine():
        return app.state.engine

    def authorized(token):
        return not settings.token or hmac.compare_digest(token.encode("utf-8"), settings.token.encode("utf-8"))

    def same_origin(origin, host):
        if origin is None:
            return True  # Clientes nativos no envían Origin.
        try:
            parsed = urlsplit(origin)
        except ValueError:
            return False
        return parsed.scheme in {"http", "https"} and parsed.netloc == host

    def error(status, message, code=None):
        return JSONResponse({"error": {"code": code or f"http_{status}", "message": message}}, status_code=status)

    @app.middleware("http")
    async def auth(request: Request, call_next):
        if request.url.path.startswith("/api/"):
            if not same_origin(request.headers.get("origin"), request.headers.get("host")):
                return error(403, "Origen no permitido", "forbidden_origin")
            token = request.headers.get("authorization", "").removeprefix("Bearer ")
            if not authorized(token):
                return error(401, "Token requerido o incorrecto", "unauthorized")
            if request.headers.get("content-type", "").startswith("application/json"):
                chunks, size = [], 0
                async for chunk in request.stream():
                    size += len(chunk)
                    if size > 120000:
                        return error(413, "Petición JSON demasiado grande")
                    chunks.append(chunk)
                request._body = b"".join(chunks)
        return await call_next(request)

    @app.exception_handler(HTTPException)
    async def http_error(request, exc):
        return error(exc.status_code, str(exc.detail))

    @app.exception_handler(ValueError)
    async def value_error(request, exc):
        return error(422, str(exc), "invalid_input")

    @app.exception_handler(KeyError)
    async def missing_error(request, exc):
        return error(404, str(exc.args[0]), "not_found")

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        return error(422, "Petición inválida; consulte /docs", "invalid_request")

    @app.exception_handler(Exception)
    async def unexpected_error(request, exc):
        logger.error("Error interno", exc_info=exc)
        return error(500, "Error interno del servidor", "internal_error")

    async def image_body(request):
        data = bytearray()
        async for chunk in request.stream():
            data.extend(chunk)
            if len(data) > settings.max_bytes:
                raise HTTPException(413, "Imagen demasiado grande")
        return await run(decode_image, bytes(data), settings)

    @app.get("/api/v1/status")
    async def status():
        return {"backend": settings.backend, "encoder": engine().vision.fingerprint,
                "sessions": len(engine().sessions), "max_pixels": settings.max_pixels,
                "max_bytes": settings.max_bytes}

    @app.post("/api/v1/sessions", status_code=201)
    async def create_session():
        session = await run(engine().create)
        for sid in list(peers):
            if sid not in engine().sessions:
                await peers.pop(sid).close()
        return {"id": session.id, "config": session.config.model_dump()}

    @app.delete("/api/v1/sessions/{sid}", status_code=204)
    async def delete_session(sid: str):
        await run(engine().session, sid)
        if sid in peers:
            await peers.pop(sid).close()
        await run(engine().delete, sid)

    @app.put("/api/v1/sessions/{sid}/config")
    async def configure(sid: str, config: FilterConfig):
        return await run(engine().configure, sid, config)

    @app.put("/api/v1/sessions/{sid}/location")
    async def location(sid: str, body: Location):
        def update():
            engine().session(sid).location = body.model_dump()
        await run(update)
        return body

    @app.post("/api/v1/sessions/{sid}/frames")
    async def frame(sid: str, request: Request):
        image = await image_body(request)
        return await run(engine().process, sid, image)

    @app.get("/api/v1/sessions/{sid}/result")
    async def result(sid: str):
        return await run(lambda: engine().session(sid).last)

    @app.post("/api/v1/sessions/{sid}/tracks/{tid}/labels", status_code=201)
    async def assign(sid: str, tid: int, assignment: Assignment):
        return await run(engine().assign, sid, tid, assignment.label)

    @app.post("/api/v1/sessions/{sid}/commands")
    async def command(sid: str, body: Command):
        return await run(engine().command, sid, body.text, body.track_id)

    @app.put("/api/v1/sessions/{sid}/reference")
    async def reference(sid: str, request: Request):
        image = await image_body(request)
        def set_reference():
            session = engine().session(sid)
            session.reference = engine().vision.embed(image)
        await run(set_reference)
        return {"active": True}

    @app.delete("/api/v1/sessions/{sid}/reference", status_code=204)
    async def delete_reference(sid: str):
        await run(lambda: setattr(engine().session(sid), "reference", None))

    @app.post("/api/v1/labels", status_code=201)
    async def learn(request: Request, label: str = Query(min_length=1, max_length=100),
                    category: str = Query(min_length=1, max_length=100)):
        assignment = Assignment(label=label, category=category)
        image = await image_body(request)
        return await run(lambda: engine().memory.learn(assignment.label, assignment.category,
                                                       [engine().vision.embed(image)]))

    @app.get("/api/v1/labels")
    async def labels(after: int = Query(0, ge=0), limit: int = Query(100, ge=1, le=200)):
        items = await run(engine().memory.labels, after, limit)
        return {"items": items, "next_cursor": items[-1]["id"] if len(items) == limit else None}

    @app.delete("/api/v1/labels/{label_id}", status_code=204)
    async def delete_label(label_id: int):
        if not await run(engine().memory.delete, label_id):
            raise HTTPException(404, "Etiqueta inexistente")

    @app.get("/api/v1/events")
    async def events(after: int = Query(0, ge=0), limit: int = Query(100, ge=1, le=200),
                     label: str | None = Query(None, max_length=100), since: float = Query(0, ge=0),
                     kind: str | None = Query(None, max_length=30)):
        items = await run(engine().memory.events, after, limit, label, since, kind)
        return {"items": items, "next_cursor": items[-1]["id"] if len(items) == limit else None}

    @app.get("/api/v1/clips")
    async def clips():
        def listing():
            directory = settings.data_dir / "clips"
            return [json.loads(p.read_text(encoding="utf-8")) for p in
                    sorted(directory.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:100]]
        return {"items": await run(listing)}

    @app.get("/api/v1/clips/{clip_id}")
    async def clip(clip_id: str):
        if not re.fullmatch(r"[a-f0-9]{32}", clip_id):
            raise HTTPException(404, "Clip inexistente")
        path = settings.data_dir / "clips" / f"{clip_id}.avi"
        if not path.with_suffix(".json").exists() or not path.exists():
            raise HTTPException(404, "Clip inexistente")
        return FileResponse(path, media_type="video/x-msvideo", filename=path.name)

    @app.websocket("/api/v1/sessions/{sid}/stream")
    async def stream(ws: WebSocket, sid: str):
        if not same_origin(ws.headers.get("origin"), ws.headers.get("host")):
            await ws.close(code=1008)
            return
        await ws.accept()
        try:
            token = await asyncio.wait_for(ws.receive_text(), 10)
            if not authorized(token):
                await ws.close(code=1008)
                return
            await run(engine().session, sid)
            await ws.send_json({"ready": True})
            while True:
                data = await asyncio.wait_for(ws.receive_bytes(), settings.session_ttl)
                try:
                    def process(data=data):
                        return engine().process(sid, decode_image(data, settings))
                    await ws.send_json(await run(process))
                except (ValueError, HTTPException) as exc:
                    await ws.send_json({"error": {"code": "frame_rejected", "message": str(exc)}})
        except (WebSocketDisconnect, asyncio.TimeoutError):
            pass
        except KeyError:
            await ws.close(code=1008)

    @app.post("/api/v1/sessions/{sid}/offer")
    async def offer(sid: str, body: Offer):
        await run(engine().session, sid)
        try:
            from .webrtc import Receiver
        except ImportError as exc:
            raise HTTPException(503, "Instale odin-vision[webrtc]") from exc
        if media_lock.locked():
            raise HTTPException(429, "Negociación WebRTC ocupada")
        async with media_lock:
            if sid in peers:
                await peers.pop(sid).close()
            receiver = Receiver(sid, engine(), run, settings)
            peers[sid] = receiver
            try:
                return await asyncio.wait_for(receiver.offer(body.sdp, body.type), 20)
            except asyncio.CancelledError:
                peers.pop(sid, None)
                await receiver.close()
                raise
            except Exception:  # noqa: BLE001 - cualquier fallo de negociación se traduce a 422
                peers.pop(sid, None)
                await receiver.close()
                raise HTTPException(422, "No se pudo negociar el stream WebRTC") from None

    static = Path(__file__).parent / "static"
    app.mount("/static", StaticFiles(directory=static), name="static")

    @app.get("/")
    async def index():
        return FileResponse(static / "index.html")

    return app
