from dataclasses import replace

import pytest
from fastapi.testclient import TestClient
from starlette.requests import Request
from starlette.websockets import WebSocketDisconnect

from odin_vision.server import create_app


@pytest.fixture
def client(settings):
    with TestClient(create_app(settings)) as client:
        yield client


def test_full_http_flow(client, encoded):
    assert client.get("/").status_code == 200
    sid = client.post("/api/v1/sessions").json()["id"]
    result = client.post(f"/api/v1/sessions/{sid}/frames", content=encoded).json()
    tid = result["objects"][0]["id"]
    saved = client.post(f"/api/v1/sessions/{sid}/tracks/{tid}/labels", json={"label": "Rojo"})
    assert saved.status_code == 201
    assert client.post(f"/api/v1/sessions/{sid}/frames", content=encoded).json()["objects"][0]["label"] == "Rojo"
    assert client.get("/api/v1/events?label=Rojo").json()["items"]
    assert client.delete(f"/api/v1/labels/{saved.json()['id']}").status_code == 204
    assert client.delete(f"/api/v1/sessions/{sid}").status_code == 204
    assert client.get(f"/api/v1/sessions/{sid}/result").status_code == 404


@pytest.mark.parametrize("chunk_size", [1, 37, 1000000])
@pytest.mark.parametrize("over_limit", [False, True])
def test_fragmented_image_body_and_exact_byte_limit(settings, encoded, monkeypatch, chunk_size, over_limit):
    async def stream(self):
        yield b""
        for start in range(0, len(encoded), chunk_size):
            yield encoded[start:start+chunk_size]
        yield b""

    with TestClient(create_app(replace(settings, max_bytes=len(encoded)-int(over_limit)))) as client:
        sid = client.post("/api/v1/sessions").json()["id"]
        monkeypatch.setattr(Request, "stream", stream)
        response = client.post(f"/api/v1/sessions/{sid}/frames", content=encoded)
        assert response.status_code == (413 if over_limit else 200)
        if not over_limit:
            assert response.json()["objects"][0]["class"] == "person"


def test_reference_http(client, encoded):
    saved = client.post("/api/v1/labels?label=Test&category=person", content=encoded)
    assert saved.status_code == 201
    assert client.post("/api/v1/labels?label=Test&category=person", content=encoded).json()["examples"] == 1
    sid = client.post("/api/v1/sessions").json()["id"]
    assert client.put(f"/api/v1/sessions/{sid}/reference", content=encoded).json()["active"]
    assert client.delete(f"/api/v1/sessions/{sid}/reference").status_code == 204
    assert client.put(f"/api/v1/sessions/{sid}/location", json={"latitude": 40, "longitude": -3}).status_code == 200


def test_repeated_config_preserves_active_clip(client, encoded):
    sid = client.post("/api/v1/sessions").json()["id"]
    path = f"/api/v1/sessions/{sid}"
    config = {"record_labels": ["Rojo"]}
    assert client.put(f"{path}/config", json=config).status_code == 200
    first = client.post(f"{path}/frames", content=encoded).json()
    tid = first["objects"][0]["id"]
    assert client.post(f"{path}/tracks/{tid}/labels", json={"label": "Rojo"}).status_code == 201
    before = client.post(f"{path}/frames", content=encoded).json()
    assert client.put(f"{path}/config", json=config).status_code == 200
    assert client.get("/api/v1/clips").json()["items"] == []
    after = client.post(f"{path}/frames", content=encoded).json()
    assert after["objects"][0]["id"] == tid
    assert after["frame"] == before["frame"] + 1
    assert client.delete(path).status_code == 204
    clips = client.get("/api/v1/clips").json()["items"]
    assert len(clips) == 1
    assert [frame["frame"] for frame in clips[0]["frames"]] == [before["frame"], after["frame"]]


def test_auth_http_and_websocket(settings, encoded):
    with TestClient(create_app(replace(settings, token="secret"))) as client:
        assert client.get("/api/v1/labels").status_code == 401
        assert client.get("/api/v1/labels", headers={"Authorization": "Bearer wrong"}).status_code == 401
        headers = {"Authorization": "Bearer secret"}
        sid = client.post("/api/v1/sessions", headers=headers).json()["id"]
        with client.websocket_connect(f"/api/v1/sessions/{sid}/stream") as ws:
            ws.send_text("wrong")
            with pytest.raises(WebSocketDisconnect):
                ws.receive_json()
        with client.websocket_connect(f"/api/v1/sessions/{sid}/stream") as ws:
            ws.send_text("secret")
            assert ws.receive_json()["ready"]
            ws.send_bytes(encoded)
            assert ws.receive_json()["objects"]
            ws.send_bytes(b"bad")
            assert ws.receive_json()["error"]["code"] == "frame_rejected"
            ws.send_bytes(encoded)
            assert ws.receive_json()["frame"] == 2


@pytest.mark.parametrize("path,body", [
    ("config", {"threshold": 2}), ("config", {"detect": [""]}),
    ("config", {"unknown": True}), ("commands", {"text": ""}),
    ("location", {"latitude": 91, "longitude": 0}),
])
def test_invalid_requests(client, path, body):
    sid = client.post("/api/v1/sessions").json()["id"]
    method = client.post if path == "commands" else client.put
    response = method(f"/api/v1/sessions/{sid}/{path}", json=body)
    assert response.status_code == 422
    assert "error" in response.json()


def test_limits_and_missing(client, encoded):
    assert client.post("/api/v1/sessions/nope/frames", content=encoded).status_code == 404
    assert client.get("/api/v1/events?limit=9999").status_code == 422
    sid = client.post("/api/v1/sessions").json()["id"]
    assert client.post(f"/api/v1/sessions/{sid}/frames", content=b"a"*2_000_001).status_code == 413
    assert client.put(f"/api/v1/sessions/{sid}/config", content=b" "*120001,
                      headers={"Content-Type": "application/json"}).status_code == 413
    assert client.get("/api/v1/clips/not-an-id").status_code == 404
    assert client.get("/openapi.json").status_code == 200


def test_cross_origin_rejected(client):
    origin = {"Origin": "https://unrelated.example"}
    assert client.post("/api/v1/sessions", headers=origin).status_code == 403
    sid = client.post("/api/v1/sessions").json()["id"]
    with pytest.raises(WebSocketDisconnect), client.websocket_connect(
            f"/api/v1/sessions/{sid}/stream", headers=origin):
        pass
    assert client.post("/api/v1/sessions", headers={"Origin": "http://testserver"}).status_code == 201
