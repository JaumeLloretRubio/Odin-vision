"""Negociación y vídeo RTP reales de aiortc, sin cámara ni pesos."""
import asyncio
import json

import pytest

pytest.importorskip("aiortc", reason="La prueba RTP requiere aiortc real")
from aiortc import RTCConfiguration, RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from av import VideoFrame

from odin_vision.webrtc import Receiver


def test_webrtc_video_and_metadata(engine, scene, settings):
    async def exercise():
        sid = engine.create().id

        async def run(fn):
            return fn()

        receiver = Receiver(sid, engine, run, settings)
        sender = RTCPeerConnection(RTCConfiguration(iceServers=[]))
        received = asyncio.get_running_loop().create_future()

        class Camera(VideoStreamTrack):
            async def recv(self):
                pts, time_base = await self.next_timestamp()
                frame = VideoFrame.from_image(scene)
                frame.pts, frame.time_base = pts, time_base
                return frame

        channel = sender.createDataChannel("results")

        @channel.on("message")
        def on_message(data):
            result = json.loads(data)
            if result.get("objects") and not received.done():
                received.set_result(result)

        sender.addTrack(Camera())
        try:
            await sender.setLocalDescription(await sender.createOffer())
            answer = await receiver.offer(sender.localDescription.sdp, "offer")
            await sender.setRemoteDescription(RTCSessionDescription(**answer))
            result = await asyncio.wait_for(received, 20)
            assert result["objects"][0]["class"] == "person"
            assert result["width"] == 200
        finally:
            await sender.close()
            await receiver.close()
        assert not receiver.tasks

    asyncio.run(exercise())
