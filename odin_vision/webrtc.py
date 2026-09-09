import asyncio
import json

from aiortc import RTCConfiguration, RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import MediaStreamError
from fastapi import HTTPException


class Receiver:
    def __init__(self, sid, engine, run, settings):
        self.sid, self.engine, self.run, self.settings = sid, engine, run, settings
        self.pc = RTCPeerConnection(RTCConfiguration(iceServers=[]))
        self.tasks = set()
        self.channel = None
        self.latest = asyncio.Queue(maxsize=1)
        self.video_started = False

        @self.pc.on("datachannel")
        def channel(dc):
            self.channel = dc

        @self.pc.on("track")
        def track_received(track):
            if track.kind != "video" or self.video_started:
                track.stop()
                return
            self.video_started = True
            self.start(self.receive(track))
            self.start(self.process())

        @self.pc.on("connectionstatechange")
        async def state_changed():
            if self.pc.connectionState in {"failed", "closed"}:
                for task in list(self.tasks):
                    task.cancel()

    def start(self, coroutine):
        task = asyncio.create_task(coroutine)
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)

    async def receive(self, track):
        try:
            while True:
                frame = await track.recv()
                if frame.width * frame.height > self.settings.max_pixels:
                    continue
                if self.latest.full():
                    self.latest.get_nowait()
                self.latest.put_nowait(frame)
        except MediaStreamError:
            if self.latest.full():
                self.latest.get_nowait()
            self.latest.put_nowait(None)

    async def process(self):
        while True:
            frame = await self.latest.get()
            if frame is None:
                return
            try:
                result = await self.run(lambda f=frame: self.engine.process(self.sid, f.to_image()))
                if self.channel and self.channel.readyState == "open" and self.channel.bufferedAmount < 65536:
                    self.channel.send(json.dumps(result))
            except HTTPException:
                continue  # Descartar frame al estar ocupado; la cola conserva el más reciente.
            except KeyError:
                return

    async def offer(self, sdp, kind):
        await self.pc.setRemoteDescription(RTCSessionDescription(sdp=sdp, type=kind))
        await self.pc.setLocalDescription(await self.pc.createAnswer())
        return {"sdp": self.pc.localDescription.sdp, "type": self.pc.localDescription.type}

    async def close(self):
        tasks = list(self.tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await self.pc.close()
