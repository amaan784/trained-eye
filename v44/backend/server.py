"""Backend server: receives camera frames from the iPhone app over a WebSocket.

Protocol (version 1), one WebSocket per watch session at /ws:

  phone -> server  text    {"type": "hello", "client": "ios", "source": "glasses", "fps": 1}
  server -> phone  text    {"type": "welcome", "session": "s1", "protocol": 1}
  phone -> server  binary  one JPEG frame per message
  server -> phone  text    {"type": "ack", "seq": 1, "bytes": 51234, "width": 1280, "height": 720}
                           or {"type": "error", "seq": 1, "error": "..."} for a bad frame
  phone -> server  text    {"type": "bye"}

Later milestones add {"type": "alert", "text": ...} from the server, which the phone
speaks through the glasses. Frames are only held in memory; nothing is written to disk.
Only frame metadata (sequence number, size, dimensions) is logged.
"""

from __future__ import annotations

import io
import itertools
import json
import logging

from PIL import Image, UnidentifiedImageError
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket, WebSocketDisconnect

PROTOCOL_VERSION = 1
MAX_FRAME_BYTES = 8 * 1024 * 1024
MAX_FRAME_PIXELS = 4096 * 4096  # DAT streams at most 720x1280
SOURCES = ("glasses", "mock", "phone")

log = logging.getLogger("trained_eye.server")


def inspect_jpeg(data: bytes) -> tuple[int, int]:
    """Decode a JPEG frame and return (width, height), or raise ValueError."""
    if len(data) > MAX_FRAME_BYTES:
        raise ValueError(f"frame is {len(data)} bytes; the limit is {MAX_FRAME_BYTES}")
    if not data.startswith(b"\xff\xd8"):
        raise ValueError("frame is not a JPEG")
    try:
        with Image.open(io.BytesIO(data)) as img:
            if img.format not in ("JPEG", "MPO"):  # MPO: a JPEG with extra images appended
                raise ValueError("frame is not a JPEG")
            width, height = img.size
            if width * height > MAX_FRAME_PIXELS:
                raise ValueError(f"frame is {width}x{height}; too large")
            img.load()  # catches truncated or corrupt frames
            return width, height
    except ValueError:
        raise
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError, SyntaxError):
        raise ValueError("frame is not a readable JPEG") from None


async def health(request):
    return JSONResponse({"ok": True, "protocol": PROTOCOL_VERSION})


def create_app() -> Starlette:
    sessions = itertools.count(1)

    async def frames(ws: WebSocket) -> None:
        await ws.accept()
        try:
            hello = json.loads(await ws.receive_text())
        except WebSocketDisconnect:
            return
        except (json.JSONDecodeError, KeyError):
            hello = None
        if not (
            isinstance(hello, dict)
            and hello.get("type") == "hello"
            and hello.get("source") in SOURCES
        ):
            await ws.send_json(
                {"type": "error", "error": f"first message must be a hello with a source, "
                 f"one of {list(SOURCES)}"}
            )
            await ws.close(code=1002)
            return

        session = f"s{next(sessions)}"
        log.info("session %s opened: client=%s source=%s", session,
                 hello.get("client"), hello["source"])
        await ws.send_json({"type": "welcome", "session": session, "protocol": PROTOCOL_VERSION})

        seq = 0
        try:
            while True:
                message = await ws.receive()
                if message["type"] == "websocket.disconnect":
                    break
                if message.get("bytes") is not None:
                    seq += 1
                    data = message["bytes"]
                    try:
                        width, height = inspect_jpeg(data)
                    except ValueError as err:
                        await ws.send_json({"type": "error", "seq": seq, "error": str(err)})
                        continue
                    log.info("session %s frame %d: %dx%d, %d bytes",
                             session, seq, width, height, len(data))
                    await ws.send_json(
                        {"type": "ack", "seq": seq, "bytes": len(data),
                         "width": width, "height": height}
                    )
                    continue
                try:
                    msg = json.loads(message.get("text") or "")
                except json.JSONDecodeError:
                    msg = {}
                if not isinstance(msg, dict):
                    msg = {}
                if msg.get("type") == "bye":
                    await ws.close()
                    break
                await ws.send_json({"type": "error", "error": "unknown message"})
        except WebSocketDisconnect:
            pass
        log.info("session %s closed after %d frames", session, seq)

    return Starlette(routes=[Route("/health", health), WebSocketRoute("/ws", frames)])


app = create_app()
