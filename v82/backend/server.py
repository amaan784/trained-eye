"""Backend server: receives camera frames from the iPhone app over a WebSocket.

Protocol (version 1), one WebSocket per watch session at /ws:

  phone -> server  text    {"type": "hello", "client": "ios", "source": "glasses", "fps": 1}
  server -> phone  text    {"type": "welcome", "session": "s1", "protocol": 1}
  phone -> server  binary  one JPEG frame per message
  server -> phone  text    {"type": "ack", "seq": 1, "bytes": 51234, "width": 1280, "height": 720}
                           or {"type": "error", "seq": 1, "error": "..."} for a bad frame
  phone -> server  text    {"type": "bye"}

Teaching from the phone (backend/teach_api.py): POST /teach/start with the 10 second
capture, GET /teach/status, POST /teach/answer for the spend and read-back questions,
each with the SERVER_TOKEN from .env. On shutdown a teach run stops before its next paid
call and the call in flight is recorded.

GET /skills lists the taught skills for the app's Skills screen:
{"skills": [{"name", "alert", "cooldown_s", "dataset", "calibrated", "threshold"}]}.

Watch mode: a hello with "watch": ["washer_done", ...] scores every frame for those taught
skills (backend/watch/watcher.py). The welcome then lists them under "watching", each ack
carries "scores" ({skill: probability, or null if the student failed}), and an alert is

  server -> phone  text    {"type": "alert", "skill": "washer_done", "text": "...",
                            "t": ..., "p": 0.93, "confirmed": true}

which the phone speaks through the glasses. Frames are only held in memory; nothing is
written to disk. Only frame metadata (sequence number, size, dimensions) is logged here,
and the watcher logs only scores and decisions.
"""

from __future__ import annotations

import contextlib
import io
import itertools
import json
import logging
import math
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from PIL import Image, UnidentifiedImageError
from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.responses import JSONResponse
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket, WebSocketDisconnect

from backend.watch.watcher import alert_message

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


def taught_skills(data_dir: Path) -> list[dict]:
    """Every taught, not refused skill under data_dir/skills, for the app."""
    from backend.skills.spec import SkillSpec

    out = []
    for spec_file in sorted((Path(data_dir) / "skills").glob("*/spec.json")):
        try:
            spec = SkillSpec.model_validate_json(spec_file.read_text())
        except (OSError, ValueError):
            continue  # a folder, a broken link, unreadable or from an older version
        if spec.refused:
            continue
        folder = spec_file.parent
        threshold_file = folder / "threshold.json"
        threshold = None
        if threshold_file.is_file():
            try:
                threshold = float(json.loads(threshold_file.read_text())["threshold"])
            except (OSError, ValueError, KeyError, TypeError):
                threshold = None
            if threshold is not None and not math.isfinite(threshold):
                threshold = None  # Infinity or NaN cannot go into JSON
        out.append({"name": spec.name, "alert": spec.alert, "cooldown_s": spec.cooldown_s,
                    "dataset": (folder / "dataset" / f"{spec.name}-teach.jsonl").is_file(),
                    "calibrated": threshold is not None, "threshold": threshold})
    return out


def default_data_dir() -> Path:
    from backend.config import load_settings

    return load_settings().data_dir


def default_watcher(names: list[str], session: str):
    """A Watcher for the taught skills `names`, from the settings: the student endpoint
    (TE_STUDENT_URL), Muse Spark for the cloud check, a frame log under data/watch/."""
    from backend.config import load_settings
    from backend.dataengine.labels import label_image
    from backend.inference.student import OpenAIStudent
    from backend.meta.client import MetaClient
    from backend.skills.spec import check_name
    from backend.watch.watcher import FrameLog, Watcher, load_skill

    settings = load_settings()
    if not settings.student_url:
        raise ValueError("watch mode needs the trained model's endpoint: set TE_STUDENT_URL")
    skills = []
    for name in names:
        folder = settings.data_dir / "skills" / check_name(name)
        if not (folder / "spec.json").is_file():
            raise ValueError(f"no taught skill named {name!r}")
        skills.append(load_skill(folder))
    client = MetaClient(settings)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    student = OpenAIStudent(settings.student_url, settings.student_model,
                            api_key=settings.secrets.get("STUDENT_API_KEY"))
    return Watcher(skills, student,
                   confirm=lambda image, spec: label_image(client, spec, image),
                   log=FrameLog(settings.data_dir / "watch" / f"{stamp}-{session}.jsonl"))


def create_app(watcher_factory: Callable[[list[str], str], object] | None = None, *,
               data_dir: Callable[[], Path] = default_data_dir,
               extra_routes: Sequence = (), on_shutdown: Sequence[Callable[[], None]] = ()
               ) -> Starlette:
    """`watcher_factory(skill names, session)` builds the Watcher for a session that asks
    to watch; without one, watch requests get an error and frames are only acked.
    `data_dir()` gives the folder GET /skills reads; `extra_routes` are mounted as well
    (the teach API), and each of `on_shutdown` runs, in a thread, when the server stops."""
    sessions = itertools.count(1)

    @contextlib.asynccontextmanager
    async def lifespan(app):
        yield
        for stop in on_shutdown:
            await run_in_threadpool(stop)

    async def skills(request):
        return JSONResponse({"skills": await run_in_threadpool(taught_skills, data_dir())})

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
        welcome = {"type": "welcome", "session": session, "protocol": PROTOCOL_VERSION}
        watcher, wanted = None, hello.get("watch")
        if wanted is not None:
            try:
                if not (isinstance(wanted, list) and wanted
                        and all(isinstance(n, str) for n in wanted)):
                    raise ValueError('"watch" must be a list of skill names')
                if watcher_factory is None:
                    raise ValueError("watch mode is not set up on this server")
                watcher = await run_in_threadpool(watcher_factory, wanted, session)
            except Exception as err:  # noqa: BLE001 - tell the phone, keep the session
                await ws.send_json({"type": "error", "error": f"cannot watch: {err}"})
            else:
                welcome["watching"] = sorted(watcher.skills)
                log.info("session %s watching %s", session, ", ".join(welcome["watching"]))
        await ws.send_json(welcome)

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
                    ack = {"type": "ack", "seq": seq, "bytes": len(data),
                           "width": width, "height": height}
                    alerts = []
                    if watcher is not None:
                        ack["scores"], alerts = await run_in_threadpool(watcher.on_frame, data)
                    await ws.send_json(ack)
                    for alert in alerts:
                        log.info("session %s alert %s (confirmed: %s)", session, alert.skill,
                                 alert.confirmed)
                        await ws.send_json(alert_message(alert))
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

    return Starlette(routes=[Route("/health", health), Route("/skills", skills),
                             WebSocketRoute("/ws", frames), *extra_routes], lifespan=lifespan)


def default_teach_api() -> tuple[list, Callable[[], None]]:
    """The phone's teach API and its stop(), with the settings and Meta client from .env."""
    from backend.config import load_settings
    from backend.meta.client import MetaClient
    from backend.teach_api import teach_api

    return teach_api(load_settings, MetaClient)


_teach_routes, _teach_stop = default_teach_api()
app = create_app(default_watcher, extra_routes=_teach_routes, on_shutdown=[_teach_stop])
