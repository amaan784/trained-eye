"""Teaching from the phone: HTTP routes that backend/server.py mounts next to /ws.

  POST /teach/start   multipart/form-data: skill, audio (a file) or request (text),
                      frames (JPEG files of the 10 second capture, in order), optional
                      positives and negatives (default 150 each), audio_s (seconds of a
                      non-WAV recording, for the estimate; else 10)
                      200 the status below, state waiting_spend, nothing spent yet
                      400 bad input, 409 another session runs, 413 too large,
                      422 refused (policy, own labels needed, or over ASK_ABOVE_USD),
                      503 MODEL_API_KEY missing; each as {"error": "...", ...}
  GET  /teach/status?skill=burner_left_on   (without skill: the latest; 404 if none)
                      {"skill", "state", "question": "spend" | "read_back" | null, "lines",
                       "estimate", "warning", "capture", "read_back", "stopped", "error",
                       "summary", "spent_usd"}
  POST /teach/answer  JSON {"skill", "step": "spend" | "read_back", "yes": true | false,
                       "correction": "..." (only with read_back and yes false)}
                      200 the status, 400 bad answer, 404 no session, 409 not asked now

Every route needs the header "Authorization: Bearer <SERVER_TOKEN>" (the secret in .env,
entered once in the app's Connect screen): these routes spend money, so a phone or anyone
else on the network without it gets 401, and without SERVER_TOKEN set they answer 503.
One session at a time, kept in memory; states in teach_session.py, checks in
teach_upload.py. teach_api() also returns stop(), which the server calls when it shuts
down, so a paid call in flight is still recorded.
"""

from __future__ import annotations

import hmac
import json
import threading
from collections.abc import Callable

from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from backend.config import ConfigError, Settings
from backend.meta.client import MetaClient
from backend.teach_session import ANSWER_TIMEOUT_S, NotWaiting, TeachSession
from backend.teach_upload import Problem, new_session, parse_multipart

MAX_UPLOAD_BYTES = 200 * 1024 * 1024
MAX_ANSWER_BYTES = 64 * 1024
TOKEN = "SERVER_TOKEN"


def error(status: int, message: str, **extra) -> JSONResponse:
    return JSONResponse({"error": message, **extra}, status_code=status)


async def read_body(request: Request, limit: int = MAX_UPLOAD_BYTES,
                    what: str = "the capture") -> bytes:
    chunks, size = [], 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > limit:
            size_text = (f"{limit // 2**20} MB" if limit >= 2**20 else f"{limit // 1024} KB")
            raise Problem(413, f"{what} is over {size_text}")
        chunks.append(chunk)
    return b"".join(chunks)


def refused(request: Request, settings_factory: Callable[[], Settings]) -> JSONResponse | None:
    """None if the request carries the server token, else the answer to send."""
    try:
        settings = settings_factory()
    except ConfigError as err:
        return error(503, str(err))
    if not settings.has_secret(TOKEN):
        return error(503, f"teaching from the phone is off: set {TOKEN} in .env on the Mac "
                     "and the same token in the app's Connect screen")
    try:
        expected = f"Bearer {settings.secret(TOKEN)}".encode()
    except ConfigError as err:  # e.g. a space or smart quotes in the token
        return error(503, str(err))
    given = request.headers.get("authorization", "").encode()
    if not hmac.compare_digest(given, expected):
        return error(401, f"wrong or missing token: enter {TOKEN} from the Mac's .env in the "
                     "app's Connect screen")
    return None


def teach_routes(settings_factory: Callable[[], Settings],
                 client_factory: Callable[[Settings], MetaClient], *,
                 timeout_s: float = ANSWER_TIMEOUT_S) -> list[Route]:
    """The routes of teach_api() without its stop()."""
    return teach_api(settings_factory, client_factory, timeout_s=timeout_s)[0]


def teach_api(settings_factory: Callable[[], Settings],
              client_factory: Callable[[Settings], MetaClient], *,
              timeout_s: float = ANSWER_TIMEOUT_S) -> tuple[list[Route], Callable[[], None]]:
    """POST /teach/start, GET /teach/status, POST /teach/answer, and stop() for shutdown.
    `settings_factory()` is read per request; `client_factory(settings)` makes the
    session's MetaClient."""
    sessions: dict[str, TeachSession] = {}  # in start order: the last is the latest
    starting = threading.Lock()  # held from the upload to the thread start

    def prepare(body: bytes, content_type: str) -> TeachSession:
        return new_session(parse_multipart(body, content_type), settings_factory(),
                           client_factory, timeout_s)

    async def start(request: Request):
        if (no := refused(request, settings_factory)) is not None:
            return no
        if not starting.acquire(blocking=False):
            return error(409, "another teach session is starting")
        try:
            busy = next((s for s in sessions.values() if not s.finished), None)
            if busy:
                return error(409, f"a teach session for {busy.skill} is running "
                             f"({busy.state}): answer it or wait for it to end")
            body = await read_body(request)
            session = await run_in_threadpool(prepare, body,
                                              request.headers.get("content-type", ""))
            sessions.pop(session.skill, None)
            sessions[session.skill] = session.start()
            return JSONResponse(session.status())
        except Problem as err:
            return error(err.status, str(err), **err.extra)
        except ConfigError as err:  # a missing key or a bad .env; never the key itself
            return error(503, str(err))
        except ValueError as err:
            return error(400, str(err))
        finally:
            starting.release()

    async def status(request: Request):
        if (no := refused(request, settings_factory)) is not None:
            return no
        skill = request.query_params.get("skill")
        session = sessions.get(skill) if skill else next(reversed(sessions.values()), None)
        if session is None:
            return error(404, f"no teach session for {skill or 'any skill'} since the server "
                         "started")
        return JSONResponse(session.status())

    async def answer(request: Request):
        if (no := refused(request, settings_factory)) is not None:
            return no
        try:
            body = json.loads(await read_body(request, MAX_ANSWER_BYTES, "the answer"))
        except Problem as err:
            return error(err.status, str(err))
        except (ValueError, RecursionError):
            body = None
        if not isinstance(body, dict) or not isinstance(body.get("yes"), bool):
            return error(400, 'send JSON like {"skill": "burner_left_on", "step": "spend", '
                         '"yes": true}')
        skill, correction = body.get("skill"), body.get("correction")
        session = sessions.get(skill) if isinstance(skill, str) else None
        if session is None:
            return error(404, f"no teach session for {skill!r}")
        try:
            if correction is not None and not isinstance(correction, str):
                raise ValueError("correction must be text")
            session.answer(body.get("step"), body["yes"], correction)
        except NotWaiting as err:
            return error(409, str(err))
        except ValueError as err:
            return error(400, str(err))
        return JSONResponse(session.status())

    def stop() -> None:
        for session in list(sessions.values()):
            if not session.finished:
                session.stop()

    return [Route("/teach/start", start, methods=["POST"]),
            Route("/teach/status", status, methods=["GET"]),
            Route("/teach/answer", answer, methods=["POST"])], stop
