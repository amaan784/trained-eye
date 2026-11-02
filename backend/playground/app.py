"""The web playground (M4): one page and a JSON API over the local results.

  GET  /                     the page; sets the visitor cookie
  GET  /static/...           its scripts and style (static/)
  GET  /api/status           whether live teaching is on, the caps, what is left today
  GET  /api/clips            recorded clips; /api/clips/{name}: one clip's timeline
  GET  /api/skills           taught skills; /api/skills/{name}: how one was built
  GET  /api/examples         precomputed "teach it yourself" results
  GET  /api/teach/keyframes  the keyframes a visitor may pick
  POST /api/teach            {"condition": "...", "keyframe": "<id>"}: starts a live run
                             (JSON from this page only: other sites cannot spend for a
                             visitor)
  GET  /api/teach/{job}      the run so far; ?since=n leaves out the first n variations
  GET  /api/results          the evaluation's SVG charts and results.md
  GET  /media/...            the files those answers link to, and nothing else

Run one worker (the budget file belongs to one process). Behind a proxy, let uvicorn
read the forwarded address (--proxy-headers), or every visitor shares the proxy's.
"""

from __future__ import annotations

import json
import re
import secrets
from collections.abc import Callable
from pathlib import Path

from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from backend.meta.client import MetaClient
from backend.playground import data, skills
from backend.playground.api import NO_KEY, WORKERS, Teacher
from backend.playground.budget import Budget
from backend.playground.runs import estimate_usd

STATIC = Path(__file__).parent / "static"
COOKIE = "te_visitor"
VISITOR_ID = re.compile(r"[A-Za-z0-9_-]{16,64}")
COOKIE_MAX_AGE = 30 * 86_400
MAX_BODY = 4096
PAGE_CSP = ("default-src 'self'; img-src 'self' data:; media-src 'self'; object-src 'none'; "
            "base-uri 'none'; form-action 'self'; frame-ancestors 'none'")
SAME_SITE = ("same-origin", "none")  # Sec-Fetch-Site of this page's own requests
SVG_CSP = "default-src 'none'; style-src 'unsafe-inline'; img-src data:"  # a chart runs no code


async def limited_body(request: Request, limit: int = MAX_BODY) -> bytes | None:
    """The body, or None once it is over `limit` bytes (read no further)."""
    length = request.headers.get("content-length", "")
    if length.isdecimal() and int(length) > limit:
        return None
    chunks, size = [], 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > limit:
            return None
        chunks.append(chunk)
    return b"".join(chunks)


def parse_since(raw: str) -> int:
    try:
        return max(0, int(raw))
    except ValueError:
        return 0


def create_app(data_dir: Path, eval_dir: Path, client_factory: Callable[[], MetaClient], *,
               visitor_cap_usd: float = 0.50, daily_cap_usd: float = 10.0,
               workers: int = WORKERS) -> Starlette:
    """`client_factory()` gives a new MetaClient for each live run (and for /api/status)."""
    data_dir, eval_dir = Path(data_dir), Path(eval_dir)
    budget = Budget(data_dir / "playground" / "budget.json", visitor_cap_usd, daily_cap_usd)
    teacher = Teacher(data_dir, client_factory, budget, workers=workers)

    def visitor(request: Request) -> tuple[str, tuple[str, ...], bool]:
        """(cookie id, budget keys, whether the id is new)."""
        raw = request.cookies.get(COOKIE, "")
        new = not VISITOR_ID.fullmatch(raw)
        vid = secrets.token_urlsafe(16) if new else raw
        return vid, budget.keys(vid, request.client.host if request.client else None), new

    def remember(response: Response, request: Request, vid: str, new: bool) -> Response:
        if new:
            response.set_cookie(COOKIE, vid, max_age=COOKIE_MAX_AGE, httponly=True,
                                samesite="lax", secure=request.url.scheme == "https")
        return response

    def found(value) -> JSONResponse:
        return JSONResponse({"error": "not found"}, 404) if value is None else JSONResponse(value)

    def status_of(keys: tuple[str, ...]) -> dict:
        try:
            settings = client_factory().settings
            live, estimate = settings.has_secret("MODEL_API_KEY"), round(estimate_usd(settings), 4)
        except Exception:  # noqa: BLE001 - unreadable settings: not live
            live, estimate = False, None
        out = {"live": live, "estimate_usd": estimate, "visitor_cap_usd": visitor_cap_usd,
               "daily_cap_usd": daily_cap_usd, **budget.left(keys)}
        return out if live else {**out, "message": NO_KEY}

    async def page(request):
        vid, _, new = visitor(request)
        response = FileResponse(STATIC / "index.html", media_type="text/html",
                                headers={"Content-Security-Policy": PAGE_CSP,
                                         "Cache-Control": "no-cache"})
        return remember(response, request, vid, new)

    async def status(request):
        vid, keys, new = visitor(request)
        out = await run_in_threadpool(status_of, keys)
        return remember(JSONResponse(out), request, vid, new)

    async def teach(request):
        vid, keys, new = visitor(request)
        if request.headers.get("sec-fetch-site", "same-origin") not in SAME_SITE:
            return JSONResponse({"error": "Start a run from the playground page."}, 403)
        if request.headers.get("content-type", "").split(";")[0].strip() != "application/json":
            return JSONResponse({"error": "Send JSON (Content-Type: application/json)."}, 415)
        body = await limited_body(request)
        if body is None:
            return JSONResponse({"error": "The request is too large."}, 413)
        try:
            payload = json.loads(body or b"null")
        except (ValueError, RecursionError):  # 3.11 recurses on deep nesting under 4 KB
            payload = None
        if not isinstance(payload, dict):
            return JSONResponse({"error": 'Send JSON: {"condition": "...", "keyframe": "..."}'},
                                400)
        code, out = await run_in_threadpool(teacher.start, payload.get("condition"),
                                            payload.get("keyframe"), keys)
        return remember(JSONResponse(out, code), request, vid, new)

    async def job(request):
        since = parse_since(request.query_params.get("since", "0"))
        return found(teacher.poll(request.path_params["job"], since))

    async def clips(request):
        return JSONResponse({"clips": await run_in_threadpool(data.list_clips, eval_dir)})

    async def clip(request):
        return found(await run_in_threadpool(data.clip_timeline, eval_dir, data_dir,
                                             request.path_params["name"]))

    async def skill_list(request):
        return JSONResponse({"skills": await run_in_threadpool(skills.list_skills, data_dir)})

    async def skill(request):
        return found(await run_in_threadpool(skills.skill_detail, data_dir,
                                             request.path_params["name"]))

    async def examples(request):
        return JSONResponse({"examples": await run_in_threadpool(skills.examples, data_dir)})

    async def keyframes(request):
        offered = await run_in_threadpool(skills.provided_keyframes, data_dir)
        return JSONResponse({"keyframes": offered})

    async def results(request):
        return JSONResponse(await run_in_threadpool(data.results, eval_dir))

    async def media(request):
        hit = data.media_path(data_dir, eval_dir, request.path_params["path"])
        if hit is None:
            return found(None)
        path, kind = hit
        headers = {"X-Content-Type-Options": "nosniff", "Cache-Control": "public, max-age=300"}
        if kind == data.SVG:
            headers["Content-Security-Policy"] = SVG_CSP
        return FileResponse(path, media_type=kind, headers=headers)

    return Starlette(routes=[
        Route("/", page), Route("/api/status", status), Route("/api/clips", clips),
        Route("/api/clips/{name}", clip), Route("/api/skills", skill_list),
        Route("/api/skills/{name}", skill), Route("/api/examples", examples),
        Route("/api/teach", teach, methods=["POST"]),
        Route("/api/teach/keyframes", keyframes), Route("/api/teach/{job}", job),
        Route("/api/results", results), Route("/media/{path:path}", media),
        Mount("/static", StaticFiles(directory=STATIC), name="static"),
    ])


def default_app() -> Starlette:
    """The playground from the settings: data/ as configured, eval/ in the project, the
    default caps. For `uvicorn --factory backend.playground.app:default_app`."""
    from backend.config import PROJECT_ROOT, load_settings

    return create_app(load_settings().data_dir, PROJECT_ROOT / "eval",
                      lambda: MetaClient(load_settings()))
