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
  GET  /how-it-works         docs/how-it-works.html, the flow in diagrams (GitHub shows an
                             .html file as source), without its web fonts

The public copy, create_app(public=True) (`te playground --public`, and the Modal app in
modal_app.py), serves only what may be published: no generated image (media/skills/*/
synthetic is a 404, the training wall and the examples come back empty with a note),
keyframes only from te blur folders (skills.py) and clip videos only with a matching
blur record (data.py). /api/status says "public", and the page hides the walls.

Run one worker (the budget file belongs to one process). Behind a proxy, let uvicorn
read the forwarded address (--proxy-headers), or every visitor shares the proxy's; under
a host that runs the app itself (Modal), wrap it in behind_proxy().
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
from starlette.responses import FileResponse, HTMLResponse, JSONResponse, Response
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
# The checkout (parent thrice: the container's /root has too few parents for parents[n]).
DIAGRAMS = Path(__file__).resolve().parent.parent.parent / "docs" / "how-it-works.html"
WEB_FONTS = re.compile(r'<link[^>]*fonts\.(?:googleapis|gstatic)\.com[^>]*>\n?')


def diagrams_page(path: Path = DIAGRAMS) -> str | None:
    """The diagram page as served here: its Google Fonts links left out, so it loads
    nothing from other sites (its fonts fall back to the system's). None when missing."""
    return WEB_FONTS.sub("", path.read_text(encoding="utf-8")) if path.is_file() else None


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


def behind_proxy(app):
    """ASGI middleware for a host whose proxy adds the visitor's address to
    X-Forwarded-For: the last entry becomes request.client (a visitor can write entries
    before it, never after). Without the header the address is left as it is."""
    async def wrapped(scope, receive, send):
        if scope["type"] in ("http", "websocket"):
            raw = b",".join(v for k, v in scope.get("headers", []) if k == b"x-forwarded-for")
            last = raw.decode("latin-1").split(",")[-1].strip() if raw else ""
            if last:
                scope = {**scope, "client": (last, 0)}
        await app(scope, receive, send)
    return wrapped


def create_app(data_dir: Path, eval_dir: Path, client_factory: Callable[[], MetaClient], *,
               visitor_cap_usd: float = 0.50, daily_cap_usd: float = 10.0,
               workers: int = WORKERS, public: bool = False) -> Starlette:
    """`client_factory()` gives a new MetaClient for each live run (and for /api/status).
    public=True: the copy for the open web (see the module docstring)."""
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
        out = {"live": live, "public": public, "estimate_usd": estimate,
               "visitor_cap_usd": visitor_cap_usd, "daily_cap_usd": daily_cap_usd,
               **budget.left(keys)}
        return out if live else {**out, "message": NO_KEY}

    async def page(request):
        vid, _, new = visitor(request)
        response = FileResponse(STATIC / "index.html", media_type="text/html",
                                headers={"Content-Security-Policy": PAGE_CSP,
                                         "Cache-Control": "no-cache"})
        return remember(response, request, vid, new)

    async def how_it_works(request):
        html = diagrams_page()
        if html is None:
            return JSONResponse({"error": "not here"}, status_code=404)
        return HTMLResponse(html, headers={"Content-Security-Policy": SVG_CSP,
                                           "Cache-Control": "no-cache"})

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
        if public and payload.get("keyframe") not in [  # a list: the id may be unhashable
                k["id"] for k in skills.provided_keyframes(data_dir, public=True)]:
            payload = {**payload, "keyframe": None}  # not on offer here: refused as unknown
        code, out = await run_in_threadpool(teacher.start, payload.get("condition"),
                                            payload.get("keyframe"), keys)
        if public and "examples" in out:  # generated images: not in the public copy
            out = {**out, "examples": [], "examples_note": skills.EXAMPLES_NOTE}
        return remember(JSONResponse(out, code), request, vid, new)

    async def job(request):
        since = parse_since(request.query_params.get("since", "0"))
        return found(teacher.poll(request.path_params["job"], since))

    async def clips(request):
        listed = await run_in_threadpool(lambda: data.list_clips(data_dir, public=public))
        return JSONResponse({"clips": listed})

    async def clip(request):
        name = request.path_params["name"]
        return found(await run_in_threadpool(
            lambda: data.clip_timeline(data_dir, name, public=public)))

    async def skill_list(request):
        listed = await run_in_threadpool(lambda: skills.list_skills(data_dir, public=public))
        return JSONResponse({"skills": listed})

    async def skill(request):
        name = request.path_params["name"]
        return found(await run_in_threadpool(
            lambda: skills.skill_detail(data_dir, name, public=public)))

    async def examples(request):
        shown = await run_in_threadpool(lambda: skills.examples(data_dir, public=public))
        return JSONResponse({"examples": shown,
                             "note": skills.EXAMPLES_NOTE if public else None})

    async def keyframes(request):
        offered = await run_in_threadpool(
            lambda: skills.provided_keyframes(data_dir, public=public))
        return JSONResponse({"keyframes": offered})

    async def results(request):
        return JSONResponse(await run_in_threadpool(data.results, eval_dir))

    async def media(request):
        hit = await run_in_threadpool(data.media_path, data_dir, eval_dir,
                                      request.path_params["path"], public=public)
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
        Route("/how-it-works", how_it_works),
        Mount("/static", StaticFiles(directory=STATIC), name="static"),
    ])


def default_app() -> Starlette:
    """The playground from the settings: data/ as configured, eval/ in the project, the
    default caps. For `uvicorn --factory backend.playground.app:default_app`."""
    from backend.config import PROJECT_ROOT, load_settings

    return create_app(load_settings().data_dir, PROJECT_ROOT / "eval",
                      lambda: MetaClient(load_settings()))
