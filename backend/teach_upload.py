"""The phone's teach upload (POST /teach/start in backend/teach_api.py): the multipart
body parsed, checked the way te teach checks its inputs, saved, priced, and turned into
a TeachSession that has not started. Nothing is spent here.

The capture is saved under data/teach/<skill>/<UTC time>/ (f000.jpg, f001.jpg, ...,
request.<ext>), so a refused or failed run can be repeated from the Mac with te teach.
Starlette's own form parser needs python-multipart, which is not a dependency, so the
body is parsed here.
"""

from __future__ import annotations

import math
import shlex
import time
from collections.abc import Callable
from email.message import Message
from email.utils import collapse_rfc2231_value
from pathlib import Path

from backend.config import ASK_ABOVE_USD, PARALLEL_CALLS, Settings, shown_path
from backend.dataengine.keyframes import load_frames, select_keyframes
from backend.meta.client import MetaClient, sniff_mime, wav_duration_s
from backend.skills.compiler import MAX_REQUEST
from backend.skills.spec import check_name, policy_check
from backend.teach import TeachOptions, check_clip, estimate, labels_problem, save_keyframes
from backend.teach_session import TeachSession

MAX_FRAMES = 600  # f000 to f599; select_keyframes keeps at most 24
MAX_NUMBER = 100_000  # far over what the $15 limit allows, far under a float's range
MAX_ASPECT = 4  # a frame longer than 4:1 is not from a camera (and would blow up memory)
FIELDS = {"skill", "request", "audio", "frames", "positives", "negatives", "audio_s"}
AUDIO_EXTENSIONS = {"wav", "m4a", "mp3", "mp4", "aac", "flac", "ogg", "webm"}

Form = dict[str, list[tuple[str | None, bytes]]]  # field -> [(filename or None, bytes)]


class Problem(Exception):
    """An upload refused with this HTTP status (and extra JSON fields)."""

    def __init__(self, status: int, message: str, **extra):
        super().__init__(message)
        self.status, self.extra = status, extra


def parse_multipart(body: bytes, content_type: str) -> Form:
    """The parts of a multipart/form-data body, in the order sent. ValueError if it is not."""
    header = Message()
    header["content-type"] = content_type
    boundary = header.get_param("boundary")
    if header.get_content_type() != "multipart/form-data" or not isinstance(boundary, str):
        raise ValueError("send the capture as multipart/form-data")
    chunks = (b"\r\n" + body).split(b"\r\n--" + boundary.encode("latin-1"))
    if len(chunks) < 2 or not chunks[-1].startswith(b"--"):
        raise ValueError("the multipart body is cut off")
    form: Form = {}
    for chunk in chunks[1:-1]:
        head, _, data = chunk.partition(b"\r\n")[2].partition(b"\r\n\r\n")
        part = Message()
        for line in head.decode("utf-8", "replace").split("\r\n"):
            key, _, value = line.partition(":")
            part[key.strip()] = value.strip()
        name = part.get_param("name", header="content-disposition")
        if not name:
            raise ValueError("a multipart part has no name")
        name = str(collapse_rfc2231_value(name))
        form.setdefault(name, []).append((part.get_filename(), data))
    return form


def one(form: Form, name: str) -> tuple[str | None, bytes] | None:
    values = form.get(name, [])
    if len(values) > 1:
        raise ValueError(f"{name} was sent more than once")
    return values[0] if values else None


def text(form: Form, name: str) -> str:
    value = one(form, name)
    try:
        return " ".join(value[1].decode("utf-8").split()) if value else ""
    except UnicodeDecodeError:
        raise ValueError(f"{name} is not UTF-8 text") from None


def number(form: Form, name: str, default: float, kind: type = int) -> float:
    raw = text(form, name)
    try:
        value = kind(raw) if raw else default
        finite = math.isfinite(value)
    except (ValueError, OverflowError):  # an int too large for a float, too
        raise ValueError(f"{name} must be a number, not {raw[:20]!r}") from None
    if not (finite and 0 <= value <= MAX_NUMBER):
        raise ValueError(f"{name} must be a finite number from 0 to {MAX_NUMBER}")
    return value


def save_capture(folder: Path, frames: list[bytes], audio: bytes | None, ext: str) -> Path:
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    capture, n = folder / stamp, 1
    while capture.exists():
        n += 1
        capture = folder / f"{stamp}-{n}"
    capture.mkdir(parents=True)
    for i, data in enumerate(frames):
        (capture / f"f{i:03d}.jpg").write_bytes(data)
    if audio is not None:
        (capture / f"request.{ext}").write_bytes(audio)
    return capture


def audio_extension(filename: str | None, data: bytes) -> str:
    if not data:
        raise ValueError("the audio file is empty")
    ext = Path(filename or "").suffix.lower().lstrip(".")
    if ext in AUDIO_EXTENSIONS:
        return ext
    if sniff_mime(data) == "audio/wav":
        return "wav"
    raise ValueError("audio must be a file ending in one of "
                     f"{', '.join(sorted(AUDIO_EXTENSIONS))}")


def new_session(form: Form, settings: Settings, client_factory: Callable[[Settings], MetaClient],
                timeout_s: float) -> TeachSession:
    """ValueError for bad input, Problem for a refusal, ConfigError without a key."""
    from backend.server import inspect_jpeg  # here: server.py imports the teach routes

    unknown = sorted(set(form) - FIELDS)
    if unknown:
        raise ValueError(f"unknown fields: {', '.join(unknown)}")
    skill, request = check_name(text(form, "skill")), text(form, "request")
    audio = one(form, "audio")
    if bool(audio) == bool(request):
        raise ValueError("send the spoken request as audio, or its text as request (one of them)")
    if len(request) > MAX_REQUEST:
        raise ValueError(f"the request is {len(request)} characters, over the {MAX_REQUEST} "
                         "allowed: say it in one sentence")
    refusal = policy_check(request) if request else None
    if refusal:
        raise Problem(422, f"I can't learn that one. {refusal}", refused=True)
    ext = audio_extension(*audio) if audio else ""
    frames = [data for _, data in form.get("frames", [])]
    if not frames or len(frames) > MAX_FRAMES:
        raise ValueError(f"send 1 to {MAX_FRAMES} JPEG frames of the capture, named frames")
    for i, data in enumerate(frames):
        try:
            width, height = inspect_jpeg(data)
        except ValueError as err:
            raise ValueError(f"frame {i}: {err}") from None
        if max(width, height) > MAX_ASPECT * min(width, height):  # keyframes scales by width
            raise ValueError(f"frame {i}: {width}x{height} is not a camera frame")
    positives, negatives = number(form, "positives", 150), number(form, "negatives", 150)
    audio_s = 0.0
    if audio:
        audio_s = wav_duration_s(audio[1]) or number(form, "audio_s", 0.0, float) or 10.0
    settings.secret("MODEL_API_KEY")  # a missing key stops here, before anything is saved
    capture = save_capture(settings.data_dir / "teach" / skill, frames,
                           audio[1] if audio else None, ext)
    skill_root = settings.data_dir / "skills" / skill
    check_clip(capture, skill_root)
    keyframes = select_keyframes(load_frames(capture))
    if not keyframes:
        raise ValueError("no usable frames in the capture")
    audio_path = capture / f"request.{ext}" if audio else None
    command = shlex.join(["te", "teach", "--skill", skill, "--clip", shown_path(capture),
                          *(["--audio", shown_path(audio_path)] if audio_path else
                            ["--request", request])])
    problem = labels_problem(settings.train_on_model_outputs, {}, skill_root / "keyframes")
    if problem:
        save_keyframes(skill_root / "keyframes", keyframes)  # so they can be labelled
        raise Problem(422, f"{problem}. Run it from the Mac: {command} --labels <file>")
    client = client_factory(settings)
    parts = estimate(client, audio_s=audio_s, keyframes=len(keyframes), positives=positives,
                     negatives=negatives)
    parts = {k: round(v, 4) for k, v in parts.items()}
    if parts["at_most"] > ASK_ABOVE_USD:
        raise Problem(422, f"this run could cost up to ${parts['at_most']:.2f}, over the "
                      f"${ASK_ABOVE_USD:.2f} that is only spent after asking on the Mac. Run "
                      f"it there: {command} --positives {positives} --negatives {negatives}, "
                      "or send fewer positives and negatives", estimate=parts)
    room = min(settings.skill_cap_usd - client.ledger.spent(skill),
               settings.global_cap_usd - client.ledger.spent())
    warning = None
    if parts["at_most"] > room:
        warning = (f"only ${max(room, 0):.2f} is left under the spend caps for {skill}: the "
                   "cap may stop the run partway (raise TE_SKILL_CAP_USD on the Mac and start "
                   "again; finished calls are cached)")
    opts = TeachOptions(skill=skill, clip=capture, audio=audio_path, request=request or None,
                        positives=positives, negatives=negatives, audio_s=audio_s or None,
                        workers=PARALLEL_CALLS)
    return TeachSession(client, opts, settings.data_dir / "skills", estimate=parts,
                        capture=shown_path(capture), warning=warning, timeout_s=timeout_s)
