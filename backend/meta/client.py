"""MetaClient: every call to the Meta Model API goes through here.

Each call runs the same steps:
  1. build the request and its cache key (media is hashed, see cache.py)
  2. cache hit: return the stored response; no network call, no spend
  3. estimate the cost and check the spend caps (ledger.py)
  4. send, retrying 429, 5xx and timeouts with backoff (a 4xx is never retried)
  5. record the actual cost from the reported usage, check the output, cache it

The check in step 3 reserves the estimate in the ledger until step 5 records the call (or
the call fails before it), so calls running at the same time in other threads or
processes count against the caps together.

Spend is counted conservatively: a retried attempt that may have reached the server
(a read timeout, a 500/502/504) is charged at the estimate, and so is a reply that
cannot be priced or a call stopped with Ctrl+C. The caps are checked again before each
such retry. Unusable outputs are charged but never cached, and a cached reply that
today's checks reject is fetched again. A failed call's error carries what it was
charged as `cost_usd`.

Structured output uses Chat Completions with a json_schema response_format, as in Meta's
cookbook recipes. The media shapes follow the API reference on dev.meta.ai (read Nov 4):
Voice Transcribe is POST /asr/transcribe, a multipart body with a JSON `request` part and
a WAV `audio` part; SAM 3.1 is a streamed Responses API call; Muse Image edits take a
JSON body with an `images` array of data URLs.

The API key is only ever sent to the configured Meta base URL, and secret values are
removed from any error text.
"""

from __future__ import annotations

import base64
import email.utils
import io
import json
import math
import random
import re
import shutil
import subprocess
import tempfile
import time
import wave
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from backend.config import SECRET_NAMES, Settings, check_model_allowed
from backend.meta.cache import Cache, cache_key
from backend.meta.ledger import (
    IMAGE_TOKENS,
    NO_SKILL,
    BudgetExceeded,
    Entry,
    Ledger,
    token_cost,
    unit_cost,
)
from backend.meta.pace import Pacer

RETRY_STATUSES = frozenset({408, 429})
# Statuses after which the server may already have done (and billed) the work.
MAYBE_PROCESSED_STATUSES = frozenset({500, 502, 504})
TRANSIENT_ERRORS = (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError)
NOT_SENT_ERRORS = (httpx.ConnectError, httpx.ConnectTimeout)
IMAGE_TOKENS_GUESS = IMAGE_TOKENS  # per image, for the cap check; cost uses reported usage


class MetaAPIError(RuntimeError):
    cost_usd: float = 0.0  # what the failed call was charged in the ledger

    def __init__(self, status: int | None, message: str):
        super().__init__(f"Meta API error {status}: {message}" if status else message)
        self.status = status


class OutputError(MetaAPIError):
    """The call succeeded (and was billed) but the output is unusable."""

    def __init__(self, message: str, finish_reason: str | None = None):
        super().__init__(None, message)
        self.finish_reason = finish_reason  # "length": the reply ran out of max_tokens


class CacheMiss(LookupError):
    """A cache-only lookup found nothing; no request was sent."""


def sniff_mime(data: bytes) -> str:
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return "audio/wav"
    return "application/octet-stream"


def data_url(data: bytes, mime: str | None = None) -> str:
    return f"data:{mime or sniff_mime(data)};base64,{base64.b64encode(data).decode('ascii')}"


def text_part(text: str) -> dict:
    return {"type": "text", "text": text}


def image_part(image: bytes) -> dict:
    """A Chat Completions image content part (base64 data URL)."""
    return {"type": "image_url", "image_url": {"url": data_url(image)}}


@dataclass
class CallInfo:
    model: str
    cache_key: str
    cached: bool
    cost_usd: float  # spent by this call: 0.0 on a cache hit
    usage: dict = field(default_factory=dict)
    billed_usd: float | None = None  # what the response cost when it was made


@dataclass
class StructuredResult:
    data: Any
    info: CallInfo
    reasoning_tokens: int | None = None


@dataclass
class _Tally:
    """Attempts of one call that may have been processed (and billed) by the server."""

    maybe_billed: int = 0
    guard: Callable[[], None] | None = None  # re-checks the spend caps before a retry


def _messages(input: Any, system: str | None) -> list[dict]:
    if isinstance(input, str):
        messages = [{"role": "user", "content": input}]
    elif isinstance(input, list) and input and all("role" in m for m in input):
        messages = list(input)
    elif isinstance(input, list):
        messages = [{"role": "user", "content": list(input)}]
    else:
        raise TypeError("input must be a string, a list of content parts or a list of messages")
    if system:
        messages = [{"role": "system", "content": system}, *messages]
    return messages


def _input_counts(messages: list[dict]) -> tuple[int, int]:
    """(estimated text tokens, images) of a request's messages."""
    images = 0
    chars = 0
    for m in messages:
        content = m.get("content")
        parts = content if isinstance(content, list) else [{"type": "text", "text": content or ""}]
        for p in parts:
            if p.get("type") in ("image_url", "input_image"):
                images += 1
            else:
                chars += len(str(p.get("text", "")))
    return chars // 3 + 50, images


def token_counts(usage: dict) -> tuple[int, int]:
    """(input, output) tokens, in Chat Completions or Responses naming."""
    inp = usage.get("prompt_tokens", usage.get("input_tokens"))
    out = usage.get("completion_tokens", usage.get("output_tokens"))
    if inp is None and out is None:
        raise ValueError("usage has no token counts")
    return int(inp or 0), int(out or 0)


def _parse_json_text(text: str) -> Any:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        stripped = stripped.removeprefix("json").strip()
    return json.loads(stripped)


# -- media endpoints (Voice Transcribe, SAM 3.1, Muse Image) -------------------

EXTENSIONS = {"image/jpeg": "jpg", "image/png": "png", "image/webp": "webp"}
TRANSCRIBE_MODE = "PUSH_TO_TALK"  # one spoken request, one transcript
WAV_RATE = 16000  # our choice: 16 kHz mono 16-bit PCM, the realtime API's smaller rate


def wav_duration_s(audio: bytes) -> float | None:
    try:
        with wave.open(io.BytesIO(audio)) as w:
            return w.getnframes() / float(w.getframerate())
    except (wave.Error, EOFError, ZeroDivisionError):
        return None


def to_wav(path: Path) -> bytes:
    """The recording at `path` as WAV bytes for Voice Transcribe, which takes WAV only: a
    WAV file as it is, anything else (m4a from the phone, ...) converted with ffmpeg to
    16 kHz mono 16-bit PCM, bit-exact so the same file gives the same bytes (and cache
    key) every time."""
    data = path.read_bytes()
    if sniff_mime(data) == "audio/wav":
        return data
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is needed to convert the audio to WAV (brew install ffmpeg)")
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "request.wav"
        done = subprocess.run(
            ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-vn", "-map_metadata", "-1",
             "-ac", "1", "-ar", str(WAV_RATE), "-c:a", "pcm_s16le", "-fflags", "+bitexact",
             "-flags:a", "+bitexact", str(out)],
            capture_output=True, text=True, check=False)
        if done.returncode != 0 or not out.is_file():
            raise ValueError(f"cannot read the audio {path.name}: "
                             f"{(done.stderr or 'ffmpeg failed').strip()[:200]}")
        return out.read_bytes()


def parse_sse(body: str) -> list[dict]:
    """Parse a server-sent-events body into its JSON data payloads.

    Lines end in CRLF, CR or LF (as the SSE format allows); an empty line ends an event.
    """
    events, data = [], []
    for line in re.split(r"\r\n|\r|\n", body):
        if not line:
            if data:
                events.append("\n".join(data))
                data = []
        elif line.startswith("data:"):
            data.append(line[5:].removeprefix(" "))
    if data:
        events.append("\n".join(data))
    return [json.loads(e) for e in events if e.strip() and e.strip() != "[DONE]"]


def sam_stream_text(events: list[dict]) -> str:
    """The text of a streamed SAM response's one output-text lane, as Meta's SAM 3
    protocol (meta-models/meta-sam, protocol/sam3.md) reads it: deltas of one lane (an
    item_id, output_index, content_index), finished by response.output_text.done or a
    response.content_part.done with an output_text part, whose text must equal the deltas
    (or is the whole text when there were none). OutputError for another lane, a
    mismatch, a refusal, a failure or no completion. Lenient on one point: a response that
    completes without the done event still gives its deltas. Empty text is a valid answer
    (SAM found nothing)."""
    lane, deltas, final, completed = None, [], None, None
    for event in events:
        kind = event.get("type", "")
        if kind == "response.output_text.delta" or kind == "response.output_text.done" or (
                kind == "response.content_part.done"
                and (event.get("part") or {}).get("type") == "output_text"):
            this = (event.get("item_id"), event.get("output_index"), event.get("content_index"))
            if lane is None:
                lane = this
            elif this != lane:
                raise OutputError("SAM interleaved more than one output text lane")
            if kind == "response.output_text.delta":
                if final is not None:
                    raise OutputError("SAM sent text after finishing its output")
                deltas.append(event.get("delta") or "")
                continue
            text = event.get("text") if kind.endswith("output_text.done") else (
                (event.get("part") or {}).get("text"))
            if not isinstance(text, str) or (final is not None and text != final):
                raise OutputError("SAM finished its output twice with different text")
            if deltas and text != "".join(deltas):
                raise OutputError("SAM's finished text does not match what it streamed")
            final = text
        elif "refusal" in kind or (kind == "response.content_part.done"
                                   and (event.get("part") or {}).get("type") == "refusal"):
            raise OutputError("SAM refused the request")
        elif kind in ("response.failed", "error", "response.incomplete"):
            raise OutputError(f"SAM stream ended with {kind}")
        elif kind == "response.completed":
            completed = event.get("response") or {}
    if completed is None:
        raise OutputError("SAM stream did not complete")
    if final is not None:
        return final
    return "".join(deltas) if lane is not None else responses_text(completed)


def responses_text(response: dict) -> str:
    """Concatenate output_text parts from a (non-streamed) Responses API object."""
    texts = []
    for item in response.get("output", []):
        for part in item.get("content", []) or []:
            if part.get("type") == "output_text":
                texts.append(part.get("text", ""))
    return "".join(texts)


@dataclass
class Transcript:
    text: str
    segments: list[dict]
    duration_s: float
    info: CallInfo


@dataclass
class SamResult:
    text: str  # raw special-token output; parsed by the data engine
    info: CallInfo


@dataclass
class ImageResult:
    images: list[bytes]  # exactly as returned, provenance metadata intact
    info: CallInfo
    revised_prompts: list[str] = field(default_factory=list)


class MetaClient:
    def __init__(
        self,
        settings: Settings,
        *,
        transport: httpx.BaseTransport | None = None,
        cache: Cache | None = None,
        ledger: Ledger | None = None,
        sleep: Callable[[float], None] = time.sleep,
        pacer: Pacer | None = None,
    ):
        self.settings = settings
        self._pacer = pacer  # spaces the requests that go out, per endpoint (te uses one)
        self.cache = cache or Cache(settings.cache_dir)
        self.ledger = ledger or Ledger(
            settings.ledger_path, settings.skill_cap_usd, settings.global_cap_usd
        )
        self._sleep = sleep
        self._http = httpx.Client(
            base_url=settings.base_url, timeout=settings.timeout_s, transport=transport
        )
        # For downloading generated images by URL: never carries the API key, so it may
        # follow CDN redirects.
        self._plain = httpx.Client(
            timeout=settings.timeout_s, transport=transport, follow_redirects=True
        )

    # -- transport ---------------------------------------------------------

    def _redact(self, text: str) -> str:
        for name in SECRET_NAMES:
            value = self.settings.secrets.get(name)
            if value:
                text = text.replace(value, "[redacted]")
        return text

    def _backoff(self, attempt: int, retry_after: str | None) -> float:
        if retry_after:
            try:
                return min(60.0, max(0.0, float(retry_after)))
            except ValueError:
                try:
                    when = email.utils.parsedate_to_datetime(retry_after)
                    return min(60.0, max(0.0, when.timestamp() - time.time()))
                except (TypeError, ValueError):
                    pass
        base = min(30.0, 2.0**attempt)
        return base + random.uniform(0, base / 4)

    def _send(self, method: str, path: str, tally: _Tally | None = None,
              **kwargs) -> httpx.Response:
        tally = tally or _Tally()
        headers = {**(kwargs.pop("headers", None) or {}),
                   "Authorization": f"Bearer {self.settings.secret('MODEL_API_KEY')}"}
        retries = self.settings.max_retries
        for attempt in range(retries + 1):
            if self._pacer:
                self._pacer.wait(path)
            try:
                resp = self._http.request(method, path, headers=headers, **kwargs)
            except TRANSIENT_ERRORS as err:
                if not isinstance(err, NOT_SENT_ERRORS):
                    tally.maybe_billed += 1
                if attempt == retries:
                    raise MetaAPIError(None, f"network error after {attempt + 1} tries: "
                                       f"{type(err).__name__}") from None
                if tally.maybe_billed and tally.guard:
                    tally.guard()
                self._sleep(self._backoff(attempt, None))
                if tally.guard:  # the caps again before a new attempt (and a server stop)
                    tally.guard()
                continue
            except httpx.TransportError as err:
                raise MetaAPIError(None, f"request failed: {type(err).__name__} "
                                   f"(check TE_BASE_URL)") from None
            status = resp.status_code
            if status in RETRY_STATUSES or status >= 500:
                if status in MAYBE_PROCESSED_STATUSES:
                    tally.maybe_billed += 1
                if attempt == retries:
                    raise MetaAPIError(status, f"gave up after {attempt + 1} tries: "
                                       f"{self._redact(resp.text)[:300]}")
                if tally.maybe_billed and tally.guard:
                    tally.guard()
                self._sleep(self._backoff(attempt, resp.headers.get("retry-after")))
                if tally.guard:
                    tally.guard()
                continue
            if not resp.is_success:
                raise MetaAPIError(status, self._redact(resp.text)[:500])
            return resp
        raise AssertionError("unreachable")

    def _json(self, resp: httpx.Response) -> dict:
        try:
            body = resp.json()
        except ValueError:
            raise MetaAPIError(resp.status_code, "response is not JSON: "
                               f"{self._redact(resp.text)[:200]!r}") from None
        if not isinstance(body, dict):
            raise MetaAPIError(resp.status_code, "response is not a JSON object")
        return body

    def _call(
        self,
        *,
        endpoint: str,
        model: str,
        key_request: Any,
        skill: str | None,
        estimate_usd: float,
        send: Callable[[_Tally], tuple[dict, dict[str, bytes]]],
        cost: Callable[[dict], float],
        check: Callable[[dict], Any],
        cache_only: bool = False,
        inputs: tuple[int, int] = (0, 0),
    ) -> tuple[dict, dict[str, bytes], CallInfo, Any]:
        """Shared cache / ledger / send logic.

        `check(response)` returns the parsed output or raises OutputError. Returns
        (response, blobs, info, parsed). With `cache_only`, a miss raises CacheMiss
        instead of sending. `inputs` (estimated text tokens, images) goes in the entry.
        """
        check_model_allowed(model)
        key = cache_key(model, endpoint, key_request)
        hit = self.cache.get(key)
        if hit is not None:
            blobs = {name: self.cache.blob(key, name) for name in hit.get("blobs", [])}
            info = CallInfo(model, key, True, 0.0, hit.get("usage", {}), hit.get("cost_usd"))
            try:
                return hit["response"], blobs, info, check(hit["response"])
            except OutputError:
                pass  # cached under older, looser checks: fetch it again
        if cache_only:
            raise CacheMiss(key)

        self.settings.secret("MODEL_API_KEY")  # fail before touching the ledger
        entry = Entry(model, endpoint, estimate_usd, 0.0, skill=skill or NO_SKILL, cache_key=key,
                      input_text_tokens=inputs[0], input_images=inputs[1])
        self.ledger.check(estimate_usd, skill)  # reserves the estimate until record()
        try:
            return self._paid(entry, key, endpoint, model, skill, estimate_usd, send, cost,
                              check)
        finally:
            self.ledger.release()  # a call that failed before record(); free after it

    def _paid(self, entry: Entry, key: str, endpoint: str, model: str, skill: str | None,
              estimate_usd: float, send, cost, check) -> tuple[dict, dict, CallInfo, Any]:
        """Send a call that passed the caps, record what it cost, check and cache it."""
        tally = _Tally()
        tally.guard = lambda: self.ledger.check(estimate_usd * (1 + tally.maybe_billed), skill)
        try:
            response, blobs = send(tally)
        except OutputError as err:
            # Served (so billed) but unreadable.
            entry.status = "output_error"
            entry.actual_usd = err.cost_usd = estimate_usd * (1 + tally.maybe_billed)
            self.ledger.record(entry)
            raise
        except (MetaAPIError, BudgetExceeded) as err:
            entry.status = "error"
            entry.actual_usd = estimate_usd * tally.maybe_billed
            if isinstance(err, MetaAPIError):
                err.cost_usd = entry.actual_usd
            self.ledger.record(entry)
            raise
        except BaseException:
            # The request went through but handling the reply failed, or Ctrl+C stopped
            # the call while Meta was working on it: assume it was billed.
            entry.status = "error"
            entry.actual_usd = estimate_usd * (1 + tally.maybe_billed)
            self.ledger.record(entry)
            raise

        usage = response.get("usage") or {}
        if not isinstance(usage, dict):
            usage = {"unreadable": repr(usage)[:100]}
        try:
            entry.actual_usd = cost(response) if usage else estimate_usd
            if not math.isfinite(entry.actual_usd) or entry.actual_usd < 0:
                raise ValueError("not a cost")  # nan or a negative would switch the caps off
        except Exception:  # noqa: BLE001 - any unpriceable reply is charged the estimate
            entry.actual_usd = estimate_usd
            usage = {**usage, "unpriced": True}
        if not usage:
            # No usage reported: record the estimate so spend is never under-counted.
            usage = {"missing": True}
        entry.actual_usd += estimate_usd * tally.maybe_billed
        if tally.maybe_billed:
            usage = {**usage, "maybe_billed_retries": tally.maybe_billed}
        entry.usage = usage
        info = CallInfo(model, key, False, entry.actual_usd, usage, entry.actual_usd)
        try:
            parsed = check(response)
        except BaseException as err:  # served and billed, whatever went wrong reading it
            entry.status = "output_error"
            if isinstance(err, MetaAPIError):
                err.cost_usd = entry.actual_usd
            self.ledger.record(entry)
            raise
        self.ledger.record(entry)
        self.cache.put(key, {"model": model, "endpoint": endpoint, "response": response,
                             "usage": usage, "cost_usd": entry.actual_usd}, blobs)
        return response, blobs, info, parsed

    # -- typed methods -----------------------------------------------------

    def structured(
        self,
        model: str,
        input: Any,
        schema: dict,
        *,
        name: str = "result",
        system: str | None = None,
        skill: str | None = None,
        max_tokens: int = 2048,
        reasoning_effort: str | None = None,
        validate: Callable[[Any], None] | None = None,
        cache_only: bool = False,
    ) -> StructuredResult:
        """JSON output constrained by `schema`, via Chat Completions.

        `validate(data)` may raise OutputError for output that parses but is unusable;
        such output is billed like any other but never cached. With `cache_only`, only the
        cache is asked (CacheMiss if the reply is not there).
        """
        endpoint = "/chat/completions"
        messages = _messages(input, system)
        payload = {
            "model": model,
            "messages": messages,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": name, "schema": schema, "strict": True},
            },
            "max_tokens": max_tokens,
            "reasoning_effort": reasoning_effort,
        }
        payload = {k: v for k, v in payload.items() if v is not None}
        price = self.settings.price(model)
        inputs = _input_counts(messages)
        estimate = token_cost(price, inputs[0] + inputs[1] * IMAGE_TOKENS_GUESS, max_tokens)

        def send(tally):
            return self._json(self._send("POST", endpoint, tally, json=payload)), {}

        def cost(r):
            return token_cost(price, *token_counts(r["usage"]))

        def check(r):
            choices = r.get("choices")
            choice = choices[0] if isinstance(choices, list) and choices else {}
            choice = choice if isinstance(choice, dict) else {}
            message = choice.get("message")
            message = message if isinstance(message, dict) else {}
            content = message.get("content") or ""
            reason = choice.get("finish_reason")
            reason = reason if isinstance(reason, str) else None
            if message.get("refusal"):
                raise OutputError(f"model refused: {message['refusal']}")
            if not isinstance(content, str):
                raise OutputError(f"output content is not text ({type(content).__name__})")
            if not content.strip():
                hint = " (raise max_tokens: reasoning used the budget)" if reason == "length" else ""
                raise OutputError(f"empty output, finish_reason={reason}{hint}", reason)
            try:
                data = _parse_json_text(content)
            except json.JSONDecodeError:
                raise OutputError(f"output is not JSON, finish_reason={reason}: "
                                  f"{content[:200]!r}", reason) from None
            if validate is not None:
                validate(data)
            return data

        _, _, info, data = self._call(endpoint=endpoint, model=model, key_request=payload,
                                      skill=skill, estimate_usd=estimate, send=send, cost=cost,
                                      check=check, cache_only=cache_only, inputs=inputs)
        details = info.usage.get("completion_tokens_details")
        reasoning = details.get("reasoning_tokens") if isinstance(details, dict) else None
        return StructuredResult(data, info, reasoning)

    def list_models(self) -> list[str]:
        """Model IDs the key can use. Free; not cached or recorded."""
        body = self._json(self._send("GET", "/models"))
        return sorted(m.get("id", "") for m in body.get("data", []) if isinstance(m, dict))

    # -- media endpoints -----------------------------------------------------

    def transcribe(
        self,
        audio: bytes,
        *,
        filename: str = "request.wav",
        skill: str | None = None,
        duration_s: float | None = None,
    ) -> Transcript:
        """Speech to text with Muse Voice Transcribe: POST /asr/transcribe, a multipart body
        with the settings as a JSON `request` part and the clip as a WAV `audio` part. The
        API takes WAV only, so convert other formats first (`to_wav`). The reply is
        {sessionId, transcript, audioDurationMs, turns}; `turns` is empty in PUSH_TO_TALK."""
        endpoint = "/asr/transcribe"
        model = self.settings.model("transcribe")
        if sniff_mime(audio) != "audio/wav":
            raise ValueError("Voice Transcribe takes WAV only: convert the audio first "
                             "(backend.meta.client.to_wav)")
        duration = duration_s if duration_s is not None else wav_duration_s(audio)
        if duration is None:
            raise ValueError("the WAV header is unreadable: pass duration_s")
        price = self.settings.price(model)
        estimate = unit_cost(price, audio_s=duration)
        settings = {"model": model, "audioEncoding": "WAV", "mode": TRANSCRIBE_MODE}

        def send(tally):
            files = {"request": (None, json.dumps(settings), "application/json"),
                     "audio": (filename, audio, "audio/wav")}
            resp = self._send("POST", endpoint, tally, files=files,
                              headers={"Accept": "application/json"})
            return _with_usage((self._json(resp), {}), {"audio_s": duration})

        def seconds_of(r) -> float:
            return float(r["audioDurationMs"]) / 1000 if r.get("audioDurationMs") else duration

        def cost(r):
            return unit_cost(price, audio_s=seconds_of(r))

        def check(r):
            text = r.get("transcript")
            if not isinstance(text, str) or not text.strip():
                raise OutputError("the recording transcribed to no words: check the audio, "
                                  "or pass the request as text with --request")
            return text

        response, _, info, text = self._call(
            endpoint=endpoint, model=model, key_request={**settings, "audio": audio},
            skill=skill, estimate_usd=estimate, send=send, cost=cost, check=check,
        )
        try:
            seconds = seconds_of(response)
        except (TypeError, ValueError):
            seconds = duration
        return Transcript(
            text=text,
            segments=response.get("turns") or [],
            duration_s=seconds,
            info=info,
        )

    def sam_image(self, prompt: str, image: bytes, *, skill: str | None = None) -> SamResult:
        """SAM 3.1 on one image with one noun-phrase concept (Responses API, streamed)."""
        endpoint = "/responses"
        model = self.settings.model("sam")
        payload = {
            "model": model,
            "input": [{
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": data_url(image)},
                ],
            }],
            "stream": True,
            "metadata": {"include_confidence": "true"},  # a c= score on each box (protocol)
        }
        price = self.settings.price(model)

        def send(tally):
            resp = self._send("POST", endpoint, tally, json=payload,
                              headers={"Accept": "text/event-stream"})
            if resp.headers.get("content-type", "").startswith("text/event-stream"):
                try:
                    events = parse_sse(resp.text)
                except json.JSONDecodeError:
                    raise OutputError("SAM stream is not readable") from None
                out = sam_stream_text(events)
                final = next((e.get("response") or {} for e in events
                              if e.get("type") == "response.completed"), {})
                body = {"text": out, "usage": final.get("usage") or {}}
            else:
                raw = self._json(resp)
                if raw.get("status") not in (None, "completed"):
                    raise OutputError(f"SAM response status {raw.get('status')}")
                body = {"text": responses_text(raw), "usage": raw.get("usage") or {}}
            return _with_usage((body, {}), {"images": 1})

        def check(r):
            if not isinstance(r.get("text"), str):
                raise OutputError("SAM returned no output text")
            return r["text"]  # empty: SAM found nothing, which is an answer

        _, _, info, text = self._call(
            endpoint=endpoint, model=model, key_request=payload, skill=skill,
            estimate_usd=unit_cost(price, images=1), send=send,
            cost=lambda r: unit_cost(price, images=1), check=check,
        )
        return SamResult(text, info)

    def image_edit(
        self, image: bytes, prompt: str, *, skill: str | None = None, n: int = 1,
        size: str | None = None, reasoning_strength: str | None = None,
    ) -> ImageResult:
        """Muse Image edit of a real keyframe. `reasoning_strength` "low" or "high" (the
        API's default when left out) is how long the model plans before it draws."""
        return self._images("/images/edits", prompt, [image], skill=skill, n=n, size=size,
                            reasoning_strength=reasoning_strength)

    def image_generate(
        self, prompt: str, refs: list[bytes] = (), *, skill: str | None = None, n: int = 1,
        size: str | None = None,
    ) -> ImageResult:
        """Muse Image generation. Reference images, if any, go through the edits endpoint."""
        endpoint = "/images/edits" if refs else "/images/generations"
        return self._images(endpoint, prompt, list(refs), skill=skill, n=n, size=size)

    def _images(self, endpoint: str, prompt: str, images: list[bytes], *, skill: str | None,
                n: int, size: str | None, reasoning_strength: str | None = None
                ) -> ImageResult:
        model = self.settings.model("image")
        price = self.settings.price(model)
        if reasoning_strength not in (None, "low", "high"):
            raise ValueError(f"reasoning_strength must be low or high, not {reasoning_strength!r}")
        fields = {"model": model, "prompt": prompt, "n": n, "size": size,
                  "reasoning_strength": reasoning_strength}
        fields = {k: v for k, v in fields.items() if v is not None}

        def send(tally):
            # Edits as JSON with an `images` array of data URLs: the API reference's raw
            # HTTP shape (its multipart form takes one PNG only; keyframes are JPEG).
            body = {**fields, "images": [{"image_url": data_url(img)} for img in images]}
            resp = self._send("POST", endpoint, tally, json=body if images else fields)
            body = self._json(resp)
            blobs: dict[str, bytes] = {}
            kept = []
            for i, item in enumerate(body.get("data") or []):
                if item.get("b64_json"):
                    raw = base64.b64decode(item["b64_json"])
                elif item.get("url"):
                    raw = self._download(item["url"])
                else:
                    continue
                blob = f"{i}.{EXTENSIONS.get(sniff_mime(raw), 'bin')}"
                blobs[blob] = raw
                kept.append({k: v for k, v in item.items() if k != "b64_json"} | {"blob": blob})
            body = {**body, "data": kept}
            return _with_usage((body, blobs), {"images": len(blobs)})

        def check(r):
            if not r.get("data"):
                raise OutputError("no images returned")
            return r["data"]

        # A reply without usable images is still billed for the n images asked for.
        _, blobs, info, items = self._call(
            endpoint=endpoint, model=model, key_request={**fields, "images": images},
            skill=skill, estimate_usd=unit_cost(price, images=n), send=send,
            cost=lambda r: unit_cost(price, images=len(r.get("data") or []) or n), check=check,
        )
        ordered = [blobs[item["blob"]] for item in items]
        prompts = [item.get("revised_prompt", "") for item in items]
        return ImageResult(ordered, info, [p for p in prompts if p])

    def _download(self, url: str, tries: int = 2) -> bytes:
        """A generated image by URL. The edit is billed already, so a failed download is
        an unusable output (OutputError), not a crash."""
        for attempt in range(tries):
            try:
                resp = self._plain.get(url)
                resp.raise_for_status()
                return resp.content
            except httpx.HTTPError as err:
                if attempt == tries - 1:
                    raise OutputError(f"image download failed: {type(err).__name__}") from None
                self._sleep(1.0)
        raise AssertionError("unreachable")


def _with_usage(result: tuple[dict, dict], fallback: dict) -> tuple[dict, dict]:
    """Unit-priced endpoints may not report usage; fill in what we billed for."""
    response, blobs = result
    if not response.get("usage"):
        response = {**response, "usage": fallback}
    return response, blobs
