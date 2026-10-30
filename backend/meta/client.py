"""MetaClient: every call to the Meta Model API goes through here.

Each call runs the same steps:
  1. build the request and its cache key (media is hashed, see cache.py)
  2. cache hit: return the stored response; no network call, no spend
  3. estimate the cost and check the spend caps (ledger.py)
  4. send, retrying 429, 5xx and timeouts with backoff (a 4xx is never retried)
  5. record the actual cost from the reported usage, check the output, cache it

Spend is counted conservatively: a retried attempt that may have reached the server
(a read timeout, a 500/502/504) is charged at the estimate, and so is a reply that
cannot be priced. Unusable outputs are charged but never cached.

Request shapes are OpenAI-compatible. Structured output uses Chat Completions with a
json_schema response_format, as in Meta's cookbook recipes.

The API key is only ever sent to the configured Meta base URL, and secret values are
removed from any error text.
"""

from __future__ import annotations

import base64
import email.utils
import json
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import httpx

from backend.config import SECRET_NAMES, Settings, check_model_allowed
from backend.meta.cache import Cache, cache_key
from backend.meta.ledger import NO_SKILL, Entry, Ledger, token_cost

RETRY_STATUSES = frozenset({408, 429})
# Statuses after which the server may already have done (and billed) the work.
MAYBE_PROCESSED_STATUSES = frozenset({500, 502, 504})
TRANSIENT_ERRORS = (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError)
NOT_SENT_ERRORS = (httpx.ConnectError, httpx.ConnectTimeout)
IMAGE_TOKENS_GUESS = 1500  # per image, for estimates only; actual cost uses reported usage


class MetaAPIError(RuntimeError):
    def __init__(self, status: int | None, message: str):
        super().__init__(f"Meta API error {status}: {message}" if status else message)
        self.status = status


class OutputError(MetaAPIError):
    """The call succeeded (and was billed) but the output is unusable."""

    def __init__(self, message: str):
        super().__init__(None, message)


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
    cost_usd: float
    usage: dict = field(default_factory=dict)


@dataclass
class StructuredResult:
    data: Any
    info: CallInfo
    reasoning_tokens: int | None = None


@dataclass
class _Tally:
    """Attempts of one call that may have been processed (and billed) by the server."""

    maybe_billed: int = 0


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


def _estimate_input_tokens(messages: list[dict]) -> int:
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
    return chars // 3 + 50 + images * IMAGE_TOKENS_GUESS


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


class MetaClient:
    def __init__(
        self,
        settings: Settings,
        *,
        transport: httpx.BaseTransport | None = None,
        cache: Cache | None = None,
        ledger: Ledger | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.settings = settings
        self.cache = cache or Cache(settings.cache_dir)
        self.ledger = ledger or Ledger(
            settings.ledger_path, settings.skill_cap_usd, settings.global_cap_usd
        )
        self._sleep = sleep
        self._http = httpx.Client(
            base_url=settings.base_url, timeout=settings.timeout_s, transport=transport
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
        headers = {"Authorization": f"Bearer {self.settings.secret('MODEL_API_KEY')}"}
        retries = self.settings.max_retries
        for attempt in range(retries + 1):
            try:
                resp = self._http.request(method, path, headers=headers, **kwargs)
            except TRANSIENT_ERRORS as err:
                if not isinstance(err, NOT_SENT_ERRORS):
                    tally.maybe_billed += 1
                if attempt == retries:
                    raise MetaAPIError(None, f"network error after {attempt + 1} tries: "
                                       f"{type(err).__name__}") from None
                self._sleep(self._backoff(attempt, None))
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
                self._sleep(self._backoff(attempt, resp.headers.get("retry-after")))
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
    ) -> tuple[dict, dict[str, bytes], CallInfo, Any]:
        """Shared cache / ledger / send logic.

        `check(response)` returns the parsed output or raises OutputError. Returns
        (response, blobs, info, parsed).
        """
        check_model_allowed(model)
        key = cache_key(model, endpoint, key_request)
        hit = self.cache.get(key)
        if hit is not None:
            blobs = {name: self.cache.blob(key, name) for name in hit.get("blobs", [])}
            info = CallInfo(model, key, True, 0.0, hit.get("usage", {}))
            return hit["response"], blobs, info, check(hit["response"])

        self.settings.secret("MODEL_API_KEY")  # fail before touching the ledger
        self.ledger.check(estimate_usd, skill)
        entry = Entry(model, endpoint, estimate_usd, 0.0, skill=skill or NO_SKILL, cache_key=key)
        tally = _Tally()
        try:
            response, blobs = send(tally)
        except OutputError:
            # Served (so billed) but unreadable.
            entry.status = "output_error"
            entry.actual_usd = estimate_usd * (1 + tally.maybe_billed)
            self.ledger.record(entry)
            raise
        except MetaAPIError:
            entry.status = "error"
            entry.actual_usd = estimate_usd * tally.maybe_billed
            self.ledger.record(entry)
            raise
        except Exception:
            # The request went through but handling the reply failed: assume it was billed.
            entry.status = "error"
            entry.actual_usd = estimate_usd * (1 + tally.maybe_billed)
            self.ledger.record(entry)
            raise

        usage = response.get("usage") or {}
        try:
            entry.actual_usd = cost(response) if usage else estimate_usd
        except (TypeError, ValueError, KeyError):
            entry.actual_usd = estimate_usd
            usage = {**usage, "unpriced": True}
        if not usage:
            # No usage reported: record the estimate so spend is never under-counted.
            usage = {"missing": True}
        entry.actual_usd += estimate_usd * tally.maybe_billed
        if tally.maybe_billed:
            usage = {**usage, "maybe_billed_retries": tally.maybe_billed}
        entry.usage = usage
        info = CallInfo(model, key, False, entry.actual_usd, usage)
        try:
            parsed = check(response)
        except OutputError:
            entry.status = "output_error"
            self.ledger.record(entry)
            raise
        self.ledger.record(entry)
        self.cache.put(key, {"model": model, "endpoint": endpoint, "response": response,
                             "usage": usage}, blobs)
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
    ) -> StructuredResult:
        """JSON output constrained by `schema`, via Chat Completions."""
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
        estimate = token_cost(price, _estimate_input_tokens(messages), max_tokens)

        def send(tally):
            return self._json(self._send("POST", endpoint, tally, json=payload)), {}

        def cost(r):
            return token_cost(price, *token_counts(r["usage"]))

        def check(r):
            choice = (r.get("choices") or [{}])[0]
            message = choice.get("message") or {}
            content = message.get("content") or ""
            if message.get("refusal"):
                raise OutputError(f"model refused: {message['refusal']}")
            if not content.strip():
                reason = choice.get("finish_reason")
                hint = " (raise max_tokens: reasoning used the budget)" if reason == "length" else ""
                raise OutputError(f"empty output, finish_reason={reason}{hint}")
            try:
                return _parse_json_text(content)
            except json.JSONDecodeError:
                raise OutputError(f"output is not JSON: {content[:200]!r}") from None

        _, _, info, data = self._call(endpoint=endpoint, model=model, key_request=payload,
                                      skill=skill, estimate_usd=estimate, send=send, cost=cost,
                                      check=check)
        details = info.usage.get("completion_tokens_details") or {}
        return StructuredResult(data, info, details.get("reasoning_tokens"))

    def list_models(self) -> list[str]:
        """Model IDs the key can use. Free; not cached or recorded."""
        body = self._json(self._send("GET", "/models"))
        return sorted(m.get("id", "") for m in body.get("data", []) if isinstance(m, dict))

