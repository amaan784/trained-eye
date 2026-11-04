"""The student: the fine-tuned Glimmer model that scores frames in watch mode.

It runs behind an OpenAI-compatible chat endpoint: vLLM on Modal, or llama.cpp's
llama-server on the Mac (which one depends on the Mac's memory; HANDOVER.md section 5).
Each frame is sent exactly as the training examples were built (dataset.py): the system
prompt, then the padded 512x512 image and the condition text.

Glimmer reasons before it answers, so the first generated token is not the answer. The
probability is read at the answer itself: the last generated token that is "yes" or
"no", with the chance of each taken from that token's top logprobs and renormalised
over the two. A reply without such a token is an error, never a guess, and so is a reply
cut off at max_tokens (finish_reason "length"): a yes or no in it comes from the reasoning,
not the answer. Raise TE_STUDENT_MAX_TOKENS (or te watch --max-tokens) for those.

Sampling follows Meta's serving notes (meta-oss-cookbook muse-glimmer/inference-server/
vllm.md): no temperature in the request, so the server uses the checkpoint's published
settings (it advises against greedy decoding), and reasoning kept short with
chat_template_kwargs.reasoning_strength "low", as the cookbook's evaluation does. The
answer token's logprobs are what is read, so a sampled reasoning path changes little.

The endpoint is the author's own model, not a Meta API call, so it is not cached and not
in the spend ledger; Modal bills for the GPU time instead.
"""

from __future__ import annotations

import base64
import io
import math
import time
from dataclasses import dataclass

import httpx

from backend.dataengine.dataset import SYSTEM_PROMPT, pad_image, user_prompt
from backend.skills.spec import SkillSpec

ANSWERS = ("yes", "no")
REASONING = {"chat_template_kwargs": {"reasoning_strength": "low"}}


class StudentError(RuntimeError):
    """The endpoint failed or its reply had no readable yes/no answer."""


@dataclass
class Score:
    p: float  # probability that the condition is true in the frame
    answer: str  # the token the model actually generated: "yes" or "no"
    latency_s: float


def frame_message(image: bytes, spec: SkillSpec, system_in_user: bool = False) -> list[dict]:
    """Chat messages for one frame, the same as a training example without the answer."""
    buf = io.BytesIO()
    pad_image(image).save(buf, format="PNG")
    url = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"
    text = user_prompt(spec.positive, spec.negatives)
    messages = []
    if system_in_user:
        text = f"{SYSTEM_PROMPT}\n\n{text}"
    else:
        messages.append({"role": "system", "content": SYSTEM_PROMPT})
    messages.append({"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": url}}, {"type": "text", "text": text}]})
    return messages


def _word(token: str) -> str:
    return token.strip().strip(".!\"'").lower()


def answer_probability(logprobs: list[dict]) -> tuple[float, str]:
    """(p_yes, answer) from an OpenAI-style `choices[0].logprobs.content` list."""
    for item in reversed(logprobs or []):
        if not isinstance(item, dict) or _word(str(item.get("token", ""))) not in ANSWERS:
            continue
        mass = dict.fromkeys(ANSWERS, 0.0)
        candidates = [item, *(item.get("top_logprobs") or [])]
        seen = set()
        for cand in candidates:
            token = str(cand.get("token", ""))
            word = _word(token)
            if word in mass and token not in seen and isinstance(cand.get("logprob"),
                                                                 (int, float)):
                seen.add(token)
                mass[word] += math.exp(cand["logprob"])
        total = mass["yes"] + mass["no"]
        if total <= 0:
            raise StudentError("the answer token has no usable logprobs")
        return mass["yes"] / total, _word(str(item["token"]))
    raise StudentError("the reply has no yes or no answer token")


class OpenAIStudent:
    """Scores frames through an OpenAI-compatible /chat/completions endpoint."""

    def __init__(self, base_url: str, model: str, *, api_key: str | None = None,
                 timeout_s: float = 60.0, max_tokens: int = 512, top_logprobs: int = 10,
                 system_in_user: bool = False, extra_body: dict | None = None,
                 transport: httpx.BaseTransport | None = None):
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        # Modal answers a request still running after 150 s with a 303 to its result URL
        self._http = httpx.Client(base_url=base_url.rstrip("/"), headers=headers,
                                  timeout=timeout_s, transport=transport,
                                  follow_redirects=True)
        self.model = model
        self.max_tokens = max_tokens
        self.top_logprobs = top_logprobs
        self.system_in_user = system_in_user
        self.extra_body = {**REASONING, **(extra_body or {})}

    def score(self, image: bytes, spec: SkillSpec) -> Score:
        body = {"model": self.model, "messages": frame_message(image, spec, self.system_in_user),
                "max_tokens": self.max_tokens, "logprobs": True,
                "top_logprobs": self.top_logprobs, **self.extra_body}
        start = time.monotonic()
        try:
            resp = self._http.post("/chat/completions", json=body)
            resp.raise_for_status()
            reply = resp.json()
        except (httpx.HTTPError, ValueError) as err:
            raise StudentError(f"student endpoint failed: {type(err).__name__}: "
                               f"{str(err)[:200]}") from None
        try:
            choice = reply["choices"][0]
            logprobs = choice["logprobs"]["content"]
        except (KeyError, IndexError, TypeError):
            raise StudentError("the reply has no logprobs (does the server support "
                               "logprobs and top_logprobs?)") from None
        if choice.get("finish_reason") == "length":
            raise StudentError(f"the reply was cut off at max_tokens ({self.max_tokens}) "
                               "before its answer: raise TE_STUDENT_MAX_TOKENS (te watch "
                               "--max-tokens)")
        p, answer = answer_probability(logprobs)
        return Score(p, answer, time.monotonic() - start)
