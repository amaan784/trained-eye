"""The regression check: does fine-tuning on yes/no frames cost general visual skill?

About 50 general visual questions written by the author, plus a few text-only ones, are
asked of the base model and the fine-tuned one at matching precision (both bf16, or both
the same quantization: the report records what the author states as `precision`, it
cannot check it). Both endpoints speak the OpenAI chat API, as in
backend/inference/student.py: the image, if any, as a data URL, then the question; no
system prompt; temperature 0 by default, so a rerun asks the same thing (Meta's serving
notes advise against greedy decoding for Glimmer: Endpoint can leave it to the server).

Questions file, one JSON object per line (`image` optional, relative to the file):

    {"id": "q01", "image": "img/q01.jpg", "question": "...", "answer": "...", "match": "exact"}

Grading ignores case, extra spaces and end punctuation:
- exact: the reply is the answer;
- contains: the answer appears in the reply as whole words;
- choice: the reply's first word, without brackets or punctuation, is the answer
  ("B", "(B)" and "B) red" all match "B"); a leading "Answer:" is skipped.
Only the reply's `content` is graded; reasoning a server returns apart is not. A request
that fails counts as wrong and is listed under `failed`: rerun before trusting the result.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import httpx

from backend.meta.client import data_url, sniff_mime

MATCHES = ("exact", "contains", "choice")


@dataclass
class Question:
    id: str
    question: str
    answer: str
    match: str
    image: Path | None = None


def load_questions(path: Path) -> list[Question]:
    """The questions, or ValueError naming every line that cannot be used."""
    path = Path(path)
    out, bad = [], []
    for n, line in enumerate(path.read_text("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            image = path.parent / row["image"] if row.get("image") else None
            q = Question(str(row["id"]), str(row["question"]), str(row["answer"]),
                         row["match"], image)
        except (ValueError, KeyError, TypeError, AttributeError) as err:
            bad.append(f"line {n}: {type(err).__name__} {err}")
            continue
        if q.match not in MATCHES:
            bad.append(f"line {n}: match must be one of {', '.join(MATCHES)}")
        elif q.image and not q.image.is_file():
            bad.append(f"line {n}: no image at {q.image}")
        elif q.image and not sniff_mime(q.image.read_bytes()).startswith("image/"):
            bad.append(f"line {n}: {q.image} is not a JPEG, PNG or WebP image")
        else:
            out.append(q)
    ids = [q.id for q in out]
    if repeated := sorted({i for i in ids if ids.count(i) > 1}):
        bad.append(f"ids used twice: {', '.join(repeated)}")
    if bad or not out:
        raise ValueError(f"{path}: " + ("; ".join(bad) or "no questions"))
    return out


def _norm(text: str) -> str:
    return " ".join(str(text).lower().split()).strip(" .!?\"'*")


def grade(reply: str, answer: str, match: str) -> bool:
    got, want = _norm(reply), _norm(answer)
    if not want:
        return False
    if match == "exact":
        return got == want
    if match == "contains":
        # a number inside a longer one ("3" in "3.5", "500" in "1,500") is not a match
        return re.search(rf"(?<!\w)(?<!\d[.,]){re.escape(want)}(?![.,]?\w)", got) is not None
    got = re.sub(r"^[*_\s]*(the answer is|answer)[*_\s]*:?[*_\s]*", "", got)
    return (got.split() or [""])[0].strip("()[]*.,:;!?\"'") == want


class Endpoint:
    """One model behind an OpenAI-compatible /chat/completions endpoint. `temperature`
    None leaves it to the server; `extra_body` is added to each request (for example the
    student's chat_template_kwargs), the same for both models."""

    def __init__(self, base_url: str, model: str, *, api_key: str | None = None,
                 max_tokens: int = 2048, timeout_s: float = 120.0,
                 temperature: float | None = 0.0, extra_body: dict | None = None,
                 transport: httpx.BaseTransport | None = None):
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._http = httpx.Client(base_url=base_url.rstrip("/"), headers=headers,
                                  timeout=timeout_s, transport=transport)
        self.model = model
        self.max_tokens = max_tokens
        self.extra = {**({} if temperature is None else {"temperature": temperature}),
                      **(extra_body or {})}

    def ask(self, q: Question) -> tuple[str, str | None]:
        """(reply text, finish reason)."""
        content: str | list = q.question
        if q.image:
            content = [{"type": "image_url", "image_url": {"url": data_url(q.image.read_bytes())}},
                       {"type": "text", "text": q.question}]
        body = {"model": self.model, "messages": [{"role": "user", "content": content}],
                "max_tokens": self.max_tokens, **self.extra}
        resp = self._http.post("/chat/completions", json=body)
        resp.raise_for_status()
        choice = resp.json()["choices"][0]
        return str(choice["message"].get("content") or ""), choice.get("finish_reason")


def _ask(endpoint: Endpoint, q: Question) -> dict:
    try:
        reply, finish = endpoint.ask(q)
    except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError, AttributeError) as err:
        return {"reply": None, "ok": False, "error": f"{type(err).__name__}: {str(err)[:200]}"}
    return {"reply": reply[:500], "ok": grade(reply, q.answer, q.match), "finish": finish}


def _side(rows: list[dict], side: str, model: str) -> dict:
    def share(subset):
        right = sum(r[side]["ok"] for r in subset)
        return {"n": len(subset), "correct": right,
                "accuracy": round(right / len(subset), 4) if subset else None}
    return {"model": model, **share(rows),
            "with_image": share([r for r in rows if r["image"]]),
            "text_only": share([r for r in rows if not r["image"]]),
            "failed": [r["id"] for r in rows if "error" in r[side]],
            "cut_off": [r["id"] for r in rows if r[side].get("finish") == "length"]}


def run_regression(questions: list[Question], base: Endpoint, finetuned: Endpoint, *,
                   precision: str) -> dict:
    """Ask both models every question; accuracy, the questions whose result changed,
    and a short markdown table."""
    rows = [{"id": q.id, "question": q.question, "answer": q.answer, "match": q.match,
             "image": q.image is not None, "base": _ask(base, q),
             "finetuned": _ask(finetuned, q)} for q in questions]
    report = {"precision": precision, "questions": len(rows),
              "base": _side(rows, "base", base.model),
              "finetuned": _side(rows, "finetuned", finetuned.model),
              "changed": [{"id": r["id"], "base": r["base"]["ok"],
                           "finetuned": r["finetuned"]["ok"]}
                          for r in rows if r["base"]["ok"] != r["finetuned"]["ok"]],
              "rows": rows}
    report["markdown"] = markdown(report)
    return report


def markdown(report: dict) -> str:
    base, ft = report["base"], report["finetuned"]

    def cell(part):
        return "n/a" if not part["n"] else f"{part['correct']}/{part['n']} ({part['accuracy']:.0%})"
    title = (f"Regression check at {report['precision']}: {base['model']} (base) vs "
             f"{ft['model']} (fine-tuned)")
    lines = [title, "", "| Questions | Base | Fine-tuned |", "|---|---|---|"]
    for label, key in (("All", None), ("With an image", "with_image"), ("Text only", "text_only")):
        b, f = (base, ft) if key is None else (base[key], ft[key])
        lines.append(f"| {label} ({b['n']}) | {cell(b)} | {cell(f)} |")
    changed = [f"{c['id']} ({'right to wrong' if c['base'] else 'wrong to right'})"
               for c in report["changed"]]
    lines += ["", f"Changed: {', '.join(changed) or 'none'}."]
    for name, side in (("base", base), ("fine-tuned", ft)):
        if side["failed"] or side["cut_off"]:
            lines.append(f"{name}: {len(side['failed'])} failed requests, {len(side['cut_off'])} "
                         "replies cut off (raise max_tokens); rerun before trusting this")
    return "\n".join(lines)
