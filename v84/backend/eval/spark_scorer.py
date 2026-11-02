"""Muse Spark as the per-frame judge, for Experiment 4 (fine-tuned Glimmer vs base
Glimmer zero-shot vs Muse Spark on every frame).

`SparkScorer(client).score(image, spec)` has the student's shape
(backend/inference/student.py), so the watcher, `te watch` and the evaluation use it as
they are. Each frame goes through labels.label_image, the same request as a training
label or a watch-mode confirmation, so it is cached and recorded in the spend ledger
under the skill (and counts to the skill's cap). Spark gives no probability: yes is p 1.0,
no 0.0 and unsure 0.5; te watch --scorer spark sets every threshold to 1.0, so only a yes
counts (at the default 0.5 an unsure would count as a yes). A
frame Spark could not label raises StudentError, which the watcher logs as a frame
without a score. Errors that are not about one frame (the key, a spend cap, the API down
after retries) are raised as they are and stop the replay, as in te teach.

Latency: a cached frame takes no time, so a second replay would log the cache lookup.
With `latency_log`, every frame actually sent is recorded there (a hash of the request,
its seconds and cost, never the image) and a cached frame reports the latency measured
when it was sent. A frame cached by something else (a watch confirmation) has no record:
its score says `latency_measured=False`, and the evaluation should leave it out of the
latency figures.

Base Glimmer zero-shot needs no code: it is OpenAIStudent pointed at an endpoint serving
the base model, e.g. `TE_STUDENT_URL=<base endpoint> TE_STUDENT_MODEL=<its served name>
te watch ...`, with the same prompt as the fine-tuned model. Untrained, it reasons at
length, so raise OpenAIStudent's max_tokens (512) if replies stop before yes or no.
"""

from __future__ import annotations

import json
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from backend.dataengine.dataset import user_prompt
from backend.dataengine.labels import label_image
from backend.inference.student import Score, StudentError
from backend.meta.cache import sha256_hex
from backend.meta.client import MetaClient
from backend.skills.spec import SkillSpec

P_OF_LABEL = {"yes": 1.0, "no": 0.0, "unsure": 0.5}


def default_log(data_dir: Path) -> Path:
    """Where `te watch --scorer spark` keeps the latency record: data/eval/."""
    return Path(data_dir) / "eval" / "spark_frames.jsonl"


@dataclass
class SparkScore(Score):
    cached: bool = False
    latency_measured: bool = True  # False: cached by another caller, latency unknown
    cost_usd: float = 0.0  # spent on this frame now: 0.0 when cached


def request_key(model: str, spec: SkillSpec, image: bytes) -> str:
    """What decides Spark's answer: the model, the question and the image."""
    prompt = user_prompt(spec.positive or "", spec.negatives)
    return sha256_hex(f"{model}|{prompt}|{sha256_hex(image)}".encode())


def read_log(path: Path) -> dict[str, dict]:
    """{request key: the latest record} from a latency log; {} if there is none."""
    path = Path(path)
    if not path.is_file():
        return {}
    rows = [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]
    return {row["key"]: row for row in rows}


def sent_summary(path: Path) -> dict:
    """Frames actually sent, their median latency and mean cost, from a latency log."""
    rows = list(read_log(path).values())
    if not rows:
        return {"checks": 0, "latency_median_s": None, "usd_per_check": None}
    return {"checks": len(rows),
            "latency_median_s": round(statistics.median(r["latency_s"] for r in rows), 3),
            "usd_per_check": round(statistics.mean(r["cost_usd"] for r in rows), 6)}


class SparkScorer:
    """Scores frames with Muse Spark's yes, no or unsure."""

    def __init__(self, client: MetaClient, *, latency_log: Path | None = None,
                 clock: Callable[[], float] = time.monotonic):
        self.client = client
        self.clock = clock
        self.latency_log = Path(latency_log) if latency_log else None
        self._sent = read_log(self.latency_log) if self.latency_log else {}
        self.frames = self.cached = self.failed = 0
        self.spent_usd = 0.0

    def score(self, image: bytes, spec: SkillSpec) -> SparkScore:
        start = self.clock()
        label = label_image(self.client, spec, image)
        seconds = self.clock() - start
        self.frames += 1
        self.spent_usd += label.cost_usd
        if label.error:
            self.failed += 1
            raise StudentError(f"Muse Spark gave no label: {label.reason}"[:300])
        key = request_key(self.client.settings.model("labeler"), spec, image)
        measured = True
        if label.cached:
            self.cached += 1
            known = self._sent.get(key)
            seconds = known["latency_s"] if known else seconds
            measured = known is not None
        elif self.latency_log:
            row = {"key": key, "skill": spec.name, "latency_s": round(seconds, 4),
                   "cost_usd": round(label.cost_usd, 6), "ts": round(time.time(), 1)}
            self._sent[key] = row
            self.latency_log.parent.mkdir(parents=True, exist_ok=True)
            with self.latency_log.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, sort_keys=True) + "\n")
        return SparkScore(P_OF_LABEL[label.label], label.label, seconds, cached=label.cached,
                          latency_measured=measured, cost_usd=label.cost_usd)

    def summary(self) -> dict:
        """Frames scored, from the cache, without a label, and dollars spent so far."""
        return {"frames": self.frames, "cached": self.cached, "failed": self.failed,
                "spent_usd": round(self.spent_usd, 4)}
