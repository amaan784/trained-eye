"""Teach it yourself: a visitor's condition and a provided keyframe, taught live.

  local policy check (free) -> Muse Wonder compiles the skill -> 12 Muse Image edits of
  the keyframe (6 showing the condition, 6 near-misses) -> Muse Spark labels each one

Every call goes through MetaClient under the skill name "playground". The run's estimate
is reserved against the visitor's and the day's budget first, and each call must fit in
it (runs.py), so a cap stops calls before they are sent. A run works in a thread that
the page polls. Images go to the browser as data URLs of the bytes Muse Image returned
(provenance kept); the only copy on disk is the client's call cache.
"""

from __future__ import annotations

import logging
import re
import secrets
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from backend.dataengine import labels, synth
from backend.meta.cache import sha256_hex
from backend.meta.client import MetaAPIError, MetaClient, data_url
from backend.meta.ledger import BudgetExceeded
from backend.playground import runs
from backend.playground.budget import Budget, CapReached
from backend.playground.skills import examples, keyframe_bytes
from backend.skills.compiler import compile_skill
from backend.skills.spec import SkillSpec, policy_check, read_back

MAX_CONDITION = 300
WORKERS = 4  # image edits and labels in flight per run
MAX_RUNNING = 4  # runs at once, for everyone
JOB_TTL_S = 900  # finished runs are forgotten after this
OPENERS = re.compile(r"(?:tell|let|remind|notify|alert|warn|when|whenever|if|once|until)\b",
                     re.IGNORECASE)
NO_KEY = "Live teaching is off here: this copy of the playground has no Meta API key."
log = logging.getLogger("trained_eye.playground")


def as_request(condition: str) -> str:
    """"the kettle light turns off" -> "Tell me when the kettle light turns off", so the
    local policy check reads it as a spoken request (it looks for a name after "when")."""
    return condition if OPENERS.match(condition) else f"Tell me when {condition}"


class Teacher:
    """`client_factory()` is called once per run and must return a new MetaClient."""

    def __init__(self, data_dir: Path, client_factory: Callable[[], MetaClient],
                 budget: Budget, *, workers: int = WORKERS):
        self.data_dir, self.client_factory, self.budget = Path(data_dir), client_factory, budget
        self.workers = max(1, workers)
        self._jobs: dict[str, runs.Job] = {}
        self._lock = threading.Lock()

    def unavailable(self, reason: str, message: str) -> dict:
        return {"unavailable": reason, "message": message, "examples": examples(self.data_dir)}

    def start(self, condition, keyframe_id, keys: tuple[str, ...]) -> tuple[int, dict]:
        """(HTTP status, body). 202 starts a run; poll() follows it."""
        condition = " ".join(condition.split()) if isinstance(condition, str) else ""
        if not condition:
            return 400, {"error": "Type a condition first, for example: the kettle light "
                                  "turns off."}
        if len(condition) > MAX_CONDITION:
            return 400, {"error": f"Keep the condition under {MAX_CONDITION} characters."}
        request = as_request(condition)
        refusal = policy_check(request)
        if refusal:
            spec = SkillSpec(name=runs.SKILL, request=request, refused=True, refusal_reason=refusal)
            return 200, {"refused": True, "read_back": read_back(spec), "cost_usd": 0.0}
        image = keyframe_bytes(self.data_dir, keyframe_id)
        if image is None:
            return 400, {"error": "Pick one of the keyframes shown."}
        try:
            client = self.client_factory()
        except Exception as err:  # noqa: BLE001 - e.g. a ConfigError from the settings
            log.error("playground: no Meta client: %s", type(err).__name__)
            return 503, self.unavailable("config", "Live teaching is off here: the "
                                         "settings could not be read.")
        if not client.settings.has_secret("MODEL_API_KEY"):
            return 503, self.unavailable("no_key", NO_KEY)
        estimate, problem = runs.estimate_usd(client.settings), None
        with self._lock:
            now = time.monotonic()
            self._jobs = {k: j for k, j in self._jobs.items()
                          if not j.done or now - j.started < JOB_TTL_S}
            running = [j for j in self._jobs.values() if not j.done]
            if any(set(j.keys) & set(keys) for j in running):
                return 409, {"error": "Your last run is still going: wait for it to finish."}
            if len(running) >= MAX_RUNNING:
                return 503, {"error": "The playground is busy. Try again in a minute."}
            try:
                # Only asks (the run's calls reserve their own); counts every call in flight.
                client.ledger.check(estimate, runs.SKILL, reserve=False)
                reservation = self.budget.reserve(keys, estimate)
            except CapReached as err:
                problem = str(err)
            except BudgetExceeded:
                problem = "The playground's total budget for live calls is used up."
            else:
                job = runs.Job(secrets.token_urlsafe(12), tuple(keys), estimate)
                self._jobs[job.id] = job
        if problem:
            return 429, self.unavailable("cap", problem)
        meter = runs.MeteredLedger(client.ledger, self.budget, reservation)
        client.ledger = meter
        threading.Thread(target=self._run, args=(job, client, meter, request, image),
                         name=f"playground-{job.id}", daemon=True).start()
        return 202, job.view()

    def poll(self, job_id: str, since: int = 0) -> dict | None:
        job = self._jobs.get(job_id)
        return job.view(since) if job else None

    def _run(self, job: runs.Job, client: MetaClient, meter: runs.MeteredLedger, request: str,
             image: bytes) -> None:
        try:
            spec = compile_skill(client, request, [image], name=runs.SKILL)
            job.read_back, job.refused = read_back(spec), spec.refused
            if spec.refused:
                return
            job.spec = {"target": spec.target.description, "look_at": spec.target.sam_prompt,
                        "positive": spec.positive, "negatives": spec.negatives,
                        "alert": spec.alert}
            plan = synth.plan_variations(spec, [sha256_hex(image)], runs.POSITIVES, runs.NEGATIVES,
                                         reasoning_strength=client.settings.image_reasoning)
            pool = ThreadPoolExecutor(self.workers, thread_name_prefix=f"playground-{job.id}")
            try:
                futures = [pool.submit(self._variation, client, spec, v, image) for v in plan]
                for future in as_completed(futures):
                    job.variations.append(future.result())
            finally:
                pool.shutdown(wait=True, cancel_futures=True)  # no new calls after a stop
        except CapReached as err:
            job.stopped = f"Stopped before the next call: {err}"
        except BudgetExceeded:
            job.stopped = "Stopped before the next call: the playground's budget is used up."
        except MetaAPIError as err:
            job.stopped = f"Stopped: a Meta API call failed (status {err.status or 'none'})."
        except Exception as err:  # noqa: BLE001 - the visitor gets a sentence, the log the type
            log.error("playground run %s failed: %s", job.id, type(err).__name__)
            job.stopped = "Stopped by an unexpected error."
        finally:
            meter.close()
            job.cost_usd = round(meter.spent, 4)
            try:
                self.budget.settle(meter.reservation, meter.spent)
            except OSError as err:  # the totals stay in memory; the next settle saves them
                log.error("playground run %s: budget not saved: %s", job.id, type(err).__name__)
            finally:
                job.done = True  # else the visitor and a run slot stay blocked until restart

    def _variation(self, client: MetaClient, spec: SkillSpec, v: synth.Variation,
                   image: bytes) -> dict:
        row = {"id": v.id, "kind": v.kind, "label": v.label, "condition": v.condition,
               "nuisance": v.nuisance, "spark_label": None, "kept": False}
        try:
            result = client.image_edit(image, v.prompt, skill=runs.SKILL)
        except MetaAPIError as err:
            if not synth.skippable(err):
                raise
            return {**row, "error": "Muse Image did not make this one."}
        made = result.images[0]
        lab = labels.label_image(client, spec, made)
        return {**row, "image": data_url(made), "spark_label": lab.label, "reason": lab.reason,
                "kept": lab.label == v.label, "cached": result.info.cached}
