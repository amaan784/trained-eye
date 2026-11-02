"""One teach run started from the phone: backend.teach.run_teach in a background thread,
with its two questions answered over HTTP (backend/teach_api.py).

  waiting_spend -> running -> waiting_read_back -> running -> done
                                      |                    \\-> stopped or failed
                                      \\-> (no, with a correction) running again

The spend question comes first, in this module, because run_teach's first paid step (the
transcription) comes before its only question (the read-back). A "no" to the read-back
that carries a correction compiles the spec again with it; the transcript comes from the
cache, so only the new compile is paid for, under the spend already agreed. Each
correction replaces the one before, as with te teach --correct. A question nobody
answers within ANSWER_TIMEOUT_S counts as no. When the server stops, stop() refuses the
next paid call, answers an open question no and waits for the call in flight, so the
ledger records it.
"""

from __future__ import annotations

import dataclasses
import logging
import threading
import time
from pathlib import Path

from backend.dataengine.synth import TooManyFailures
from backend.meta.client import MetaAPIError, MetaClient
from backend.meta.ledger import BudgetExceeded
from backend.teach import TeachOptions, run_teach

ANSWER_TIMEOUT_S = 600.0
STOP_WAIT_S = 120.0  # how long stop() waits for a paid call in flight (MetaClient's timeout)
STATES = ("waiting_spend", "running", "waiting_read_back", "done", "stopped", "failed")
FINISHED = ("done", "stopped", "failed")
WAITING = {"spend": "waiting_spend", "read_back": "waiting_read_back"}
MAX_CORRECTION = 300  # compile_skill keeps this many characters

log = logging.getLogger("trained_eye.teach")


class NotWaiting(Exception):
    """The session is not waiting for an answer to that question."""


class ServerStopping(BudgetExceeded):
    """The server is stopping, so no new paid call starts."""


class StopGuard:
    """A client's ledger that refuses every new call once `stopping` is set."""

    def __init__(self, inner, stopping: threading.Event):
        self.inner, self.stopping = inner, stopping

    def __getattr__(self, name):  # spent(), record(), caps: the ledger's own
        return getattr(self.inner, name)

    def check(self, estimate_usd: float, skill: str | None = None) -> None:
        if self.stopping.is_set():
            raise ServerStopping("the server is stopping")
        self.inner.check(estimate_usd, skill)


class TeachSession:
    def __init__(self, client: MetaClient, opts: TeachOptions, out_root: Path, *,
                 estimate: dict, capture: str | None = None, warning: str | None = None,
                 timeout_s: float = ANSWER_TIMEOUT_S):
        self.client, self.opts, self.out_root = client, opts, Path(out_root)
        self.skill, self.timeout_s = opts.skill, timeout_s
        self.state = "waiting_spend"
        self.lines: list[str] = []
        self.info = {"skill": self.skill, "estimate": estimate, "warning": warning,
                     "capture": capture, "read_back": None, "stopped": None, "error": None,
                     "summary": None, "spent_usd": None}
        self._asking: str | None = "spend"  # open from the start, so an early answer counts
        self._answer: tuple[bool, str | None] | None = None
        self._timed_out: str | None = None
        self._correction: str | None = None
        self._cond = threading.Condition()
        self._stopping = threading.Event()
        client.ledger = StopGuard(client.ledger, self._stopping)
        if client._sleep is time.sleep:  # a retry's backoff then ends when the server stops
            client._sleep = self._stopping.wait
        self._thread = threading.Thread(target=self._run, name=f"teach-{self.skill}",
                                        daemon=True)

    def start(self) -> TeachSession:
        self._thread.start()
        return self

    def join(self, timeout: float | None = None) -> bool:
        """Wait for the run to end; True if it has."""
        self._thread.join(timeout)
        return not self._thread.is_alive()

    def stop(self, timeout: float | None = STOP_WAIT_S) -> bool:
        """No new paid call, an open question answered no, then wait for the run to end
        (a call in flight is recorded when it returns). True if it has ended."""
        self._stopping.set()
        with self._cond:
            self._cond.notify_all()
        if self._thread.ident is None:  # never started
            return True
        if self._thread.is_alive():
            log.info("teach %s: waiting up to %g s for the paid call in flight, so the ledger "
                     "records it", self.skill, timeout)
        return self.join(timeout)

    @property
    def finished(self) -> bool:
        return self.state in FINISHED

    def status(self) -> dict:
        with self._cond:
            question = self._asking if self._answer is None else None
            return {**self.info, "state": self.state, "question": question,
                    "lines": list(self.lines)}

    def answer(self, step: str, yes: bool, correction: str | None = None) -> None:
        """Answer the open question. ValueError for an answer that makes no sense,
        NotWaiting if that question is not open."""
        if not isinstance(step, str) or step not in WAITING:
            raise ValueError(f'step must be "spend" or "read_back", not {step!r}')
        correction = " ".join((correction or "").split()) or None
        if correction and (yes or step != "read_back"):
            raise ValueError("a correction goes with the read_back step and yes: false")
        if correction and len(correction) > MAX_CORRECTION:
            raise ValueError(f"the correction is {len(correction)} characters, over the "
                             f"{MAX_CORRECTION} allowed")
        with self._cond:
            if self.finished or self._asking != step or self._answer is not None:
                raise NotWaiting(f"not waiting for a {step} answer (state: {self.state})")
            self._answer = (bool(yes), correction)
            if yes or correction:
                self.state = "running"
            self._cond.notify_all()

    def _ask(self, step: str) -> tuple[bool, str | None]:
        with self._cond:
            if self._asking != step:
                self._asking, self._answer, self.state = step, None, WAITING[step]
            answered = self._cond.wait_for(
                lambda: self._answer is not None or self._stopping.is_set(), self.timeout_s)
            reply = self._answer if self._answer is not None else (False, None)
            if not answered:
                self._timed_out = step
            self._asking = self._answer = None
            return reply

    def _confirm(self, sentence: str) -> bool:
        with self._cond:
            self.info["read_back"] = sentence
        yes, self._correction = self._ask("read_back")
        return yes

    def _say(self, line: str) -> None:
        log.info("teach %s: %s", self.skill, line)
        with self._cond:
            self.lines.append(line)

    def _finish(self, state: str, **info) -> None:
        # Log first: once the state reads finished, the thread does nothing more.
        log.info("teach %s %s: %s", self.skill, state, info.get("stopped") or info.get("error")
                 or "dataset written")
        with self._cond:
            self.info.update(info)
            self.state = state
            self._asking = None  # a run that ended early must not leave a question open

    def _spent_since(self, before: float | None) -> float | None:
        try:
            return round(self.client.ledger.spent(self.skill) - before, 4)
        except (OSError, ValueError, TypeError):  # an unreadable ledger, or none before
            return None

    def _run(self) -> None:
        before = None
        try:
            before = self.client.ledger.spent(self.skill)
            if not self._ask("spend")[0]:
                return self._finish("stopped", spent_usd=0.0, stopped=self._no(
                    "spend", "the spend was not agreed") + " (nothing was spent)")
            while True:
                report = run_teach(self.client, self.opts, self.out_root,
                                   confirm=self._confirm, say=self._say)
                if report.get("stopped") != "not confirmed" or not self._correction:
                    break
                self.opts = dataclasses.replace(self.opts, correction=self._correction)
            stopped = report.get("stopped")
            if stopped == "not confirmed":
                stopped = self._no("read_back", stopped)
            self._finish("stopped" if stopped else "done", stopped=stopped,
                         spent_usd=self._spent_since(before),
                         read_back=report.get("read_back", self.info["read_back"]),
                         summary=summarise(report) if "label_stats" in report else None)
        except ServerStopping:
            self._finish("stopped", stopped="the server stopped before the next paid call",
                         spent_usd=self._spent_since(before))
        except Exception as err:  # the phone is told; the server goes on
            log.exception("teach %s failed", self.skill)
            self._finish("failed", error=failure(err), spent_usd=self._spent_since(before))

    def _no(self, step: str, reason: str) -> str:
        if self._stopping.is_set():
            return f"{reason}: the server stopped"
        if self._timed_out != step:
            return reason
        wait = (f"{self.timeout_s / 60:g} minutes" if self.timeout_s >= 60
                else f"{self.timeout_s:g} seconds")
        return f"{reason}: no answer in {wait}"


def summarise(report: dict) -> dict:
    """What the phone shows at the end: dataset counts and how the labels went."""
    dataset, stats = report.get("dataset") or {}, report.get("label_stats") or {}
    counts = dataset.get("counts") or {}
    return {"examples": dataset.get("examples", 0), "counts": counts,
            "yes": sum(n for k, n in counts.items() if k.endswith("/yes")),
            "no": sum(n for k, n in counts.items() if k.endswith("/no")),
            "generated": stats.get("total"), "generated_kept": stats.get("kept"),
            "agreement": stats.get("agreement"),
            "generation_failed": report.get("generation_failed"),
            "roi_frames_with_box": report.get("roi_frames_with_box"),
            "warning": report.get("warning")}


def failure(err: Exception) -> str:
    """The error for the phone, with what to do about it."""
    if isinstance(err, TooManyFailures):
        return (f"{err}. Failed calls are not cached and are paid for again: look at the "
                "skill folder on the Mac before starting again")
    if isinstance(err, BudgetExceeded):
        return (f"{err}. Raise TE_SKILL_CAP_USD or TE_GLOBAL_CAP_USD on the Mac; finished "
                "calls are cached")
    if isinstance(err, MetaAPIError):
        return f"{err}. Finished calls are cached, so starting again does not pay for them twice"
    if isinstance(err, ValueError):
        return str(err)
    return f"{type(err).__name__}: {err}"
