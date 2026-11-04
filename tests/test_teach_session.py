"""The teach session behind the phone's teach API (backend/teach_session.py): questions,
timeouts and failures, against the fake Meta API of test_teach. Each test waits for the
session's thread with a timeout."""

import time

import pytest

from backend.teach import TeachOptions
from backend.teach_session import NotWaiting, ServerStopping, TeachSession, failure, summarise
from tests.test_teach import FakeMeta, client_for, clip, wav

REQUEST = "let me know when the washer says done"


def session(tmp_path, fake, timeout_s=5.0, **opts):
    opts = {"request": REQUEST, "positives": 6, "negatives": 6, **opts}
    return TeachSession(client_for(tmp_path, fake),
                        TeachOptions(skill="burner_left_on", clip=clip(tmp_path / "clip"), **opts),
                        tmp_path / "skills", estimate={"total": 0.1}, timeout_s=timeout_s)


def test_an_unanswered_spend_question_counts_as_no(tmp_path):
    fake = FakeMeta()
    s = session(tmp_path, fake, timeout_s=0.05, request=None, audio=wav(tmp_path / "r.wav"))
    assert s.status()["question"] == "spend"
    assert s.start().join(5)
    status = s.status()
    assert status["state"] == "stopped" and status["question"] is None
    assert status["stopped"] == ("the spend was not agreed: no answer in 0.05 seconds "
                                 "(nothing was spent)")
    assert fake.calls == {} and status["spent_usd"] == 0.0


def test_an_unanswered_read_back_counts_as_no(tmp_path):
    fake = FakeMeta()
    s = session(tmp_path, fake, timeout_s=0.3)
    s.answer("spend", True)  # before the thread asks: the question is open from the start
    assert s.start().join(5)
    status = s.status()
    assert status["state"] == "stopped" and status["read_back"].startswith("I'll watch")
    assert status["stopped"] == "not confirmed: no answer in 0.3 seconds"
    assert fake.calls == {"/chat/completions": 1}  # the compile only
    assert any(line.startswith("keyframes:") for line in status["lines"])


def test_answers_must_fit_the_open_question(tmp_path):
    s = session(tmp_path, FakeMeta()).start()
    with pytest.raises(NotWaiting, match="not waiting for a read_back answer"):
        s.answer("read_back", True)
    with pytest.raises(ValueError, match="goes with the read_back step"):
        s.answer("spend", False, "the lid")
    with pytest.raises(ValueError, match="goes with the read_back step"):
        s.answer("read_back", True, "the lid")
    with pytest.raises(ValueError, match="over the 300 allowed"):
        s.answer("read_back", False, "x" * 301)
    with pytest.raises(ValueError, match="step must be"):
        s.answer("later", True)
    with pytest.raises(ValueError, match="step must be"):
        s.answer(["spend"], True)
    s.answer("spend", False)
    with pytest.raises(NotWaiting):
        s.answer("spend", True)  # already answered
    assert s.join(5) and s.status()["state"] == "stopped"


def test_a_request_the_compiler_refuses_stops_without_a_paid_call(tmp_path):
    fake = FakeMeta()
    s = session(tmp_path, fake, request="Tell me when Sam gets here").start()
    s.answer("spend", True)
    assert s.join(5)
    status = s.status()
    assert status["state"] == "stopped" and status["stopped"] == "refused"
    assert status["read_back"].startswith("I can't learn that one.") and fake.calls == {}


def test_a_failed_run_says_why_and_what_a_rerun_costs(tmp_path):
    fake = FakeMeta(reject_edit=lambda body: True)
    s = session(tmp_path, fake, positives=20, negatives=20).start()
    s.answer("spend", True)
    deadline = time.monotonic() + 5
    while s.status()["question"] != "read_back":
        assert time.monotonic() < deadline, s.status()
        time.sleep(0.01)
    s.answer("read_back", True)
    assert s.join(5)
    status = s.status()
    assert status["state"] == "failed" and "image edits failed so far" in status["error"]
    assert "paid for again" in status["error"] and status["summary"] is None
    assert status["spent_usd"] > 0  # the compile and SAM calls were paid for


def test_summary_and_failure_wording():
    report = {"dataset": {"examples": 5, "counts": {"a/real/no": 2, "a/synthetic/yes": 3}},
              "label_stats": {"total": 4, "kept": 3, "agreement": 0.75},
              "generation_failed": 1, "roi_frames_with_box": 20}
    assert summarise(report) == {
        "examples": 5, "counts": {"a/real/no": 2, "a/synthetic/yes": 3}, "yes": 3, "no": 2,
        "generated": 4, "generated_kept": 3, "agreement": 0.75, "generation_failed": 1,
        "roi_frames_with_box": 20, "warning": None}
    assert failure(ValueError("the request is too long")) == "the request is too long"
    assert failure(KeyError("x")) == "KeyError: 'x'"


def test_a_run_that_fails_before_the_spend_question_closes_it(tmp_path, monkeypatch):
    s = session(tmp_path, FakeMeta())

    def unreadable(skill=None):
        raise ValueError("a half-written ledger line")

    monkeypatch.setattr(s.client.ledger.inner, "spent", unreadable)
    assert s.start().join(5)
    status = s.status()
    assert status["state"] == "failed" and status["question"] is None
    with pytest.raises(NotWaiting):
        s.answer("spend", True)  # a dead session cannot be started again
    assert s.status()["state"] == "failed"


def test_stop_answers_an_open_question_no(tmp_path):
    fake = FakeMeta()
    s = session(tmp_path, fake, timeout_s=30).start()
    assert s.stop(5)
    status = s.status()
    assert status["state"] == "stopped" and status["question"] is None
    assert status["stopped"] == ("the spend was not agreed: the server stopped "
                                 "(nothing was spent)")
    with pytest.raises(ServerStopping):
        s.client.ledger.check(0.01, "burner_left_on")
    assert fake.calls == {}


def test_stop_refuses_the_next_paid_call(tmp_path):
    fake = FakeMeta()
    s = session(tmp_path, fake, timeout_s=30)
    s.answer("spend", True)
    assert s.stop(5)  # not started: returns at once
    assert s.start().join(5)
    status = s.status()
    assert status["state"] == "stopped" and fake.calls == {}
    assert status["stopped"] == "the server stopped before the next paid call"
    assert status["spent_usd"] == 0.0


def test_stop_cuts_a_rate_limited_retry_short(tmp_path):
    import threading

    import httpx

    from backend.config import load_settings
    from backend.meta.client import MetaClient

    sent, first = [], threading.Event()

    def busy(request):
        sent.append(request.url.path)
        first.set()
        return httpx.Response(429, headers={"retry-after": "60"}, json={"error": "slow down"})

    settings = load_settings(env={"TE_DATA_DIR": str(tmp_path / "data"),
                                  "MODEL_API_KEY": "sk-test"}, env_file=None)
    client = MetaClient(settings, transport=httpx.MockTransport(busy))  # real backoff sleeps
    s = TeachSession(client, TeachOptions(skill="burner_left_on", clip=clip(tmp_path / "clip"),
                                          request=REQUEST, positives=6, negatives=6),
                     tmp_path / "skills", estimate={"total": 0.1}, timeout_s=30)
    s.answer("spend", True)
    s.start()
    assert first.wait(5)
    began = time.monotonic()
    assert s.stop(10) and time.monotonic() - began < 5  # not the 60 s the server asked for
    status = s.status()
    assert status["state"] == "stopped" and len(sent) == 1  # no attempt after the stop
    assert status["stopped"] == "the server stopped before the next paid call"
