"""Muse Spark as the per-frame judge, against a fake Meta API: nothing is sent or billed."""

import base64
import io
import json

import httpx
import pytest
from PIL import Image

from backend.eval.spark_scorer import SparkScorer, default_log, request_key, sent_summary
from backend.inference.student import StudentError
from backend.meta.client import MetaAPIError
from backend.skills.spec import SkillSpec, Target
from backend.watch.trigger import Trigger
from backend.watch.watcher import FrameLog, WatchedSkill, Watcher
from tests.test_teach import chat, client_for, png

SPEC = SkillSpec(name="burner_left_on", request="tell me when the washer says End",
                 target=Target(sam_prompt="washer display", description="the washer display"),
                 positive="the display shows End", negatives=["display off"],
                 alert="Laundry's done.", confirm_with_cloud=False)
GREEN, RED, GREY, BLUE = (0, 220, 0), (220, 0, 0), (128, 128, 128), (0, 0, 220)


class FakeSpark:
    """Green frames are yes, red no, grey unsure; blue is refused; status forces an error."""

    def __init__(self, status=200):
        self.calls = 0
        self.status = status

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        if self.status != 200:
            return httpx.Response(self.status, json={"error": "nope"})
        body = json.loads(request.content)
        url = body["messages"][1]["content"][0]["image_url"]["url"]
        colour = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).getpixel((9, 9))
        if colour == BLUE:
            return httpx.Response(200, json={
                "choices": [{"message": {"content": None, "refusal": "no"},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1600, "completion_tokens": 5}})
        label = {GREEN: "yes", RED: "no"}.get(colour[:3], "unsure")
        return httpx.Response(200, json=chat({"label": label, "reason": "fake"}))


def ticking(step=0.75):
    now = [0.0]

    def clock():
        now[0] += step
        return now[0]
    return clock


@pytest.mark.parametrize("colour, p, answer", [(GREEN, 1.0, "yes"), (RED, 0.0, "no"),
                                               (GREY, 0.5, "unsure")])
def test_spark_labels_become_probabilities_and_are_billed_to_the_skill(tmp_path, colour, p,
                                                                       answer):
    fake = FakeSpark()
    client = client_for(tmp_path, fake)
    scorer = SparkScorer(client, clock=ticking())
    score = scorer.score(png(colour), SPEC)
    assert (score.p, score.answer, score.cached) == (p, answer, False)
    assert score.latency_s == pytest.approx(0.75) and score.latency_measured
    assert score.cost_usd > 0
    assert [e["skill"] for e in client.ledger.entries()] == ["burner_left_on"]


def test_a_cached_frame_reports_the_latency_measured_when_it_was_sent(tmp_path):
    fake = FakeSpark()
    log = default_log(tmp_path / "data")
    first = SparkScorer(client_for(tmp_path, fake), latency_log=log, clock=ticking(2.5))
    sent = first.score(png(GREEN), SPEC)
    assert fake.calls == 1 and log.is_file()
    # a new replay, a new process: the reply comes from the cache, the latency from the log
    again = SparkScorer(client_for(tmp_path, fake), latency_log=log, clock=ticking(0.001))
    score = again.score(png(GREEN), SPEC)
    assert fake.calls == 1 and score.cached and score.cost_usd == 0.0
    assert score.latency_s == pytest.approx(2.5) and score.latency_measured
    assert again.summary() == {"frames": 1, "cached": 1, "failed": 0, "spent_usd": 0.0}
    summary = sent_summary(log)
    assert summary["checks"] == 1 and summary["latency_median_s"] == 2.5
    assert summary["usd_per_check"] == pytest.approx(sent.cost_usd, abs=1e-6)
    assert "base64" not in log.read_text()  # only a hash of the request, never the image


def test_a_frame_cached_by_someone_else_has_no_measured_latency(tmp_path):
    fake = FakeSpark()
    client = client_for(tmp_path, fake)
    SparkScorer(client).score(png(RED), SPEC)  # e.g. a watch-mode confirmation
    score = SparkScorer(client, latency_log=tmp_path / "log.jsonl").score(png(RED), SPEC)
    assert score.cached and not score.latency_measured and fake.calls == 1


def test_the_request_key_follows_the_question_and_the_image():
    other = SPEC.model_copy(update={"positive": "the display shows 0:00"})
    keys = {request_key("muse-spark-1.3", SPEC, png(GREEN)),
            request_key("muse-spark-1.3", other, png(GREEN)),
            request_key("muse-spark-1.3", SPEC, png(RED)),
            request_key("muse-spark-1.4", SPEC, png(GREEN))}
    assert len(keys) == 4
    assert request_key("muse-spark-1.3", SPEC.model_copy(update={"alert": "Done."}),
                       png(GREEN)) in keys  # the alert text is not part of the question


def test_a_failed_label_is_a_frame_without_a_score_in_the_watcher(tmp_path):
    fake = FakeSpark()
    scorer = SparkScorer(client_for(tmp_path, fake))
    with pytest.raises(StudentError, match="no label"):
        scorer.score(png(BLUE), SPEC)
    assert scorer.summary()["failed"] == 1 and scorer.summary()["spent_usd"] > 0
    log = FrameLog(tmp_path / "frames.jsonl")
    watcher = Watcher([WatchedSkill(SPEC, Trigger(0.5, 600.0), False)], scorer, log=log)
    for t, colour in enumerate([BLUE, GREEN, GREEN, GREEN]):
        watcher.on_frame(png(colour), t=float(t))
    rows = [json.loads(line) for line in log.path.read_text().splitlines()]
    assert rows[0]["p"] is None and "no label" in rows[0]["error"]
    assert [r["p"] for r in rows[1:]] == [1.0, 1.0, 1.0] and rows[-1]["fires"]


def test_errors_beyond_one_frame_stop_the_replay(tmp_path):
    scorer = SparkScorer(client_for(tmp_path, FakeSpark(status=401)))
    with pytest.raises(MetaAPIError):
        scorer.score(png(GREEN), SPEC)
