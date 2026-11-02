"""Teach it yourself against the fake Meta API: 12 labelled variations and no images
outside the cache, refusals for free, caps that stop calls before they are sent."""

import base64
import json
import threading
import time

import httpx
import pytest

from backend.config import load_settings
from backend.meta.client import MetaClient
from backend.playground import api, skills
from backend.playground.budget import Budget
from backend.skills.spec import REFUSALS
from tests.test_playground_skills import make_skill
from tests.test_teach import FakeMeta

KEYS = ("visitor:abc", "ip:1")


def setup(tmp_path, fake=None, key=True, visitor=5.0, **env):
    data_dir = make_skill(tmp_path / "taught")
    fake = fake or FakeMeta()

    def factory():
        values = {"TE_DATA_DIR": str(tmp_path / "live"), **env}
        if key:
            values["MODEL_API_KEY"] = "sk-test"
        return MetaClient(load_settings(env=values, env_file=None),
                          transport=httpx.MockTransport(fake), sleep=lambda s: None)

    budget = Budget(data_dir / "playground" / "budget.json", visitor, 50.0)
    teacher = api.Teacher(data_dir, factory, budget, workers=1)
    return teacher, fake, skills.provided_keyframes(data_dir)[0]["id"]


def finish(teacher, job_id):
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        view = teacher.poll(job_id)
        if view["done"]:
            return view
        time.sleep(0.01)
    raise AssertionError("the run did not finish")


def ledger(tmp_path):
    path = tmp_path / "live" / "ledger.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def test_live_run_gives_12_labelled_variations_and_writes_no_images(tmp_path):
    teacher, fake, kf = setup(tmp_path)
    before = {p for p in tmp_path.rglob("*") if p.is_file()}
    status, body = teacher.start("the washer display shows End", kf, KEYS)
    assert status == 202 and body["planned"] == 12 and body["variations"] == []
    view = finish(teacher, body["job"])
    assert view["stopped"] is None and not view["refused"]
    assert view["read_back"].startswith("I'll watch the digital display")
    assert view["spec"]["alert"] == "Your laundry's done."
    rows = view["variations"]
    assert sorted(r["kind"] for r in rows) == ["hard_negative"] * 6 + ["positive"] * 6
    for r in rows:
        header, b64 = r["image"].split(",", 1)
        assert header == "data:image/png;base64"
        assert base64.b64decode(b64) in fake.returned  # the bytes as made: provenance kept
        assert r["spark_label"] in ("yes", "no") and r["kept"] == (r["spark_label"] == r["label"])
    assert fake.edits == 12 and fake.calls["/chat/completions"] == 13  # compile + 12 labels
    entries = ledger(tmp_path)
    assert {e["skill"] for e in entries} == {"playground"} and len(entries) == 25
    assert view["cost_usd"] == round(sum(e["actual_usd"] for e in entries), 4)
    left = teacher.budget.left(KEYS)["visitor_left_usd"]
    assert left == pytest.approx(5.0 - view["cost_usd"], abs=1e-4)

    new = {p for p in tmp_path.rglob("*") if p.is_file()} - before
    cache = tmp_path / "live" / "cache"
    outside = {p.relative_to(tmp_path).as_posix() for p in new if cache not in p.parents}
    assert outside == {"live/ledger.jsonl", "taught/data/playground/budget.json"}
    assert sum(1 for p in new if p.suffix == ".png") == 12  # one per variation, cached


def test_policy_refusal_costs_nothing(tmp_path):
    teacher, fake, kf = setup(tmp_path, key=False)
    for condition, rule in (("Sam gets here", "person"), ("my pills are low", "health")):
        status, body = teacher.start(condition, kf, KEYS)
        assert status == 200 and body["refused"] and body["cost_usd"] == 0.0
        assert body["read_back"] == f"I can't learn that one. {REFUSALS[rule]}"
    assert fake.calls == {} and ledger(tmp_path) == []
    assert not (tmp_path / "taught" / "data" / "playground" / "budget.json").exists()


def test_bad_input_and_the_ledger_cap_send_nothing(tmp_path):
    teacher, fake, kf = setup(tmp_path, TE_SKILL_CAP_USD="0.1")
    for condition, keyframe in (("", kf), ("   ", kf), ("x" * 301, kf), (None, kf),
                                ("the kettle light is off", "nope"),
                                ("the kettle light is off", None)):
        assert teacher.start(condition, keyframe, KEYS)[0] == 400
    status, body = teacher.start("the kettle light is off", kf, KEYS)  # over the skill cap
    assert status == 429 and "total budget" in body["message"]
    assert [e["skill"] for e in body["examples"]] == ["washer_done"]  # precomputed instead
    assert fake.calls == {}


class Costly(FakeMeta):
    """Labels that report far more tokens than their budget, so spend outruns the estimate."""

    def __call__(self, request):
        response = super().__call__(request)
        if b"frame_label" not in request.content:
            return response
        usage = {"prompt_tokens": 1600, "completion_tokens": 40000}
        return httpx.Response(200, json={**response.json(), "usage": usage})


def test_cap_stops_a_run_between_calls(tmp_path):
    teacher, fake, kf = setup(tmp_path, fake=Costly(), visitor=0.35)
    view = finish(teacher, teacher.start("the kettle light is off", kf, KEYS)[1]["job"])
    assert view["stopped"].startswith("Stopped before the next call")
    assert "per visitor" in view["stopped"]
    assert view["count"] == 2 and fake.edits == 2  # the third edit was never sent
    spent = sum(e["actual_usd"] for e in ledger(tmp_path))
    assert teacher.budget.left(KEYS)["day_left_usd"] == pytest.approx(50 - spent, abs=1e-4)


def test_one_run_at_a_time_per_visitor(tmp_path):
    gate, slow = threading.Event(), FakeMeta()
    teacher, _, kf = setup(tmp_path, fake=lambda request: gate.wait(10) and slow(request))
    first = teacher.start("the kettle light is off", kf, KEYS)[1]["job"]
    assert teacher.start("the kettle light is off", kf, ("visitor:abc",))[0] == 409
    status, other = teacher.start("the kettle light is off", kf, ("visitor:xyz", "ip:2"))
    assert status == 202
    gate.set()
    assert finish(teacher, first)["count"] == 12 and finish(teacher, other["job"])["count"] == 12


def test_a_budget_file_that_cannot_be_saved_does_not_block_the_visitor(tmp_path, monkeypatch):
    teacher, _, kf = setup(tmp_path)

    def full_disk(reservation, spent):
        raise OSError("No space left on device")

    monkeypatch.setattr(teacher.budget, "settle", full_disk)
    view = finish(teacher, teacher.start("the kettle light is off", kf, KEYS)[1]["job"])
    assert view["done"] and view["count"] == 12
    assert teacher.start("the kettle light is off", kf, KEYS)[0] == 202  # not 409
