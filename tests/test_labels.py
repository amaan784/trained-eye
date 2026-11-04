import json

import httpx
import pytest

from backend.config import load_settings
from backend.dataengine.labels import filter_generated, label_image
from backend.meta.client import MetaAPIError, MetaClient
from backend.skills.spec import SkillSpec, Target

SPEC = SkillSpec(name="burner_left_on", request="tell me if I leave the burner on",
                 target=Target(sam_prompt="stove burner",
                               description="the stove burner"),
                 positive="the burner is lit with no pan on it", negatives=["a pan being stirred on it"],
                 alert="The burner is still on.")
JPEG = b"\xff\xd8\xff\xe0frame"


def client_for(tmp_path, replies):
    budgets = []

    def handler(request):
        budgets.append(json.loads(request.content)["max_tokens"])
        reply = replies.pop(0)
        return reply if isinstance(reply, httpx.Response) else httpx.Response(200, json=reply)

    settings = load_settings(env={"TE_DATA_DIR": str(tmp_path), "MODEL_API_KEY": "k"},
                             env_file=None)
    return MetaClient(settings, transport=httpx.MockTransport(handler),
                      sleep=lambda s: None), budgets


def reply(content, finish_reason="stop", refusal=None):
    message = {"content": content, **({"refusal": refusal} if refusal else {})}
    return {"choices": [{"message": message, "finish_reason": finish_reason}],
            "usage": {"prompt_tokens": 1500, "completion_tokens": 100}}


def test_filter_generated_stats():
    rows = [
        {"label": "yes", "spark_label": "yes", "kind": "positive"},
        {"label": "yes", "spark_label": "no", "kind": "positive"},
        {"label": "no", "spark_label": "no", "kind": "hard_negative"},
        {"label": "no", "spark_label": "unsure", "kind": "hard_negative"},
    ]
    kept, stats = filter_generated(rows)
    assert len(kept) == 2
    assert stats == {
        "total": 4, "kept": 2, "unsure": 1, "disagree": 1, "agreement": 0.5,
        "by_kind": {"positive": {"total": 2, "agree": 1, "unsure": 0, "agreement": 0.5},
                    "hard_negative": {"total": 2, "agree": 1, "unsure": 1, "agreement": 0.5}},
    }


def test_label_cut_off_for_length_is_asked_again_with_a_bigger_budget(tmp_path):
    client, budgets = client_for(tmp_path, [reply("", "length"),
                                            reply('{"label": "yes", "reason": "End shows"}')])
    lab = label_image(client, SPEC, JPEG)
    assert (lab.label, lab.error) == ("yes", False) and budgets == [2048, 4096]


@pytest.mark.parametrize("bad", [
    reply(None, refusal="I can't help with that."),
    reply("not json"),
    httpx.Response(400, json={"error": "content_policy_violation"}),
])
def test_one_unusable_label_is_unsure_with_error_not_a_crash(tmp_path, bad):
    client, _ = client_for(tmp_path, [bad])
    lab = label_image(client, SPEC, JPEG)
    assert (lab.label, lab.error) == ("unsure", True) and lab.reason.startswith("no label:")


def test_auth_errors_still_stop_the_run(tmp_path):
    client, _ = client_for(tmp_path, [httpx.Response(401, json={"error": "bad key"})])
    with pytest.raises(MetaAPIError):
        label_image(client, SPEC, JPEG)


def test_rows_whose_labelling_failed_are_counted_apart():
    rows = [{"label": "yes", "spark_label": "yes", "kind": "positive"},
            {"label": "yes", "spark_label": "unsure", "kind": "positive", "label_error": True}]
    kept, stats = filter_generated(rows)
    assert len(kept) == 1 and stats["total"] == 1 and stats["label_errors"] == 1
    assert stats["agreement"] == 1.0


def test_a_label_that_needed_the_bigger_budget_costs_nothing_on_a_rerun(tmp_path):
    client, budgets = client_for(tmp_path, [reply("", "length"),
                                            reply('{"label": "yes", "reason": "End shows"}')])
    first = label_image(client, SPEC, JPEG)
    assert first.cost_usd == pytest.approx(sum(e["actual_usd"]
                                               for e in client.ledger.entries()))
    again = label_image(client, SPEC, JPEG)  # no replies left: any request would fail
    assert (again.label, again.cached, again.cost_usd) == ("yes", True, 0.0)
    assert budgets == [2048, 4096] and len(client.ledger.entries()) == 2


def test_a_label_cut_off_mid_answer_is_asked_again(tmp_path):
    client, budgets = client_for(tmp_path, [reply('{"label": "ye', "length"),
                                            reply('{"label": "no", "reason": "dark"}')])
    lab = label_image(client, SPEC, JPEG)
    assert (lab.label, lab.error, lab.cached) == ("no", False, False)
    assert budgets == [2048, 4096]


def test_a_failed_label_still_reports_what_it_cost(tmp_path):
    client, _ = client_for(tmp_path, [reply("not json")])
    lab = label_image(client, SPEC, JPEG)
    [entry] = client.ledger.entries()
    assert lab.error and lab.cost_usd == entry["actual_usd"] > 0
