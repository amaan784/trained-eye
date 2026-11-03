"""The playground over HTTP (TestClient, fake Meta API): the page and static files, the
data APIs on a fixture folder, a live run, and the visitor and daily caps."""

import re
import time

import httpx
from starlette.testclient import TestClient

from backend.config import load_settings
from backend.meta.client import MetaClient
from backend.playground import skills
from backend.playground.app import COOKIE, STATIC, behind_proxy, create_app
from tests.test_playground_data import make_clip
from tests.test_playground_skills import blur, make_skill
from tests.test_teach import FakeMeta, png

CONDITION = {"condition": "the washer display shows End"}


def make_app(tmp_path, key=True, public=False, **caps):
    data_dir, eval_dir = make_skill(tmp_path / "taught"), tmp_path / "eval"
    make_clip(data_dir, labels={"true": [[2.5, 4.0]]}, blurred=True)
    eval_dir.mkdir()
    (eval_dir / "cost_vs_accuracy.svg").write_text('<svg xmlns="http://www.w3.org/2000/svg"/>')
    (eval_dir / "results.md").write_text("# Results\n")
    fake = FakeMeta()

    def factory():
        env = {"TE_DATA_DIR": str(tmp_path / "live"), **({"MODEL_API_KEY": "sk-t"} if key else {})}
        return MetaClient(load_settings(env=env, env_file=None),
                          transport=httpx.MockTransport(fake), sleep=lambda s: None)

    app = create_app(data_dir, eval_dir, factory, workers=2, public=public, **caps)
    return TestClient(app), fake


def run(client, **extra):
    kf = client.get("/api/teach/keyframes").json()["keyframes"][0]["id"]
    r = client.post("/api/teach", json={**CONDITION, "keyframe": kf, **extra})
    if r.status_code != 202:
        return r.status_code, r.json()
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        view = client.get(f"/api/teach/{r.json()['job']}").json()
        if view["done"]:
            return 202, view
        time.sleep(0.01)
    raise AssertionError("the run did not finish")


def test_page_and_static_files_load(tmp_path):
    client, _ = make_app(tmp_path)
    page = client.get("/")
    assert page.headers["content-type"].startswith("text/html") and COOKIE in page.cookies
    assert "default-src 'self'" in page.headers["content-security-policy"]
    html = page.text
    for words in ("Precomputed", "Faces are blurred", "Frames are not stored", "provenance"):
        assert words in html
    linked = re.findall(r'(?:src|href)="(static/[^"]+)"', html)
    assert sorted(linked) == ["static/app.js", "static/style.css", "static/timeline.js",
                              "static/wall.js"]
    for path in linked:
        r = client.get("/" + path)
        assert r.status_code == 200 and r.content == (STATIC / path[7:]).read_bytes()
    for f in STATIC.iterdir():  # plain wording: no em or en dashes, no remote loads
        if f.is_dir():
            continue
        text = f.read_text(encoding="utf-8")
        assert "—" not in text and "–" not in text and "https://" not in text
    font = client.get("/static/fonts/InterVariable.woff2")  # bundled, not fetched from a CDN
    assert font.status_code == 200 and font.content[:4] == b"wOF2"
    assert client.get("/static/fonts/LICENSE.txt").text.count("SIL Open Font License") >= 1


def test_data_apis_read_the_fixture(tmp_path):
    client, fake = make_app(tmp_path)
    assert [c["name"] for c in client.get("/api/clips").json()["clips"]] == ["washer_1"]
    timeline = client.get("/api/clips/washer_1").json()
    assert timeline["skills"]["washer_done"]["alerts"][0]["text"] == "Your laundry's done."
    assert client.get("/api/clips/nope").status_code == 404
    (skill,) = client.get("/api/skills").json()["skills"]
    detail = client.get(f"/api/skills/{skill['name']}").json()
    assert len(detail["wall"]) == 12 and "found no box" in detail["roi"]["note"]
    assert client.get("/api/skills/nope").status_code == 404
    results = client.get("/api/results").json()
    assert results["results_md"] == "# Results\n" and len(results["charts"]) == 1
    assert len(client.get("/api/examples").json()["examples"][0]["variations"]) == 12
    urls = [timeline["video"], results["charts"][0]["url"], detail["wall"][0]["url"],
            detail["keyframes"][0]["url"]]
    for url in urls:
        r = client.get("/" + url)
        assert r.status_code == 200 and r.headers["x-content-type-options"] == "nosniff"
    assert "default-src 'none'" in client.get("/" + urls[1]).headers["content-security-policy"]
    for bad in ("/media/clips/washer_1/log.jsonl", "/media/skills/washer_done/spec.json",
                "/media/results/..%2F..%2Fledger.jsonl", "/media/"):
        assert client.get(bad).status_code == 404
    assert fake.calls == {}  # reading costs nothing


def test_status_and_a_live_run(tmp_path):
    client, fake = make_app(tmp_path)
    status = client.get("/api/status").json()
    assert status["live"] and status["visitor_left_usd"] == 0.5 and status["estimate_usd"] > 0
    code, view = run(client)
    assert code == 202 and view["count"] == 12 and view["stopped"] is None and fake.edits == 12
    left = client.get("/api/status").json()["visitor_left_usd"]
    assert left == round(0.5 - view["cost_usd"], 4)
    assert client.get("/api/teach/no-such-job").status_code == 404


def test_visitor_cap_stops_calls_before_they_are_made(tmp_path):
    client, fake = make_app(tmp_path, visitor_cap_usd=0.35)
    assert run(client)[0] == 202
    calls = dict(fake.calls)
    code, body = run(client)  # same cookie: the second run would pass the cap
    assert code == 429 and "per visitor" in body["message"] and body["examples"]
    fresh = TestClient(client.app)  # no cookie, same address: still the same visitor
    assert run(fresh)[0] == 429
    assert fake.calls == calls


def test_daily_cap_stops_calls_before_they_are_made(tmp_path):
    client, fake = make_app(tmp_path, visitor_cap_usd=5.0, daily_cap_usd=0.2)
    code, body = run(client)
    assert code == 429 and "Today's budget" in body["message"] and fake.calls == {}


def test_refusal_no_key_and_bad_bodies(tmp_path):
    client, fake = make_app(tmp_path, key=False)
    assert "no Meta API key" in client.get("/api/status").json()["message"]
    code, body = run(client, condition="Sam gets here")
    assert code == 200 and body["refused"] and body["read_back"].startswith("I can't learn")
    code, body = run(client)
    assert code == 503 and body["unavailable"] == "no_key" and body["examples"]
    json_type = {"Content-Type": "application/json"}
    for content in (b"not json", b"[1, 2]", b""):
        assert client.post("/api/teach", content=content, headers=json_type).status_code == 400
    assert client.post("/api/teach", content=b"x" * 5000, headers=json_type).status_code == 413
    assert fake.calls == {}


def test_only_the_page_itself_can_start_a_run(tmp_path):
    client, fake = make_app(tmp_path)
    assert "frame-ancestors 'none'" in client.get("/").headers["content-security-policy"]
    kf = client.get("/api/teach/keyframes").json()["keyframes"][0]["id"]
    body = {**CONDITION, "keyframe": kf}
    other = client.post("/api/teach", json=body, headers={"Sec-Fetch-Site": "cross-site"})
    assert other.status_code == 403
    form = client.post("/api/teach", content=str(body).encode(),
                       headers={"Content-Type": "text/plain"})  # what a form can send
    assert form.status_code == 415
    claimed = client.post("/api/teach", content=b"{}", headers={
        "Content-Type": "application/json", "Content-Length": "999999"})
    assert claimed.status_code == 413  # refused on the header, before reading
    assert fake.calls == {}


def test_odd_since_values_read_as_zero(tmp_path):
    client, _ = make_app(tmp_path)
    _, view = run(client)
    for since in ("abc", "\u00b2", "-3", ""):
        again = client.get(f"/api/teach/{view['job']}", params={"since": since})
        assert again.status_code == 200 and again.json()["count"] == len(again.json()["variations"])


def test_public_copy_serves_no_generated_image_and_no_original_keyframe(tmp_path):
    local, _ = make_app(tmp_path / "local")
    detail = local.get("/api/skills/washer_done").json()
    generated, original = detail["wall"][0]["url"], detail["keyframes"][0]["url"]
    assert local.get("/" + generated).status_code == local.get("/" + original).status_code == 200
    client, fake = make_app(tmp_path / "public", public=True)
    status = client.get("/api/status").json()
    assert status["public"] is True and local.get("/api/status").json()["public"] is False
    for url in (generated, original):  # the same files exist in this copy's data too
        assert (tmp_path / "public/taught/data" / url.removeprefix("media/")).is_file()
        assert client.get("/" + url).status_code == 404
    detail = client.get("/api/skills/washer_done").json()
    assert detail["wall"] == [] and detail["wall_note"] == skills.WALL_NOTE
    assert detail["keyframes"] == []  # nothing blurred with te blur yet
    shown = client.get("/api/examples").json()
    assert shown == {"examples": [], "note": skills.EXAMPLES_NOTE}
    assert client.get("/api/teach/keyframes").json()["keyframes"] == []
    taken = local.get("/api/teach/keyframes").json()["keyframes"][0]["id"]
    r = client.post("/api/teach", json={**CONDITION, "keyframe": taken})
    assert r.status_code == 400 and "Pick one" in r.json()["error"]
    for odd in (["a"], {"x": 1}):
        assert client.post("/api/teach", json={**CONDITION, "keyframe": odd}).status_code == 400
    timeline = client.get("/api/clips/washer_1").json()  # blurred: its video is served
    assert client.get("/" + timeline["video"]).status_code == 200
    assert fake.calls == {}


def test_public_copy_teaches_from_blurred_keyframes(tmp_path):
    client, fake = make_app(tmp_path, public=True)
    data_dir = tmp_path / "taught" / "data"
    folder = data_dir / "skills" / "washer_done"
    blur(folder / "keyframes", folder / "keyframes_blurred")
    picked = tmp_path / "picked"
    picked.mkdir()
    (picked / "kettle.png").write_bytes(png((40, 50, 60)))
    blur(picked, data_dir / "playground" / "keyframes")
    detail = client.get("/api/skills/washer_done").json()
    assert detail["keyframes"] and all(client.get("/" + k["url"]).status_code == 200
                                       for k in detail["keyframes"])
    (offered,) = client.get("/api/teach/keyframes").json()["keyframes"]
    assert offered["id"] == "playground/kettle.png"
    assert client.get("/" + offered["url"]).status_code == 200
    code, view = run(client)
    assert code == 202 and view["count"] == 12 and fake.edits == 12


def test_public_copy_without_a_key_shows_no_examples(tmp_path):
    client, fake = make_app(tmp_path, key=False, public=True)
    picked = tmp_path / "picked"
    picked.mkdir()
    (picked / "kettle.png").write_bytes(png((40, 50, 60)))
    blur(picked, tmp_path / "taught" / "data" / "playground" / "keyframes")
    code, body = run(client)
    assert code == 503 and body["unavailable"] == "no_key"
    assert body["examples"] == [] and body["examples_note"] == skills.EXAMPLES_NOTE
    assert fake.calls == {}


def test_behind_proxy_takes_the_address_the_proxy_added():
    from starlette.applications import Starlette
    from starlette.responses import PlainTextResponse
    from starlette.routing import Route

    async def who(request):
        return PlainTextResponse(request.client.host)

    client = TestClient(behind_proxy(Starlette(routes=[Route("/", who)])))
    assert client.get("/").text == "testclient"  # no header: left as it is
    spoofed = {"X-Forwarded-For": "6.6.6.6, 203.0.113.9"}  # the visitor wrote the first
    assert client.get("/", headers=spoofed).text == "203.0.113.9"
    assert client.get("/", headers={"X-Forwarded-For": " 198.51.100.4 "}).text == "198.51.100.4"
