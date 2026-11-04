import xml.etree.ElementTree as ET

import pytest

from backend.eval.timeline import clock, timeline

NS = "{http://www.w3.org/2000/svg}"


def rows(n=120, alerts=(), errors=(), threshold=0.62):
    out = []
    for t in range(n):
        if t in errors:
            out.append({"t": float(t), "skill": "burner_left_on", "threshold": threshold,
                        "p": None, "error": "endpoint down"})
            continue
        p = 0.9 if 60 <= t <= 90 else 0.1
        r = {"t": float(t), "skill": "burner_left_on", "threshold": threshold, "p": p,
             "over": p >= threshold, "fires": t in alerts, "latency_s": 0.3}
        if t in alerts:
            r.update(alert=True, cloud="yes")
        out.append(r)
    return out


def draw(**kwargs):
    svg = timeline("Live timeline: burner_left_on", "clip 1, fine-tuned student",
                   rows(**kwargs), [[58, 92]], alert_text="Laundry's <done> & dry")
    return svg, ET.fromstring(svg)


def texts(root):
    return ["".join(t.itertext()) for t in root.iter(f"{NS}text")]


def test_the_timeline_has_the_line_threshold_ranges_and_spoken_alerts():
    svg, root = draw(alerts=(62,))
    labels = texts(root)
    assert "\"Laundry's <done> & dry\" at 1:02" in labels  # escaped in the file, read back
    assert "&lt;done&gt; &amp; dry" in svg
    assert {"yes-probability", "threshold 0.62", "labelled true", "alert"} <= set(labels)
    assert "time in the clip (minutes:seconds)" in labels and "1:00" in labels
    lines = list(root.iter(f"{NS}polyline"))
    dashed = [p for p in lines if p.get("stroke-dasharray")]
    assert len(dashed) == 1 and dashed[0].get("class") == "t2-s"
    ys = {pt.split(",")[1] for pt in dashed[0].get("points").split()}
    assert len(ys) == 1  # a constant threshold is a straight line
    assert [p.get("class") for p in lines if not p.get("stroke-dasharray")] == ["acc-s"]
    shaded = [r for r in root.iter(f"{NS}rect") if r.get("class") == "wash"]
    assert len(shaded) == 2  # the range and the legend key
    assert shaded[1].find(f"{NS}title").text == "labelled true from 0:58 to 1:32"
    tips = [g.find(f"{NS}title").text for g in root.iter(f"{NS}g")]
    assert tips == ["alert at 1:02: Laundry's <done> & dry (Spark: yes)"]
    style = root.find(f"{NS}style").text
    assert "@media (prefers-color-scheme: dark)" in style and "Inter" in style


def test_student_errors_leave_gaps_and_close_alerts_take_two_label_rows():
    _, root = draw(alerts=(62, 70, 100), errors=(30, 31, 50))
    lines = [p for p in root.iter(f"{NS}polyline") if not p.get("stroke-dasharray")]
    assert len(lines) == 3  # 0-29, 32-49, 51-119
    spoken = [t for t in root.iter(f"{NS}text") if (t.text or "").startswith('"Laundry')]
    assert len(spoken) == 3
    assert spoken[0].get("y") != spoken[1].get("y")  # 8 s apart: the second row
    assert spoken[2].get("text-anchor") == "end"  # near the right edge: it reads leftwards


def test_ranges_outside_the_clip_are_not_drawn_and_empty_logs_fail():
    svg = timeline("t", "s", rows(30), [[100, 120]])
    root = ET.fromstring(svg)
    assert len([r for r in root.iter(f"{NS}rect") if r.get("class") == "wash"]) == 1
    with pytest.raises(ValueError):
        timeline("t", "s", [], [])
    assert (clock(0), clock(62.4), clock(3725)) == ("0:00", "1:02", "62:05")
