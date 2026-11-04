import re
import xml.etree.ElementTree as ET

import pytest

from backend.eval.charts import grouped_bars, line_chart, scatter

NS = "{http://www.w3.org/2000/svg}"


def parse(svg):
    root = ET.fromstring(svg)  # raises if it is not well-formed XML
    assert root.tag == f"{NS}svg"
    return root


def texts(root):
    return ["".join(t.itertext()) for t in root.iter(f"{NS}text")]


def test_every_chart_has_the_light_and_dark_style_and_the_font():
    charts = [grouped_bars("Experiment 1", "F1", ["real"], [("F1", [0.5])]),
              line_chart("Experiment 3", "F1", [(25, 0.4, "$0.25")], "images"),
              scatter("Experiment 4", "cost", [{"label": "Spark", "x": 10.8, "y": 0.9}])]
    for svg in charts:
        root = parse(svg)
        style = root.find(f"{NS}style").text
        light, _, dark = style.partition("@media (prefers-color-scheme: dark){")
        assert ".bg{fill:#F4F5F2}" in light and ".bg{fill:#000000}" in dark
        assert "font-family:Archivo, system-ui" in style
        assert "gradient" not in svg.lower()
        assert root.find(f"{NS}title").text and root.find(f"{NS}rect").get("class") == "bg"
        outside_style = svg.replace(style, "")
        assert not re.search(r"#[0-9a-f]{6}", outside_style)  # colours only through classes


def test_grouped_bars_label_every_value_and_skip_missing_ones():
    svg = grouped_bars("Experiment 1: real, synthetic, mixed", "F1 and event recall",
                       ["real only", "synthetic only", "mixed"],
                       [("frame F1", [0.61, 0.0, 0.83]),
                        ("events caught within 5 s", [0.5, None, 1.0])])
    root = parse(svg)
    labels = texts(root)
    for want in ("0.61", "0.83", "0.00", "n/a", "1.00", "mixed", "frame F1",
                 "events caught within 5 s"):
        assert want in labels
    bars = [p for p in root.iter(f"{NS}path")]
    assert len(bars) == 4  # no bar for 0.0 or the missing value
    assert {b.get("class") for b in bars} == {"acc", "neu"}
    assert bars[0].find(f"{NS}title").text == "real only, frame F1: 0.61"
    single = parse(grouped_bars("t", "s", ["a"], [("F1", [0.5])]))
    assert "F1" not in texts(single)  # one series: no legend
    with pytest.raises(ValueError):
        grouped_bars("t", "s", ["a", "b"], [("F1", [0.5])])


def test_text_is_escaped():
    nasty = 'mixed <b>&"x"\x07'
    root = parse(grouped_bars(nasty, "a & b", [nasty], [("F1", [0.5])]))
    assert texts(root)[0] == 'mixed <b>&"x"' and "a & b" in texts(root)
    assert root.find(f"{NS}path/{NS}title").text == 'mixed <b>&"x", F1: 0.50'


def test_line_chart_labels_value_and_cost_on_a_log_scale():
    points = [(300, 0.88, "$3.00"), (25, 0.42, "$0.25"), (100, None, "$1.00"),
              (50, 0.6, "$0.50")]
    root = parse(line_chart("Experiment 3", "F1 against images", points,
                            "synthetic images per skill"))
    labels = texts(root)
    for want in ("0.42", "0.60", "0.88", "$0.25", "$0.50", "$3.00", "n/a", "25", "300"):
        assert want in labels
    line = root.find(f"{NS}polyline")
    xs = [float(pt.split(",")[0]) for pt in line.get("points").split()]
    assert len(xs) == 3 and xs == sorted(xs)
    assert xs[1] - xs[0] == pytest.approx((xs[2] - xs[0]) * 0.279, abs=1)  # log(2)/log(12)
    with pytest.raises(ValueError):
        line_chart("t", "s", [(0, 0.5, "")], "images")


def test_scatter_highlights_the_fine_tuned_point():
    points = [{"label": "fine-tuned Glimmer", "x": 0.012, "y": 0.91, "highlight": True},
              {"label": "base Glimmer", "x": 0.012, "y": 0.9},
              {"label": "Muse Spark per frame", "x": 10.8, "y": 0.86},
              {"label": "not run", "x": None, "y": None}]
    root = parse(scatter("Experiment 4", "accuracy against cost", points))
    labels = texts(root)
    assert "fine-tuned Glimmer" in labels and "F1 0.91, $0.012 per hour" in labels
    assert "F1 0.86, $10.80 per hour" in labels
    assert "not plotted, no numbers yet: not run" in labels
    rings = [c for c in root.iter(f"{NS}circle") if "ring" in (c.get("class") or "")]
    assert [c.get("class") for c in rings].count("mark ring") == 1 and len(rings) == 3
    ys = [float(t.get("y")) for t in root.iter(f"{NS}text")
          if t.text in ("fine-tuned Glimmer", "base Glimmer")]
    assert abs(ys[0] - ys[1]) >= 32  # close points: their labels do not overlap

