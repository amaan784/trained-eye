import xml.etree.ElementTree as ET

from backend.eval.svg import (
    DARK,
    LIGHT,
    STYLE,
    Box,
    bar_path,
    document,
    esc,
    legend,
    nice_ticks,
    usd,
)


def test_escaping_keeps_markup_out_and_drops_what_xml_cannot_hold():
    assert esc("it's <ok> & \"fine\"") == "it&#x27;s &lt;ok&gt; &amp; &quot;fine&quot;"
    assert esc("bell\x07 nul\x00 tab\t") == "bell nul tab\t"
    svg = document(200, 100, "a </svg> b", 'x="1"', [])
    root = ET.fromstring(svg)
    assert root.find("{http://www.w3.org/2000/svg}title").text == "a </svg> b"


def test_one_style_light_by_default_and_dark_under_the_media_query():
    light, _, dark = STYLE.partition("@media (prefers-color-scheme: dark){")
    assert dark.endswith("}") and "<" not in STYLE and "&" not in STYLE
    for key in ("bg", "t1", "acc"):
        assert f".{key}{{fill:{LIGHT[key]}}}" in light
        assert f".{key}{{fill:{DARK[key]}}}" in dark
    assert LIGHT["acc"] != DARK["acc"] and ".ring{stroke:#1a1a19}" in dark


def test_ticks_money_and_marks():
    assert nice_ticks(1) == [0, 0.25, 0.5, 0.75, 1.0]
    assert nice_ticks(12.4) == [0, 5, 10, 15] and nice_ticks(0) == [0.0, 1.0]
    assert nice_ticks(0.05) == [0, 0.02, 0.04, 0.06]
    assert (usd(0.006), usd(3.95), usd(0)) == ("$0.006", "$3.95", "$0.00")
    box = Box(10, 20, 100, 50)
    assert (box.bottom, box.x(0.5), box.y(1.0), box.y(2.0), box.y(0)) == (70, 60, 20, 20, 70)
    assert bar_path(0, 10, 24, 50).startswith("M0.0,50.0V14.0Q0.0,10.0 4.0,10.0")
    assert "V10.0Q0.0,9.0 1.0,9.0" in bar_path(0, 9, 24, 10)  # short bar: rounding fits
    keys = "".join(legend(0, 10, [("box", "acc", "F1"), ("dash", "t2-s", "threshold")]))
    assert 'class="acc"' in keys and 'stroke-dasharray="5 4"' in keys
