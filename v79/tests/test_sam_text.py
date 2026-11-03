"""SAM 3.1's segmentation text (backend/dataengine/sam_text.py): records, boxes and
confidence as Meta's SAM 3 protocol defines them. The examples are written for these
tests; mask payloads are only checked, never decoded, so any well-formed one will do."""

import pytest

from backend.dataengine.sam_text import parse

MASK = "<|mask;x=0;y=0;data=4,6,!!!!#Kq|>"


def box(x1, y1, x2, y2, w=64, h=48, extra=""):
    return f"<|box;x1={x1};y1={y1};x2={x2};y2={y2};w={w};h={h}{extra}|>"


def test_one_object_with_its_confidence():
    parsed = parse(f"<0f>0{box(4, 6, 20, 30, extra=';c=0.82')}{MASK}\n")
    [record] = parsed.records
    assert record.object_id == "0" and record.frame is None
    assert record.box == (4, 6, 21, 31)  # inclusive corners become a half-open box
    assert record.size == (64, 48) and record.confidence == 0.82
    assert record.mask_encoding == "one_bit" and parsed.errors == parsed.warnings == []


def test_several_objects_commas_and_order_of_fields_and_tokens():
    line = (f"<0f>,3{box(0, 0, 9, 9)}{MASK},12<|mask;data=2,2,~!!!!(;y=0;x=0|>"
            "<|box;h=48;w=64;y2=40;x2=63;y1=1;x1=-0|>\n")
    parsed = parse(line)
    assert [(r.object_id, r.box) for r in parsed.records] == [("3", (0, 0, 10, 10)),
                                                             ("12", (0, 1, 64, 41))]
    assert parsed.records[1].mask_encoding == "lossless" and parsed.records[1].confidence is None


def test_unknown_fields_and_tokens_are_ignored_once_with_a_warning():
    line = f"<0f;seq=4>0{box(1, 1, 5, 5, extra=';area=25')}<|note;why=test|>{MASK}"
    parsed = parse(f"{line}\n{line}\n")
    assert len(parsed.records) == 2 and parsed.errors == []
    assert len(parsed.warnings) == 3 and all(w.startswith("line 1:") for w in parsed.warnings)


@pytest.mark.parametrize("c", ["1.5", "-0.1", "", "nan", "0.3;c=0.4", "high"])
def test_a_bad_confidence_is_dropped_and_the_record_kept(c):
    parsed = parse(f"<0f>0{box(1, 1, 5, 5, extra=';c=' + c)}{MASK}")
    assert parsed.records[0].confidence is None and parsed.records[0].box == (1, 1, 6, 6)
    assert parsed.warnings and parsed.errors == []


@pytest.mark.parametrize("c, value", [("0", 0.0), ("-0", 0.0), ("1", 1.0), (".5", 0.5),
                                      ("6.25e-1", 0.625)])
def test_confidence_spellings(c, value):
    assert parse(f"<0f>0{box(1, 1, 5, 5, extra=';c=' + c)}{MASK}").records[0].confidence == value


@pytest.mark.parametrize("segment", [
    box(5, 5, 4, 9) + MASK,  # reversed
    box(0, 0, 64, 10) + MASK,  # x2 must be inside a 64 wide picture
    box(0, 0, 5, 5, w=0) + MASK,
    "<|box;x1=0;y1=0;x2=5;w=64;h=48|>" + MASK,  # y2 missing
    "<|box;x1=0;x1=0;y1=0;x2=5;y2=5;w=64;h=48|>" + MASK,  # repeated key
    "<|box;x1=a;y1=0;x2=5;y2=5;w=64;h=48|>" + MASK,
    box(0, 0, 5, 5),  # no mask
    box(0, 0, 5, 5) + box(0, 0, 5, 5) + MASK,  # two boxes
    box(0, 0, 5, 5) + "stray" + MASK,
    box(0, 0, 5, 5) + "<|mask;x=1;y=0;data=4,6,!!!!#Kq|>",
    box(0, 0, 5, 5) + "<|mask;x=0;y=0;data=0,6,!!!!#Kq|>",
    box(0, 0, 5, 5) + "<|mask;x=0;y=0;data=4,6,#Kq!!!|>",  # no ~ or ! marker
    box(0, 0, 5, 5) + "<|mask;x=0;y=0;data=4,6,!!!!}Kq|>",  # } is not a base85 digit
])
def test_a_malformed_segment_drops_the_rest_of_its_line(segment):
    good = f"0{box(1, 1, 5, 5)}{MASK}"
    parsed = parse(f"<0f>{good},7{segment},8{box(2, 2, 6, 6)}{MASK}\n<0f>{good}\n")
    assert [r.object_id for r in parsed.records] == ["0", "0"]  # the 8 after it is lost
    assert len(parsed.errors) == 1 and parsed.errors[0].startswith("line 1:")


def test_prose_bare_headers_ids_and_frames():
    text = ("Here is what I found:\n<0f>\n<0f>x" + box(1, 1, 5, 5) + MASK + "\n"
            "<3f>0" + box(1, 1, 5, 5) + MASK + "\n\n")
    parsed = parse(text)
    assert parsed.records == [] and len(parsed.errors) == 3  # prose is skipped silently
    assert parse("").records == [] and parse("").errors == []  # nothing found is fine


def test_video_lines_keep_their_frames_in_any_order():
    text = "".join(f"<{n}f>5{box(n, 1, n + 4, 5)}{MASK}\n" for n in (9, 10, 8))
    parsed = parse(text, "video")
    assert [(r.object_id, r.frame, r.box[0]) for r in parsed.records] == [
        ("5", 9, 9), ("5", 10, 10), ("5", 8, 8)]
