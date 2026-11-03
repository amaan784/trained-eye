"""Reading SAM 3.1's segmentation text: the object boxes and their confidence.

SAM 3.1 answers in one Responses API output-text lane, one line per frame, in the
special-token grammar of Meta's SAM 3 protocol (protocol/sam3.md in
github.com/meta-models/meta-sam, read at commit 9348a8a; this is our own implementation
of that grammar, not Meta's parser). For an image the line is

    <0f>0<|box;x1=10;y1=20;x2=14;y2=24;w=100;h=80;c=0.75|><|mask;x=0;y=0;data=5,5,!...|>

a frame marker, then comma-separated segments: an object id, then exactly one box token
and one mask token in either order. Box corners are inclusive source pixels in a w by h
picture; the records here use half-open boxes (right = x2 + 1), as the protocol's own
normalized records do. Mask payloads are kept as text: the marker and alphabet are
checked, but they are never decoded (the region of interest needs the box only).

Rules followed from the protocol: unknown fields and tokens are ignored with a warning,
a confidence that is not a number from 0 to 1 is dropped with a warning, a malformed
segment drops the rest of its line (earlier segments stay), an image line must be frame
0, and a line that does not start with "<" is prose and is skipped.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

FRAME = re.compile(r"<(\d+)f((?:;[^>]*)?)>")
TOKEN = re.compile(r"<\|([A-Za-z][A-Za-z0-9_.-]*)((?:;.*?)?)\|>")
OBJECT_ID = re.compile(r"\d+")
SIGNED = re.compile(r"-?\d+")
UNSIGNED = re.compile(r"\d+")
CONFIDENCE = re.compile(r"-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
# base85 digits: printable ASCII "!" to "{" without the grammar's delimiters
PAYLOAD_DIGITS = frozenset(chr(c) for c in range(ord("!"), ord("{") + 1)) - set("\"\\,;<>|")
BOX_FIELDS, MASK_FIELDS = {"x1", "y1", "x2", "y2", "w", "h", "c"}, {"x", "y", "data", "c"}


class Malformed(ValueError):
    """The rest of the line is dropped."""


@dataclass(frozen=True)
class Record:
    object_id: str
    frame: int | None  # None for an image
    box: tuple[int, int, int, int]  # left, top, right, bottom; right and bottom exclusive
    size: tuple[int, int]  # the w and h the box is in
    confidence: float | None  # the box token's c (None when absent: not zero)
    mask_confidence: float | None
    mask_encoding: str  # "lossless" or "one_bit"
    mask_payload: str = field(repr=False)


@dataclass
class Parsed:
    records: list[Record] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)  # "line 3: ...": data was dropped
    warnings: list[str] = field(default_factory=list)  # input was ignored, records kept


def parse(text: str, media: str = "image") -> Parsed:
    """Every record in SAM's output text. `media` is "image" or "video"."""
    out, seen = Parsed(), set()
    for number, line in enumerate(text.split("\n"), 1):
        line = line.rstrip("\r")
        if not line.strip() or not line.lstrip().startswith("<"):
            continue  # an empty line, or prose around the records
        warnings: list[tuple[str, str]] = []
        try:
            _line(line, media, out.records, warnings)
        except Malformed as err:
            out.errors.append(f"line {number}: {err}")
            continue  # a malformed segment reports no warnings
        for key, text_ in warnings:
            if key not in seen:  # each warning once per stream
                seen.add(key)
                out.warnings.append(f"line {number}: {text_}")
    return out


def _line(line: str, media: str, records: list[Record], warnings: list) -> None:
    """Append the line's records; a malformed segment raises, keeping the ones before it."""
    head = FRAME.match(line)
    if not head:
        raise Malformed("no frame marker")
    frame = int(head.group(1))
    if head.group(2):
        warnings.append(("frame-fields", "fields on the frame marker ignored"))
    if media == "image" and frame != 0:
        raise Malformed("an image record must be frame 0")
    pos = head.end()
    if pos == len(line):
        raise Malformed("a frame marker with no records")
    while pos < len(line):
        if line.startswith(",", pos):
            pos += 1
        ident = OBJECT_ID.match(line, pos)
        if not ident:
            raise Malformed("an object id must be a decimal number")
        pos, tokens = ident.end(), []
        while (token := TOKEN.match(line, pos)):
            tokens.append((token.group(1), token.group(2)))
            pos = token.end()
        if pos < len(line) and not line.startswith(",", pos):
            raise Malformed("text between tokens")
        records.append(_record(ident.group(), None if media == "image" else frame, tokens,
                               warnings))


def _record(object_id: str, frame: int | None, tokens: list, warnings: list) -> Record:
    boxes = [body for name, body in tokens if name == "box"]
    masks = [body for name, body in tokens if name == "mask"]
    for name, _ in tokens:
        if name not in ("box", "mask"):
            warnings.append((f"token:{name}", f"token {name!r} ignored"))
    if len(boxes) != 1 or len(masks) != 1:
        raise Malformed("a record needs exactly one box and one mask")
    box, box_c = _fields(boxes[0], BOX_FIELDS, "box", warnings)
    mask, mask_c = _fields(masks[0], MASK_FIELDS, "mask", warnings)
    try:
        x1, y1, x2, y2 = (_integer(box[k], SIGNED) for k in ("x1", "y1", "x2", "y2"))
        w, h = (_integer(box[k], UNSIGNED) for k in ("w", "h"))
    except KeyError as err:
        raise Malformed(f"box field {err.args[0]} missing") from None
    if not (w > 0 and h > 0 and 0 <= x1 <= x2 < w and 0 <= y1 <= y2 < h):
        raise Malformed("box coordinates outside the picture or reversed")
    if mask.get("x") != "0" or mask.get("y") != "0" or "data" not in mask:
        raise Malformed("a mask needs x=0, y=0 and data")
    encoding, payload = _payload(mask["data"])
    return Record(object_id, frame, (x1, y1, x2 + 1, y2 + 1), (w, h), box_c, mask_c, encoding,
                  payload)


def _fields(body: str, known: set[str], token: str, warnings: list) -> tuple[dict, float | None]:
    values, confidences = {}, []
    for part in body.split(";"):
        key, has_value, value = part.partition("=")
        key, value = key.strip(), value.strip()
        if not key:
            continue
        if key == "c":
            confidences.append(value if has_value else None)
            continue
        if key not in known:
            warnings.append((f"field:{token}.{key}", f"{token} field {key!r} ignored"))
            continue
        if key in values:
            raise Malformed(f"{token} field {key} repeated")
        values[key] = value if has_value else None
    confidence = None
    if confidences:
        confidence = _confidence(confidences[0]) if len(confidences) == 1 else None
        if confidence is None:
            warnings.append(("confidence",
                             "a confidence that is not a number from 0 to 1 was ignored"))
    return values, confidence


def _integer(value: str | None, pattern: re.Pattern) -> int:
    if value is None or not pattern.fullmatch(value):
        raise Malformed(f"{value!r} is not a whole number")
    return int(value)


def _confidence(value: str | None) -> float | None:
    if value is None or not CONFIDENCE.fullmatch(value):
        return None
    number = float(value)
    return number + 0.0 if math.isfinite(number) and 0 <= number <= 1 else None  # -0 is 0


def _payload(data: str | None) -> tuple[str, str]:
    """(encoding, payload) from "height,width,payload", checked but not decoded."""
    height, _, rest = (data or "").partition(",")
    width, _, payload = rest.partition(",")
    if not (UNSIGNED.fullmatch(height) and UNSIGNED.fullmatch(width)
            and int(height) > 0 and int(width) > 0):
        raise Malformed("mask data must start with a positive height and width")
    if len(payload) < 6 or payload[0] not in "~!":
        raise Malformed("mask payload must start with ~ or ! and a length prefix")
    if not set(payload[1:]) <= PAYLOAD_DIGITS:
        raise Malformed("mask payload has characters outside its alphabet")
    return ("lossless" if payload[0] == "~" else "one_bit"), payload
