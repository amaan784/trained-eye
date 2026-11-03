"""Spend ledger for Meta API calls.

Every paid call is checked against the caps before it is sent (using its estimated
cost) and recorded after it returns (using the usage the API reports). Entries are
appended to a JSONL file, one per call. Cache hits cost nothing and are not recorded.

The check also reserves the estimate until the call is recorded, so calls in flight
count against the caps together: te serve's phone teach thread and its watch sessions,
the playground's workers, and a te teach in another process. Check-and-reserve and
record-and-release each run under a lock for this process's threads (one per ledger
path) and an flock on <ledger>.lock beside the ledger, which holds the reservations as
JSON. A reservation of a thread or process that ended without recording is dropped.

image_tokens() reads back the usage of image calls, for estimates. `te spend` prints the
totals.
"""

from __future__ import annotations

import contextlib
import json
import os
import threading
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

from backend.config import Price, Settings, shown_path

try:
    import fcntl
except ImportError:  # Windows: this process's threads still share one lock
    fcntl = None

NO_SKILL = "(none)"
IMAGE_TOKENS = 1500  # input tokens per image, a guess until the ledger has measured calls
STALE_S = 3600.0  # a reservation this old belongs to a call that never recorded
_PATHS: dict[str, tuple[threading.Lock, set[str]]] = {}  # path -> (lock, keys held here)
_PATHS_LOCK = threading.Lock()


class BudgetExceeded(RuntimeError):
    pass


def token_cost(price: Price, input_tokens: int, output_tokens: int) -> float:
    """Cost of a token-priced call. Reasoning tokens are billed as output tokens."""
    return (input_tokens * price.input_per_m + output_tokens * price.output_per_m) / 1e6


def unit_cost(price: Price, images: int = 0, audio_s: float = 0.0) -> float:
    return images * price.per_image + audio_s / 3600 * price.per_audio_hour


@dataclass
class Entry:
    model: str
    endpoint: str
    estimate_usd: float
    actual_usd: float
    skill: str = NO_SKILL
    status: str = "ok"
    usage: dict = field(default_factory=dict)
    cache_key: str = ""
    ts: float = field(default_factory=time.time)
    # Token-priced calls: images in the input and the estimated tokens of its text, so
    # image_tokens() can tell what an image costs from the reported input tokens.
    input_images: int = 0
    input_text_tokens: int = 0


class Ledger:
    def __init__(self, path: Path | str, skill_cap_usd: float, global_cap_usd: float):
        self.path = Path(path)
        self.skill_cap_usd = skill_cap_usd
        self.global_cap_usd = global_cap_usd

    @property
    def lock_path(self) -> Path:
        return self.path.with_name(self.path.name + ".lock")

    def entries(self) -> list[dict]:
        if not self.path.is_file():
            return []
        lines = self.path.read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines if line.strip()]

    def spent(self, skill: str | None = None) -> float:
        return sum(
            e["actual_usd"]
            for e in self.entries()
            if skill is None or e.get("skill", NO_SKILL) == skill
        )

    def check(self, estimate_usd: float, skill: str | None = None, *,
              reserve: bool = True) -> None:
        """Raise BudgetExceeded if this call could push spend past a cap, counting what
        is recorded and every other call in flight. Otherwise reserve the estimate for
        this thread's call until record() or release(); reserve=False only asks.

        The client calls this before each call, and again before retrying an attempt that
        may have been billed, with the estimate grown by those attempts: a thread's new
        reservation replaces its last one.
        """
        me = _key()
        with self._reservations() as held:
            others = [r for key, r in held.items() if key != me]
            _fits(estimate_usd, "global cap", self.global_cap_usd, self.spent(),
                  sum(r["usd"] for r in others))
            if skill:
                _fits(estimate_usd, f"skill {skill!r} cap", self.skill_cap_usd,
                      self.spent(skill), sum(r["usd"] for r in others if r["skill"] == skill))
            if reserve:
                held[me] = {"usd": float(estimate_usd), "skill": skill, "ts": time.time()}

    def record(self, entry: Entry) -> None:
        """Append the entry and release this thread's reservation, in one step."""
        with self._reservations() as held:
            try:
                with self.path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(asdict(entry), sort_keys=True) + "\n")
            finally:
                held.pop(_key(), None)

    def release(self, thread: int | None = None) -> None:
        """Drop a reservation without recording (the call failed before record): this
        thread's, or that of `thread` in this process. Free when there is none."""
        key = _key(thread)
        if key not in _path_state(self.path)[1]:
            return
        with self._reservations() as held:
            held.pop(key, None)

    def in_flight_usd(self, skill: str | None = None) -> float:
        """What the calls checked but not recorded yet may cost, every process."""
        with self._reservations() as held:
            return sum(r["usd"] for r in held.values() if skill is None or r["skill"] == skill)

    @contextlib.contextmanager
    def _reservations(self):
        """The reservations of every process, {"<pid>:<thread>": {usd, skill, ts}}, under
        this path's lock and an flock on lock_path; changes are saved on the way out."""
        lock, here = _path_state(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with lock, open(os.open(self.lock_path, os.O_RDWR | os.O_CREAT, 0o644), "r+",
                        encoding="utf-8") as fh:
            if fcntl is not None:
                with contextlib.suppress(OSError):  # no flock here: threads still serialize
                    fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            try:
                saved = json.loads(fh.read() or "{}")
            except ValueError:  # cut short by a crash: the calls it held are over
                saved = {}
            saved = saved if isinstance(saved, dict) else {}
            held = {k: {"usd": float(r["usd"]), "skill": r.get("skill"), "ts": r["ts"]}
                    for k, r in saved.items() if _live(k, r)}
            try:
                yield held
            finally:
                # Changed one key at a time: release() reads it without the lock.
                now = {k for k in held if k.startswith(f"{os.getpid()}:")}
                here.difference_update(here - now)
                here.update(now)
                if held != saved:
                    fh.seek(0)
                    fh.truncate()
                    fh.write(json.dumps(held, sort_keys=True))
                    fh.flush()

    def totals(self) -> dict:
        by_model: dict[str, dict] = defaultdict(lambda: {"calls": 0, "usd": 0.0})
        by_skill: dict[str, dict] = defaultdict(lambda: {"calls": 0, "usd": 0.0})
        estimate = actual = 0.0
        entries = self.entries()
        for e in entries:
            estimate += e["estimate_usd"]
            actual += e["actual_usd"]
            for bucket, name in ((by_model, e["model"]), (by_skill, e.get("skill", NO_SKILL))):
                bucket[name]["calls"] += 1
                bucket[name]["usd"] += e["actual_usd"]
        return {
            "calls": len(entries),
            "errors": sum(1 for e in entries if e.get("status") != "ok"),
            "estimate_usd": estimate,
            "actual_usd": actual,
            "by_model": dict(by_model),
            "by_skill": dict(by_skill),
        }


def _fits(estimate_usd: float, label: str, cap: float, used: float, flight: float) -> None:
    if used + flight + estimate_usd > cap:
        pending = f" and ${flight:.4f} in flight" if flight else ""
        raise BudgetExceeded(f"{label} ${cap:.2f}: spent ${used:.4f}{pending}, "
                             f"this call may cost ${estimate_usd:.4f}")


def _key(thread: int | None = None) -> str:
    return f"{os.getpid()}:{thread if thread is not None else threading.current_thread().ident}"


def _path_state(path: Path) -> tuple[threading.Lock, set[str]]:
    """This process's lock for a ledger path, and the reservations it holds there."""
    with _PATHS_LOCK:
        return _PATHS.setdefault(str(Path(path).resolve()), (threading.Lock(), set()))


def _live(key: str, reservation) -> bool:
    """False for a reservation whose thread (here) or process (elsewhere) has ended, an
    old one, or one that cannot be read."""
    try:
        pid, thread = (int(part) for part in key.split(":"))
        usd, ts = reservation["usd"], reservation["ts"]
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (usd, ts)):
            return False
        if usd < 0 or time.time() - ts > STALE_S:
            return False
    except (AttributeError, KeyError, TypeError, ValueError):
        return False
    if pid == os.getpid():
        return thread in {t.ident for t in threading.enumerate()}
    if os.name != "posix":  # no safe liveness probe: STALE_S ends it
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:  # alive, but another user's
        return True
    return True


def _count(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


@dataclass(frozen=True)
class ImageTokens:
    per_image: int
    calls: int = 0  # ledger calls it was measured over; 0: the guess

    def describe(self) -> str:
        if self.calls:
            return (f"image tokens: {self.per_image} per image, measured over {self.calls} "
                    f"call{'' if self.calls == 1 else 's'} in the ledger")
        return (f"image tokens: {self.per_image} per image, a guess (no compile or label "
                "calls with images in the ledger yet)")


def image_tokens(settings: Settings, ledger=None) -> ImageTokens:
    """Input tokens per image for estimates: the mean over the ledger's ok calls to the
    compiler and labeler models that sent images (their reported input tokens less the
    estimated text, pooled over the images), or IMAGE_TOKENS before there is such a call.
    `ledger` (anything with entries()) defaults to the settings' ledger file."""
    if ledger is None:
        ledger = Ledger(settings.ledger_path, settings.skill_cap_usd, settings.global_cap_usd)
    models = {settings.model("compiler"), settings.model("labeler")}
    try:
        entries = ledger.entries()
    except (OSError, ValueError):
        entries = []
    tokens = images = calls = 0
    for e in entries:
        if not isinstance(e, dict) or e.get("status") != "ok" or e.get("model") not in models:
            continue
        usage, n, text = e.get("usage"), e.get("input_images"), e.get("input_text_tokens")
        if not isinstance(usage, dict) or usage.get("unpriced") or usage.get("missing"):
            continue  # older entries have no input_images; these had no usable usage
        reported = usage.get("prompt_tokens", usage.get("input_tokens"))
        if all(_count(v) for v in (n, text, reported)) and n > 0:
            tokens, images, calls = tokens + reported - text, images + n, calls + 1
    if not calls or tokens <= 0:
        return ImageTokens(IMAGE_TOKENS)
    return ImageTokens(round(tokens / images), calls)


def format_report(ledger: Ledger) -> str:
    t = ledger.totals()
    remaining = ledger.global_cap_usd - t["actual_usd"]
    lines = [
        f"Meta API spend (ledger: {shown_path(ledger.path)})",
        f"  calls: {t['calls']}  errors: {t['errors']}",
        f"  actual: ${t['actual_usd']:.4f}  (estimated before calls: ${t['estimate_usd']:.4f})",
        f"  global cap: ${ledger.global_cap_usd:.2f}  remaining: ${remaining:.4f}",
        f"  per-skill cap: ${ledger.skill_cap_usd:.2f}",
    ]
    for title, key in (("by model", "by_model"), ("by skill", "by_skill")):
        lines.append(f"  {title}:")
        rows = sorted(t[key].items())
        if not rows:
            lines.append("    (no calls yet)")
        width = max([28, *(len(name) for name, _ in rows)])  # skill names run to 40
        for name, row in rows:
            calls = f"{row['calls']:>5} call{'' if row['calls'] == 1 else 's'}"
            lines.append(f"    {name:<{width}} {calls:<12} ${row['usd']:.4f}")
    return "\n".join(lines)
