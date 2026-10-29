"""Spend ledger for Meta API calls.

Every paid call is checked against the caps before it is sent (using its estimated
cost) and recorded after it returns (using the usage the API reports). Entries are
appended to a JSONL file, one per call. Cache hits cost nothing and are not recorded.

`te spend` prints the totals.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

from backend.config import Price

NO_SKILL = "(none)"


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


class Ledger:
    def __init__(self, path: Path | str, skill_cap_usd: float, global_cap_usd: float):
        self.path = Path(path)
        self.skill_cap_usd = skill_cap_usd
        self.global_cap_usd = global_cap_usd

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

    def check(self, estimate_usd: float, skill: str | None = None) -> None:
        """Raise BudgetExceeded if this call could push spend past a cap."""
        total = self.spent()
        if total + estimate_usd > self.global_cap_usd:
            raise BudgetExceeded(
                f"global cap ${self.global_cap_usd:.2f}: spent ${total:.4f}, "
                f"this call may cost ${estimate_usd:.4f}"
            )
        if skill:
            used = self.spent(skill)
            if used + estimate_usd > self.skill_cap_usd:
                raise BudgetExceeded(
                    f"skill {skill!r} cap ${self.skill_cap_usd:.2f}: spent ${used:.4f}, "
                    f"this call may cost ${estimate_usd:.4f}"
                )

    def record(self, entry: Entry) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(entry), sort_keys=True) + "\n")

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


def format_report(ledger: Ledger) -> str:
    t = ledger.totals()
    remaining = ledger.global_cap_usd - t["actual_usd"]
    lines = [
        f"Meta API spend (ledger: {ledger.path})",
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
        for name, row in rows:
            lines.append(f"    {name:<28} {row['calls']:>5} calls  ${row['usd']:.4f}")
    return "\n".join(lines)
