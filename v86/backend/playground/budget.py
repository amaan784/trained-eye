"""Spend caps for the playground's live Meta calls: per visitor and per day (UTC).

A visitor is known by a random id in a cookie and by their IP address (an IPv6 address
by its /64, which one home or phone usually gets whole), kept only as a salted hash; each
counts against the visitor cap on its own, so clearing cookies does not
reset it. Before a live run, its estimated cost is reserved against what the visitor
and the day have left; each call the run makes must fit in that reservation, which grows
while the caps allow. When the run ends, the spend the ledger recorded replaces the
reservation. Totals are kept in a small JSON file (the last few days only). One process
owns the file: run the playground with a single worker.
"""

from __future__ import annotations

import datetime
import hashlib
import ipaddress
import json
import secrets
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from backend.meta.ledger import BudgetExceeded

KEEP_DAYS = 7
EPSILON = 1e-9


class CapReached(BudgetExceeded):
    """A playground cap would be passed. Raised before the call, so nothing was sent."""


@dataclass
class Reservation:
    keys: tuple[str, ...]
    day: str
    usd: float = 0.0


def network_of(ip: str) -> str:
    """What one visitor's address counts as: an IPv4 address itself (also when written as
    IPv6), an IPv6 address its /64. Anything else is kept as given."""
    try:
        address = ipaddress.ip_address(ip.strip())
    except ValueError:
        return ip
    if address.version == 6 and address.ipv4_mapped:
        return str(address.ipv4_mapped)
    if address.version == 6:
        return str(ipaddress.ip_network(f"{address}/64", strict=False))
    return str(address)


def utc_today() -> str:
    return datetime.datetime.now(datetime.UTC).date().isoformat()


class Budget:
    def __init__(self, path: Path, visitor_cap_usd: float, daily_cap_usd: float,
                 today: Callable[[], str] = utc_today):
        self.path = Path(path)
        self.visitor_cap_usd = float(visitor_cap_usd)
        self.daily_cap_usd = float(daily_cap_usd)
        self._today = today
        self._lock = threading.Lock()
        self._pending: dict[tuple[str, str], float] = {}  # (day, key or "") -> reserved
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raw = {}
        raw = raw if isinstance(raw, dict) else {}
        days = raw.get("days")
        self._data = {"salt": str(raw.get("salt") or secrets.token_hex(16)),
                      "days": days if isinstance(days, dict) else {}}

    def keys(self, visitor: str, ip: str | None) -> tuple[str, ...]:
        """The budget keys of one visitor: the cookie id and, if known, the hashed IP."""
        out = [f"visitor:{visitor}"]
        if ip:
            ip = network_of(ip)
            digest = hashlib.sha256(f"{self._data['salt']}|{ip}".encode()).hexdigest()
            out.append(f"ip:{digest[:20]}")
        return tuple(out)

    def _used(self, day: str, key: str = "") -> float:
        """Recorded plus reserved spend of a key on a day ("" for the whole day)."""
        record = self._data["days"].get(day) or {}
        spent = record.get("by_visitor", {}).get(key, 0.0) if key else record.get("usd", 0.0)
        return float(spent) + self._pending.get((day, key), 0.0)

    def left(self, keys: tuple[str, ...]) -> dict:
        with self._lock:
            day = self._today()
            used = max((self._used(day, k) for k in keys), default=0.0)
            return {"visitor_left_usd": round(max(0.0, self.visitor_cap_usd - used), 4),
                    "day_left_usd": round(max(0.0, self.daily_cap_usd - self._used(day)), 4)}

    def reserve(self, keys: tuple[str, ...], usd: float) -> Reservation:
        """Reserve a run's estimate, or raise CapReached."""
        reservation = Reservation(tuple(keys), self._today())
        self.grow(reservation, usd)
        return reservation

    def grow(self, reservation: Reservation, usd: float) -> None:
        """Make the reservation at least `usd`, or raise CapReached and leave it as it is."""
        with self._lock:
            extra = usd - reservation.usd
            if extra <= 0:
                return
            day = reservation.day
            if self._used(day) + extra > self.daily_cap_usd + EPSILON:
                raise CapReached(f"Today's budget for live calls (${self.daily_cap_usd:.2f} "
                                 "for everyone) is used up. Try again tomorrow (UTC).")
            if any(self._used(day, k) + extra > self.visitor_cap_usd + EPSILON
                   for k in reservation.keys):
                raise CapReached(f"Your live budget for today (${self.visitor_cap_usd:.2f} "
                                 "per visitor) does not cover another run. Try again tomorrow "
                                 "(UTC).")
            for key in ("", *reservation.keys):
                self._pending[(day, key)] = self._pending.get((day, key), 0.0) + extra
            reservation.usd = usd

    def settle(self, reservation: Reservation, spent_usd: float) -> None:
        """Replace the reservation with the recorded spend and save the totals."""
        with self._lock:
            day = reservation.day
            for key in ("", *reservation.keys):
                left = self._pending.get((day, key), 0.0) - reservation.usd
                if left > EPSILON:
                    self._pending[(day, key)] = left
                else:
                    self._pending.pop((day, key), None)
            reservation.usd = 0.0
            record = self._data["days"].setdefault(day, {"usd": 0.0, "by_visitor": {}})
            record["usd"] = round(record.get("usd", 0.0) + spent_usd, 6)
            by_visitor = record.setdefault("by_visitor", {})
            for key in reservation.keys:
                by_visitor[key] = round(by_visitor.get(key, 0.0) + spent_usd, 6)
            for old in sorted(self._data["days"])[:-KEEP_DAYS]:
                del self._data["days"][old]
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._data, indent=1, sort_keys=True) + "\n", "utf-8")
            tmp.replace(self.path)
