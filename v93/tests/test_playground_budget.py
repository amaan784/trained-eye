"""Per-visitor and per-day caps: reservations before calls, recorded spend after."""

import json

import pytest

from backend.meta.ledger import BudgetExceeded
from backend.playground.budget import Budget, CapReached


def budget(tmp_path, visitor=0.5, daily=1.0, day="2026-11-02"):
    return Budget(tmp_path / "playground" / "budget.json", visitor, daily, today=lambda: day)


def test_keys_hash_the_ip(tmp_path):
    b = budget(tmp_path)
    keys = b.keys("abc", "203.0.113.7")
    assert keys[0] == "visitor:abc" and keys[1].startswith("ip:")
    assert "203.0.113.7" not in keys[1]
    assert b.keys("abc", None) == ("visitor:abc",)
    assert b.keys("other", "203.0.113.7")[1] == keys[1]  # same IP, same key
    assert b.keys("abc", "::ffff:203.0.113.7") == keys  # the same address written as IPv6


def test_an_ipv6_visitor_counts_by_their_64(tmp_path):
    b = budget(tmp_path)
    home = b.keys("abc", "2001:db8:1:2:aaaa::1")[1]
    assert b.keys("abc", "2001:db8:1:2:bbbb::9")[1] == home  # a new address, same home
    assert b.keys("abc", "2001:db8:1:3::1")[1] != home


def test_visitor_cap_counts_reservations_and_spend(tmp_path):
    b = budget(tmp_path)
    keys = b.keys("abc", "203.0.113.7")
    first = b.reserve(keys, 0.3)
    with pytest.raises(CapReached, match="per visitor"):  # 0.3 reserved + 0.3 > 0.5
        b.reserve(keys, 0.3)
    b.settle(first, 0.1)  # it cost less than reserved
    assert b.left(keys) == {"visitor_left_usd": 0.4, "day_left_usd": 0.9}
    second = b.reserve(keys, 0.3)
    b.settle(second, 0.2)
    with pytest.raises(CapReached):
        b.reserve(keys, 0.3)
    # a new cookie from the same address is still the same visitor
    with pytest.raises(CapReached):
        b.reserve(b.keys("new-cookie", "203.0.113.7"), 0.3)
    b.reserve(b.keys("someone", "198.51.100.1"), 0.3)  # others still can


def test_daily_cap_covers_everyone(tmp_path):
    b = budget(tmp_path, visitor=5.0, daily=1.0)
    runs = [b.reserve(b.keys(f"v{i}", f"10.0.0.{i}"), 0.3) for i in range(3)]
    with pytest.raises(CapReached, match="Today's budget"):
        b.reserve(b.keys("v9", "10.0.0.9"), 0.3)
    for run in runs:
        b.settle(run, 0.3)
    assert b.left(b.keys("v9", "10.0.0.9"))["day_left_usd"] == 0.1


def test_grow_within_the_caps_only(tmp_path):
    b = budget(tmp_path)
    run = b.reserve(b.keys("abc", None), 0.2)
    b.grow(run, 0.1)  # smaller: nothing changes
    assert run.usd == 0.2
    b.grow(run, 0.45)
    with pytest.raises(BudgetExceeded):  # what the client's ledger check catches
        b.grow(run, 0.6)
    assert run.usd == 0.45
    b.settle(run, 0.0)
    assert b.left(b.keys("abc", None))["visitor_left_usd"] == 0.5


def test_totals_persist_per_day(tmp_path):
    b = budget(tmp_path)
    keys = b.keys("abc", "203.0.113.7")
    b.settle(b.reserve(keys, 0.3), 0.25)
    saved = json.loads((tmp_path / "playground" / "budget.json").read_text())
    assert saved["days"]["2026-11-02"]["usd"] == 0.25
    assert saved["days"]["2026-11-02"]["by_visitor"] == {k: 0.25 for k in keys}
    assert "203.0.113.7" not in json.dumps(saved)

    again = budget(tmp_path)  # a restart reads the file, salt included
    assert again.keys("abc", "203.0.113.7") == keys
    assert again.left(keys)["visitor_left_usd"] == 0.25
    tomorrow = budget(tmp_path, day="2026-11-03")
    assert tomorrow.left(keys) == {"visitor_left_usd": 0.5, "day_left_usd": 1.0}


def test_old_days_are_dropped(tmp_path):
    for d in range(1, 10):
        b = budget(tmp_path, day=f"2026-11-{d:02d}")
        b.settle(b.reserve(b.keys("abc", None), 0.1), 0.1)
    saved = json.loads((tmp_path / "playground" / "budget.json").read_text())
    assert sorted(saved["days"]) == [f"2026-11-{d:02d}" for d in range(3, 10)]


def test_unreadable_file_starts_fresh(tmp_path):
    path = tmp_path / "playground" / "budget.json"
    path.parent.mkdir(parents=True)
    path.write_text("not json")
    b = budget(tmp_path)
    assert b.left(b.keys("abc", None)) == {"visitor_left_usd": 0.5, "day_left_usd": 1.0}
