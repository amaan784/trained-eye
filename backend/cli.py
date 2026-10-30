"""The `te` command line.

  te spend          print Meta API spend from the ledger
"""

from __future__ import annotations

import argparse
import sys

from backend.config import load_settings
from backend.meta.ledger import Ledger, format_report


def make_ledger(settings) -> Ledger:
    return Ledger(settings.ledger_path, settings.skill_cap_usd, settings.global_cap_usd)


def cmd_spend(args) -> int:
    print(format_report(make_ledger(load_settings())))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="te", description="Trained Eye")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("spend", help="print Meta API spend").set_defaults(func=cmd_spend)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
