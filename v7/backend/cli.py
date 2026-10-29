"""The `te` command line.

  te spend          print Meta API spend from the ledger
  te models         list the models your MODEL_API_KEY can use (free)
  te api-smoke      one tiny paid Muse Spark call through the cache and ledger
"""

from __future__ import annotations

import argparse
import json
import sys

from backend.config import ConfigError, load_settings
from backend.meta.client import MetaAPIError, MetaClient
from backend.meta.ledger import BudgetExceeded, Ledger, format_report, token_cost

SMOKE_SCHEMA = {
    "type": "object",
    "properties": {"ok": {"type": "boolean"}},
    "required": ["ok"],
    "additionalProperties": False,
}


def make_ledger(settings) -> Ledger:
    return Ledger(settings.ledger_path, settings.skill_cap_usd, settings.global_cap_usd)


def make_client(settings) -> MetaClient:
    return MetaClient(settings)


def cmd_spend(args) -> int:
    print(format_report(make_ledger(load_settings())))
    return 0


def cmd_models(args) -> int:
    settings = load_settings()
    offered = make_client(settings).list_models()
    for model_id in offered:
        print(model_id)
    wanted = {m for job, m in settings.models.items() if job != "student_base"}
    missing = sorted(wanted - set(offered))
    print(f"configured models missing for this key: {', '.join(missing) or 'none'}")
    return 1 if missing else 0


def cmd_api_smoke(args) -> int:
    settings = load_settings()
    model = settings.model("labeler")
    bound = token_cost(settings.price(model), 200, args.max_tokens)
    print(f"one structured call to {model}, at most about ${bound:.4f}")
    result = make_client(settings).structured(
        model, "Reply with ok set to true.", SMOKE_SCHEMA, name="smoke", skill="smoke",
        max_tokens=args.max_tokens,
    )
    note = "  (cache hit: no call made)" if result.info.cached else ""
    print(f"result: {json.dumps(result.data)}")
    print(f"cost: ${result.info.cost_usd:.6f}{note}")
    print(f"usage: {json.dumps(result.info.usage, sort_keys=True)}")
    return 0 if result.data.get("ok") is True else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="te", description="Trained Eye")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("spend", help="print Meta API spend").set_defaults(func=cmd_spend)
    sub.add_parser("models", help="list models for this key").set_defaults(func=cmd_models)

    smoke = sub.add_parser("api-smoke", help="one tiny paid call to check the key")
    smoke.add_argument("--max-tokens", type=int, default=512)
    smoke.set_defaults(func=cmd_api_smoke)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (ConfigError, MetaAPIError, BudgetExceeded) as err:
        print(f"te: {err}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
