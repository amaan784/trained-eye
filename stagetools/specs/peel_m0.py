"""Peel spec for M0 (stages v1 to v11). Historical: this is how v1 to v11 were built,
when the repo root was the finished M0. Do not run it against a later root."""

BASE, FIRST, LAST = 0, 1, 11

CLI_SERVE_BLOCK_START = "def cmd_serve(args) -> int:"
CLI_PARSER_START = "def build_parser() -> argparse.ArgumentParser:"

PEEL: dict[int, dict] = {
    11: {"remove": ["CLAUDE.md", "PROGRESS.md"]},
    10: {"remove": ["docs/glasses-check.md", "LICENSE", ".gitattributes"]},
    9: {
        "remove": ["backend/training/wandb_smoke.py", "tests/test_wandb_smoke.py"],
        "edit": {"pyproject.toml": [
            ('dev = ["pytest>=8", "ruff>=0.6"]\ntrain = ["modal>=1.0", "wandb>=0.19"]\n',
             'dev = ["pytest>=8", "ruff>=0.6"]\n'),
        ]},
    },
    8: {
        "remove": ["backend/server.py", "tests/test_server.py"],
        "edit": {
            "pyproject.toml": [
                ('dependencies = [\n    "httpx>=0.27",\n    "pillow>=10",\n    "starlette>=0.37",\n'
                 '    "uvicorn[standard]>=0.30",\n]\n',
                 'dependencies = [\n    "httpx>=0.27",\n]\n'),
            ],
            "backend/cli.py": [
                ("  te serve          run the backend (WebSocket at /ws)\n"
                 "  te ws-selftest    stream synthetic frames through the WebSocket, in process\n",
                 ""),
                ("import argparse\nimport io\nimport json\n", "import argparse\nimport json\n"),
                ("CUT:" + CLI_SERVE_BLOCK_START, CLI_PARSER_START),
                ('\n    serve = sub.add_parser("serve", help="run the backend server")\n'
                 '    serve.add_argument("--host", default="0.0.0.0")\n'
                 '    serve.add_argument("--port", type=int, default=8765)\n'
                 '    serve.set_defaults(func=cmd_serve)\n\n'
                 '    selftest = sub.add_parser("ws-selftest", help="send synthetic frames over the WebSocket")\n'
                 '    selftest.add_argument("--frames", type=int, default=3)\n'
                 '    selftest.set_defaults(func=cmd_ws_selftest)\n',
                 ""),
            ],
        },
    },
    7: {
        "remove": ["tests/test_cli.py", ".env.example"],
        "edit": {"backend/cli.py": [
            ("  te models         list the models your MODEL_API_KEY can use (free)\n"
             "  te api-smoke      one tiny paid Muse Spark call through the cache and ledger\n",
             ""),
            ("import argparse\nimport json\nimport sys\n\n"
             "from backend.config import ConfigError, load_settings\n"
             "from backend.meta.client import MetaAPIError, MetaClient\n"
             "from backend.meta.ledger import BudgetExceeded, Ledger, format_report, token_cost\n"
             "\nSMOKE_SCHEMA = {\n"
             '    "type": "object",\n'
             '    "properties": {"ok": {"type": "boolean"}},\n'
             '    "required": ["ok"],\n'
             '    "additionalProperties": False,\n'
             "}\n",
             "import argparse\nimport sys\n\n"
             "from backend.config import load_settings\n"
             "from backend.meta.ledger import Ledger, format_report\n"),
            ("def make_client(settings) -> MetaClient:\n    return MetaClient(settings)\n\n\n", ""),
            ("CUT:def cmd_models(args) -> int:", CLI_PARSER_START),
            ('    sub.add_parser("models", help="list models for this key").set_defaults(func=cmd_models)\n'
             '\n    smoke = sub.add_parser("api-smoke", help="one tiny paid call to check the key")\n'
             '    smoke.add_argument("--max-tokens", type=int, default=512)\n'
             '    smoke.set_defaults(func=cmd_api_smoke)\n',
             ""),
            ("    args = build_parser().parse_args(argv)\n    try:\n        return args.func(args)\n"
             "    except (ConfigError, MetaAPIError, BudgetExceeded) as err:\n"
             '        print(f"te: {err}", file=sys.stderr)\n        return 2\n',
             "    args = build_parser().parse_args(argv)\n    return args.func(args)\n"),
        ]},
    },
    6: {
        "edit": {
            "backend/meta/client.py": [
                ("json_schema response_format, as in Meta's cookbook recipes. SAM 3.1 uses the Responses\n"
                 "API and Muse Image / Voice Transcribe use the images and audio endpoints, as described\n"
                 "in SPEC.md; those three shapes still need confirming with a live call.\n",
                 "json_schema response_format, as in Meta's cookbook recipes.\n"),
                ("import email.utils\nimport io\nimport json\nimport random\nimport re\nimport time\nimport wave\n",
                 "import email.utils\nimport json\nimport random\nimport time\n"),
                ("from backend.meta.ledger import NO_SKILL, Entry, Ledger, token_cost, unit_cost\n",
                 "from backend.meta.ledger import NO_SKILL, Entry, Ledger, token_cost\n"),
                ("CUT:# -- media endpoints (Voice Transcribe, SAM 3.1, Muse Image)", "class MetaClient:"),
                ("        )\n        # For downloading generated images by URL: never carries the API key, so it may\n"
                 "        # follow CDN redirects.\n        self._plain = httpx.Client(\n"
                 "            timeout=settings.timeout_s, transport=transport, follow_redirects=True\n"
                 "        )\n",
                 "        )\n"),
                ("CUT_TO_END:\n    # -- media endpoints ----", ""),
            ],
            "tests/test_client.py": [
                ("import base64\nimport io\nimport json\nimport wave\n", "import base64\nimport json\n"),
                ("CUT_TO_END:\n\n# -- media endpoints ----", ""),
            ],
        },
    },
    5: {
        "remove": ["backend/meta/client.py", "tests/test_client.py"],
        "edit": {"pyproject.toml": [
            ('dependencies = [\n    "httpx>=0.27",\n]\n', "dependencies = []\n"),
        ]},
    },
    4: {
        "remove": ["backend/cli.py", "tests/conftest.py"],
        "edit": {
            "pyproject.toml": [('[project.scripts]\nte = "backend.cli:main"\n\n', "")],
            "tests/test_ledger.py": [
                ("from backend.cli import main\n", ""),
                ("CUT_TO_END:\n\ndef test_spend_command(", ""),
            ],
        },
    },
    3: {"remove": ["backend/meta/ledger.py", "tests/test_ledger.py"]},
    2: {"remove": ["backend/meta/cache.py", "tests/test_cache.py", ".gitignore"]},
}
