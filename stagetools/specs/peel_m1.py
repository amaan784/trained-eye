"""Peel spec for M1 (stages v12 to v21), on top of v11. Historical: this is how v12
to v21 were built, when the repo root was the finished M1. Do not run it against a
later root."""

BASE, FIRST, LAST = 11, 12, 21

PARSER = "def build_parser() -> argparse.ArgumentParser:"

PEEL: dict[int, dict] = {
    21: {"remove": ["docs/recording-protocol.md"], "restore": ["PROGRESS.md", "CLAUDE.md"]},
    20: {"edit": {
        "backend/cli.py": [
            ("  te teach          teach a skill: request + 10 s clip -> spec, generated data, "
             "labels, dataset\n", ""),
            ("import json\nimport shutil\nimport subprocess\nimport sys\nfrom pathlib import Path\n",
             "import json\nimport sys\n"),
            ("CUT:def audio_duration_s(path: Path) -> float | None:", PARSER),
            ('\n    teach = sub.add_parser("teach", help="teach a skill from a request and a 10 s clip")\n'
             '    teach.add_argument("--skill", required=True, help="skill name, e.g. washer_done")\n'
             '    teach.add_argument("--clip", required=True, help="video file or folder of JPEG frames")\n'
             '    teach.add_argument("--audio", help="the spoken request (wav, m4a, ...)")\n'
             '    teach.add_argument("--request", help="the request as text, instead of --audio")\n'
             '    teach.add_argument("--positives", type=int, default=150)\n'
             '    teach.add_argument("--negatives", type=int, default=150)\n'
             '    teach.add_argument("--labels", help="JSON of your own keyframe labels {file: yes|no}")\n'
             '    teach.add_argument("--estimate-only", action="store_true", help="print the cost estimate")\n'
             '    teach.add_argument("--yes", action="store_true", help="skip both confirmations")\n'
             '    teach.set_defaults(func=cmd_teach)\n', ""),
        ],
        "tests/test_cli.py": [("CUT_TO_END:\n\n\ndef write_clip(", "")],
    }},
    19: {
        "remove": ["backend/teach.py", "tests/test_teach.py"],
        "edit": {"backend/config.py": [
            ("    # False = SPEC.md's fallback: train on real frames and the user's own labels only.\n"
             "    train_on_model_outputs: bool = True\n", ""),
            ('        train_on_model_outputs=merged.get("TE_TRAIN_ON_MODEL_OUTPUTS", "1").strip().lower()\n'
             '        not in ("0", "false", "no", "off"),\n', ""),
        ]},
    },
    18: {"remove": ["backend/dataengine/labels.py", "tests/test_labels.py"]},
    17: {"remove": ["backend/dataengine/synth.py", "tests/test_synth.py"]},
    16: {"remove": ["backend/dataengine/roi.py", "tests/test_roi.py"]},
    15: {"remove": ["backend/dataengine/dataset.py", "tests/test_dataset.py"]},
    14: {
        "remove": ["backend/dataengine/keyframes.py", "tests/test_keyframes.py"],
        "edit": {"pyproject.toml": [('    "numpy>=1.26",\n', "")]},
    },
    13: {
        "remove": ["backend/skills/compiler.py", "tests/test_compiler.py"],
        "edit": {"backend/meta/client.py": [
            ('        reasoning_effort: str | None = None,\n'
             '        validate: Callable[[Any], None] | None = None,\n'
             '    ) -> StructuredResult:\n'
             '        """JSON output constrained by `schema`, via Chat Completions.\n\n'
             '        `validate(data)` may raise OutputError for output that parses but is unusable;\n'
             '        such output is billed like any other but never cached.\n'
             '        """\n',
             '        reasoning_effort: str | None = None,\n'
             '    ) -> StructuredResult:\n'
             '        """JSON output constrained by `schema`, via Chat Completions."""\n'),
            ('            try:\n'
             '                data = _parse_json_text(content)\n'
             '            except json.JSONDecodeError:\n'
             '                raise OutputError(f"output is not JSON: {content[:200]!r}") from None\n'
             '            if validate is not None:\n'
             '                validate(data)\n'
             '            return data\n',
             '            try:\n'
             '                return _parse_json_text(content)\n'
             '            except json.JSONDecodeError:\n'
             '                raise OutputError(f"output is not JSON: {content[:200]!r}") from None\n'),
        ]},
    },
    12: {
        "remove": ["backend/skills/spec.py", "tests/test_skills.py"],
        "edit": {
            "pyproject.toml": [('    "pydantic>=2.6",\n', "")],
            "backend/cli.py": [
                ("  te policy         check a request against the local policy rules (free, offline)\n",
                 ""),
                ("CUT:def cmd_policy(args) -> int:", PARSER),
                ('\n    policy = sub.add_parser("policy", help="check a request against the local policy rules")\n'
                 '    policy.add_argument("request")\n'
                 '    policy.set_defaults(func=cmd_policy)\n', ""),
            ],
        },
        "restore_readme": True,
    },
}
