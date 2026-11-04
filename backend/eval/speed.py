"""Why one student answers faster than another: the tokens each spends per frame.

    python -m backend.eval.speed --url <endpoint>/v1 --model trained-eye --name finetuned \\
        --clip <clip> --skill fridge_left_open --frames 20 --out eval/speed.json

Sends the student's own request (backend/inference/student.py: the same prompt, image,
logprobs and reasoning setting) for `--frames` frames of a clip, one at a time so the
latencies are not queued behind each other, and records per frame the completion tokens,
whether the reply carried reasoning text, and the seconds it took. The medians go into
--out under --name (other names in the file are kept), for docs/evaluation.md.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import httpx

from backend.config import load_settings
from backend.dataengine.keyframes import load_frames
from backend.inference.student import REASONING, frame_message
from backend.skills.spec import SkillSpec


def measure(http: httpx.Client, model: str, spec: SkillSpec, frames: list,
            max_tokens: int = 512) -> list[dict]:
    rows = []
    for frame in frames:
        body = {"model": model, "messages": frame_message(frame.data, spec),
                "max_tokens": max_tokens, "logprobs": True, "top_logprobs": 10, **REASONING}
        start = time.monotonic()
        reply = http.post("/chat/completions", json=body)
        reply.raise_for_status()
        seconds = time.monotonic() - start
        data = reply.json()
        message = data["choices"][0].get("message") or {}
        reasoning = message.get("reasoning") or message.get("reasoning_content") or ""
        usage = data.get("usage") or {}
        rows.append({"t": frame.t_s, "seconds": round(seconds, 3),
                     "completion_tokens": usage.get("completion_tokens"),
                     "prompt_tokens": usage.get("prompt_tokens"),
                     "reasoning_chars": len(reasoning),
                     "answer": (message.get("content") or "").strip()[:20]})
    return rows


def summarize(rows: list[dict]) -> dict:
    def median(key):
        values = [r[key] for r in rows if r.get(key) is not None]
        return statistics.median(values) if values else None

    return {"frames": len(rows), "median_seconds": median("seconds"),
            "median_completion_tokens": median("completion_tokens"),
            "median_prompt_tokens": median("prompt_tokens"),
            "frames_with_reasoning": sum(1 for r in rows if r["reasoning_chars"]),
            "rows": rows}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m backend.eval.speed", description=__doc__)
    p.add_argument("--url", required=True, help="the endpoint, ending in /v1")
    p.add_argument("--model", required=True, help="the served model name")
    p.add_argument("--name", required=True, help="the key in --out, e.g. finetuned")
    p.add_argument("--clip", required=True, help="a video or frame folder")
    p.add_argument("--skill", required=True)
    p.add_argument("--frames", type=int, default=20)
    p.add_argument("--out", default="eval/speed.json")
    args = p.parse_args(argv)
    settings = load_settings()
    spec = SkillSpec.model_validate_json(
        (settings.data_dir / "skills" / args.skill / "spec.json").read_text())
    frames = load_frames(Path(args.clip), fps=1.0)
    step = max(1, len(frames) // args.frames)
    picked = frames[::step][: args.frames]
    key = settings.secrets.get("STUDENT_API_KEY")
    with httpx.Client(base_url=args.url.rstrip("/"), timeout=900, follow_redirects=True,
                      headers={"Authorization": f"Bearer {key}"} if key else {}) as http:
        measure(http, args.model, spec, picked[:1])  # a cold endpoint answers the first late
        result = summarize(measure(http, args.model, spec, picked))
    out = Path(args.out)
    saved = json.loads(out.read_text()) if out.is_file() else {}
    saved[args.name] = {"model": args.model, "skill": args.skill, **result}
    out.write_text(json.dumps(saved, indent=1) + "\n")
    print(f"{args.name}: {result['frames']} frames, median {result['median_seconds']} s and "
          f"{result['median_completion_tokens']} completion tokens a frame, reasoning text "
          f"on {result['frames_with_reasoning']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
