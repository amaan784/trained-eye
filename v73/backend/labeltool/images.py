"""Hand labels for generated training images (Experiment 2: how often Spark is wrong).

One image at a time with the question Spark was asked about it, never Spark's answer. Keys
y, n, u (unsure), Backspace (back). Each answer is saved at once as {"labels": {"<file
name>": "yes" | "no" | "unsure"}}; the page resumes at the first image without an answer,
and labels in the file for other images are kept.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from backend.dataengine.dataset import user_prompt
from backend.skills.spec import SkillSpec

ANSWERS = ("yes", "no", "unsure")
STATIC = Path(__file__).parent / "static"  # the pages and style.css of both labelling tools


def write_json(path: Path, data, indent: int | None = 1) -> None:
    """Write `data` to `path` in one step, so a crash never leaves half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=indent) + "\n", encoding="utf-8")
    tmp.replace(path)


def sample_images(labels_jsonl: Path, n: int, seed: int) -> list[Path]:
    """`n` generated images picked at random from a skill's labels.jsonl (te teach), the
    same ones in the same order for the same rows and seed. Real keyframes are never
    picked. Raises ValueError unless 1 <= n <= the number of generated images listed."""
    labels_jsonl = Path(labels_jsonl)
    rows = [json.loads(x) for x in labels_jsonl.read_text("utf-8").splitlines() if x.strip()]
    files = sorted({r["file"] for r in rows if r.get("source") == "synthetic" and r.get("file")})
    if not 1 <= n <= len(files):
        raise ValueError(f"{labels_jsonl} lists {len(files)} generated images; ask for 1 to "
                         f"{len(files)}, not {n}")
    return [labels_jsonl.parent / f for f in random.Random(seed).sample(files, n)]


def skill_question(images: list[Path]) -> str:
    """The question Spark was asked, from spec.json in the skill folder above the images
    (data/skills/<skill>/synthetic/...), or "" if there is none."""
    spec_path = images[0].parent.parent / "spec.json" if images else None
    if spec_path is None or not spec_path.is_file():
        return ""
    spec = SkillSpec.model_validate_json(spec_path.read_text(encoding="utf-8"))
    return user_prompt(spec.positive or "", spec.negatives)


def create_image_app(images: list[Path], out: Path, *, question: str | None = None) -> Starlette:
    """The labelling page for `images`, saving each answer to `out`; `question` defaults to
    skill_question. ValueError for no or same-named images, FileNotFoundError if one is gone."""
    images, out = [Path(p) for p in images], Path(out)
    names = [p.name for p in images]
    if not images or len(set(names)) != len(names):
        raise ValueError("give at least one image, each with its own file name")
    if missing := [str(p) for p in images if not p.is_file()]:
        raise FileNotFoundError(f"{len(missing)} images are missing, first {missing[0]}")
    question = skill_question(images) if question is None else question
    labels = json.loads(out.read_text(encoding="utf-8"))["labels"] if out.is_file() else {}

    async def state(request: Request):
        return JSONResponse({"question": question, "images": names, "out": out.name,
                             "labels": {n: labels[n] for n in names if n in labels}})

    async def image(request: Request):
        i = request.path_params["i"]
        if i >= len(images):
            return JSONResponse({"error": "no such image"}, status_code=404)
        return FileResponse(images[i])

    async def answer(request: Request):
        try:
            body = await request.json()
            name, label = body["file"], body["label"]
            if name not in names or label not in ANSWERS:
                raise ValueError(f"file must be one of the images and label one of {ANSWERS}")
        except (ValueError, KeyError, TypeError) as err:
            return JSONResponse({"error": str(err)}, status_code=400)
        labels[name] = label
        write_json(out, {"labels": labels})
        return JSONResponse({"done": sum(n in labels for n in names), "total": len(names)})

    return Starlette(routes=[Route("/", lambda request: FileResponse(STATIC / "images.html")),
                             Route("/state", state),
                             Route("/image/{i:int}", image),
                             Route("/answer", answer, methods=["POST"]),
                             Mount("/static", StaticFiles(directory=STATIC))])
