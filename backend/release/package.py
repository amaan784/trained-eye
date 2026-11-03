"""The Hugging Face release folder and its publishing: prepare_release assembles the folder
(the card from model_card.py, the weights, the prompt manifests, the base model's LICENSE
and release.json), upload sends it, and make_collection groups release repos into a
Hugging Face collection (`te release-collection`).

`upload` imports huggingface_hub only when called and refuses without confirmed=True (the
author's yes).

Generated images are never published. What regenerates them ships instead: the prompt
manifests, and with `te release --keyframes` the face-blurred keyframes the edits start
from (keyframes/<skill>/, from a `te blur` folder with its sources.json). upload() allows
exactly those images: each listed in its sources.json, its original's SHA-256 a
keyframe_sha256 of that skill's manifest, and neither it nor its original any manifest's
image_sha256. Every other image or video in a model release is refused. A test-set folder
(backend/release/test_set.py) goes to a dataset repo with its own checks.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from collections.abc import Iterable, Mapping
from pathlib import Path

from backend.config import load_settings
from backend.release.faces import IMAGE_SUFFIXES, SOURCES_FILE
from backend.release.model_card import model_card, synthetic_skills

BASE_LICENSE_FILE = Path(__file__).with_name("BASE_LICENSE.txt")
WEIGHT_SUFFIXES = (".safetensors", ".json", ".jinja", ".model", ".txt", ".gguf")
MEDIA_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".gif", ".heic", ".mp4", ".mov")
REPO_ID = re.compile(r"[A-Za-z0-9][\w.-]*/[A-Za-z0-9][\w.-]*")
KEYFRAMES_DIR = "keyframes"  # in a release folder: keyframes/<skill>/ from te blur
TEST_SET_KIND = "test_set"  # release.json kind of a te release-test-set folder
DATASET_PREFIX = "datasets/"  # te release-collection: datasets/owner/name is a dataset repo
HUB_MISSING = ("needs huggingface_hub, which is not installed and not on the approved "
               "downloads list: ask the author, then pip install huggingface_hub")


def _sha256(path: Path) -> str:
    with path.open("rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


def manifest_hashes(manifests_dir: Path,
                    skills: Iterable[str] | None = None) -> tuple[dict[str, set[str]], set[str]]:
    """({skill: the keyframe_sha256 its edits start from}, every image_sha256) of the prompt
    manifests <skill>.jsonl in manifests_dir (only `skills`' when given)."""
    keyframes, generated = {}, set()
    for path in sorted(Path(manifests_dir).glob("*.jsonl")):
        if skills is not None and path.stem not in skills:
            continue
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()]
        rows = [r for r in rows if isinstance(r, dict)]
        keyframes[path.stem] = {r["keyframe_sha256"] for r in rows
                                if isinstance(r.get("keyframe_sha256"), str)}
        generated |= {r["image_sha256"] for r in rows if isinstance(r.get("image_sha256"), str)}
    return keyframes, generated


def shippable_keyframes(folder: Path, needed: set[str], generated: set[str]) -> dict[str, str]:
    """{file: its original's SHA-256} for the images of a te blur folder that may be
    published: listed in its sources.json with an original in `needed` (keyframes the
    manifest's edits start from). Raises ValueError for a listed image that is, or was made
    from, a generated image (`generated`: image_sha256 values). Images not listed, and those
    of keyframes no edit starts from, are left out."""
    folder = Path(folder)
    if not (folder / SOURCES_FILE).is_file():
        raise ValueError(f"{folder} has no {SOURCES_FILE}: give the folder te blur wrote")
    sources = json.loads((folder / SOURCES_FILE).read_text(encoding="utf-8"))
    out = {}
    for original, name in sources.items() if isinstance(sources, dict) else []:
        path = folder / str(name)
        if (Path(str(name)).name != name or not path.is_file()
                or path.suffix.lower() not in IMAGE_SUFFIXES):
            continue
        if original in generated or _sha256(path) in generated:
            raise ValueError(f"{path} is a generated image or was made from one (its SHA-256 "
                             "or its original's is a manifest's image_sha256): generated "
                             "images are never published")
        if original in needed:
            out[name] = original
    return out


def pick_keyframes(keyframes: Mapping[str, Path], skills: list[str],
                   manifests_dir: Path) -> dict[str, tuple[Path, dict[str, str]]]:
    """{skill: (its te blur folder, the files to ship)}; refuses a skill without generated
    images and a folder that misses a keyframe the manifest needs (te regenerate would
    stop)."""
    if extra := sorted(set(keyframes) - set(skills)):
        raise ValueError(f"--keyframes for {', '.join(extra)}, which has no generated images "
                         "in this bundle: only skills with a manifest need keyframes")
    needed, generated = manifest_hashes(manifests_dir, skills)
    picked = {}
    for skill, folder in sorted(keyframes.items()):
        files = shippable_keyframes(folder, needed[skill], generated)
        if missing := needed[skill] - set(files.values()):
            raise ValueError(f"{folder} misses {len(missing)} of the {len(needed[skill])} "
                             f"keyframes {skill}'s edits start from: te blur the skill's whole "
                             "keyframes folder")
        picked[skill] = (Path(folder), files)
    return picked


def prepare_release(out_dir: Path, *, run: str, bundle_manifest: Path, manifests_dir: Path,
                    weights_dir: Path | None = None,
                    keyframes: Mapping[str, Path] | None = None, **card) -> dict:
    """Assemble the folder to upload (uploading nothing): README.md (the card; `card` holds
    kind, results, license_id, models), weights_dir's top-level weight and config files (no
    pickles, checkpoints or trainer README), the manifests of the skills with synthetic data,
    the face-blurred keyframes of `keyframes` ({skill: te blur folder}) under
    keyframes/<skill>/ with their sources.json, and release.json listing every file with
    its SHA-256."""
    out_dir, manifests_dir = Path(out_dir), Path(manifests_dir)
    if out_dir.exists() and any(out_dir.iterdir()) and not (out_dir / "release.json").is_file():
        raise ValueError(f"{out_dir} is not empty and not a release folder: pick another")
    synthetic = synthetic_skills(json.loads(Path(bundle_manifest).read_text()))
    if missing := [s for s in synthetic if not (manifests_dir / f"{s}.jsonl").is_file()]:
        raise ValueError(f"no prompt manifest for {', '.join(missing)} in {manifests_dir}: "
                         "run te manifests first")
    picked = pick_keyframes(dict(keyframes or {}), synthetic, manifests_dir)
    shutil.rmtree(out_dir, ignore_errors=True)
    (out_dir / "manifests").mkdir(parents=True)
    skipped = []
    for path in sorted(Path(weights_dir).iterdir()) if weights_dir else []:
        if not path.is_file() or path.suffix not in WEIGHT_SUFFIXES or path.name == "README.md":
            skipped.append(path.name)
            continue
        shutil.copy2(path, out_dir / path.name)
    for skill in synthetic:
        shutil.copyfile(manifests_dir / f"{skill}.jsonl", out_dir / "manifests" / f"{skill}.jsonl")
    for skill, (folder, files) in picked.items():
        target = out_dir / KEYFRAMES_DIR / skill
        target.mkdir(parents=True)
        for name in files:
            shutil.copyfile(folder / name, target / name)
        (target / SOURCES_FILE).write_text(json.dumps(
            {original: name for name, original in files.items()}, indent=1, sort_keys=True)
            + "\n")
    model_card(out_dir / "README.md", run=run, bundle_manifest=bundle_manifest,
               keyframes=list(picked), **card)
    shutil.copyfile(BASE_LICENSE_FILE, out_dir / "LICENSE")  # Apache-2.0 4(a): ship the text
    release = {"run": run, "kind": card.get("kind", "adapter"), "skipped": skipped,
               "keyframes": {skill: len(files) for skill, (_, files) in picked.items()},
               "files": [{"path": p.relative_to(out_dir).as_posix(), "bytes": p.stat().st_size,
                          "sha256": _sha256(p)} for p in sorted(out_dir.rglob("*")) if p.is_file()]}
    (out_dir / "release.json").write_text(json.dumps(release, indent=1) + "\n")
    return release


def media_problems(folder: Path) -> list[str]:
    """The images and videos in a model release folder that may not be published: all but
    the face-blurred keyframes under keyframes/<skill>/ that shippable_keyframes allows
    against the folder's own manifests."""
    folder = Path(folder)
    needed, generated = manifest_hashes(folder / "manifests")
    allowed: dict[str, dict[str, str]] = {}
    bad = []
    for path in sorted(folder.rglob("*")):
        if path.suffix.lower() not in MEDIA_SUFFIXES:
            continue
        parts = path.relative_to(folder).parts
        if len(parts) == 3 and parts[0] == KEYFRAMES_DIR and parts[1] in needed:
            if parts[1] not in allowed:
                try:
                    allowed[parts[1]] = shippable_keyframes(folder / KEYFRAMES_DIR / parts[1],
                                                            needed[parts[1]], generated)
                except ValueError:  # a generated image in it: none of that folder ships
                    allowed[parts[1]] = {}
            if path.is_file() and parts[2] in allowed[parts[1]]:
                continue
        bad.append("/".join(parts))
    return bad


def _hub(token_env: str):
    """(HfApi with the token from the environment or .env, a function that turns an error
    into one whose text does not carry the token)."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        raise RuntimeError(f"Hugging Face {HUB_MISSING}") from None
    token = load_settings().secret(token_env)

    def failed(err: Exception, what: str = "upload") -> RuntimeError:
        return RuntimeError(f"{what} failed ({type(err).__name__}): "
                            f"{str(err).replace(token, '[redacted]')[:300]}")

    return HfApi(token=token), failed


def _visibility(private) -> str:
    return "private" if private else "public"


def upload(folder: Path, repo_id: str, token_env: str = "HF_TOKEN", *, confirmed: bool = False,
           private: bool = True) -> str:
    """Publish a prepare_release folder to the model repo `repo_id`, or a prepare_test_set
    folder to the dataset repo `repo_id` (private unless private=False). Refuses without
    confirmed=True, a model release without the base model's LICENSE or with an image or
    video other than its checked keyframes, and a test set that fails test_set.check_folder."""
    if not confirmed:
        raise PermissionError("publishing needs the author's yes: nothing was uploaded")
    folder = Path(folder)
    if not REPO_ID.fullmatch(repo_id) or not (folder / "release.json").is_file():
        raise ValueError(f"need a repo id like owner/name and a folder from te release: "
                         f"{repo_id!r}, {folder}")
    release = json.loads((folder / "release.json").read_text())
    kind = release.get("kind", "adapter")
    if kind == TEST_SET_KIND:
        from backend.release.test_set import check_folder

        check_folder(folder, release)
        repo_type, url = "dataset", f"https://huggingface.co/datasets/{repo_id}"
    else:
        if not (folder / "LICENSE").is_file():
            raise ValueError("the release folder has no LICENSE (the base model's): run te "
                             "release again")
        if media := media_problems(folder):
            raise ValueError(f"images or video in the release folder ({', '.join(media[:5])}): "
                             "generated images are never published, and keyframes only "
                             "face-blurred by te blur, listed in their sources.json and in "
                             "the manifest")
        names = {f["path"] for f in release["files"]}
        needed = "adapter_config.json" if kind == "adapter" else "config.json"
        if needed not in names or not any(n.endswith(".safetensors") for n in names):
            raise ValueError(f"the release folder has no {kind} weights ({needed} and "
                             ".safetensors): run te fetch-weights, then te release")
        repo_type, url = "model", f"https://huggingface.co/{repo_id}"
    api, failed = _hub(token_env)
    try:
        api.create_repo(repo_id, repo_type=repo_type, private=private, exist_ok=True)
        # an existing repo keeps its visibility: never publish after a "private" question
        actual = getattr(api.repo_info(repo_id, repo_type=repo_type), "private", None)
    except Exception as err:  # noqa: BLE001
        raise failed(err) from None
    if actual is not private:
        raise ValueError(f"{repo_id} already exists and is {_visibility(actual)}, not "
                         f"{_visibility(private)}: nothing was uploaded")
    try:
        api.upload_folder(repo_id=repo_id, folder_path=str(folder), repo_type=repo_type,
                          commit_message=f"Trained Eye release {folder.name}")
    except Exception as err:  # noqa: BLE001
        raise failed(err) from None
    return url


def collection_item(spec: str) -> tuple[str, str]:
    """(item type, repo id) of a te release-collection repo: owner/name is a model repo,
    datasets/owner/name a dataset repo (as in their huggingface.co addresses)."""
    kind, repo = ("dataset", spec[len(DATASET_PREFIX):]) if spec.startswith(
        DATASET_PREFIX) else ("model", spec)
    if not REPO_ID.fullmatch(repo):
        raise ValueError(f"{spec!r}: give owner/name for a model repo or datasets/owner/name "
                         "for a dataset repo")
    return kind, repo


def make_collection(title: str, repos: Iterable[str], token_env: str = "HF_TOKEN", *,
                    confirmed: bool = False, private: bool = True, namespace: str | None = None,
                    description: str | None = None) -> str:
    """Group release repos (see collection_item) into the Hugging Face collection `title`
    (private unless private=False) and return its address. Refuses without confirmed=True,
    and, before making anything, a repo that is missing or not reachable or whose visibility
    is not the one asked for. An existing collection of that title is reused only if its
    visibility is the one asked for."""
    if not confirmed:
        raise PermissionError("publishing needs the author's yes: nothing was made")
    items = [collection_item(spec) for spec in repos]
    if not title.strip() or not items:
        raise ValueError("a collection needs a title and at least one repo")
    if twice := sorted({r for r in items if items.count(r) > 1}):
        raise ValueError(f"repos given twice: {', '.join(r for _, r in twice)}")
    api, failed = _hub(token_env)
    for kind, repo in items:
        try:
            actual = getattr(api.repo_info(repo, repo_type=kind), "private", None)
        except Exception as err:  # noqa: BLE001
            raise ValueError(f"{kind} repo {repo} is missing or not reachable "
                             f"({failed(err, 'lookup')}): nothing was made") from None
        if actual is not private:
            raise ValueError(f"{kind} repo {repo} is {_visibility(actual)}, not "
                             f"{_visibility(private)}: nothing was made")
    try:
        collection = api.create_collection(title, namespace=namespace, description=description,
                                           private=private, exists_ok=True)
    except Exception as err:  # noqa: BLE001
        raise failed(err, "creating the collection") from None
    if getattr(collection, "private", None) is not private:
        raise ValueError(f"a collection titled {title!r} already exists and is "
                         f"{_visibility(collection.private)}, not {_visibility(private)}: "
                         "nothing was added")
    for kind, repo in items:
        try:
            api.add_collection_item(collection.slug, item_id=repo, item_type=kind,
                                    exists_ok=True)
        except Exception as err:  # noqa: BLE001
            raise failed(err, f"adding {repo}") from None
    return f"https://huggingface.co/collections/{collection.slug}"
