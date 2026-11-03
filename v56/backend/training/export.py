"""Merge a run's QLoRA adapter into the bf16 base on Modal, and check the merge.

    modal run backend/training/export.py --run exp1-mixed [--reference nf4] [--result out.json]

Loads the base at the pinned revision in bf16 with the adapter modal_train.py left in
/vol/runs/<run>/out, merges it with peft's merge_and_unload(), saves safetensors and the
processor to /vol/runs/<run>/merged (what modal_serve.py serves), reloads that folder and
checks it answers yes or no like base + adapter on up to 50 examples of the run's bundle (the
local copy te train made, validation split if it has one; export_check.py picks and compares).
Loading and prompting copy the cookbook's evaluator (meta-oss-cookbook b30ca2c,
muse-glimmer/post-training/sft/axolotl/scripts/evaluate.py, Apache 2.0).

Precision (results/README.md): "Use the same IDs, generation settings, and base precision for
every comparison. Pass --load-in-4bit for both the untuned NF4 baseline and its QLoRA adapter;
use bf16 for both the untuned baseline and LoRA or full checkpoint." Meta evaluated its QLoRA
adapters on an NF4 base only. The merge is bf16, so the default reference is bf16 base +
adapter; --reference nf4 compares with the NF4 base + adapter that training saw.

The result's `seconds` is the remote call timed here (so the container start is in it), for
te export's Modal ledger (backend/training/gpu_ledger.py); a failed call still writes the
run, ok false and its seconds before the error goes on.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import modal

APP = "trained-eye-export"
VOLUME = "trained-eye"
MOUNT = "/vol"
GPU = "H100"  # 80 GB; the cookbook says bf16 needs about 60 GB (muse-glimmer/README.md)
HF_SECRET = "huggingface"
TIMEOUT_S = 3 * 3600  # not measured: two loads of the 30B base and up to 100 generations
CUDA_IMAGE = "nvidia/cuda:13.0.0-devel-ubuntu24.04"  # as modal_train.py
RUN_NAME = re.compile(r"[a-z0-9][a-z0-9-]{1,39}")  # as te train checks it
MODEL_CLASSES = ("AutoModelForImageTextToText", "AutoModelForVision2Seq", "AutoModelForCausalLM")
SMALL_FILES = (".json", ".jinja", ".txt", ".model")  # never weights; index files are skipped
SEED = 17  # evaluate.py's default --seed

app = modal.App(APP)
volume = modal.Volume.from_name(VOLUME, create_if_missing=True)
# The training pins (modal_train.py) without flash-attn and Cut Cross Entropy: evaluate.py
# sets no attention implementation by default, and CCE is a training plugin. Axolotl brings
# the transformers, peft and bitsandbytes versions the adapter was trained with.
image = (
    modal.Image.from_registry(CUDA_IMAGE, add_python="3.12")
    .apt_install("git", "build-essential")
    .pip_install("uv")
    .env({"UV_TORCH_BACKEND": "cu130"})
    .run_commands('uv pip install --system "torch==2.12.1" "torchvision==0.27.1"',
                  'uv pip install --system "axolotl==0.19.0"')
    .env({"HF_HOME": f"{MOUNT}/hf", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"})
)


def load_model(transformers, torch, source: str, revision: str | None, nf4: bool = False):
    """(model, class name): the first class that loads, as evaluate.py's _load_model."""
    kwargs = {"revision": revision, "low_cpu_mem_usage": True, "torch_dtype": torch.bfloat16,
              "device_map": "auto"}
    if nf4:  # evaluate.py's --load-in-4bit; bf16 storage keeps the vision path in bf16
        kwargs["quantization_config"] = transformers.BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_storage=torch.bfloat16)
    errors = []
    for name in MODEL_CLASSES:
        cls = getattr(transformers, name, None)
        if cls is None:
            continue
        try:
            return cls.from_pretrained(source, **kwargs).eval(), name
        except (KeyError, TypeError, ValueError) as err:
            errors.append(f"{name}: {err}")
    raise RuntimeError(f"no model class loads {source}: " + "; ".join(errors))


def generate(model, processor, torch, messages: list[dict], strength: str, today: str,
             max_new_tokens: int) -> str:
    """One greedy generation, as evaluate.py's generate_one; the decoded text."""
    from PIL import Image

    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                           reasoning_strength=strength, current_date=today)
    images = []
    for part in (p for m in messages if isinstance(m["content"], list) for p in m["content"]):
        if part.get("type") == "image":
            with Image.open(part["path"]) as img:
                images.append(img.convert("RGB"))
    inputs = processor(text=prompt, return_tensors="pt", **({"images": images} if images else {}))
    device = next(model.parameters()).device
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
    tok = getattr(processor, "tokenizer", processor)
    eos = [tok.eos_token_id] if isinstance(tok.eos_token_id, int) else []
    if "<|eot|>" in tok.get_vocab():  # results/README.md: stops repeated answer turns
        eos.append(tok.convert_tokens_to_ids("<|eot|>"))
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                             num_beams=1, use_cache=True, pad_token_id=tok.pad_token_id,
                             eos_token_id=list(dict.fromkeys(eos)) or None)
    return tok.decode(out[0, inputs["input_ids"].shape[-1]:], skip_special_tokens=True)


def finish_files(merged: Path, base: str, revision: str) -> dict:
    """Copy the base's small files the save did not write (vLLM reads chat_template.jinja and
    generation_config.json from the folder) and compare tensor names and architectures."""
    import shutil

    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    snap = Path(snapshot_download(base, revision=revision,
                                  allow_patterns=[f"*{s}" for s in SMALL_FILES]))
    copied = []
    for f in sorted(snap.iterdir()):
        if (f.is_file() and f.suffix in SMALL_FILES and not f.name.endswith(".index.json")
                and not (merged / f.name).exists()):
            shutil.copyfile(f, merged / f.name)
            copied.append(f.name)
    names: set[str] = set()
    for shard in merged.glob("*.safetensors"):
        with safe_open(str(shard), framework="pt") as weights:
            names.update(weights.keys())
    index = snap / "model.safetensors.index.json"
    want = set(json.loads(index.read_text())["weight_map"]) if index.is_file() else set()
    arch = {k: json.loads((d / "config.json").read_text()).get("architectures")
            for k, d in (("base", snap), ("merged", merged))}
    return {"copied_from_base": copied, "bin_files": len(list(merged.glob("*.bin"))),
            "tensor_names_match": names == want if want else None,
            "missing_tensors": sorted(want - names)[:5], "extra_tensors": sorted(names - want)[:5],
            "architectures": arch}


@app.function(image=image, gpu=GPU, volumes={MOUNT: volume}, timeout=TIMEOUT_S,
              secrets=[modal.Secret.from_name(HF_SECRET)])
def export(run: str, base: str, revision: str, examples: list[dict], reference: str = "bf16",
           strength: str = "low", max_new_tokens: int = 512) -> dict:
    """Merge the run's adapter into /vol/runs/<run>/merged; generate on `examples` with the
    reference (base + adapter) and with the merged folder reloaded from the volume."""
    import gc
    import shutil
    from datetime import UTC, datetime

    import torch
    import transformers
    from peft import PeftModel

    if not RUN_NAME.fullmatch(run) or reference not in ("bf16", "nf4"):
        raise ValueError(f"bad run name {run!r} or reference {reference!r} (bf16 or nf4)")
    run_dir = Path(MOUNT) / "runs" / run
    adapter, merged_dir = run_dir / "out", run_dir / "merged"
    if not (adapter / "adapter_config.json").is_file():
        raise FileNotFoundError(f"no adapter in {adapter}: train run {run} first")
    torch.manual_seed(SEED)  # evaluate.py's set_deterministic_state
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = False, True
    today = datetime.now(UTC).date().isoformat()  # one date for both (--current-date)

    def generations(model, processor) -> list[str]:
        return [generate(model, processor, torch, e["messages"][:-1], strength, today,
                         max_new_tokens) for e in examples]

    def with_adapter(nf4: bool):
        model, name = load_model(transformers, torch, base, revision, nf4=nf4)
        return PeftModel.from_pretrained(model, str(adapter), is_trainable=False).eval(), name

    processor = transformers.AutoProcessor.from_pretrained(base, revision=revision)
    model, model_class = with_adapter(nf4=reference == "nf4")
    before = generations(model, processor)
    if reference == "nf4":  # the merge itself always goes into the bf16 base
        del model
        gc.collect()
        torch.cuda.empty_cache()
        model, _ = with_adapter(nf4=False)
    merged = model.merge_and_unload()
    shutil.rmtree(merged_dir, ignore_errors=True)
    merged.save_pretrained(str(merged_dir))  # safetensors by default; bin_files checks it
    processor.save_pretrained(str(merged_dir))
    del model, merged
    gc.collect()
    torch.cuda.empty_cache()
    files = finish_files(merged_dir, base, revision)
    volume.commit()
    reloaded, _ = load_model(transformers, torch, str(merged_dir), None)
    after = generations(reloaded, transformers.AutoProcessor.from_pretrained(str(merged_dir)))
    return {"merged_dir": str(merged_dir), "base": base, "revision": revision,
            "model_class": model_class, "reasoning_strength": strength, "current_date": today,
            **files, "reference_generations": before, "merged_generations": after}


@app.local_entrypoint()
def main(run: str, bundle: str = "", limit: int = 50, reference: str = "bf16",
         result: str = "") -> None:
    from backend import config
    from backend.training.export_check import compare, pick, yes_no

    folder = Path(bundle) if bundle else config.load_settings().data_dir / "train" / run
    manifest = json.loads((folder / "manifest.json").read_text())
    if manifest["image_root"] != f"{MOUNT}/bundles/{run}":
        raise SystemExit(f"the bundle was made for {manifest['image_root']}, not run {run}")
    split = "val" if manifest.get("val_examples") else "train"
    if limit < 1:
        raise SystemExit("--limit must be 1 or more")
    examples = pick((folder / f"{split}.jsonl").read_text().splitlines(), limit)
    if not examples:  # before the paid GPU call, never after it
        raise SystemExit(f"no examples in the bundle's {split}.jsonl to compare")
    base = config.DEFAULT_MODELS["student_base"]
    config.check_model_allowed(base)
    start = time.monotonic()
    try:
        out = export.remote(run, base, config.STUDENT_BASE_REVISION, examples,
                            reference=reference)
    except BaseException:
        if result:  # the GPU time it took, for te export's ledger
            Path(result).write_text(json.dumps({"run": run, "ok": False, "seconds": round(
                time.monotonic() - start, 1)}, indent=1) + "\n")
        raise
    seconds = round(time.monotonic() - start, 1)
    before = [yes_no(text) for text in out.pop("reference_generations")]
    after = [yes_no(text) for text in out.pop("merged_generations")]  # raw text never kept
    summary = {"run": run, "split": split, "reference": f"{reference} base + adapter",
               "seconds": seconds, **out,
               **compare([e["id"] for e in examples],
                         [e["messages"][-1]["content"] for e in examples], before, after)}
    print(json.dumps(summary))
    if result:  # for a te export command
        Path(result).write_text(json.dumps(summary, indent=1) + "\n")
