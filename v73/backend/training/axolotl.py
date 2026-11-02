"""The Axolotl config for one training run, from Meta's QLoRA recipe.

Source: meta-oss-cookbook at commit b30ca2c,
muse-glimmer/post-training/sft/axolotl/configs/muse-glimmer-30b-qlora.yaml (Axolotl
0.19.0). Every training value below is the recipe's own. What changes per run: the
paths (absolute, on the Modal volume), the W&B names, and, on one H100, gradient
accumulation 32 to keep the recipe's global batch of 32 (the recipe's README says so).

The config is written as JSON, which YAML readers (Axolotl's included) accept as is.
"""

from __future__ import annotations

from backend.config import DEFAULT_MODELS, STUDENT_BASE_REVISION

WANDB_PROJECT = "trained-eye"
GLOBAL_BATCH = 32
CHAT_DATASET = {"type": "chat_template", "field_messages": "messages",
                "roles_to_train": ["assistant"], "train_on_eos": "turn"}

RECIPE = {
    "processor_type": "AutoProcessor",
    "plugins": ["axolotl.integrations.cut_cross_entropy.CutCrossEntropyPlugin"],
    "cut_cross_entropy": True,
    "adapter": "qlora",
    "load_in_4bit": True,
    "lora_r": 32,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    # language projections only: the vision tower stays frozen
    "lora_target_modules": (r"model\.language_model\.layers\.\d+\."
                            r"(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"),
    # Muse's gated attention is incompatible with Axolotl's fused LoRA kernels
    "lora_mlp_kernel": False,
    "lora_qkv_kernel": False,
    "lora_o_kernel": False,
    "lora_embedding_kernel": False,
    "skip_prepare_dataset": True,
    "remove_unused_columns": False,
    "sample_packing": False,
    "eval_sample_packing": False,
    "train_on_inputs": False,
    "val_set_size": 0.0,
    "sequence_len": 4096,
    "pad_to_sequence_len": False,
    "wandb_mode": "online",
    "wandb_log_model": None,  # artifact uploads only after the author approves
    "micro_batch_size": 1,
    "num_epochs": 1,
    "optimizer": "adamw_bnb_8bit",
    "lr_scheduler": "cosine",
    "learning_rate": 0.0002,
    "bf16": True,
    "fp16": False,
    "tf32": True,
    "gradient_checkpointing": True,
    "logging_steps": 1,
    "include_tkps": True,
    "attn_implementation": "flash_attention_2",
    "warmup_ratio": 0.1,
    "eval_strategy": "epoch",
    "save_strategy": "epoch",
    "save_total_limit": 2,
    "weight_decay": 0.0,
    "seed": 42,
}
# The recipe's two-step smoke: crosses two optimizer steps and saves at step 2.
SMOKE = {"max_steps": 2, "gradient_accumulation_steps": 1, "eval_strategy": "no",
         "save_strategy": "steps", "save_steps": 2}


def qlora_config(run: str, train_jsonl: str, output_dir: str, *, val_jsonl: str | None = None,
                 gpus: int = 1, smoke: bool = False, seed: int = 42) -> dict:
    """The full Axolotl config for `run`. Paths must be absolute inside the container."""
    for path in (train_jsonl, output_dir, val_jsonl):
        if path is not None and not path.startswith("/"):
            raise ValueError(f"paths in the training config must be absolute: {path}")
    config = {
        "base_model": DEFAULT_MODELS["student_base"],
        "model_revision": STUDENT_BASE_REVISION,
        **RECIPE,
        "datasets": [{"path": train_jsonl, **CHAT_DATASET}],
        "dataset_prepared_path": f"{output_dir}/prepared",
        "output_dir": output_dir,
        "wandb_project": WANDB_PROJECT,
        "wandb_name": run,
        "gradient_accumulation_steps": GLOBAL_BATCH // gpus,
        "seed": seed,
    }
    if val_jsonl:
        config["test_datasets"] = [{"path": val_jsonl, "split": "train", **CHAT_DATASET}]
    if smoke:
        config.update(SMOKE)
    return config


def optimizer_steps(examples: int, gpus: int = 1) -> int:
    """Optimizer steps for one epoch at the recipe's global batch of 32."""
    per_step = (GLOBAL_BATCH // gpus) * gpus
    return max(1, -(-examples // per_step))
