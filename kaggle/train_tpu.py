"""
TPU-adapted training entry point for CodeForge ASM.

Key differences from the standard trainer:
  - No bitsandbytes / no 4-bit quantization
  - No unsloth
  - Uses torch_xla for TPU device management
  - xm.mark_step() / xm.optimizer_step() replaces plain optimizer.step()
  - bf16 throughout (TPU natively supports it)
  - NASM subprocesses run on the CPU host — unchanged from GPU version
"""
from __future__ import annotations

import os

# Must be set before any JAX/torch_xla import to prevent JAX from
# pre-allocating all TPU HBM (default is 75% pre-allocation).
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# ── torch_xla (must import before torch on TPU) ──────────────────────────────
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    _XLA_AVAILABLE = True
except ImportError:
    torch_xla = None
    xm = None
    _XLA_AVAILABLE = False

_DEVICE = None  # initialized lazily


def _get_device():
    global _DEVICE
    if _DEVICE is not None:
        return _DEVICE
    if _XLA_AVAILABLE:
        _DEVICE = torch_xla.device()
        print(f"[TPU] device: {_DEVICE}")
    else:
        _DEVICE = torch.device("cpu")
        print("[TPU] torch_xla not found — using CPU")
    return _DEVICE

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add project root so we can reuse src modules
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.best_of_n import BestOfN, BestOfNConfig
from src.mcts import MCTSConfig, MCTSLineSearch
from src.prompt_engine import PromptEngine, PromptItem
from src.reward import RewardPipeline
from src.utils import SYS_PROMPT, ensure_dir, sanitize_model_output

try:
    import wandb
except ImportError:
    wandb = None


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class RuntimeConfig:
    raw: dict[str, Any]

    @property
    def iterations(self) -> int:
        return int(self.raw["training"]["iterations"])

    @property
    def prompts_per_iteration(self) -> int:
        return int(self.raw["training"]["prompts_per_iteration"])


def load_config(path: str | Path) -> RuntimeConfig:
    return RuntimeConfig(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


# ── Model loading (no quantization) ──────────────────────────────────────────

def build_model_and_tokenizer(cfg: RuntimeConfig) -> tuple[Any, Any]:
    model_cfg = cfg.raw["model"]
    training = cfg.raw["training"]

    model_name = model_cfg["name_or_path"]
    trust_remote_code = bool(model_cfg.get("trust_remote_code", True))

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = torch.bfloat16  # TPU native; override if config says otherwise
    dtype_str = model_cfg.get("torch_dtype", "bfloat16")
    if hasattr(torch, dtype_str):
        torch_dtype = getattr(torch, dtype_str)

    attn_impl = str(model_cfg.get("attn_implementation", "eager"))

    print(f"[TPU] Loading {model_name} in {torch_dtype} ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        dtype=torch_dtype,
        attn_implementation=attn_impl,
    )

    lora_cfg = LoraConfig(
        r=int(model_cfg.get("lora_r", 16)),
        lora_alpha=int(model_cfg.get("lora_alpha", 32)),
        lora_dropout=float(model_cfg.get("lora_dropout", 0.05)),
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    if bool(training.get("gradient_checkpointing", True)):
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            print("[TPU] Gradient checkpointing enabled")
        except Exception as e:
            print(f"[TPU] Gradient checkpointing skipped: {e}")

    # Move to XLA device
    device = _get_device()
    model = model.to(device)
    print(f"[TPU] Model moved to {device}")

    return model, tokenizer


# ── Generation ────────────────────────────────────────────────────────────────

try:
    from transformers import LogitsProcessor, LogitsProcessorList

    class _XLAMarkStepProcessor(LogitsProcessor):
        """Calls xm.mark_step() after each token so XLA executes eagerly."""
        def __call__(self, input_ids: Any, scores: Any) -> Any:
            if _XLA_AVAILABLE:
                xm.mark_step()
            return scores

    _XLA_LOGITS_PROCESSOR = LogitsProcessorList([_XLAMarkStepProcessor()])
except Exception:
    _XLA_LOGITS_PROCESSOR = None


class TPUTextGenerator:
    def __init__(self, model: Any, tokenizer: Any) -> None:
        self.model = model
        self.tokenizer = tokenizer

    def __call__(self, prompt: str, n: int, max_new_tokens: int, temperature: float, top_p: float) -> list[str]:
        encoded = self.tokenizer(
            [prompt] * n,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        device = _get_device()
        encoded = {k: v.to(device) for k, v in encoded.items()}

        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **encoded,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    logits_processor=_XLA_LOGITS_PROCESSOR,
                )
                if _XLA_AVAILABLE:
                    xm.mark_step()
        finally:
            if was_training:
                self.model.train()

        prompt_len = encoded["input_ids"].shape[1]
        completions = outputs[:, prompt_len:]
        return self.tokenizer.batch_decode(completions, skip_special_tokens=True)


# ── GRPO update (TPU) ─────────────────────────────────────────────────────────

def _group_relative_weights(rows: list[dict[str, Any]]) -> list[float]:
    by_prompt: dict[str, list[int]] = {}
    for i, row in enumerate(rows):
        by_prompt.setdefault(row["prompt_id"], []).append(i)

    weights = [0.0] * len(rows)
    for _, indices in by_prompt.items():
        rewards = [float(rows[i]["reward"]) for i in indices]
        mean = sum(rewards) / len(rewards)
        variance = sum((r - mean) ** 2 for r in rewards) / max(1, len(rewards))
        std = variance ** 0.5
        advantages = [(r - mean) / (std + 1e-6) for r in rewards]

        pos = [max(0.0, a) for a in advantages]
        if sum(pos) == 0.0:
            best = max(range(len(indices)), key=lambda j: rewards[j])
            pos[best] = 1.0

        s = sum(pos)
        norm = [p / s for p in pos]
        for local_j, row_idx in enumerate(indices):
            weights[row_idx] = norm[local_j]

    return weights


def run_grpo_update_tpu(
    rows: list[dict[str, Any]],
    cfg: RuntimeConfig,
    model: Any,
    tokenizer: Any,
    optimizer: Any,
) -> dict[str, float]:
    training = cfg.raw["training"]
    grad_acc = int(training.get("gradient_accumulation_steps", 2))
    max_len = int(training.get("train_max_seq_len", 1024))
    grad_clip = float(training.get("grad_clip_norm", 1.0))

    device = _get_device()
    weights = _group_relative_weights(rows)
    total_loss = 0.0
    used = 0

    model.train()
    optimizer.zero_grad(set_to_none=True)

    for i, row in enumerate(rows):
        weight = float(weights[i])
        if weight <= 0.0:
            continue

        prompt_text = f"{SYS_PROMPT}\n\nTask: {row['instruction']}\n"
        completion = row["asm"].strip() + "\n"
        full_text = prompt_text + completion

        full_enc = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=max_len)
        prompt_enc = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=max_len)
        input_ids = full_enc["input_ids"].to(device)
        attention_mask = full_enc["attention_mask"].to(device)
        labels = input_ids.clone()

        prompt_len = min(prompt_enc["input_ids"].shape[1], labels.shape[1])
        labels[:, :prompt_len] = -100

        out = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits[:, :-1, :].contiguous()
        target = labels[:, 1:].contiguous()

        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            target.view(-1),
            ignore_index=-100,
            reduction="mean",
        )
        weighted_loss = loss * weight
        (weighted_loss / grad_acc).backward()

        total_loss += float(weighted_loss.detach().cpu())
        used += 1

        if used % grad_acc == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            if _XLA_AVAILABLE:
                xm.optimizer_step(optimizer)
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    if used > 0 and used % grad_acc != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        if _XLA_AVAILABLE:
            xm.optimizer_step(optimizer)
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    return {"grpo_loss": total_loss / max(1, used), "kl": 0.0}


# ── Helpers ───────────────────────────────────────────────────────────────────

def evaluate_candidates(
    reward_pipeline: RewardPipeline,
    prompt_item: PromptItem,
    candidates: list[str],
    sample_prefix: str,
) -> list[dict[str, Any]]:
    items = [(prompt_item, c, f"{sample_prefix}-{i}") for i, c in enumerate(candidates)]
    results = reward_pipeline.evaluate_batch(items, workers=min(32, len(items)))
    return [
        {
            "prompt_id": prompt_item.id,
            "instruction": prompt_item.instruction,
            "tier": prompt_item.tier,
            "asm": sanitize_model_output(asm),
            "reward": r.reward,
            "assembled": r.assembled,
            "linked": r.linked,
            "ran": r.ran,
            "correct": r.correct,
            "stage_failed": r.stage_failed,
            "source": "bon",
        }
        for asm, r in zip(candidates, results)
    ]


def _per_tier_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    by_tier: dict[int, list] = {}
    for r in rows:
        by_tier.setdefault(int(r.get("tier", 0)), []).append(r)
    metrics: dict[str, float] = {}
    for tier, tier_rows in sorted(by_tier.items()):
        n = max(1, len(tier_rows))
        metrics[f"reward/tier{tier}_assemble"] = sum(1 for r in tier_rows if r["assembled"]) / n
        metrics[f"reward/tier{tier}_correct"] = sum(1 for r in tier_rows if r["correct"]) / n
        metrics[f"reward/tier{tier}_avg_reward"] = sum(r["reward"] for r in tier_rows) / n
    return metrics


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="kaggle/configs/grpo_config_tpu.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    random.seed(int(cfg.raw["project"]["seed"]))
    training = cfg.raw["training"]
    paths = cfg.raw["paths"]

    artifacts_dir = Path(paths["artifacts_dir"])
    checkpoints_dir = Path(paths["checkpoints_dir"])
    ensure_dir(artifacts_dir)
    ensure_dir(checkpoints_dir)

    model, tokenizer = build_model_and_tokenizer(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training.get("learning_rate", 5e-6)))
    generator = TPUTextGenerator(model=model, tokenizer=tokenizer)

    prompts = PromptEngine(paths["prompt_dataset"])
    reward_pipeline = RewardPipeline(
        artifacts_dir=artifacts_dir,
        timeout_seconds=int(cfg.raw["reward"]["timeout_seconds"]),
    )

    bon_cfg = BestOfNConfig(
        n=int(training["generations_per_prompt"]),
        max_new_tokens=int(training["max_new_tokens"]),
        temperature=float(training["temperature"]),
        top_p=float(training["top_p"]),
    )
    best_of_n = BestOfN(generator=generator, cfg=bon_cfg)

    use_wandb = bool(training.get("use_wandb", False)) and wandb is not None
    if use_wandb and os.environ.get("WANDB_API_KEY"):
        wandb.login(key=os.environ["WANDB_API_KEY"], relogin=True)
        run = wandb.init(project=cfg.raw["project"]["name"], config=cfg.raw, resume="allow")
    else:
        run = None

    print(f"[TPU] Starting training: {cfg.iterations} iterations, {cfg.prompts_per_iteration} prompts/iter")

    for it in range(cfg.iterations):
        batch_prompts = prompts.sample_random(cfg.prompts_per_iteration)
        all_rows: list[dict[str, Any]] = []

        for p in batch_prompts:
            user_prompt = f"{SYS_PROMPT}\n\nTask: {p.instruction}"
            candidates = best_of_n.generate(user_prompt)
            rows = evaluate_candidates(reward_pipeline, p, candidates, sample_prefix=f"it{it}-{p.id}")
            all_rows.extend(rows)

        grpo_metrics = run_grpo_update_tpu(all_rows, cfg, model, tokenizer, optimizer)

        rewards = [r["reward"] for r in all_rows]
        n = max(1, len(all_rows))
        iter_metrics = {
            "iteration": it,
            "reward/mean": sum(rewards) / n,
            "reward/assembly_rate": sum(1 for r in all_rows if r["assembled"]) / n,
            "reward/correct_rate": sum(1 for r in all_rows if r["correct"]) / n,
            **grpo_metrics,
            **_per_tier_metrics(all_rows),
        }

        print(json.dumps(iter_metrics, ensure_ascii=False))
        if run is not None:
            run.log(iter_metrics)

        out_path = artifacts_dir / f"iteration_{it}.json"
        out_path.write_text(json.dumps(all_rows, ensure_ascii=False, indent=2), encoding="utf-8")

        ckpt_path = checkpoints_dir / f"iter_{it}"
        ensure_dir(ckpt_path)
        model.save_pretrained(ckpt_path)
        tokenizer.save_pretrained(ckpt_path)
        print(f"[TPU] Checkpoint saved: {ckpt_path}")

    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
