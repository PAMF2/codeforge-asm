from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .best_of_n import BestOfN, BestOfNConfig
from .prompt_engine import PromptEngine, PromptItem
from .reward import RewardPipeline
from .utils import SYS_PROMPT, ensure_dir

try:
    import wandb
except Exception:  # pragma: no cover
    wandb = None

try:
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None
    F = None

try:
    from peft import LoraConfig, get_peft_model
except Exception:  # pragma: no cover
    LoraConfig = None
    get_peft_model = None

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
except Exception:  # pragma: no cover
    AutoModelForCausalLM = None
    AutoTokenizer = None
    BitsAndBytesConfig = None


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


class DummyGenerator:
    """MVP generator to validate the full RL data path before model wiring."""

    def __call__(self, prompt: str, n: int, max_new_tokens: int, temperature: float, top_p: float) -> list[str]:
        del prompt, max_new_tokens, temperature, top_p
        return [
            """global _start
section .text
_start:
    mov rax, 60
    mov rdi, 0
    syscall"""
            for _ in range(n)
        ]


class HFTextGenerator:
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
        encoded = {k: v.to(self.model.device) for k, v in encoded.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **encoded,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        prompt_len = encoded["input_ids"].shape[1]
        completions = outputs[:, prompt_len:]
        return self.tokenizer.batch_decode(completions, skip_special_tokens=True)


@dataclass
class TrainBundle:
    model: Any
    tokenizer: Any
    optimizer: Any
    generator: Any


def maybe_build_train_bundle(cfg: RuntimeConfig) -> TrainBundle | None:
    training = cfg.raw["training"]
    if bool(training.get("dry_run", True)):
        return None

    if AutoModelForCausalLM is None or AutoTokenizer is None or torch is None:
        raise RuntimeError("Missing ML dependencies. Install requirements and rerun.")

    model_cfg = cfg.raw["model"]
    model_name = model_cfg["name_or_path"]
    trust_remote_code = bool(model_cfg.get("trust_remote_code", True))
    load_in_4bit = bool(model_cfg.get("load_in_4bit", True))

    quant_cfg = None
    if load_in_4bit:
        if BitsAndBytesConfig is None:
            raise RuntimeError("bitsandbytes/transformers quantization config unavailable.")
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        device_map="auto",
        quantization_config=quant_cfg,
    )

    if get_peft_model is not None and LoraConfig is not None:
        lora_cfg = LoraConfig(
            r=int(model_cfg.get("lora_r", 16)),
            lora_alpha=int(model_cfg.get("lora_alpha", 32)),
            lora_dropout=float(model_cfg.get("lora_dropout", 0.05)),
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "up_proj",
                "down_proj",
                "gate_proj",
            ],
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

    learning_rate = float(training.get("learning_rate", 5e-6))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    generator = HFTextGenerator(model=model, tokenizer=tokenizer)

    return TrainBundle(model=model, tokenizer=tokenizer, optimizer=optimizer, generator=generator)


def _group_relative_weights(rows: list[dict[str, Any]]) -> list[float]:
    by_prompt: dict[str, list[int]] = {}
    for i, row in enumerate(rows):
        by_prompt.setdefault(row["prompt_id"], []).append(i)

    weights = [0.0 for _ in rows]
    for _, indices in by_prompt.items():
        rewards = [float(rows[i]["reward"]) for i in indices]
        mean = sum(rewards) / len(rewards)
        variance = sum((r - mean) ** 2 for r in rewards) / max(1, len(rewards))
        std = variance ** 0.5
        advantages = [(r - mean) / (std + 1e-6) for r in rewards]

        pos = [max(0.0, a) for a in advantages]
        if sum(pos) == 0.0:
            best_idx = max(range(len(indices)), key=lambda j: rewards[j])
            pos[best_idx] = 1.0

        s = sum(pos)
        norm = [p / s for p in pos]
        for local_j, row_idx in enumerate(indices):
            weights[row_idx] = norm[local_j]

    return weights


def run_grpo_update(rows: list[dict[str, Any]], cfg: RuntimeConfig, bundle: TrainBundle | None) -> dict[str, float]:
    """Group-relative update (GRPO-inspired objective) over generated candidates."""
    if bundle is None:
        return {"grpo_loss": 0.0, "kl": 0.0}

    if torch is None or F is None:
        raise RuntimeError("Torch is required for non-dry training mode.")

    model = bundle.model
    tokenizer = bundle.tokenizer
    optimizer = bundle.optimizer
    model.train()

    grad_acc = int(cfg.raw["training"].get("gradient_accumulation_steps", 4))
    max_len = int(cfg.raw["training"].get("train_max_seq_len", 1024))
    grad_clip = float(cfg.raw["training"].get("grad_clip_norm", 1.0))

    weights = _group_relative_weights(rows)
    total_loss = 0.0
    used = 0

    optimizer.zero_grad(set_to_none=True)
    for i, row in enumerate(rows):
        weight = float(weights[i])
        if weight <= 0.0:
            continue

        prompt_text = f"{SYS_PROMPT}\\n\\nTask: {row['instruction']}\\n"
        completion = row["asm"].strip() + "\\n"
        full_text = prompt_text + completion

        full_enc = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=max_len)
        prompt_enc = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=max_len)
        input_ids = full_enc["input_ids"].to(model.device)
        attention_mask = full_enc["attention_mask"].to(model.device)
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
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    if used > 0 and used % grad_acc != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    avg_loss = total_loss / max(1, used)
    return {"grpo_loss": avg_loss, "kl": 0.0}


def evaluate_candidates(
    reward_pipeline: RewardPipeline,
    prompt_item: PromptItem,
    candidates: list[str],
    sample_prefix: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, asm in enumerate(candidates):
        sample_id = f"{sample_prefix}-{idx}"
        result = reward_pipeline.evaluate(prompt_item, asm, sample_id)
        rows.append(
            {
                "prompt_id": prompt_item.id,
                "instruction": prompt_item.instruction,
                "asm": asm,
                "reward": result.reward,
                "assembled": result.assembled,
                "linked": result.linked,
                "ran": result.ran,
                "correct": result.correct,
                "stage_failed": result.stage_failed,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/grpo_config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    random.seed(int(cfg.raw["project"]["seed"]))

    paths = cfg.raw["paths"]
    artifacts_dir = Path(paths["artifacts_dir"])
    ensure_dir(artifacts_dir)

    prompts = PromptEngine(paths["prompt_dataset"])
    reward_pipeline = RewardPipeline(
        artifacts_dir=artifacts_dir,
        timeout_seconds=int(cfg.raw["reward"]["timeout_seconds"]),
    )

    bon_cfg = BestOfNConfig(
        n=int(cfg.raw["training"]["generations_per_prompt"]),
        max_new_tokens=int(cfg.raw["training"]["max_new_tokens"]),
        temperature=float(cfg.raw["training"]["temperature"]),
        top_p=float(cfg.raw["training"]["top_p"]),
    )

    bundle = maybe_build_train_bundle(cfg)
    generator = bundle.generator if bundle is not None else DummyGenerator()
    best_of_n = BestOfN(generator=generator, cfg=bon_cfg)
    print(f"[CodeForge] Training mode: {'real' if bundle is not None else 'dry_run'}")

    use_wandb = bool(cfg.raw["training"].get("use_wandb", True)) and wandb is not None
    run = None
    if use_wandb:
        run = wandb.init(project=cfg.raw["project"]["name"], config=cfg.raw)

    print("[CodeForge] Starting loop")
    for it in range(cfg.iterations):
        batch_prompts = prompts.sample(cfg.prompts_per_iteration)
        all_rows: list[dict[str, Any]] = []

        for p in batch_prompts:
            user_prompt = f"{SYS_PROMPT}\\n\\nTask: {p.instruction}"
            candidates = best_of_n.generate(user_prompt)
            rows = evaluate_candidates(reward_pipeline, p, candidates, sample_prefix=f"it{it}-{p.id}")
            all_rows.extend(rows)

        rewards = [r["reward"] for r in all_rows]
        avg_reward = sum(rewards) / max(1, len(rewards))
        success_assemble = sum(1 for r in all_rows if r["assembled"]) / max(1, len(all_rows))
        success_correct = sum(1 for r in all_rows if r["correct"]) / max(1, len(all_rows))

        grpo_metrics = run_grpo_update(all_rows, cfg, bundle=bundle)

        iter_metrics = {
            "iteration": it,
            "avg_reward": avg_reward,
            "assemble_success_rate": success_assemble,
            "correctness_rate": success_correct,
            **grpo_metrics,
        }

        print(json.dumps(iter_metrics, ensure_ascii=False))
        if run is not None:
            run.log(iter_metrics)

        out_path = artifacts_dir / f"iteration_{it}.json"
        out_path.write_text(json.dumps(all_rows, ensure_ascii=False, indent=2), encoding="utf-8")

        if bundle is not None:
            checkpoints_dir = Path(paths["checkpoints_dir"])
            ensure_dir(checkpoints_dir)
            ckpt_path = checkpoints_dir / f"iter_{it}"
            ensure_dir(ckpt_path)
            bundle.model.save_pretrained(ckpt_path)
            bundle.tokenizer.save_pretrained(ckpt_path)

    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
