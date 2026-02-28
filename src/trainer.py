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


def run_grpo_update(_: list[dict[str, Any]], __: RuntimeConfig) -> dict[str, float]:
    """Stub for TRL GRPO integration.

    Replace this with a real GRPOTrainer step once model loading is wired.
    """
    return {"grpo_loss": 0.0, "kl": 0.0}


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
    generator = DummyGenerator()
    best_of_n = BestOfN(generator=generator, cfg=bon_cfg)

    use_wandb = bool(cfg.raw["training"].get("use_wandb", True)) and wandb is not None
    run = None
    if use_wandb:
        run = wandb.init(project=cfg.raw["project"]["name"], config=cfg.raw)

    print("[CodeForge] Starting loop")
    for it in range(cfg.iterations):
        batch_prompts = prompts.sample(cfg.prompts_per_iteration)
        all_rows: list[dict[str, Any]] = []

        for p in batch_prompts:
            user_prompt = f"{SYS_PROMPT}\n\nTask: {p.instruction}"
            candidates = best_of_n.generate(user_prompt)
            rows = evaluate_candidates(reward_pipeline, p, candidates, sample_prefix=f"it{it}-{p.id}")
            all_rows.extend(rows)

        rewards = [r["reward"] for r in all_rows]
        avg_reward = sum(rewards) / max(1, len(rewards))
        success_assemble = sum(1 for r in all_rows if r["assembled"]) / max(1, len(all_rows))
        success_correct = sum(1 for r in all_rows if r["correct"]) / max(1, len(all_rows))

        grpo_metrics = run_grpo_update(all_rows, cfg)

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

    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
