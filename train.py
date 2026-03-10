from __future__ import annotations

# This is the only file the agent edits during autoresearch.

import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = ROOT / "artifacts"
WORKSPACE_DIR = ROOT / "workspace"


# ─────────────────────────────────────────────────────────────────────────────
# RESEARCH CONFIG — agent edits this section
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ResearchConfig:
    # Iterations per experiment (each iteration = N prompts × K generations)
    iterations: int = 5
    prompts_per_iteration: int = 10
    use_random_sampling: bool = True
    generations_per_prompt: int = 8

    # Optimization
    learning_rate: float = 5e-6
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    grad_clip_norm: float = 1.0
    kl_beta: float = 0.1
    grpo_backend: str = "manual"  # "manual" or "trl"

    # Generation
    train_max_seq_len: int = 1024
    max_new_tokens: int = 512
    temperature: float = 0.8
    top_p: float = 0.95

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    # Reward weights (assemble + link + run + correctness should sum to 1.0)
    reward_assemble: float = 0.25
    reward_link: float = 0.25
    reward_run: float = 0.20
    reward_correctness: float = 0.30
    reward_timeout: float = 12.0

    # MCTS (set use_mcts_after_iteration < iterations to enable)
    use_mcts_after_iteration: int = 999
    mcts_simulations: int = 32
    mcts_min_tier: int = 3


RESEARCH = ResearchConfig()

# ─────────────────────────────────────────────────────────────────────────────


def ensure_system_deps() -> None:
    if platform.system().lower() != "linux":
        return
    if shutil.which("nasm") and shutil.which("ld"):
        return
    subprocess.run(["apt-get", "install", "-y", "nasm", "binutils"], check=False,
                   capture_output=True)


def build_yaml(cfg: ResearchConfig) -> dict:
    return {
        "project": {"name": "codeforge-asm", "seed": 42},
        "model": {
            "name_or_path": "mistralai/Ministral-8B-Instruct-2410",
            "trust_remote_code": True,
            "load_in_4bit": True,
            "lora_r": cfg.lora_r,
            "lora_alpha": cfg.lora_alpha,
            "lora_dropout": cfg.lora_dropout,
        },
        "training": {
            "grpo_backend": cfg.grpo_backend,
            "iterations": cfg.iterations,
            "prompts_per_iteration": cfg.prompts_per_iteration,
            "use_random_sampling": cfg.use_random_sampling,
            "generations_per_prompt": cfg.generations_per_prompt,
            "learning_rate": cfg.learning_rate,
            "batch_size": cfg.batch_size,
            "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
            "grad_clip_norm": cfg.grad_clip_norm,
            "train_max_seq_len": cfg.train_max_seq_len,
            "max_new_tokens": cfg.max_new_tokens,
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
            "kl_beta": cfg.kl_beta,
            "use_mcts_after_iteration": cfg.use_mcts_after_iteration,
            "use_wandb": False,
            "push_to_hub": False,
            "dry_run": False,
            "use_unsloth": False,
        },
        "reward": {
            "stage_weights": {
                "assemble": cfg.reward_assemble,
                "link": cfg.reward_link,
                "run": cfg.reward_run,
                "correctness": cfg.reward_correctness,
            },
            "timeout_seconds": cfg.reward_timeout,
        },
        "mcts": {
            "simulations": cfg.mcts_simulations,
            "max_lines": 30,
            "branch_factor": 4,
            "exploration_constant": 1.414,
            "max_depth": 15,
            "min_tier": cfg.mcts_min_tier,
        },
        "paths": {
            "prompt_dataset": "prompts/dataset.json",
            "artifacts_dir": "artifacts",
            "checkpoints_dir": "checkpoints",
        },
    }


def read_metrics() -> dict:
    """Read metrics from the last artifact file produced by training."""
    artifact_files = sorted(ARTIFACTS_DIR.glob("iteration_*.json"))
    if not artifact_files:
        return {}

    rows = json.loads(artifact_files[-1].read_text(encoding="utf-8"))
    if not rows:
        return {}

    total = len(rows)
    correct = sum(1 for r in rows if r.get("correct", False))
    assembled = sum(1 for r in rows if r.get("assembled", False))
    rewards = [r.get("reward", 0.0) for r in rows]

    return {
        "correct_rate": correct / total if total else 0.0,
        "assembly_rate": assembled / total if total else 0.0,
        "reward_mean": sum(rewards) / len(rewards) if rewards else 0.0,
        "n_samples": total,
    }


def main() -> None:
    ensure_system_deps()

    cfg_dict = build_yaml(RESEARCH)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", dir=ROOT, delete=False, encoding="utf-8"
    ) as f:
        yaml.dump(cfg_dict, f, default_flow_style=False)
        tmp_config = f.name

    t0 = time.time()

    try:
        ret = subprocess.run(
            [sys.executable, "-m", "src.trainer", "--config", tmp_config, "--start-iter", "0"],
            cwd=ROOT,
            env={**os.environ, "PYTHONUTF8": "1"},
            check=False,
        ).returncode
    finally:
        Path(tmp_config).unlink(missing_ok=True)

    elapsed = time.time() - t0

    try:
        import torch
        peak_vram_mb = float(torch.cuda.max_memory_allocated() / 1024 / 1024) if torch.cuda.is_available() else 0.0
    except Exception:
        peak_vram_mb = 0.0

    metrics = read_metrics()

    print("---")
    print(f"primary_metric:    {metrics.get('correct_rate', 0.0):.6f}")
    print(f"correct_rate:      {metrics.get('correct_rate', 0.0):.6f}")
    print(f"assembly_rate:     {metrics.get('assembly_rate', 0.0):.6f}")
    print(f"reward_mean:       {metrics.get('reward_mean', 0.0):.6f}")
    print(f"training_seconds:  {elapsed:.1f}")
    print(f"iterations_done:   {RESEARCH.iterations}")
    print(f"n_samples:         {metrics.get('n_samples', 0)}")
    print(f"peak_vram_mb:      {peak_vram_mb:.1f}")
    print("config_json:")
    print(json.dumps(asdict(RESEARCH), indent=2))

    if ret != 0:
        raise SystemExit(f"Training subprocess exited with code {ret}")


if __name__ == "__main__":
    main()
