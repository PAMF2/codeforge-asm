#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import yaml


def sh(cmd: str) -> None:
    print(f"$ {cmd}")
    subprocess.run(cmd, shell=True, check=True)


def get_secret(name: str) -> str:
    from kaggle_secrets import UserSecretsClient

    return UserSecretsClient().get_secret(name)


def ensure_utils_exec_error_guard(repo_root: Path) -> None:
    utils_path = repo_root / "src" / "utils.py"
    src = utils_path.read_text(encoding="utf-8")
    if "except OSError as exc" in src:
        print("[kaggle_bootstrap] src/utils.py already handles OSError")
        return

    marker = "    except FileNotFoundError as exc:\n"
    if marker not in src:
        print("[kaggle_bootstrap] could not patch src/utils.py automatically")
        return

    patch = """    except OSError as exc:
        return subprocess.CompletedProcess(
            cmd_list,
            returncode=126,
            stdout="",
            stderr=f"OS error while executing command: {exc}",
        )
"""
    src = src.replace(marker, patch + marker, 1)
    utils_path.write_text(src, encoding="utf-8")
    print("[kaggle_bootstrap] patched src/utils.py with OSError guard")


def write_kaggle_config(repo_root: Path) -> Path:
    cfg_path = repo_root / "configs" / "grpo_config.yaml"
    out_path = repo_root / "configs" / "grpo_config.kaggle.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    cfg["model"]["name_or_path"] = "mistralai/Ministral-8B-Instruct-2410"
    cfg["model"]["load_in_4bit"] = True
    cfg["model"]["trust_remote_code"] = True
    cfg["model"]["device_map"] = "balanced"
    cfg["model"]["max_memory_per_gpu_gb"] = 14

    tr = cfg["training"]
    tr["dry_run"] = False
    tr["grpo_backend"] = "trl"
    tr["use_wandb"] = True
    tr["push_to_hub"] = True
    tr["hub_repo_id"] = "mistral-hackaton-2026/codeforge"
    tr["hub_fallback_repo_id"] = "PAMF2/codeforge"
    tr["batch_size"] = 1
    tr["gradient_accumulation_steps"] = 8
    tr["max_new_tokens"] = 128
    tr["generations_per_prompt"] = 4
    tr["prompts_per_iteration"] = 8

    out_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    print(f"[kaggle_bootstrap] wrote {out_path}")
    return out_path


def main() -> int:
    repo_root = Path("/kaggle/working/codeforge-asm")

    # Ensure Kaggle dual-T4 setup is visible to torch/accelerate.
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    sh("nvidia-smi || true")

    # System + Python deps
    sh("apt-get update -y && apt-get install -y nasm binutils")
    sh(
        "pip install -q -U pyyaml wandb accelerate datasets peft trl "
        "huggingface_hub mistralai transformers 'bitsandbytes>=0.46.1'"
    )

    # Load secrets from Kaggle
    os.environ["HF_TOKEN"] = get_secret("HF_TOKEN")
    os.environ["MISTRAL_API_KEY"] = get_secret("MISTRAL_API_KEY")
    try:
        os.environ["WANDB_API_KEY"] = get_secret("WANDB_API_KEY")
    except Exception:
        os.environ["WANDB_API_KEY"] = get_secret("WANDB _API_KEY")

    ensure_utils_exec_error_guard(repo_root)
    kaggle_cfg = write_kaggle_config(repo_root)

    # Launch training
    sh(f"python train.py --config {kaggle_cfg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
