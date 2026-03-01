#!/usr/bin/env python3
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path("/kaggle/working/codeforge-asm")
BASE_CONFIG = REPO_ROOT / "configs" / "grpo_config.yaml"
OUT_CONFIG = REPO_ROOT / "configs" / "grpo_config.kaggle.resume.yaml"


def sh(cmd: str, check: bool = True) -> int:
    print(f"$ {cmd}", flush=True)
    proc = subprocess.run(cmd, shell=True, check=False)
    if check and proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {cmd}")
    return proc.returncode


def get_secret_or_env(name: str) -> str | None:
    if os.getenv(name):
        return os.getenv(name)
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret(name)
    except Exception:
        return None


def install_python_stack() -> None:
    # Keep this stack consistent to avoid numpy/scipy/sklearn and torchvision issues.
    sh(
        "pip install -q --upgrade --force-reinstall "
        "'numpy==2.1.3' "
        "'scipy==1.14.1' "
        "'scikit-learn==1.5.2' "
        "'transformers==4.57.1' "
        "'tokenizers>=0.22,<0.23' "
        "'trl>=0.21,<0.24' "
        "'accelerate>=1.8,<1.12' "
        "'peft>=0.17,<0.19' "
        "'datasets>=4.0,<5.0' "
        "'huggingface_hub>=0.34,<1.0' "
        "'wandb>=0.20,<0.26' "
        "'pyyaml>=6.0.2' "
        "'bitsandbytes>=0.46.1' "
        "'mistralai>=1.0,<2.0' "
        "'sentencepiece>=0.2.0' "
        "'tiktoken>=0.7.0' "
        "'protobuf<6' "
        "'grpcio-status<1.72'"
    )
    sh("pip uninstall -y torchvision", check=False)


def install_system_deps() -> None:
    sh("apt-get update -qq && apt-get install -y -qq nasm binutils", check=False)


def check_ministral_import() -> None:
    probe = (
        "from transformers.models.ministral.modeling_ministral import MinistralForCausalLM; "
        "print('OK: Ministral import')"
    )
    sh(f"python -c \"{probe}\"")


def parse_iters(repo_files: list[str]) -> list[int]:
    out: set[int] = set()
    for path in repo_files:
        m = re.match(r"^checkpoints/iter_(\\d+)/", path)
        if m:
            out.add(int(m.group(1)))
    return sorted(out)


def download_hf_checkpoints(repo_id: str, token: str, checkpoints_dir: Path) -> int:
    from huggingface_hub import HfApi, snapshot_download

    api = HfApi(token=token)
    try:
        repo_files = list(api.list_repo_files(repo_id=repo_id, repo_type="model"))
    except Exception as exc:
        print(f"[kaggle_resume_final] failed listing HF repo {repo_id}: {exc}", flush=True)
        return -1

    iters = parse_iters(repo_files)
    if not iters:
        print("[kaggle_resume_final] no checkpoints found on HF repo", flush=True)
        return -1

    print(f"[kaggle_resume_final] Found HF checkpoints for iters: {iters}", flush=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    for it in iters:
        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            token=token,
            local_dir=str(REPO_ROOT),
            allow_patterns=[f"checkpoints/iter_{it}/*"],
            resume_download=True,
        )
    print(f"[kaggle_resume_final] Downloaded up to iter_{iters[-1]} from HF", flush=True)
    return iters[-1]


def write_resume_config(cfg: dict[str, Any]) -> Path:
    cfg["model"]["name_or_path"] = "mistralai/Ministral-8B-Instruct-2410"
    cfg["model"]["trust_remote_code"] = True
    cfg["model"]["load_in_4bit"] = True
    cfg["model"]["device_map"] = "balanced"
    cfg["model"]["max_memory_per_gpu_gb"] = 14
    cfg["model"]["attn_implementation"] = "sdpa"

    tr = cfg["training"]
    tr["dry_run"] = False
    tr["grpo_backend"] = "trl"
    tr["use_vllm"] = False
    tr["use_wandb"] = bool(os.getenv("WANDB_API_KEY"))
    tr["push_to_hub"] = bool(os.getenv("HF_TOKEN"))
    tr["hub_repo_id"] = "mistral-hackaton-2026/codeforge"
    tr["hub_fallback_repo_id"] = "PAMF2/codeforge"
    tr["hub_private"] = True

    # Faster settings on Kaggle T4 while keeping training signal.
    tr["max_new_tokens"] = int(tr.get("max_new_tokens", 96))
    tr["prompts_per_iteration"] = int(tr.get("prompts_per_iteration", 12))
    tr["generations_per_prompt"] = int(tr.get("generations_per_prompt", 8))
    tr["batch_size"] = int(tr.get("batch_size", 2))
    tr["gradient_accumulation_steps"] = int(tr.get("gradient_accumulation_steps", 4))
    tr["use_random_sampling"] = True
    tr["gradient_checkpointing"] = True

    OUT_CONFIG.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    print(f"[kaggle_resume_final] wrote {OUT_CONFIG}", flush=True)
    return OUT_CONFIG


def main() -> int:
    os.chdir(REPO_ROOT)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"

    sh("nvidia-smi || true", check=False)
    install_python_stack()
    install_system_deps()

    hf_token = get_secret_or_env("HF_TOKEN")
    mistral_key = get_secret_or_env("MISTRAL_API_KEY")
    wandb_key = get_secret_or_env("WANDB_API_KEY") or get_secret_or_env("WANDB _API_KEY")

    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
    if mistral_key:
        os.environ["MISTRAL_API_KEY"] = mistral_key
    if wandb_key:
        os.environ["WANDB_API_KEY"] = wandb_key

    check_ministral_import()

    cfg = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    checkpoints_dir = REPO_ROOT / str(cfg["paths"]["checkpoints_dir"])
    max_saved_iter = -1
    if hf_token:
        max_saved_iter = download_hf_checkpoints("mistral-hackaton-2026/codeforge", hf_token, checkpoints_dir)
    else:
        print("[kaggle_resume_final] HF_TOKEN missing, skipping checkpoint download", flush=True)

    start_iter = max_saved_iter + 1 if max_saved_iter >= 0 else 0
    total_iters = int(cfg["training"]["iterations"])
    print(f"[kaggle_resume_final] max_saved_iter={max_saved_iter}  start_iter={start_iter}", flush=True)
    if start_iter >= total_iters:
        print("[kaggle_resume_final] nothing to run (already completed all iterations)", flush=True)
        return 0

    out_cfg = write_resume_config(cfg)
    cmd = (
        f"{sys.executable} train.py --config {out_cfg} "
        f"--start-iter {start_iter} --ensure-system-deps"
    )
    print(f"[kaggle_resume_final] running: {cmd}", flush=True)
    return sh(cmd, check=False)


if __name__ == "__main__":
    raise SystemExit(main())
