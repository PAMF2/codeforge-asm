#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import yaml


def sh(cmd: str, check: bool = True) -> int:
    print(f"$ {cmd}", flush=True)
    proc = subprocess.run(cmd, shell=True, check=False)
    if check and proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {cmd}")
    return proc.returncode


def get_secret(name: str) -> str:
    from kaggle_secrets import UserSecretsClient

    return UserSecretsClient().get_secret(name)


def download_checkpoints_from_hub(repo_id: str, local_ckpts_dir: Path, token: str) -> int:
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError:
        print("[kaggle_resume_final] huggingface_hub not available", flush=True)
        return -1

    api = HfApi(token=token)
    try:
        all_files = list(api.list_repo_files(repo_id, repo_type="model"))
    except Exception as exc:
        print(f"[kaggle_resume_final] Cannot list HF repo files: {exc}", flush=True)
        return -1

    ckpt_files = [f for f in all_files if f.startswith("checkpoints/iter_")]
    if not ckpt_files:
        print("[kaggle_resume_final] No checkpoints found in HF repo.", flush=True)
        return -1

    iters: set[int] = set()
    for fp in ckpt_files:
        parts = fp.split("/")
        if len(parts) >= 2:
            try:
                iters.add(int(parts[1].split("_")[1]))
            except Exception:
                pass

    if not iters:
        return -1

    max_iter = max(iters)
    print(f"[kaggle_resume_final] Found HF checkpoints for iters: {sorted(iters)}", flush=True)

    for it in sorted(iters):
        iter_files = [f for f in ckpt_files if f.startswith(f"checkpoints/iter_{it}/")]
        local_iter_dir = local_ckpts_dir / f"iter_{it}"
        local_iter_dir.mkdir(parents=True, exist_ok=True)
        for fp in iter_files:
            filename = fp.split("/", 2)[-1]
            dest = local_iter_dir / filename
            if dest.exists():
                continue
            try:
                hf_hub_download(
                    repo_id=repo_id,
                    filename=fp,
                    repo_type="model",
                    local_dir=str(local_ckpts_dir.parent),
                    token=token,
                )
            except Exception as exc:
                print(f"[kaggle_resume_final] Warning downloading {fp}: {exc}", flush=True)

    print(f"[kaggle_resume_final] Downloaded up to iter_{max_iter} from HF", flush=True)
    return max_iter


def write_full_config(repo_root: Path, start_iter: int) -> Path:
    cfg_path = repo_root / "configs" / "grpo_config.yaml"
    out_path = repo_root / "configs" / "grpo_config.kaggle.resume.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    cfg["model"]["name_or_path"] = "mistralai/Ministral-8B-Instruct-2410"
    cfg["model"]["load_in_4bit"] = True
    cfg["model"]["trust_remote_code"] = True
    cfg["model"]["device_map"] = "balanced"
    cfg["model"]["max_memory_per_gpu_gb"] = 14
    cfg["model"]["attn_implementation"] = "sdpa"
    cfg["model"]["gradient_checkpointing"] = True

    tr = cfg["training"]
    tr["dry_run"] = False
    tr["grpo_backend"] = "trl"
    tr["use_wandb"] = True
    tr["push_to_hub"] = True
    tr["hub_repo_id"] = "mistral-hackaton-2026/codeforge"
    tr["hub_fallback_repo_id"] = "PAMF2/codeforge"
    tr["hub_private"] = True

    tr["use_vllm"] = False
    tr["num_train_epochs"] = 1
    tr["max_new_tokens"] = 80
    tr["iterations"] = 10
    tr["prompts_per_iteration"] = 10
    tr["generations_per_prompt"] = 8
    tr["batch_size"] = 2
    tr["gradient_accumulation_steps"] = 4
    tr["use_random_sampling"] = True
    tr["use_mcts_after_iteration"] = 2

    out_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    print(f"[kaggle_resume_final] config written  {out_path}  (start_iter={start_iter})", flush=True)
    return out_path


def main() -> int:
    repo_root = Path("/kaggle/working/codeforge-asm")
    os.chdir(repo_root)

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    sh("nvidia-smi || true", check=False)

    # Clean stable stack to avoid notebook pollution/patch side effects.
    sh(
        "pip install -q --upgrade --force-reinstall "
        "'transformers==4.57.1' "
        "'tokenizers>=0.22,<0.23' "
        "'trl>=0.21,<0.24' "
        "'accelerate>=1.8,<1.12' "
        "'peft>=0.17,<0.19' "
        "'bitsandbytes>=0.46.1' "
        "'mistralai>=1.0,<2.0' "
        "'sentencepiece>=0.2.0' "
        "'tiktoken>=0.7.0' "
        "'datasets>=4.0,<5.0' "
        "'huggingface_hub>=0.34,<1.0' "
        "'wandb>=0.20,<0.26' "
        "'pyyaml>=6.0.2' "
        "'protobuf<6' "
        "'grpcio-status<1.72'"
    )

    os.environ["HF_TOKEN"] = get_secret("HF_TOKEN")
    os.environ["MISTRAL_API_KEY"] = get_secret("MISTRAL_API_KEY")
    try:
        os.environ["WANDB_API_KEY"] = get_secret("WANDB_API_KEY")
    except Exception:
        os.environ["WANDB_API_KEY"] = get_secret("WANDB _API_KEY")

    sh("apt-get update -qq && apt-get install -y -qq nasm binutils", check=False)

    try:
        import torch

        gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 1
    except Exception:
        gpu_count = 2
    print(f"[kaggle_resume_final] GPU count: {gpu_count}", flush=True)

    cfg_base = yaml.safe_load((repo_root / "configs" / "grpo_config.yaml").read_text())
    checkpoints_dir = Path(cfg_base["paths"]["checkpoints_dir"])
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    hf_token = os.environ["HF_TOKEN"]
    hub_repo = "mistral-hackaton-2026/codeforge"
    max_saved_iter = download_checkpoints_from_hub(hub_repo, checkpoints_dir, hf_token)

    start_iter = max(0, max_saved_iter + 1)
    print(f"[kaggle_resume_final] max_saved_iter={max_saved_iter}  start_iter={start_iter}", flush=True)

    cfg_out = write_full_config(repo_root, start_iter)

    cmd = [
        sys.executable,
        "train.py",
        "--config",
        str(cfg_out),
        "--start-iter",
        str(start_iter),
        "--ensure-system-deps",
    ]
    print(f"[kaggle_resume_final] running: {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, cwd=str(repo_root)).returncode


if __name__ == "__main__":
    raise SystemExit(main())
