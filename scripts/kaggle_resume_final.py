#!/usr/bin/env python3
"""
kaggle_resume_final.py — Single-cell Kaggle launcher.

Usage (3 cells):
    %cd /kaggle/working/codeforge-asm
    !git pull
    !python scripts/kaggle_resume_final.py

Behaviour:
- Always loads Ministral-8B base model (not the hub adapter)
- Downloads any existing LoRA checkpoints from HF to local checkpoints_dir
- Auto-detects the highest saved iter and sets --start-iter accordingly
- Runs all remaining iterations (up to cfg.iterations=10)
- Pushes each checkpoint back to mistral-hackaton-2026/codeforge
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import yaml


def sh(cmd: str, check: bool = True) -> int:
    print(f"$ {cmd}", flush=True)
    r = subprocess.run(cmd, shell=True, check=False)
    if check and r.returncode != 0:
        raise RuntimeError(f"Command failed ({r.returncode}): {cmd}")
    return r.returncode


def get_secret(name: str) -> str:
    from kaggle_secrets import UserSecretsClient
    return UserSecretsClient().get_secret(name)


def download_checkpoints_from_hub(
    repo_id: str,
    local_ckpts_dir: Path,
    token: str,
) -> int:
    """
    Download all checkpoints/iter_* from the HF repo into local_ckpts_dir.
    Returns the highest iteration number found (or -1 if none).
    """
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

    # Find all checkpoints/iter_N/** files
    ckpt_files = [f for f in all_files if f.startswith("checkpoints/iter_")]
    if not ckpt_files:
        print("[kaggle_resume_final] No checkpoints found in HF repo.", flush=True)
        return -1

    # Collect unique iter numbers
    iter_nums: set[int] = set()
    for fp in ckpt_files:
        parts = fp.split("/")  # ["checkpoints", "iter_N", "file"]
        if len(parts) >= 2:
            try:
                iter_nums.add(int(parts[1].split("_")[1]))
            except (IndexError, ValueError):
                pass

    if not iter_nums:
        return -1

    max_iter = max(iter_nums)
    print(f"[kaggle_resume_final] Found HF checkpoints for iters: {sorted(iter_nums)}", flush=True)

    # Download files for all iters (latest is what we need)
    for it in sorted(iter_nums):
        iter_files = [f for f in ckpt_files if f.startswith(f"checkpoints/iter_{it}/")]
        local_iter_dir = local_ckpts_dir / f"iter_{it}"
        local_iter_dir.mkdir(parents=True, exist_ok=True)
        for fp in iter_files:
            filename = fp.split("/", 2)[-1]  # strip "checkpoints/iter_N/"
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


def write_full_config(repo_root: Path, start_iter: int, gpu_count: int = 2) -> Path:
    cfg_path = repo_root / "configs" / "grpo_config.yaml"
    out_path = repo_root / "configs" / "grpo_config.kaggle.resume.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    # Always use the base model — LoRA adapter is loaded separately via --start-iter
    cfg["model"]["name_or_path"] = "mistralai/Ministral-8B-Instruct-2410"
    cfg["model"]["load_in_4bit"] = True
    cfg["model"]["trust_remote_code"] = True
    cfg["model"]["device_map"] = "balanced"
    cfg["model"]["max_memory_per_gpu_gb"] = 14
    # SDPA: faster than eager attention, works on T4 (SM75).
    # Flash Attention 2 requires Ampere (SM80+) — not available on T4.
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

    # ── Speed optimizations ──────────────────────────────────────────
    # num_train_epochs=1: 3x fewer gradient steps per iteration vs default 3.
    # max_new_tokens=128: assembly programs are short; 256 wastes generation time.
    # vLLM: 5-10x faster generation than HF generate().
    #   tensor_parallel_size=2 shards model across both T4s (8GB/GPU in fp16),
    #   leaving ~7GB per GPU for the 4-bit training model (~4.5GB).
    #   gpu_memory_utilization=0.45 so vLLM + training fit on same GPUs.
    tr["num_train_epochs"] = 1
    tr["max_new_tokens"] = 128
    tr["use_vllm"] = True
    tr["vllm_tensor_parallel_size"] = gpu_count
    tr["vllm_gpu_memory_utilization"] = 0.45
    tr["vllm_max_model_len"] = 512

    # ── Training config ───────────────────────────────────────────────
    tr["iterations"] = 10
    tr["prompts_per_iteration"] = 20
    tr["generations_per_prompt"] = 8
    tr["batch_size"] = 2
    tr["gradient_accumulation_steps"] = 4
    tr["use_random_sampling"] = True
    tr["use_mcts_after_iteration"] = 2

    out_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    print(f"[kaggle_resume_final] config written → {out_path}  (start_iter={start_iter})", flush=True)
    return out_path


def main() -> int:
    repo_root = Path("/kaggle/working/codeforge-asm")
    os.chdir(repo_root)

    # 1. Expose both T4s
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    sh("nvidia-smi || true", check=False)

    # 2. Install dependencies (vllm for fast generation, accelerate for DDP)
    sh(
        "pip install -q -U pyyaml wandb accelerate datasets peft trl "
        "huggingface_hub mistralai transformers sentencepiece tiktoken "
        "'bitsandbytes>=0.46.1'"
    )
    # vLLM: fast generation engine (5-10x vs HF generate).
    # Install separately so a vLLM failure doesn't block the rest.
    sh("pip install -q vllm || echo '[kaggle_resume_final] vllm install failed, will use HF generate'",
       check=False)

    # 3. Load Kaggle secrets
    os.environ["HF_TOKEN"] = get_secret("HF_TOKEN")
    os.environ["MISTRAL_API_KEY"] = get_secret("MISTRAL_API_KEY")
    try:
        os.environ["WANDB_API_KEY"] = get_secret("WANDB_API_KEY")
    except Exception:
        try:
            os.environ["WANDB_API_KEY"] = get_secret("WANDB _API_KEY")
        except Exception:
            print("[kaggle_resume_final] WARNING: WANDB_API_KEY not found", flush=True)

    # 4. Install system deps (nasm/binutils)
    sh("apt-get update -qq && apt-get install -y -qq nasm binutils", check=False)

    # 5. Detect GPU count
    try:
        import torch
        gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 1
    except Exception:
        gpu_count = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0,1").count(",") + 1)
    print(f"[kaggle_resume_final] GPU count: {gpu_count}", flush=True)

    # 6. Download existing LoRA checkpoints from HF
    cfg_base = yaml.safe_load((repo_root / "configs" / "grpo_config.yaml").read_text())
    checkpoints_dir = Path(cfg_base["paths"]["checkpoints_dir"])
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    hf_token = os.environ["HF_TOKEN"]
    hub_repo = "mistral-hackaton-2026/codeforge"
    max_saved_iter = download_checkpoints_from_hub(hub_repo, checkpoints_dir, hf_token)

    # start_iter = one past the highest saved checkpoint
    start_iter = max(0, max_saved_iter + 1)
    print(f"[kaggle_resume_final] max_saved_iter={max_saved_iter}  start_iter={start_iter}", flush=True)

    # 7. Write optimized config
    cfg_out = write_full_config(repo_root, start_iter, gpu_count=gpu_count)

    def run_training(cfg_path: Path, extra_args: list[str] | None = None) -> int:
        cmd = [
            sys.executable, "train.py",
            "--config", str(cfg_path),
            "--start-iter", str(start_iter),
            "--ensure-system-deps",
        ] + (extra_args or [])
        print(f"[kaggle_resume_final] running: {' '.join(cmd)}", flush=True)
        return subprocess.run(cmd, cwd=str(repo_root)).returncode

    # 8. Run — try with vLLM, fall back to HF generate on failure
    rc = run_training(cfg_out)
    if rc != 0:
        print("[kaggle_resume_final] Training failed (possibly vLLM OOM). Retrying without vLLM...", flush=True)
        cfg = yaml.safe_load(cfg_out.read_text())
        cfg["training"]["use_vllm"] = False
        cfg_out.write_text(yaml.safe_dump(cfg, sort_keys=False))
        rc = run_training(cfg_out)

    return rc


if __name__ == "__main__":
    raise SystemExit(main())
