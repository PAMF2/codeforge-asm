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
import time
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

    # ── Speed optimizations (no vLLM — torch 2.10 incompatible with all TRL-supported vLLM) ──
    # num_train_epochs=1: 3x fewer gradient steps vs default 3.
    # max_new_tokens=80: NASM programs are short; 80 tokens covers all tier 1-3 tasks.
    # prompts_per_iteration=10: halved from 20 → ~2.5x faster generation per iter.
    # generations_per_prompt=8: keep diversity for GRPO variance reduction.
    # Total tokens/iter: 10 × 8 × 80 = 6400 (vs 20 × 8 × 128 = 20480 before → 3.2x faster).
    # Expected: ~15-20 min/iter × 8 iters ≈ 2-2.5h on 2x T4.
    tr["num_train_epochs"] = 1
    tr["max_new_tokens"] = 80
    tr["use_vllm"] = False

    # ── Training config ───────────────────────────────────────────────
    tr["iterations"] = 10
    tr["prompts_per_iteration"] = 10
    tr["generations_per_prompt"] = 8
    tr["batch_size"] = 2
    tr["gradient_accumulation_steps"] = 4
    tr["use_random_sampling"] = True
    tr["use_mcts_after_iteration"] = 2

    out_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    print(f"[kaggle_resume_final] config written → {out_path}  (start_iter={start_iter})", flush=True)
    return out_path


def start_vllm_server(model_name: str, gpu_count: int, hf_token: str) -> "subprocess.Popen | None":
    """
    Start `trl vllm-serve` as a background process, then poll /health until ready.

    Why a separate process: TRL's GRPOTrainer(use_vllm=True) expects the server
    already running at port 8000. It does NOT start it — if nothing is there it
    times out after 240s and crashes.

    Memory math (2x T4, 15GB each):
      vLLM fp16 + tensor_parallel=2 → 8GB/GPU weights + 1GB KV cache = 9GB/GPU
      Training 4-bit balanced        → ~2.25GB/GPU weights + ~0.85GB opt/act
      Total per GPU                  → ~12GB < 15GB  ✓

    Returns the Popen handle if the server becomes healthy within 10 min, else None.
    """
    import requests  # available in Kaggle

    log_path = Path("/kaggle/working/vllm_server.log")
    env = os.environ.copy()
    env["HF_TOKEN"] = hf_token

    cmd = [
        "trl", "vllm-serve",
        "--model", model_name,
        "--tensor-parallel-size", str(gpu_count),
        "--gpu-memory-utilization", "0.60",
        "--max-model-len", "512",
        "--dtype", "half",
        "--port", "8000",
    ]
    print(f"[vllm] Starting server (logs → {log_path})", flush=True)
    log_fh = open(log_path, "w")
    proc = subprocess.Popen(cmd, env=env, stdout=log_fh, stderr=subprocess.STDOUT)

    health_url = "http://127.0.0.1:8000/health"
    for i in range(120):       # 120 × 5s = 600s = 10 min
        time.sleep(5)
        if proc.poll() is not None:
            log_fh.flush()
            print(f"[vllm] Server process died (exit {proc.returncode}). See {log_path}", flush=True)
            log_fh.close()
            return None
        try:
            r = requests.get(health_url, timeout=2)
            if r.ok:
                print(f"[vllm] Server ready after {(i + 1) * 5}s ✓", flush=True)
                log_fh.close()
                return proc
        except Exception:
            pass
        if i % 6 == 0:
            print(f"[vllm] Waiting for server... ({(i + 1) * 5}s elapsed)", flush=True)

    print("[vllm] Server not ready after 600s — killing.", flush=True)
    proc.terminate()
    log_fh.close()
    return None


def main() -> int:
    repo_root = Path("/kaggle/working/codeforge-asm")
    os.chdir(repo_root)

    # 1. Expose both T4s
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    sh("nvidia-smi || true", check=False)

    # 2. Install only training-specific packages.
    #    Pin transformers>=5.0 so pip CANNOT downgrade from Kaggle's 5.2.0:
    #    TRL's dependency metadata says transformers<5 on older releases, which
    #    causes a silent downgrade that removes MinistralForCausalLM.
    #    By declaring >=5.0 here, pip must find a TRL that co-exists with 5.x
    #    or raise a visible conflict — either outcome is better than silent failure.
    #    DO NOT touch torch/tokenizers — Kaggle's torch 2.10.0 is incompatible
    #    with vLLM and must stay as-is.
    sh(
        "pip install -q "
        "pyyaml wandb "
        "'trl>=0.12' "
        "'accelerate>=1.0' "
        "'peft>=0.10' "
        "'bitsandbytes>=0.46.1' "
        "'mistralai>=1.0,<2.0' "
        "'sentencepiece>=0.2.0' "
        "datasets "
        "huggingface_hub "
        "'transformers>=5.0,<6.0'"  # guard: prevents TRL from downgrading 5.2.0 → 4.x
    )

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

    def run_training(cfg_path: Path) -> int:
        cmd = [
            sys.executable, "train.py",
            "--config", str(cfg_path),
            "--start-iter", str(start_iter),
            "--ensure-system-deps",
        ]
        print(f"[kaggle_resume_final] running: {' '.join(cmd)}", flush=True)
        return subprocess.run(cmd, cwd=str(repo_root)).returncode

    # 8. Run training (HF generate, no vLLM)
    return run_training(cfg_out)


if __name__ == "__main__":
    raise SystemExit(main())
