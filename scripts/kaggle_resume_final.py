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


# ── helpers ───────────────────────────────────────────────────────────────────

def sh(cmd: str, check: bool = True) -> int:
    print(f"$ {cmd}", flush=True)
    r = subprocess.run(cmd, shell=True, check=False)
    if check and r.returncode != 0:
        raise RuntimeError(f"Command failed ({r.returncode}): {cmd}")
    return r.returncode


def get_secret(name: str) -> str:
    from kaggle_secrets import UserSecretsClient
    return UserSecretsClient().get_secret(name)


# ── checkpoint download ────────────────────────────────────────────────────────

def download_checkpoints_from_hub(
    repo_id: str,
    local_ckpts_dir: Path,
    token: str,
) -> int:
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

    iter_nums: set[int] = set()
    for fp in ckpt_files:
        parts = fp.split("/")
        if len(parts) >= 2:
            try:
                iter_nums.add(int(parts[1].split("_")[1]))
            except (IndexError, ValueError):
                pass

    if not iter_nums:
        return -1

    max_iter = max(iter_nums)
    print(f"[kaggle_resume_final] Found HF checkpoints for iters: {sorted(iter_nums)}", flush=True)

    for it in sorted(iter_nums):
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


# ── config ────────────────────────────────────────────────────────────────────

def write_full_config(repo_root: Path, start_iter: int) -> Path:
    cfg_path = repo_root / "configs" / "grpo_config.yaml"
    out_path = repo_root / "configs" / "grpo_config.kaggle.resume.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

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

    # ── Speed: no vLLM (torch 2.10 incompatible), reduced generation load ──
    # max_new_tokens=80: NASM programs are short; covers all tier 1-3 tasks.
    # prompts_per_iteration=10: halved from 20 → ~2.5x faster generation.
    # generations_per_prompt=8: keep diversity for GRPO variance reduction.
    # Total tokens/iter: 10 × 8 × 80 = 6400 (vs 20 × 8 × 128 = 20480 → 3.2x faster).
    # Expected: ~15-20 min/iter × 8 iters ≈ 2-2.5h on 2x T4.
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
    print(f"[kaggle_resume_final] config written → {out_path}  (start_iter={start_iter})", flush=True)
    return out_path


# ── torchvision compat patch ───────────────────────────────────────────────────

def _patch_torchvision_compat() -> None:
    """
    Fix the torchvision circular-import crash that blocks MinistralForCausalLM loading.

    Root cause: torchvision/_meta_registrations.py calls
      torchvision.extension._has_ops()
    at module-load time (inside a decorator applied at line 25).
    But torchvision.extension hasn't been set yet — we're still inside
    torchvision/__init__.py — so Python raises:
      AttributeError: partially initialized module 'torchvision' has no attribute 'extension'

    This crashes the entire import chain:
      MinistralForCausalLM → modeling_layers → processing_utils
        → image_utils → torchvision (boom)

    Fix 1 (root cause): patch torchvision/_meta_registrations.py to guard with hasattr().
    Fix 2 (comprehensive): scan ALL transformers .py files for unguarded
      `from torchvision.transforms import InterpolationMode` and wrap in try/except.
      Ministral-8B is text-only — InterpolationMode is never actually used at runtime.
    """
    import importlib.util as _ilu

    # ── Fix 1: torchvision/_meta_registrations.py (root cause) ──────────────
    try:
        tv_spec = _ilu.find_spec("torchvision")
        if tv_spec and tv_spec.origin:
            meta = Path(tv_spec.origin).parent / "_meta_registrations.py"
            if meta.exists():
                text = meta.read_text(encoding="utf-8")
                old = "if torchvision.extension._has_ops():"
                new = "if hasattr(torchvision, 'extension') and torchvision.extension._has_ops():"
                count = text.count(old)
                if count and new not in text:
                    meta.write_text(text.replace(old, new), encoding="utf-8")
                    print(f"[fix_tv] patched torchvision/_meta_registrations.py ({count} sites) ✓", flush=True)
                else:
                    print("[fix_tv] _meta_registrations.py already patched / pattern absent", flush=True)
        else:
            print("[fix_tv] torchvision not found — skipping Fix 1", flush=True)
    except Exception as exc:
        print(f"[fix_tv] Fix 1 error: {exc}", flush=True)

    # ── Fix 2: scan ALL transformers .py files ────────────────────────────────
    try:
        tf_spec = _ilu.find_spec("transformers")
        if not tf_spec or not tf_spec.origin:
            print("[fix_tv] transformers not found — skipping Fix 2", flush=True)
            return
        tf_dir = Path(tf_spec.origin).parent

        TARGET = "from torchvision.transforms import InterpolationMode"
        STUB = (
            "try:\n"
            "    from torchvision.transforms import InterpolationMode\n"
            "except Exception:  # torchvision circular-import bug in some versions\n"
            "    from enum import IntEnum\n"
            "    class InterpolationMode(IntEnum):  # stub — text models don't use this\n"
            "        NEAREST = 0\n"
            "        BILINEAR = 2\n"
            "        BICUBIC = 3\n"
            "        LANCZOS = 1"
        )

        patched: list[str] = []
        for py_file in sorted(tf_dir.rglob("*.py")):
            try:
                text = py_file.read_text(encoding="utf-8")
            except Exception:
                continue
            if TARGET not in text:
                continue
            idx = text.find(TARGET)
            ctx = text[max(0, idx - 60): idx + 10]
            if "try:" in ctx or "except" in ctx:
                continue  # already guarded
            py_file.write_text(text.replace(TARGET, STUB, 1), encoding="utf-8")
            patched.append(py_file.name)

        if patched:
            print(f"[fix_tv] Fix 2: patched {len(patched)} files: {patched}", flush=True)
        else:
            print("[fix_tv] Fix 2: no unguarded InterpolationMode imports found", flush=True)
    except Exception as exc:
        print(f"[fix_tv] Fix 2 error: {exc}", flush=True)


# ── pre-training diagnostic ────────────────────────────────────────────────────

def _diagnose_model_import() -> bool:
    """
    Run a subprocess to test MinistralForCausalLM import and print the FULL traceback.
    Returns True if import succeeds.
    """
    diag = Path("/tmp/codeforge_diag.py")
    diag.write_text(
        "import sys, traceback\n"
        "try:\n"
        "    from transformers.models.ministral.modeling_ministral import MinistralForCausalLM\n"
        "    print('[diag] MinistralForCausalLM import OK ✓')\n"
        "    sys.exit(0)\n"
        "except Exception:\n"
        "    print('[diag] IMPORT FAILED — full traceback:')\n"
        "    traceback.print_exc()\n"
        "    sys.exit(1)\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, str(diag)], capture_output=True, text=True
    )
    output = (result.stdout + result.stderr).strip()
    print(output, flush=True)
    return result.returncode == 0


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    repo_root = Path("/kaggle/working/codeforge-asm")
    os.chdir(repo_root)

    # 1. Expose both T4s
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"
    sh("nvidia-smi || true", check=False)

    # 2. Install training-specific packages.
    #    CRITICAL: pin transformers>=5.0 to keep Kaggle's 5.2.0 which has
    #    MinistralForCausalLM. Versions <5.0 do NOT have this class.
    #    DO NOT touch torch/tokenizers/torchvision — Kaggle's torch 2.10.0
    #    is incompatible with vLLM and must stay as-is.
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
        "'transformers>=5.0,<6.0'"  # MUST be >=5.0 for MinistralForCausalLM
    )

    # 3. Text-only training; remove torchvision to avoid torch/vision operator mismatch.
    sh("pip uninstall -y torchvision", check=False)

    # 4. Load Kaggle secrets
    os.environ["HF_TOKEN"] = get_secret("HF_TOKEN")
    os.environ["MISTRAL_API_KEY"] = get_secret("MISTRAL_API_KEY")
    try:
        os.environ["WANDB_API_KEY"] = get_secret("WANDB_API_KEY")
    except Exception:
        try:
            os.environ["WANDB_API_KEY"] = get_secret("WANDB _API_KEY")
        except Exception:
            print("[kaggle_resume_final] WARNING: WANDB_API_KEY not found", flush=True)

    # 5. Install system deps (nasm/binutils)
    sh("apt-get update -qq && apt-get install -y -qq nasm binutils", check=False)

    # 6. Detect GPU count
    try:
        import torch
        gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 1
    except Exception:
        gpu_count = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0,1").count(",") + 1)
    print(f"[kaggle_resume_final] GPU count: {gpu_count}", flush=True)

    # 7. Verify MinistralForCausalLM can be imported (shows full error if not)
    print("[kaggle_resume_final] Verifying model import...", flush=True)
    if not _diagnose_model_import():
        print("[kaggle_resume_final] Model import FAILED — check [diag] output above", flush=True)
        return 1

    # 8. Download existing LoRA checkpoints from HF
    cfg_base = yaml.safe_load((repo_root / "configs" / "grpo_config.yaml").read_text())
    checkpoints_dir = Path(cfg_base["paths"]["checkpoints_dir"])
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    hf_token = os.environ["HF_TOKEN"]
    hub_repo = "mistral-hackaton-2026/codeforge"
    max_saved_iter = download_checkpoints_from_hub(hub_repo, checkpoints_dir, hf_token)

    start_iter = max(0, max_saved_iter + 1)
    print(f"[kaggle_resume_final] max_saved_iter={max_saved_iter}  start_iter={start_iter}", flush=True)

    # 9. Write optimised config
    cfg_out = write_full_config(repo_root, start_iter)

    # 10. Run training (HF generate, no vLLM)
    cmd = [
        sys.executable, "train.py",
        "--config", str(cfg_out),
        "--start-iter", str(start_iter),
        "--ensure-system-deps",
    ]
    print(f"[kaggle_resume_final] running: {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, cwd=str(repo_root)).returncode


if __name__ == "__main__":
    raise SystemExit(main())
