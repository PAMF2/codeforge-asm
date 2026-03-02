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
OUT_CONFIG = REPO_ROOT / "configs" / "grpo_config.kaggle.resume.fast.yaml"


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


def parse_iters(repo_files: list[str]) -> list[int]:
    out: set[int] = set()
    for path in repo_files:
        m = re.match(r"^checkpoints/iter_(\d+)/", path)
        if m:
            out.add(int(m.group(1)))
    return sorted(out)


def find_local_max_iter(checkpoints_dir: Path) -> int:
    if not checkpoints_dir.exists():
        return -1
    iters = []
    for p in checkpoints_dir.iterdir():
        if not p.is_dir():
            continue
        m = re.match(r"^iter_(\d+)$", p.name)
        if m:
            iters.append(int(m.group(1)))
    return max(iters) if iters else -1


def download_hf_checkpoints_if_needed(repo_id: str, token: str, checkpoints_dir: Path) -> int:
    from huggingface_hub import HfApi, snapshot_download

    api = HfApi(token=token)
    try:
        repo_files = list(api.list_repo_files(repo_id=repo_id, repo_type="model"))
    except Exception as exc:
        print(f"[kaggle_resume_noreinstall] failed listing HF repo {repo_id}: {exc}", flush=True)
        return -1

    iters = parse_iters(repo_files)
    if not iters:
        print("[kaggle_resume_noreinstall] no checkpoints found on HF repo", flush=True)
        return -1

    print(f"[kaggle_resume_noreinstall] HF iters available: {iters}", flush=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    for it in iters:
        local_iter = checkpoints_dir / f"iter_{it}"
        if local_iter.exists():
            continue
        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            token=token,
            local_dir=str(REPO_ROOT),
            allow_patterns=[f"checkpoints/iter_{it}/*"],
            resume_download=True,
        )
        print(f"[kaggle_resume_noreinstall] downloaded iter_{it}", flush=True)
    return iters[-1]


def write_resume_config(cfg: dict[str, Any]) -> Path:
    cfg["model"]["name_or_path"] = "mistralai/Ministral-8B-Instruct-2410"
    cfg["model"]["trust_remote_code"] = True
    cfg["model"]["load_in_4bit"] = True
    cfg["model"]["torch_dtype"] = "float16"
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

    # Quality-2h profile
    tr["iterations"] = min(int(tr.get("iterations", 10)), 5)
    tr["max_new_tokens"] = 80
    tr["prompts_per_iteration"] = 10
    tr["generations_per_prompt"] = 6
    tr["batch_size"] = 2
    tr["gradient_accumulation_steps"] = 4
    tr["use_mcts_after_iteration"] = 3
    tr["use_random_sampling"] = True
    tr["gradient_checkpointing"] = True

    mcts = cfg.setdefault("mcts", {})
    mcts["simulations"] = 8
    mcts["branch_factor"] = 2
    mcts["max_depth"] = 8
    mcts["min_tier"] = 4

    OUT_CONFIG.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    print(f"[kaggle_resume_noreinstall] wrote {OUT_CONFIG}", flush=True)
    return OUT_CONFIG


def main() -> int:
    os.chdir(REPO_ROOT)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"

    # No pip reinstall here: assumes environment already prepared.
    sh("nvidia-smi || true", check=False)

    hf_token = get_secret_or_env("HF_TOKEN")
    mistral_key = get_secret_or_env("MISTRAL_API_KEY")
    wandb_key = get_secret_or_env("WANDB_API_KEY") or get_secret_or_env("WANDB _API_KEY")

    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
    if mistral_key:
        os.environ["MISTRAL_API_KEY"] = mistral_key
    if wandb_key:
        os.environ["WANDB_API_KEY"] = wandb_key

    cfg = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    checkpoints_dir = REPO_ROOT / str(cfg["paths"]["checkpoints_dir"])

    local_max = find_local_max_iter(checkpoints_dir)
    hf_max = -1
    if local_max < 0 and hf_token:
        hf_max = download_hf_checkpoints_if_needed("mistral-hackaton-2026/codeforge", hf_token, checkpoints_dir)
    max_saved_iter = max(local_max, hf_max)

    start_iter = max_saved_iter + 1 if max_saved_iter >= 0 else 0
    total_iters = int(cfg["training"]["iterations"])
    print(
        f"[kaggle_resume_noreinstall] local_max={local_max} hf_max={hf_max} "
        f"start_iter={start_iter}",
        flush=True,
    )
    if start_iter >= total_iters:
        print("[kaggle_resume_noreinstall] nothing to run (already completed all iterations)", flush=True)
        return 0

    out_cfg = write_resume_config(cfg)
    cmd = (
        f"{sys.executable} train.py --config {out_cfg} "
        f"--start-iter {start_iter} --ensure-system-deps"
    )
    print(f"[kaggle_resume_noreinstall] running: {cmd}", flush=True)
    return sh(cmd, check=False)


if __name__ == "__main__":
    raise SystemExit(main())

