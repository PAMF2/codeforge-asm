"""
hf_job.py — CodeForge ASM training job for HuggingFace Jobs API.

This script is uploaded and executed by `run_uv_job`. It:
1. Installs system deps (nasm, binutils) via apt-get
2. Clones the codeforge-asm repo from GitHub
3. Runs the GRPO trainer with real GPU config
4. Pushes checkpoints to mistral-hackaton-2026/codeforge
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], cwd: str | None = None, check: bool = True) -> int:
    print(f"$ {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, check=False, cwd=cwd)
    if check and r.returncode != 0:
        raise RuntimeError(f"Command failed ({r.returncode}): {' '.join(cmd)}")
    return r.returncode


def main() -> None:
    # ── 1. System deps ────────────────────────────────────────────────────────
    run(["apt-get", "update", "-y"])
    run(["apt-get", "install", "-y", "nasm", "binutils", "git"])

    # ── 2. Clone repo ─────────────────────────────────────────────────────────
    repo_dir = Path("/tmp/codeforge-asm")
    if not repo_dir.exists():
        run(["git", "clone", "https://github.com/PAMF2/codeforge-asm.git", str(repo_dir)])
    else:
        run(["git", "pull"], cwd=str(repo_dir))

    # ── 3. Patch config for GPU run ───────────────────────────────────────────
    import yaml  # installed via dependencies

    cfg_path = repo_dir / "configs" / "grpo_config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())

    cfg["training"]["dry_run"] = False
    cfg["training"]["use_wandb"] = bool(os.getenv("WANDB_API_KEY"))
    cfg["training"]["push_to_hub"] = True
    cfg["training"]["hub_repo_id"] = "mistral-hackaton-2026/codeforge"
    cfg["training"]["hub_fallback_repo_id"] = "PAMF2/codeforge"
    cfg["training"]["hub_private"] = True
    cfg["training"]["use_unsloth"] = False
    cfg["training"]["grpo_backend"] = "trl"
    cfg["training"]["iterations"] = 10
    cfg["training"]["prompts_per_iteration"] = 20
    cfg["training"]["generations_per_prompt"] = 8
    cfg["training"]["batch_size"] = 1
    cfg["training"]["gradient_accumulation_steps"] = 8
    cfg["training"]["max_new_tokens"] = 256
    cfg["training"]["use_random_sampling"] = True

    out_cfg = repo_dir / "configs" / "grpo_config.hf_job.yaml"
    out_cfg.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"Config written: {out_cfg}", flush=True)

    # ── 4. Run trainer ────────────────────────────────────────────────────────
    run(
        [sys.executable, "-m", "src.trainer", "--config", str(out_cfg)],
        cwd=str(repo_dir),
    )

    print("[hf_job] Done.", flush=True)


if __name__ == "__main__":
    main()
