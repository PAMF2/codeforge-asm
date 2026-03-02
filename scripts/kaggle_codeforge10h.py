#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{ts()}][codeforge10h] {msg}", flush=True)


def load_secret(name: str) -> str | None:
    if os.getenv(name):
        return os.getenv(name)
    try:
        from kaggle_secrets import UserSecretsClient  # type: ignore
    except Exception:
        return None
    try:
        return UserSecretsClient().get_secret(name)
    except Exception:
        return None


def load_env() -> None:
    hf = load_secret("HF_TOKEN") or load_secret("HUGGINGFACE_HUB_TOKEN")
    wb = load_secret("WANDB_API_KEY") or load_secret("WANDB _API_KEY")
    mi = load_secret("MISTRAL_API_KEY")
    if hf:
        os.environ["HF_TOKEN"] = hf.strip()
        os.environ["HUGGINGFACE_HUB_TOKEN"] = hf.strip()
    if wb:
        os.environ["WANDB_API_KEY"] = wb.strip()
    if mi:
        os.environ["MISTRAL_API_KEY"] = mi.strip()
    log(f"HF_TOKEN loaded: {bool(os.getenv('HF_TOKEN'))}")
    log(f"WANDB_API_KEY loaded: {bool(os.getenv('WANDB_API_KEY'))}")
    log(f"MISTRAL_API_KEY loaded: {bool(os.getenv('MISTRAL_API_KEY'))}")


def ensure_system_deps() -> None:
    if shutil.which("nasm") and shutil.which("ld"):
        log("System deps already present: nasm + ld")
        return
    log("Installing system deps: nasm + binutils")
    subprocess.run(["apt-get", "update", "-y"], check=False)
    subprocess.run(["apt-get", "install", "-y", "nasm", "binutils"], check=False)


def read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def latest_checkpoint_iter(ckpt_dir: Path) -> int:
    if not ckpt_dir.exists():
        return -1
    out = -1
    for p in ckpt_dir.iterdir():
        if not p.is_dir():
            continue
        m = re.match(r"iter_(\d+)$", p.name)
        if m:
            out = max(out, int(m.group(1)))
    return out


def iteration_correct_rate(artifacts_dir: Path, iteration: int) -> float:
    p = artifacts_dir / f"iteration_{iteration}.json"
    if not p.exists():
        return 0.0
    rows = json.loads(p.read_text(encoding="utf-8"))
    if not rows:
        return 0.0
    return sum(1 for r in rows if r.get("correct")) / len(rows)


def gpu_line() -> str:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        out = subprocess.check_output(cmd, text=True).strip()
    except Exception:
        return "gpu: n/a"
    return " | ".join(x.strip() for x in out.splitlines())


@dataclass
class Policy:
    name: str
    temperature: float
    top_p: float
    generations_per_prompt: int
    prompts_per_iteration: int
    max_new_tokens: int
    use_mcts_after_iteration: int
    mcts_simulations: int
    mcts_branch_factor: int
    mcts_max_depth: int
    mcts_min_tier: int


POLICIES: dict[str, Policy] = {
    # Agent57-like portfolio for code RL:
    # - Explorer: high-diversity search
    # - Verifier: balanced with verifier-heavy reward
    # - Exploiter: conservative decode + stronger MCTS
    "explorer": Policy(
        name="explorer",
        temperature=0.95,
        top_p=0.98,
        generations_per_prompt=10,
        prompts_per_iteration=8,
        max_new_tokens=128,
        use_mcts_after_iteration=999,
        mcts_simulations=8,
        mcts_branch_factor=2,
        mcts_max_depth=8,
        mcts_min_tier=4,
    ),
    "verifier": Policy(
        name="verifier",
        temperature=0.70,
        top_p=0.92,
        generations_per_prompt=8,
        prompts_per_iteration=10,
        max_new_tokens=128,
        use_mcts_after_iteration=0,
        mcts_simulations=12,
        mcts_branch_factor=3,
        mcts_max_depth=10,
        mcts_min_tier=2,
    ),
    "exploiter": Policy(
        name="exploiter",
        temperature=0.35,
        top_p=0.85,
        generations_per_prompt=6,
        prompts_per_iteration=10,
        max_new_tokens=96,
        use_mcts_after_iteration=0,
        mcts_simulations=20,
        mcts_branch_factor=3,
        mcts_max_depth=12,
        mcts_min_tier=2,
    ),
}


def apply_policy(base_cfg: dict[str, Any], policy: Policy, target_iterations: int) -> dict[str, Any]:
    cfg = json.loads(json.dumps(base_cfg))
    tr = cfg.setdefault("training", {})
    mcts = cfg.setdefault("mcts", {})
    rw = cfg.setdefault("reward", {}).setdefault("stage_weights", {})

    tr["grpo_backend"] = "trl"
    tr["dry_run"] = False
    tr["iterations"] = target_iterations
    tr["use_random_sampling"] = True
    tr["batch_size"] = 1
    tr["gradient_accumulation_steps"] = 4
    tr["learning_rate"] = float(tr.get("learning_rate", 5e-6))
    tr["temperature"] = policy.temperature
    tr["top_p"] = policy.top_p
    tr["generations_per_prompt"] = policy.generations_per_prompt
    tr["prompts_per_iteration"] = policy.prompts_per_iteration
    tr["max_new_tokens"] = policy.max_new_tokens
    tr["use_mcts_after_iteration"] = policy.use_mcts_after_iteration
    tr["use_wandb"] = bool(os.getenv("WANDB_API_KEY"))
    tr["push_to_hub"] = bool(os.getenv("HF_TOKEN"))

    # VerIF-style: put more weight on verifiable correctness signal.
    rw["assemble"] = 0.20
    rw["link"] = 0.20
    rw["run"] = 0.15
    rw["correctness"] = 0.45

    mcts["simulations"] = policy.mcts_simulations
    mcts["branch_factor"] = policy.mcts_branch_factor
    mcts["max_depth"] = policy.mcts_max_depth
    mcts["min_tier"] = policy.mcts_min_tier
    mcts.setdefault("exploration_constant", 1.414)
    mcts.setdefault("max_lines", 30)
    return cfg


def run_train_once(root: Path, cfg_path: Path, start_iter: int) -> int:
    cmd = [
        sys.executable,
        str(root / "train.py"),
        "--config",
        str(cfg_path),
        "--start-iter",
        str(start_iter),
        "--ensure-system-deps",
    ]
    log(f"run: {' '.join(cmd)}")
    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert proc.stdout is not None
    last_hb = time.time()
    for line in proc.stdout:
        print(line.rstrip(), flush=True)
        if time.time() - last_hb >= 30:
            last_hb = time.time()
            log(f"heartbeat | {gpu_line()}")
    return proc.wait()


def ucb_score(avg: float, pulls: int, total: int, c: float = 1.4) -> float:
    bonus = c * math.sqrt(math.log(max(2, total)) / max(1, pulls))
    return avg + bonus


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CodeForge 10h architecture runner (Agent57+AlphaCodium+VerIF+AlphaMath-style)")
    p.add_argument("--root", default="/kaggle/working/codeforge-asm")
    p.add_argument("--base-config", default="configs/grpo_config.codeforge10h.yaml")
    p.add_argument("--runtime-config", default="configs/grpo_config.runtime.yaml")
    p.add_argument("--hours", type=float, default=10.0)
    p.add_argument("--target-iterations", type=int, default=200)
    p.add_argument("--retry-delay-sec", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    root = Path(args.root)
    os.chdir(root)

    load_env()
    ensure_system_deps()

    base_cfg_path = root / args.base_config
    runtime_cfg_path = root / args.runtime_config
    base_cfg = read_yaml(base_cfg_path)
    paths = base_cfg.get("paths", {})
    ckpt_dir = root / str(paths.get("checkpoints_dir", "checkpoints"))
    artifacts_dir = root / str(paths.get("artifacts_dir", "artifacts"))
    deadline = time.time() + args.hours * 3600

    stats: dict[str, dict[str, float]] = {
        name: {"pulls": 0.0, "reward_sum": 0.0} for name in POLICIES
    }
    warmup_order = ["explorer", "verifier", "exploiter"]

    log(
        f"starting | base={base_cfg_path.name} deadline_hours={args.hours} "
        f"target_iterations={args.target_iterations}"
    )

    while time.time() < deadline:
        last = latest_checkpoint_iter(ckpt_dir)
        start_iter = last + 1
        if start_iter >= args.target_iterations:
            log(f"target reached: start_iter={start_iter}")
            break

        if warmup_order:
            chosen = warmup_order.pop(0)
        else:
            total_pulls = int(sum(v["pulls"] for v in stats.values()))
            scored = []
            for name, st in stats.items():
                avg = st["reward_sum"] / max(1.0, st["pulls"])
                scored.append((ucb_score(avg, int(st["pulls"]), total_pulls + 1), name))
            scored.sort(reverse=True)
            chosen = scored[0][1]

        policy = POLICIES[chosen]
        # Run one new iteration at a time for online policy selection.
        cfg = apply_policy(base_cfg, policy, target_iterations=start_iter + 1)
        write_yaml(runtime_cfg_path, cfg)
        log(
            f"iter={start_iter} policy={chosen} "
            f"temp={policy.temperature} gens={policy.generations_per_prompt} "
            f"prompts={policy.prompts_per_iteration} mcts_after={policy.use_mcts_after_iteration}"
        )

        rc = run_train_once(root, runtime_cfg_path, start_iter=start_iter)
        if rc != 0:
            log(f"train exit={rc}; retry in {args.retry_delay_sec}s")
            time.sleep(args.retry_delay_sec)
            continue

        corr = iteration_correct_rate(artifacts_dir, start_iter)
        stats[chosen]["pulls"] += 1.0
        stats[chosen]["reward_sum"] += corr
        log(
            f"iter={start_iter} complete | correct_rate={corr:.3f} "
            f"bandit={ {k: {'pulls': int(v['pulls']), 'avg': round(v['reward_sum']/max(1.0,v['pulls']),3)} for k,v in stats.items()} }"
        )

    log("finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
