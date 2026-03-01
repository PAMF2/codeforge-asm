"""
analyze_results.py — Post-training analysis for CodeForge ASM.

Reads iteration artifact JSONs and/or W&B run, prints a full summary.

Usage (local, after Kaggle finishes and artifacts are downloaded):
    python scripts/analyze_results.py --artifacts artifacts/

Usage (live, via W&B API while training runs):
    python scripts/analyze_results.py --wandb-run pedroafonsomalheiros30-aaa/codeforge-asm/botgv1kf

Usage (Kaggle cell — reads local artifacts in /kaggle/working/codeforge-asm/):
    python /kaggle/working/codeforge-asm/scripts/analyze_results.py \
        --artifacts /kaggle/working/codeforge-asm/artifacts_kaggle
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


# ── helpers ──────────────────────────────────────────────────────────────────

def _pct(v: float) -> str:
    return f"{v * 100:.1f}%"

def _bar(v: float, width: int = 20) -> str:
    filled = round(v * width)
    return "#" * filled + "." * (width - filled)

def _trend(values: list[float]) -> str:
    if len(values) < 2:
        return "-"
    delta = values[-1] - values[0]
    if delta > 0.02:
        return f"UP  +{delta:.3f}"
    if delta < -0.02:
        return f"DN   {delta:.3f}"
    return f"    {delta:+.3f}"


# ── local artifacts analysis ──────────────────────────────────────────────────

def analyze_artifacts(artifacts_dir: Path) -> None:
    files = sorted(artifacts_dir.glob("iteration_*.json"), key=lambda p: int(p.stem.split("_")[1]))
    if not files:
        print(f"No iteration_*.json found in {artifacts_dir}")
        return

    print(f"\n{'='*60}")
    print(f"  CodeForge ASM — Training Analysis")
    print(f"  Artifacts: {artifacts_dir}")
    print(f"  Iterations found: {len(files)}")
    print(f"{'='*60}\n")

    # Collect per-iteration summary
    iter_summaries: list[dict] = []
    tier_curves: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for f in files:
        rows: list[dict] = json.loads(f.read_text(encoding="utf-8"))
        it = int(f.stem.split("_")[1])
        n = max(1, len(rows))

        summary = {
            "iteration": it,
            "n_rows": n,
            "reward_mean": sum(r["reward"] for r in rows) / n,
            "assembly_rate": sum(1 for r in rows if r.get("assembled")) / n,
            "link_rate": sum(1 for r in rows if r.get("linked")) / n,
            "run_rate": sum(1 for r in rows if r.get("ran")) / n,
            "correct_rate": sum(1 for r in rows if r.get("correct")) / n,
            "mcts_rows": sum(1 for r in rows if r.get("source") == "mcts"),
            "bon_rows": sum(1 for r in rows if r.get("source") != "mcts"),
            "stage_fails": defaultdict(int),
        }
        for r in rows:
            sf = r.get("stage_failed") or "none"
            summary["stage_fails"][sf] += 1

        iter_summaries.append(summary)

        # Per-tier breakdown
        by_tier: dict[int, list[dict]] = defaultdict(list)
        for r in rows:
            by_tier[r.get("tier", 0)].append(r)
        for tier, tier_rows in by_tier.items():
            tn = max(1, len(tier_rows))
            tier_curves[tier]["assemble"].append(sum(1 for r in tier_rows if r.get("assembled")) / tn)
            tier_curves[tier]["correct"].append(sum(1 for r in tier_rows if r.get("correct")) / tn)
            tier_curves[tier]["reward"].append(sum(r["reward"] for r in tier_rows) / tn)

    # ── Per-iteration table ───────────────────────────────────────────────────
    print("ITERATION SUMMARY")
    print("-" * 60)
    header = f"{'Iter':>4}  {'Reward':>6}  {'Asm':>5}  {'Link':>5}  {'Run':>5}  {'Correct':>7}  {'MCTS':>5}  Trend"
    print(header)
    print("-" * 60)
    rewards = [s["reward_mean"] for s in iter_summaries]
    for i, s in enumerate(iter_summaries):
        trend = "↑" if i > 0 and s["reward_mean"] > iter_summaries[i-1]["reward_mean"] else "↓" if i > 0 else " "
        mcts_flag = "✓" if s["mcts_rows"] > 0 else " "
        print(
            f"{s['iteration']:>4}  {s['reward_mean']:>6.3f}  "
            f"{_pct(s['assembly_rate']):>5}  {_pct(s['link_rate']):>5}  "
            f"{_pct(s['run_rate']):>5}  {_pct(s['correct_rate']):>7}  "
            f"{mcts_flag:>5}  {trend}"
        )

    # ── Overall trend ─────────────────────────────────────────────────────────
    print(f"\nOVERALL TREND  ({len(iter_summaries)} iterations)")
    print("-" * 60)
    metrics = [
        ("Reward",       [s["reward_mean"]    for s in iter_summaries]),
        ("Assembly",     [s["assembly_rate"]  for s in iter_summaries]),
        ("Correctness",  [s["correct_rate"]   for s in iter_summaries]),
    ]
    for name, vals in metrics:
        bar = _bar(vals[-1])
        print(f"  {name:<12} {bar} {_pct(vals[-1]):>6}  {_trend(vals)}")

    # ── Per-tier breakdown ────────────────────────────────────────────────────
    print(f"\nPER-TIER BREAKDOWN (final iteration)")
    print("-" * 60)
    for tier in sorted(tier_curves.keys()):
        curves = tier_curves[tier]
        asm_last  = curves["assemble"][-1] if curves["assemble"] else 0.0
        cor_last  = curves["correct"][-1]  if curves["correct"]  else 0.0
        rew_last  = curves["reward"][-1]   if curves["reward"]   else 0.0
        asm_trend = _trend(curves["assemble"])
        cor_trend = _trend(curves["correct"])
        print(f"  Tier {tier}:  asm={_pct(asm_last)} {asm_trend}  |  correct={_pct(cor_last)} {cor_trend}  |  reward={rew_last:.3f}")

    # ── Stage failure breakdown ───────────────────────────────────────────────
    last = iter_summaries[-1]
    print(f"\nFAILURE BREAKDOWN (last iteration, {last['n_rows']} rows)")
    print("-" * 60)
    total = last["n_rows"]
    for stage, count in sorted(last["stage_fails"].items(), key=lambda x: -x[1]):
        print(f"  {stage:<15}  {count:>4} / {total}  {_bar(count/total, 15)}  {_pct(count/total)}")

    # ── Best programs ─────────────────────────────────────────────────────────
    print(f"\nBEST PROGRAMS (last iteration, top 3 correct)")
    print("-" * 60)
    last_rows: list[dict] = json.loads(files[-1].read_text(encoding="utf-8"))
    correct = [r for r in last_rows if r.get("correct")]
    correct.sort(key=lambda r: r["reward"], reverse=True)
    for i, r in enumerate(correct[:3]):
        print(f"\n  [{i+1}] tier={r.get('tier')}  reward={r['reward']:.3f}  source={r.get('source')}")
        print(f"       Prompt: {r.get('instruction','')[:80]}")
        asm_lines = (r.get("asm") or "").strip().splitlines()
        for line in asm_lines[:8]:
            print(f"       {line}")
        if len(asm_lines) > 8:
            print(f"       ... ({len(asm_lines)-8} more lines)")

    print(f"\n{'='*60}\n")


# ── W&B live analysis ─────────────────────────────────────────────────────────

def analyze_wandb(run_path: str) -> None:
    try:
        import wandb
    except ImportError:
        print("Install wandb: pip install wandb")
        return

    api = wandb.Api()
    run = api.run(run_path)

    print(f"\n{'='*60}")
    print(f"  W&B Live Monitor — {run.name}")
    print(f"  State: {run.state.upper()}")
    print(f"  URL: {run.url}")
    print(f"{'='*60}\n")

    summary = dict(run.summary)
    elapsed = summary.get("_runtime", 0)
    if elapsed:
        print(f"Elapsed: {elapsed/3600:.1f}h  ({elapsed/60:.0f} min)\n")

    # ── TRL step-level metrics (prefixed with train/) ─────────────────────────
    trl_keys = ["train/loss", "train/kl", "train/grad_norm", "train/epoch",
                "train/completions/mean_length", "train/rewards/reward_fn/mean",
                "train/global_step"]
    trl_history = run.history(keys=trl_keys, pandas=False)

    if trl_history:
        print(f"TRL STEP METRICS  ({len(trl_history)} steps logged)")
        print("-" * 65)
        print(f"{'Step':>5}  {'Reward':>6}  {'KL':>8}  {'Loss':>7}  {'Epoch':>5}  {'GradN':>6}")
        print("-" * 65)
        for r in trl_history:
            step = int(r.get("train/global_step", 0) or 0)
            rw   = r.get("train/rewards/reward_fn/mean", 0.0) or 0.0
            kl   = r.get("train/kl", 0.0) or 0.0
            lo   = r.get("train/loss", 0.0) or 0.0
            ep   = r.get("train/epoch", 0.0) or 0.0
            gn   = r.get("train/grad_norm", 0.0) or 0.0
            print(f"{step:>5}  {rw:>6.3f}  {kl:>8.5f}  {lo:>7.4f}  {ep:>5.2f}  {gn:>6.3f}")

        trl_rewards = [r.get("train/rewards/reward_fn/mean", 0) or 0 for r in trl_history]
        trl_rewards = [v for v in trl_rewards if v > 0]
        if trl_rewards:
            print(f"\nStep reward — best: {max(trl_rewards):.3f}  latest: {trl_rewards[-1]:.3f}  trend: {_trend(trl_rewards)}")
    else:
        print("No TRL step metrics yet.")

    # ── Custom per-iteration metrics (logged once per iteration) ──────────────
    iter_keys = ["iteration", "reward/mean", "reward/assembly_rate", "reward/correct_rate",
                 "mcts/rows", "mcts/avg_depth", "source",
                 "reward/tier1_correct", "reward/tier2_correct",
                 "reward/tier3_correct", "reward/tier4_correct"]
    iter_history = [r for r in run.scan_history(keys=iter_keys) if r.get("iteration") is not None]

    if iter_history:
        print(f"\nPER-ITERATION METRICS  ({len(iter_history)} iterations logged)")
        print("-" * 65)
        print(f"{'Iter':>4}  {'Reward':>6}  {'Asm%':>5}  {'OK%':>5}  {'Src':<4}  T1✓   T2✓   T3✓   T4✓")
        print("-" * 65)
        for r in iter_history:
            it  = r.get("iteration", "?")
            rw  = r.get("reward/mean", 0.0) or 0.0
            asm = (r.get("reward/assembly_rate", 0.0) or 0.0) * 100
            ok  = (r.get("reward/correct_rate",  0.0) or 0.0) * 100
            src = (r.get("source", "bon") or "bon")[:3]
            t1  = (r.get("reward/tier1_correct", 0.0) or 0.0) * 100
            t2  = (r.get("reward/tier2_correct", 0.0) or 0.0) * 100
            t3  = (r.get("reward/tier3_correct", 0.0) or 0.0) * 100
            t4  = (r.get("reward/tier4_correct", 0.0) or 0.0) * 100
            print(f"{it:>4}  {rw:>6.3f}  {asm:>4.0f}%  {ok:>4.0f}%  {src:<4}  {t1:>3.0f}%  {t2:>3.0f}%  {t3:>3.0f}%  {t4:>3.0f}%")

        rewards = [r.get("reward/mean", 0) or 0 for r in iter_history]
        correct = [r.get("reward/correct_rate", 0) or 0 for r in iter_history]
        print(f"\nReward trend:   {_trend(rewards)}")
        print(f"Correct trend:  {_trend(correct)}")
    else:
        print("\nNo per-iteration metrics yet (logged after each full iteration).")

    # ── HF checkpoints ────────────────────────────────────────────────────────
    print("\nHF CHECKPOINTS")
    print("-" * 60)
    try:
        from huggingface_hub import HfApi
        import os
        hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
        hf = HfApi(token=hf_token)
        files = [f for f in hf.list_repo_files("mistral-hackaton-2026/codeforge", repo_type="model")
                 if f.startswith("checkpoints/")]
        if files:
            iters = sorted(set(f.split("/")[1] for f in files))
            print(f"  Pushed iterations: {', '.join(iters)}")
        else:
            print("  No checkpoints pushed yet.")
    except Exception as e:
        print(f"  (HF check skipped: {e})")

    print()


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=str, default=None,
                        help="Path to artifacts dir (iteration_N.json files)")
    parser.add_argument("--wandb-run", type=str, default=None,
                        help="W&B run path: entity/project/run_id")
    args = parser.parse_args()

    if args.wandb_run:
        analyze_wandb(args.wandb_run)
    elif args.artifacts:
        analyze_artifacts(Path(args.artifacts))
    else:
        # Try both defaults
        local = Path("artifacts_kaggle")
        if local.exists():
            analyze_artifacts(local)
        else:
            print("Provide --artifacts <dir> or --wandb-run <entity/project/run_id>")
            print("\nExample (W&B live):")
            print("  python scripts/analyze_results.py \\")
            print("    --wandb-run pedroafonsomalheiros30-aaa/codeforge-asm/botgv1kf")
            print("\nExample (local artifacts):")
            print("  python scripts/analyze_results.py --artifacts artifacts_kaggle")


if __name__ == "__main__":
    main()
