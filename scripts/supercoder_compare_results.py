#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_summary(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "summary" not in data:
        raise ValueError(f"Missing 'summary' in {path}")
    return data["summary"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare SuperCoder eval summaries.")
    parser.add_argument(
        "--results-root",
        required=True,
        help="Root path like /content/SuperCoder/results/main/llm_superoptimizer_ds_val",
    )
    parser.add_argument("--model-under-test-tag", required=True, help="Folder tag of model under test")
    parser.add_argument("--paper-model-tag", default="Superoptimizer_Qwen7B", help="Folder tag of paper model")
    args = parser.parse_args()

    root = Path(args.results_root)
    test_json = root / args.model_under_test_tag / "num_iterations_1/0-shot/best_of_1/problem_results.json"
    paper_json = root / args.paper_model_tag / "num_iterations_1/0-shot/best_of_1/problem_results.json"

    test = load_summary(test_json)
    paper = load_summary(paper_json)

    metrics = [
        "compilation_rate",
        "accuracy",
        "avg_speedup",
        "median_speedup",
        "p75_speedup",
        "max_speedup",
    ]

    print("=== SuperCoder Eval Comparison ===")
    print(f"model_under_test: {test.get('model')}")
    print(f"paper_model     : {paper.get('model')}")
    print(f"test_json       : {test_json}")
    print(f"paper_json      : {paper_json}")
    print("")
    for k in metrics:
        t = float(test.get(k, 0.0))
        p = float(paper.get(k, 0.0))
        print(f"{k:16s} | test={t:10.4f} | paper={p:10.4f} | delta={t-p:+10.4f}")


if __name__ == "__main__":
    main()

