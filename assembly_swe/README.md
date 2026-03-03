# Assembly-SWE (CodeForge)

Assembly-SWE is a SWE-bench-inspired benchmark protocol for NASM x86-64 tasks.
It standardizes task format, deterministic verification, and leaderboard metrics.

## Layout

```text
assembly_swe/
  README.md
  schema.task.json
  datasets/
    sample_dev.jsonl
  examples/
    sample_predictions.jsonl
  tools/
    evaluate.py
```

## Task format (JSONL)

One task per line:

```json
{
  "task_id": "asm_hello_001",
  "tier": 1,
  "instruction": "Write NASM x86-64 Linux program that prints Hello\\n and exits 0.",
  "expected_stdout": "Hello\n",
  "expected_exit_code": 0
}
```

Supported keys:
- `task_id` (string, required)
- `tier` (int, required)
- `instruction` (string, required)
- `expected_stdout` (string, optional)
- `expected_exit_code` (int, optional)
- `tags` (list[string], optional)

## Predictions format (JSONL)

One candidate per line:

```json
{
  "task_id": "asm_hello_001",
  "candidate_id": "c0",
  "candidate_rank": 0,
  "asm": "global _start\nsection .text\n_start:\n..."
}
```

Supported keys:
- `task_id` (string, required)
- `asm` (string, required)
- `candidate_id` (string, optional)
- `candidate_rank` (int, optional, lower is better)

If `candidate_rank` is absent, file order is used.

## Metrics

- `pass@k`: fraction of tasks with at least one correct candidate in top-k.
- `assembly_rate@1`: fraction of top-1 candidates that assemble.
- `link_rate@1`: fraction of top-1 candidates that link.
- `run_rate@1`: fraction of top-1 candidates that execute.
- `correct_rate@1`: same as `pass@1`.
- `avg_reward@1`: average reward of top-1 candidates.
- Tier metrics: `tier_<n>_pass@1`, `tier_<n>_pass@k`.

## Run evaluation

```bash
python assembly_swe/tools/evaluate.py \
  --tasks assembly_swe/datasets/sample_dev.jsonl \
  --predictions assembly_swe/examples/sample_predictions.jsonl \
  --ks 1,3,5 \
  --outdir assembly_swe/results/latest
```

Outputs:
- `summary.json`
- `leaderboard.md`
- `rows_top1.jsonl`

## Notes

- Evaluator uses the same deterministic verification core (`nasm` + `ld` + run)
  via `src.reward.RewardPipeline`.
- This protocol is intentionally simple and reproducible for local and Kaggle runs.
