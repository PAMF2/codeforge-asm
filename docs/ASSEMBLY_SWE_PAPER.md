# Assembly-SWE and CodeForge: A Practical Benchmark and Training Framework for Assembly Code Generation

## Abstract
Large language models have rapidly improved at high-level code generation, but robust evaluation of low-level systems code remains underdeveloped. We present **Assembly-SWE**, a SWE-bench-inspired benchmark protocol for NASM x86-64 Linux tasks, and **CodeForge**, a reinforcement-learning training pipeline designed to improve executable correctness under deterministic verification. Assembly-SWE emphasizes end-to-end task success (assemble, link, run, and semantic correctness), while CodeForge optimizes model behavior with reward signals grounded in real execution outcomes. On internal checkpoint sweeps over iterations 1..30, we observe strong variability across checkpoints and a clear gap between syntactic validity and semantic correctness, highlighting the need for execution-grounded evaluation beyond text overlap. We release a reproducible evaluation toolchain, aggregate reporting artifacts, and practical recommendations for publication-grade benchmarking.

## 1. Introduction
Evaluation standards for code agents have progressed significantly in repository-level software engineering settings. However, assembly code generation introduces distinct constraints:

1. Small syntax errors often yield immediate assembly/link failures.
2. Runtime behavior is highly sensitive to ABI details and syscall conventions.
3. Surface-form correctness is insufficient; executable behavior is the true objective.

To address this, we define **Assembly-SWE**, a benchmark protocol focused on deterministic execution outcomes. We pair this benchmark with **CodeForge**, a training approach that combines policy optimization and execution-based rewards to improve practical reliability on assembly tasks.

This document has two goals:

1. Define Assembly-SWE as a rigorous and reproducible benchmark interface.
2. Describe the CodeForge pipeline and empirical behavior over checkpoint evolution.

## 2. Related Context
Recent work on coding agents has shifted from static pass/fail unit tests toward broader software engineering workflows. In parallel, reinforcement learning for code generation has benefited from verifiable rewards and execution feedback. Assembly workloads stress these paradigms because they magnify toolchain errors and low-level semantic mistakes. Assembly-SWE is designed as a compact, execution-first benchmark to close this gap for low-level generation.

## 3. Assembly-SWE: Benchmark Definition
### 3.1 Design Principles
Assembly-SWE is built around:

1. **Deterministic verification** using `nasm`, `ld`, and constrained runtime execution.
2. **Structured tasks** with explicit expected stdout and/or exit code.
3. **Candidate ranking support**, enabling pass@k analysis.
4. **Tiered difficulty**, allowing per-tier breakdown and curriculum-aware analysis.

### 3.2 Task Schema
Each task is a JSON object with required fields:

- `task_id` (string)
- `tier` (int >= 1)
- `instruction` (string)

Optional fields:

- `expected_stdout` (string)
- `expected_exit_code` (int)
- `tags` (string list)

This schema is provided in `assembly_swe/schema.task.json`.

### 3.3 Prediction Schema
Each candidate prediction includes:

- `task_id`
- `asm` (assembly source)
- optional `candidate_id`
- optional `candidate_rank`

Candidates are sorted by rank (or file order fallback), which enables consistent `pass@k` reporting.

### 3.4 Verification Pipeline
For each candidate:

1. Assemble (`nasm`)
2. Link (`ld`)
3. Execute (bounded timeout)
4. Compare runtime outputs against expected behavior

This yields stage-level status and a final correctness signal.

## 4. Metrics
Assembly-SWE reports:

- **pass@k**: fraction of tasks solved by at least one candidate within top-k.
- **assembly_rate@1**: top-1 assemble success.
- **link_rate@1**: top-1 link success.
- **run_rate@1**: top-1 runtime success.
- **correct_rate@1**: top-1 semantic correctness (equivalent to pass@1).
- **avg_reward@1**: mean evaluator reward on top-1 candidate.
- **tier breakdown**: per-tier pass@1.

These metrics separate tooling robustness from semantic correctness, which is critical for low-level code.

## 5. CodeForge Approach
### 5.1 Training Loop
CodeForge uses iterative policy updates with reward-driven optimization and checkpointed adapters. At each iteration, candidate generations are verified with the same deterministic execution logic used by the evaluator.

### 5.2 Why Execution-Grounded Rewards
Assembly generation suffers from high false confidence if measured by textual similarity alone. CodeForge rewards are tied to executable outcomes, aligning optimization with real task success.

### 5.3 Checkpoint-Centric Evaluation
We evaluate many checkpoints (`iter_1..iter_30`) rather than only the latest model. This reveals non-monotonic behavior and enables selecting a deployment checkpoint by objective (e.g., peak correctness vs. peak assembly robustness).

## 6. Empirical Snapshot (Internal Run)
Using `assembly_swe/results/all_iters/aggregate.json`:

- Best `correct_rate@1`: **0.8** (iterations 5 and 24)
- Best `assembly_rate@1`: **1.0** (iterations 25 and 29)
- Best `avg_reward@1`: **0.88** (iterations 25 and 29)

Observed pattern:

1. Large iteration-to-iteration variance.
2. High assembly success does not always imply high correctness.
3. Final checkpoint is not guaranteed to be best checkpoint.

## 7. Practical Reproducibility
### 7.1 Core Evaluation
```bash
python assembly_swe/tools/eval_all_iters.py \
  --repo-root . \
  --tasks assembly_swe/datasets/sample_dev.jsonl \
  --iter-start 1 --iter-end 30 \
  --ks 1,3,5 \
  --outdir assembly_swe/results/all_iters \
  --load-in-4bit \
  --hub-repo-id mistral-hackaton-2026/codeforge
```

### 7.2 Dataset Validation
```bash
python assembly_swe/tools/validate_dataset.py \
  --tasks assembly_swe/datasets/sample_dev.jsonl
```

### 7.3 Paper-Friendly HTML Report
```bash
python assembly_swe/tools/report_html.py \
  --aggregate assembly_swe/results/all_iters/aggregate.json
```

## 8. Limitations and Threats to Validity
1. **Small task set bias**: tiny dev sets can inflate variance and overfit conclusions.
2. **Toolchain dependence**: results depend on exact environment (`nasm`, linker, libc/kernel behavior).
3. **Checkpoint selection bias**: cherry-picking best iteration can overstate practical stability.
4. **Reward-design sensitivity**: metric outcomes depend on reward shaping choices.

## 9. Improving Assembly-SWE (Roadmap)
To make Assembly-SWE publication-strong, we recommend:

1. Expand to larger held-out dev/test sets with broader task families.
2. Add multi-architecture tracks (e.g., x86-64 and ARM64).
3. Track resource-aware metrics (latency, binary size, runtime constraints).
4. Add adversarial and perturbation-based robustness tasks.
5. Standardize public leaderboard submission format and seed protocol.

## 10. Conclusion
Assembly-SWE provides an execution-grounded benchmark protocol for low-level code generation, and CodeForge demonstrates a practical path to optimize toward real executable correctness. The main empirical lesson is that robustness and correctness evolve non-monotonically across checkpoints; therefore, evaluation must be broad, deterministic, and checkpoint-aware. We release this framework as a reproducible baseline for future low-level coding-agent research.

