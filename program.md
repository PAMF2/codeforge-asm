# autoresearch — CodeForge ASM

*You are an autonomous research agent. Your job is to train Ministral-8B to generate correct NASM x86-64 Linux assembly and find the highest `correct_rate` possible. You work alone. You never stop.*

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar9`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: Read these for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed setup. Do not modify.
   - `train.py` — the file you modify. Contains `ResearchConfig` and training wrapper.
   - `src/reward.py` — the 4-stage reward pipeline (read-only, defines the benchmark).
   - `prompts/dataset.json` — 100 tasks across 4 tiers (read-only).
4. **Verify data**: Check that `prompts/dataset.json` exists and `nasm`/`ld` are available.
5. **Run prepare**: `python prepare.py`
6. **Confirm**: `results.tsv` and `workspace/progress.log` exist.
7. **Confirm and go**: Confirm setup looks good, then kick off experimentation.

## Experimentation

Each experiment trains for a fixed number of iterations defined in `ResearchConfig.iterations`. You launch training as:

```
python train.py > run.log 2>&1
```

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything in `ResearchConfig` is fair game: iterations, prompts_per_iteration, generations_per_prompt, learning_rate, batch_size, temperature, top_p, kl_beta, lora_r, reward weights, MCTS settings, etc.

**What you CANNOT do:**
- Modify `prepare.py`, `src/reward.py`, `prompts/dataset.json`. These define the fixed benchmark.
- Modify `src/prompt_engine.py` (task loading).
- Install new packages or add dependencies.

**The goal is simple: get the highest `correct_rate`.** This is the fraction of generated assembly programs that produce the correct output when compiled and executed. Higher is better.

**VRAM** is a soft constraint. 4-bit quantization is on by default — changing LoRA rank or batch size can affect it.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement from a complex change is not worth it. Equal performance with simpler config is a win.

**The first run**: Always establish the baseline — run train.py as-is. Expected baseline: `correct_rate ≈ 0.10–0.20` (model starts weak, improves over iterations).

## Output format

Once training finishes it prints a summary like this:

```
---
primary_metric:    0.350000
correct_rate:      0.350000
assembly_rate:     0.650000
reward_mean:       0.480000
training_seconds:  1800.0
iterations_done:   5
n_samples:         400
peak_vram_mb:      12000.0
config_json:
{ ... }
```

Extract the key metric:

```
grep "^primary_metric:\|^peak_vram_mb:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated):

```
commit	primary_metric	correct_rate	assembly_rate	reward_mean	iterations	memory_gb	status	description
```

1. git commit hash (7 chars)
2. primary_metric = correct_rate (e.g. 0.350000)
3. correct_rate (same)
4. assembly_rate
5. reward_mean
6. iterations_done
7. memory_gb (peak_vram_mb / 1024, rounded to .1f)
8. status: `keep`, `discard`, or `crash`
9. short description of what this experiment tried

Example:
```
commit	primary_metric	correct_rate	assembly_rate	reward_mean	iterations	memory_gb	status	description
a1b2c3d	0.150000	0.150000	0.450000	0.380000	5	12.0	keep	baseline
b2c3d4e	0.200000	0.200000	0.520000	0.420000	5	12.1	keep	lr=1e-5 faster convergence
c3d4e5f	0.130000	0.130000	0.400000	0.350000	5	12.0	discard	lr=1e-4 too aggressive
```

Also append one line to `workspace/progress.log`:
```
iteration	status	primary_metric	note
```

Print a line like:
```
ITERATION 3 keep 0.200000 lr=1e-5 improved
```

## The experiment loop

LOOP FOREVER:

1. Look at the git state: current branch/commit.
2. Tune `train.py` with one experimental idea (edit `ResearchConfig` or the surrounding logic).
3. `git commit` with a short description.
4. Run: `python train.py > run.log 2>&1`
5. Extract results: `grep "^primary_metric:\|^peak_vram_mb:" run.log`
6. If grep is empty, the run crashed. Read `tail -n 50 run.log`, log as crash, reset and move on.
7. Record in `results.tsv`.
8. Append to `workspace/progress.log`.
9. If `primary_metric` improved (higher), keep the commit.
10. If not improved, `git reset --hard HEAD~1` and continue.

**Timeout**: If a run exceeds 60 minutes, kill it and treat as failure.

**Crashes**: If trivial fix (typo, OOM with smaller batch), fix and re-run. If fundamentally broken, log crash and move on.

**NEVER STOP**: Once the experiment loop begins, do NOT pause to ask the human. You are autonomous. The loop runs until the human interrupts you, period.

## Good experiment ideas

**Learning rate & optimization:**
- `learning_rate`: 1e-6, 3e-6, 5e-6, 1e-5, 3e-5
- `kl_beta`: 0.01, 0.05, 0.1, 0.2, 0.5 (lower = less KL penalty, more aggressive)
- `grad_clip_norm`: 0.5, 1.0, 2.0
- `grpo_backend`: "trl" vs "manual"

**Generation diversity:**
- `temperature`: 0.6, 0.7, 0.8, 0.9, 1.0
- `top_p`: 0.85, 0.90, 0.95
- `generations_per_prompt`: 4, 8, 16, 32 (more = better GRPO signal, slower)

**Data throughput:**
- `prompts_per_iteration`: 5, 10, 20 (more prompts per iter = slower but more diverse)
- `iterations`: 3, 5, 10 (more iterations = more training steps)
- `use_random_sampling`: True (diverse) vs False (deterministic, same prompts each iter)

**LoRA capacity:**
- `lora_r`: 8, 16, 32, 64 (higher rank = more capacity, more VRAM)
- `lora_alpha`: 16, 32, 64 (usually 2x lora_r)

**Reward shaping:**
- Boost `reward_correctness` (e.g. 0.5) and reduce others — focus on what matters
- Boost `reward_assemble` (e.g. 0.4) early to teach syntax first
- `reward_timeout`: 8, 12, 20 (longer = allow slow programs)

**MCTS (advanced):**
- Set `use_mcts_after_iteration=2` to enable MCTS from iteration 2
- `mcts_simulations`: 16, 32, 64
- `mcts_min_tier`: 2 (apply to easier tasks too)

**Sequence length:**
- `max_new_tokens`: 256, 512, 768 (shorter = faster, may truncate complex programs)
- `train_max_seq_len`: 512, 1024, 2048
