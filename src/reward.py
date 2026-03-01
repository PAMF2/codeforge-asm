from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

from .prompt_engine import PromptItem
from .utils import ensure_dir, run_cmd


@dataclass
class RewardResult:
    reward: float
    assembled: bool
    linked: bool
    ran: bool
    correct: bool
    stdout: str
    stderr: str
    exit_code: int | None
    stage_failed: str | None


# ── Structural partial reward ─────────────────────────────────────────────────

_ASM_MNEMONICS = frozenset({
    "mov", "push", "pop", "jmp", "je", "jne", "jl", "jg", "jle", "jge",
    "jz", "jnz", "call", "ret", "xor", "add", "sub", "cmp", "inc", "dec",
    "lea", "imul", "idiv", "mul", "div", "and", "or", "not", "shl", "shr",
    "test", "nop", "neg", "cbw", "cdq", "cqo", "movzx", "movsx", "cmov",
    "cmovl", "cmovg", "cmovle", "cmovge", "cmove", "cmovne",
})


def _structural_score(asm_code: str) -> float:
    """
    Partial reward (0.0–0.12) for structurally-correct NASM even when assembly fails.
    Teaches the model the correct x86-64 Linux skeleton before it can fully assemble.

    Scoring breakdown (all cumulative, capped at 0.12):
      +0.025  global _start declaration
      +0.025  section .text present
      +0.025  _start: label
      +0.020  uses syscall (correct x86-64 ABI) and NOT int 0x80 (32-bit ABI)
      -0.020  uses int 0x80 (wrong ABI — penalise)
      +0.015  correct sys_exit number (rax=60, x86-64)
      +0.010  has ≥2 ASM mnemonics (actual code, not just skeleton)
      +0.010  has ≥5 ASM mnemonics (substantive program)
      zero    if no mnemonics at all (pure prose output)
    """
    if not asm_code.strip():
        return 0.0

    low = asm_code.lower()
    s = 0.0

    # Core skeleton
    if "global _start" in low:
        s += 0.025
    if "section .text" in low:
        s += 0.025
    if "_start:" in low:
        s += 0.025

    # Syscall ABI
    has_syscall = "syscall" in low
    has_int80 = "int 0x80" in low or "int 80h" in low
    if has_syscall and not has_int80:
        s += 0.020  # correct x86-64 convention
    if has_int80:
        s -= 0.020  # wrong 32-bit convention — teach the model not to use it

    # Correct sys_exit (x86-64: rax=60, NOT rax=1 which is 32-bit)
    has_exit60 = any(p in low for p in (
        "mov rax, 60", "mov eax, 60", "mov rax,60", "mov eax,60",
        "mov rax, 0x3c", "mov eax, 0x3c",
    ))
    if has_exit60:
        s += 0.015

    # Substantive instructions
    found = sum(1 for m in _ASM_MNEMONICS if m in low)
    if found >= 2:
        s += 0.010
    if found >= 5:
        s += 0.010

    # Penalise pure-prose output (no mnemonics at all)
    if found == 0:
        s = 0.0

    return max(0.0, min(s, 0.12))


def _stdout_partial(got: str, expected: str) -> float:
    """
    Partial correctness bonus for stdout tasks (0.0–0.20).
    Only rewards near-perfect output to prevent reward hacking
    (e.g. model outputting 'e\\n' to score against many tier-2 tasks).

    Thresholds are intentionally strict:
      ≥ 0.95 → +0.20  (e.g. correct word, just missing trailing newline)
      < 0.95 → +0.00  (anything more wrong gets nothing)

    We deliberately skip the 0.60–0.94 band — too exploitable.
    The model would learn short character sequences that happen to match
    multiple expected strings across the dataset (reward hacking).
    """
    if not expected:
        return 0.0
    if got == expected:
        return 0.30  # exact match handled by caller, but guard here too
    sim = SequenceMatcher(None, got, expected).ratio()
    if sim >= 0.95:
        return 0.20
    return 0.0


# ── RewardPipeline ────────────────────────────────────────────────────────────

class RewardPipeline:
    def __init__(self, artifacts_dir: str | Path, timeout_seconds: int = 5) -> None:
        self.artifacts_dir = Path(artifacts_dir)
        self.timeout_seconds = timeout_seconds
        ensure_dir(self.artifacts_dir)

    def evaluate(self, prompt: PromptItem, asm_code: str, sample_id: str) -> RewardResult:
        workdir = self.artifacts_dir / sample_id
        ensure_dir(workdir)

        asm_path = workdir / "prog.asm"
        obj_path = workdir / "prog.o"
        bin_path = workdir / "prog"
        asm_path.write_text(asm_code + "\n", encoding="utf-8")

        # ── Stage 1: assemble ─────────────────────────────────────────
        assemble = run_cmd(
            ["nasm", "-f", "elf64", str(asm_path), "-o", str(obj_path)],
            self.timeout_seconds,
        )
        if assemble.returncode != 0:
            return RewardResult(
                reward=_structural_score(asm_code),
                assembled=False, linked=False, ran=False, correct=False,
                stdout=assemble.stdout, stderr=assemble.stderr,
                exit_code=None, stage_failed="assemble",
            )

        reward = 0.25

        # ── Stage 2: link ─────────────────────────────────────────────
        link = run_cmd(["ld", str(obj_path), "-o", str(bin_path)], self.timeout_seconds)
        if link.returncode != 0:
            return RewardResult(
                reward=reward,
                assembled=True, linked=False, ran=False, correct=False,
                stdout=link.stdout, stderr=link.stderr,
                exit_code=None, stage_failed="link",
            )
        reward += 0.25

        # ── Stage 3: run ──────────────────────────────────────────────
        run = run_cmd([str(bin_path)], self.timeout_seconds)
        if run.returncode < 0:
            return RewardResult(
                reward=reward,
                assembled=True, linked=True, ran=False, correct=False,
                stdout=run.stdout, stderr=run.stderr,
                exit_code=run.returncode, stage_failed="run",
            )
        reward += 0.20

        # ── Stage 4: correctness ──────────────────────────────────────
        correct = False
        correctness_bonus = 0.0

        if prompt.expected_stdout is not None:
            if run.stdout == prompt.expected_stdout:
                correct = True
                correctness_bonus = 0.30
            else:
                # Strict partial: only near-perfect output (≥0.95 sim) gets credit.
                # Avoids reward hacking via short strings matching many expected values.
                correctness_bonus = _stdout_partial(run.stdout, prompt.expected_stdout)

        elif prompt.expected_exit_code is not None:
            if run.returncode == prompt.expected_exit_code:
                correct = True
                correctness_bonus = 0.30
            # No partial for exit codes — arbitrary numbers, proximity is meaningless.

        else:
            if run.returncode == 0:
                correct = True
                correctness_bonus = 0.30

        # Fallback tiny bonus for clean exit when no expectation specified
        if not correct and prompt.expected_stdout is None and prompt.expected_exit_code is None and run.returncode == 0:
            correctness_bonus = 0.15

        reward += correctness_bonus

        return RewardResult(
            reward=reward,
            assembled=True, linked=True, ran=True, correct=correct,
            stdout=run.stdout, stderr=run.stderr,
            exit_code=run.returncode,
            stage_failed=None if correct else "correctness",
        )

    def evaluate_batch(
        self,
        items: list[tuple[PromptItem, str, str]],
        workers: int = 32,
    ) -> list[RewardResult]:
        """
        Evaluate (prompt, asm_code, sample_id) tuples in parallel.

        - ThreadPoolExecutor: subprocess.run() releases the GIL, so threads give
          true parallelism without pickling overhead of multiprocessing.
        - workers=32: saturates nasm/ld/run on large batches (160 samples) while
          keeping thread creation overhead negligible.
        - Per-future exception handling: a single broken sample (e.g. nasm crash,
          timeout edge-case) gets reward=0 and does NOT abort the whole batch.
          Order is fully preserved.
        """
        if not items:
            return []

        _zero = RewardResult(
            reward=0.0, assembled=False, linked=False, ran=False, correct=False,
            stdout="", stderr="evaluation_error", exit_code=None, stage_failed="error",
        )

        n_workers = min(workers, len(items))
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = [pool.submit(self.evaluate, p, asm, sid) for p, asm, sid in items]
            results: list[RewardResult] = []
            for f in futures:
                try:
                    results.append(f.result())
                except Exception:
                    results.append(_zero)
            return results
