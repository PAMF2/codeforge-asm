from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
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


def _structural_score(asm_code: str) -> float:
    """
    Partial reward for structurally-correct ASM even when nasm fails.
    Gives the model learning signal before it can fully assemble.
    Capped at 0.08 (well below the 0.25 full-assemble reward).
    """
    if not asm_code.strip():
        return 0.0
    s = 0.0
    low = asm_code.lower()
    if "global _start" in low:
        s += 0.02
    if "section .text" in low:
        s += 0.02
    if "_start:" in low:
        s += 0.02
    if "syscall" in low:
        s += 0.02
    if ("mov rax" in low or "mov eax" in low) and "syscall" in low:
        s += 0.02
    # Zero out if output is pure prose (no ASM mnemonics at all)
    asm_keywords = ("mov", "push", "pop", "jmp", "call", "xor", "add", "sub",
                    "cmp", "inc", "dec", "ret", "nop", "lea", "imul", "idiv")
    if not any(kw in low for kw in asm_keywords):
        s = 0.0
    return min(s, 0.08)


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

        assemble = run_cmd(["nasm", "-f", "elf64", str(asm_path), "-o", str(obj_path)], self.timeout_seconds)
        if assemble.returncode != 0:
            return RewardResult(
                reward=_structural_score(asm_code),
                assembled=False,
                linked=False,
                ran=False,
                correct=False,
                stdout=assemble.stdout,
                stderr=assemble.stderr,
                exit_code=None,
                stage_failed="assemble",
            )

        reward = 0.25

        link = run_cmd(["ld", str(obj_path), "-o", str(bin_path)], self.timeout_seconds)
        if link.returncode != 0:
            return RewardResult(
                reward=reward,
                assembled=True,
                linked=False,
                ran=False,
                correct=False,
                stdout=link.stdout,
                stderr=link.stderr,
                exit_code=None,
                stage_failed="link",
            )
        reward += 0.25

        run = run_cmd([str(bin_path)], self.timeout_seconds)
        if run.returncode < 0:
            return RewardResult(
                reward=reward,
                assembled=True,
                linked=True,
                ran=False,
                correct=False,
                stdout=run.stdout,
                stderr=run.stderr,
                exit_code=run.returncode,
                stage_failed="run",
            )
        reward += 0.20

        correct = False
        if prompt.expected_stdout is not None:
            correct = run.stdout == prompt.expected_stdout
        elif prompt.expected_exit_code is not None:
            correct = run.returncode == prompt.expected_exit_code
        else:
            correct = run.returncode == 0

        if correct:
            reward += 0.30
        elif prompt.expected_stdout is None and prompt.expected_exit_code is None and run.returncode == 0:
            reward += 0.15

        return RewardResult(
            reward=reward,
            assembled=True,
            linked=True,
            ran=True,
            correct=correct,
            stdout=run.stdout,
            stderr=run.stderr,
            exit_code=run.returncode,
            stage_failed=None if correct else "correctness",
        )

    def evaluate_batch(
        self,
        items: list[tuple[PromptItem, str, str]],
        workers: int = 8,
    ) -> list[RewardResult]:
        """
        Evaluate (prompt, asm_code, sample_id) tuples in parallel.
        subprocess.run releases the GIL, so ThreadPoolExecutor gives true
        parallelism here without multiprocessing pickling overhead.
        """
        if not items:
            return []
        with ThreadPoolExecutor(max_workers=min(workers, len(items))) as pool:
            futures = [pool.submit(self.evaluate, p, asm, sid) for p, asm, sid in items]
            return [f.result() for f in futures]
