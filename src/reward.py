from __future__ import annotations

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

        reward = 0.0

        assemble = run_cmd(["nasm", "-f", "elf64", str(asm_path), "-o", str(obj_path)], self.timeout_seconds)
        if assemble.returncode != 0:
            return RewardResult(
                reward=0.0,
                assembled=False,
                linked=False,
                ran=False,
                correct=False,
                stdout=assemble.stdout,
                stderr=assemble.stderr,
                exit_code=None,
                stage_failed="assemble",
            )
        reward += 0.25

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
