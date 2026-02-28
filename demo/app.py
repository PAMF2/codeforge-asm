from __future__ import annotations

import gradio as gr

from src.prompt_engine import PromptItem
from src.reward import RewardPipeline
from src.utils import sanitize_model_output


def baseline_model(_: str) -> str:
    return """global _start
section .text
_start:
    mov rax, 60
    mov rdi, 0
    syscall"""


def tuned_model(prompt: str) -> str:
    # TODO: load tuned LoRA adapter and generate.
    return baseline_model(prompt)


def run_demo(task: str) -> tuple[str, str, float, str]:
    asm = sanitize_model_output(tuned_model(task))
    reward = RewardPipeline(artifacts_dir="artifacts/demo", timeout_seconds=5)
    fake_prompt = PromptItem(id="demo", tier=1, instruction=task)
    result = reward.evaluate(fake_prompt, asm, sample_id="demo-run")
    return asm, result.stdout, result.reward, str(result.exit_code)


ui = gr.Interface(
    fn=run_demo,
    inputs=gr.Textbox(label="Task", lines=3, value="Write a NASM x86-64 Linux program that exits with code 42."),
    outputs=[
        gr.Code(language="nasm", label="Generated Assembly"),
        gr.Textbox(label="Program stdout"),
        gr.Number(label="Reward"),
        gr.Textbox(label="Exit code"),
    ],
    title="CodeForge ASM Demo",
    description="Generate NASM code and verify assemble/link/run in real time.",
)


if __name__ == "__main__":
    ui.launch()
