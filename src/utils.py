from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Iterable


SYS_PROMPT = (
    "You are an expert NASM x86-64 Linux programmer. "
    "Output only assembly code, no markdown fences, no explanations."
)


def sanitize_model_output(text: str) -> str:
    """Extract plain assembly from possible markdown/code-fenced responses."""
    text = text.strip()

    fence = re.search(r"```(?:asm|nasm|x86asm)?\s*(.*?)```", text, flags=re.S | re.I)
    if fence:
        text = fence.group(1).strip()

    # Remove obvious non-code preambles.
    lines = []
    for line in text.splitlines():
        if line.strip().lower().startswith(("here", "explanation", "note:")):
            continue
        lines.append(line)

    return "\n".join(lines).strip()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def run_cmd(cmd: Iterable[str], timeout_seconds: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(cmd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
