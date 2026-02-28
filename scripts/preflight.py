#!/usr/bin/env python
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from typing import Any


def cmd_ok(cmd: list[str], timeout: int = 20) -> tuple[bool, str]:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=timeout)
        ok = out.returncode == 0
        msg = (out.stdout or out.stderr).strip()
        return ok, msg[:500]
    except Exception as exc:
        return False, str(exc)


def find_bin(name: str) -> str | None:
    return shutil.which(name)


def main() -> int:
    report: dict[str, Any] = {
        "python": sys.version,
        "env": {
            "HF_TOKEN": bool(os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")),
            "WANDB_API_KEY": bool(os.getenv("WANDB_API_KEY")),
            "MISTRAL_API_KEY": bool(os.getenv("MISTRAL_API_KEY")),
        },
        "bins": {
            "hf": find_bin("hf"),
            "huggingface-cli": find_bin("huggingface-cli"),
            "nasm": find_bin("nasm"),
            "ld": find_bin("ld"),
        },
        "checks": {},
    }

    report["checks"]["hf_auth_whoami"] = cmd_ok(["hf", "auth", "whoami"])
    report["checks"]["hf_jobs_help"] = cmd_ok(["hf", "jobs", "--help"])

    for mod in ["yaml", "transformers", "datasets", "trl", "peft", "wandb"]:
        ok, msg = cmd_ok([sys.executable, "-c", f"import {mod}; print('ok')"], timeout=90)
        report["checks"][f"python_import_{mod}"] = (ok, msg)

    print(json.dumps(report, indent=2, ensure_ascii=False))

    hard_fail = [
        not report["checks"]["hf_auth_whoami"][0],
        not report["checks"]["hf_jobs_help"][0],
    ]
    return 1 if any(hard_fail) else 0


if __name__ == "__main__":
    raise SystemExit(main())
