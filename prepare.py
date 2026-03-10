from __future__ import annotations

# Fixed setup file — do not modify.
# Run once before starting autoresearch: python prepare.py

import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORKSPACE_DIR = ROOT / "workspace"
RESULTS_PATH = ROOT / "results.tsv"
PROGRESS_PATH = WORKSPACE_DIR / "progress.log"
DATASET_PATH = ROOT / "prompts" / "dataset.json"


def ensure_system_deps() -> None:
    if platform.system().lower() != "linux":
        print("[prepare] Skipping nasm/ld install: non-Linux")
        return
    if shutil.which("nasm") and shutil.which("ld"):
        print("[prepare] nasm and ld already available")
        return
    print("[prepare] Installing nasm + binutils via apt-get")
    subprocess.run(["apt-get", "update", "-q"], check=False)
    subprocess.run(["apt-get", "install", "-y", "nasm", "binutils"], check=False)


def verify_dataset() -> int:
    if not DATASET_PATH.exists():
        raise SystemExit(f"Dataset not found: {DATASET_PATH}")
    tasks = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    print(f"[prepare] Dataset OK — {len(tasks)} tasks")
    return len(tasks)


def init_workspace() -> None:
    WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)

    if not RESULTS_PATH.exists():
        RESULTS_PATH.write_text(
            "commit\tprimary_metric\tcorrect_rate\tassembly_rate\treward_mean\titerations\tmemory_gb\tstatus\tdescription\n",
            encoding="utf-8",
        )
        print(f"[prepare] Created {RESULTS_PATH}")
    else:
        print(f"[prepare] {RESULTS_PATH} already exists")

    if not PROGRESS_PATH.exists():
        PROGRESS_PATH.write_text(
            "iteration\tstatus\tprimary_metric\tnote\n",
            encoding="utf-8",
        )
        print(f"[prepare] Created {PROGRESS_PATH}")
    else:
        print(f"[prepare] {PROGRESS_PATH} already exists")


def main() -> None:
    ensure_system_deps()
    n_tasks = verify_dataset()
    init_workspace()

    payload = {
        "results_tsv": str(RESULTS_PATH),
        "progress_log": str(PROGRESS_PATH),
        "dataset": str(DATASET_PATH),
        "n_tasks": n_tasks,
        "workspace": str(WORKSPACE_DIR),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
