"""
generate_prompts.py — Auto-generate NASM x86-64 prompts via Mistral Large 3 API.

Usage:
    python prompts/generate_prompts.py \
        --tier 3 \
        --count 10 \
        --out prompts/dataset.json \
        --append

Requires:
    pip install mistralai
    export MISTRAL_API_KEY=sk-...

The script asks Mistral to produce (instruction, expected_exit_code/expected_stdout,
hint_lines, reference_solution) tuples for the requested tier, validates each
reference_solution via the reward pipeline (assemble + run), deduplicates by id,
then writes / appends to the output JSON.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
import tempfile
from pathlib import Path

# ── Tier definitions ─────────────────────────────────────────────────────────

_TIER_DESCRIPTIONS = {
    1: "exit-code only (no output) — mov rdi, N; mov rax, 60; syscall",
    2: "print a short string to stdout using sys_write (rax=1)",
    3: "arithmetic result as exit code (add, sub, imul, idiv, cmp/cmov, bit ops)",
    4: "loops and control flow — sum, fibonacci, factorial, gcd, power",
}

_SYSTEM_PROMPT = (
    "You are an expert NASM x86-64 Linux assembly programmer. "
    "You generate concise, correct NASM programs for pedagogical use. "
    "Output ONLY valid JSON, no markdown, no explanations."
)

_USER_TEMPLATE = """\
Generate {count} distinct NASM x86-64 Linux assembly programming challenges for tier {tier}.

Tier {tier} description: {tier_desc}

Rules:
- Use sys_exit=60, sys_write=1, syscall (NOT int 0x80).
- All exit codes must be in range 0-255.
- Each reference_solution must assemble with nasm and run correctly.
- IDs must be unique and follow the pattern t{tier}_<short_slug>.
- hint_lines = number of non-empty non-comment lines in reference_solution.

Return a JSON array of objects with these fields:
  id: string
  tier: {tier}
  instruction: string (imperative sentence describing the task)
  expected_exit_code: integer or null
  expected_stdout: string or null
  hint_lines: integer
  reference_solution: string (complete NASM source, use \\n for newlines)

Avoid duplicating these existing IDs:
{existing_ids}
"""

# ── Mistral client ────────────────────────────────────────────────────────────


def _mistral_generate(prompt: str, model: str = "mistral-large-latest") -> str:
    """Call Mistral chat completions and return the assistant reply text."""
    try:
        from mistralai import Mistral  # type: ignore
    except ImportError:
        print("[generate_prompts] Install mistralai: pip install mistralai", file=sys.stderr)
        sys.exit(1)

    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        print("[generate_prompts] Set MISTRAL_API_KEY env var.", file=sys.stderr)
        sys.exit(1)

    client = Mistral(api_key=api_key)
    response = client.chat.complete(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.9,
        max_tokens=4096,
    )
    return response.choices[0].message.content.strip()


# ── Validation helpers ────────────────────────────────────────────────────────


def _validate_solution(item: dict) -> bool:
    """Assemble and run the reference_solution; return True if it passes."""
    src = item.get("reference_solution", "")
    if not src:
        return False

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        asm_file = td_path / "prog.asm"
        obj_file = td_path / "prog.o"
        bin_file = td_path / "prog"

        asm_file.write_text(src, encoding="utf-8")

        # assemble
        r = subprocess.run(
            ["nasm", "-f", "elf64", str(asm_file), "-o", str(obj_file)],
            capture_output=True,
            timeout=10,
        )
        if r.returncode != 0:
            return False

        # link
        r = subprocess.run(
            ["ld", str(obj_file), "-o", str(bin_file)],
            capture_output=True,
            timeout=10,
        )
        if r.returncode != 0:
            return False

        # run
        r = subprocess.run(
            [str(bin_file)],
            capture_output=True,
            timeout=5,
        )
        exit_code = r.returncode
        stdout = r.stdout.decode("utf-8", errors="replace")

        expected_exit = item.get("expected_exit_code")
        expected_stdout = item.get("expected_stdout")

        if expected_exit is not None and exit_code != expected_exit:
            return False
        if expected_stdout is not None and stdout != expected_stdout:
            return False

    return True


# ── JSON extraction ───────────────────────────────────────────────────────────


def _extract_json_array(text: str) -> list[dict]:
    """Extract a JSON array from model output, tolerating markdown fences."""
    # Strip markdown fences if present
    text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
    # Find first [ ... ]
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON array found in model output")
    return json.loads(match.group(0))


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate NASM prompts via Mistral API")
    parser.add_argument("--tier", type=int, choices=[1, 2, 3, 4], required=True)
    parser.add_argument("--count", type=int, default=10, help="Number of new prompts to generate")
    parser.add_argument("--out", type=str, default="prompts/dataset.json")
    parser.add_argument("--append", action="store_true", help="Append to existing dataset")
    parser.add_argument("--model", type=str, default="mistral-large-latest")
    parser.add_argument("--validate", action="store_true", help="Validate via nasm/ld before adding")
    parser.add_argument("--retries", type=int, default=3, help="Retry attempts per batch")
    args = parser.parse_args()

    out_path = Path(args.out)
    existing: list[dict] = []
    if args.append and out_path.exists():
        existing = json.loads(out_path.read_text(encoding="utf-8"))

    existing_ids: set[str] = {p["id"] for p in existing}

    tier_desc = _TIER_DESCRIPTIONS[args.tier]

    collected: list[dict] = []
    attempts = 0

    while len(collected) < args.count and attempts < args.retries:
        attempts += 1
        need = args.count - len(collected)
        prompt = _USER_TEMPLATE.format(
            count=need,
            tier=args.tier,
            tier_desc=tier_desc,
            existing_ids=json.dumps(sorted(existing_ids | {p["id"] for p in collected}), indent=2),
        )

        print(f"[generate_prompts] Attempt {attempts}: requesting {need} tier-{args.tier} prompts...")
        try:
            reply = _mistral_generate(prompt, model=args.model)
            items = _extract_json_array(reply)
        except Exception as exc:
            print(f"[generate_prompts] Parse error: {exc}", file=sys.stderr)
            continue

        for item in items:
            # Basic sanity checks
            if not item.get("id") or not item.get("instruction"):
                continue
            if item.get("tier") != args.tier:
                item["tier"] = args.tier
            if item["id"] in existing_ids or any(c["id"] == item["id"] for c in collected):
                # Patch duplicate id with a random suffix
                item["id"] = item["id"] + f"_{random.randint(100, 999)}"

            if args.validate:
                print(f"  Validating {item['id']}...", end=" ")
                try:
                    ok = _validate_solution(item)
                except Exception:
                    ok = False
                print("OK" if ok else "FAIL (skipping)")
                if not ok:
                    continue

            collected.append(item)
            if len(collected) >= args.count:
                break

    print(f"[generate_prompts] Collected {len(collected)} valid prompts.")

    if args.append:
        merged = existing + collected
    else:
        merged = collected

    out_path.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[generate_prompts] Wrote {len(merged)} prompts to {out_path}")

    # Print tier distribution
    from collections import Counter
    counts = Counter(p["tier"] for p in merged)
    for t in sorted(counts):
        print(f"  Tier {t}: {counts[t]}")


if __name__ == "__main__":
    main()
