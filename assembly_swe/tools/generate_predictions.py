from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.utils import SYS_PROMPT, sanitize_model_output


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def load_model(
    base_model: str,
    adapter_path: Path,
    load_in_4bit: bool,
) -> tuple[Any, Any]:
    bnb = None
    if load_in_4bit:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    model_kwargs: dict[str, Any] = {
        "device_map": "auto",
        "trust_remote_code": True,
    }
    if bnb is not None:
        model_kwargs["quantization_config"] = bnb
    else:
        model_kwargs["torch_dtype"] = torch.float16

    base = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
    model = PeftModel.from_pretrained(base, str(adapter_path), is_trainable=False)
    model.eval()

    # Prefer tokenizer from checkpoint (keeps special tokens aligned), fallback to base model.
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(adapter_path), trust_remote_code=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def generate_one(
    model: Any,
    tokenizer: Any,
    instruction: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    prompt = f"{SYS_PROMPT}\n\nTask: {instruction}"
    enc = tokenizer(prompt, return_tensors="pt", truncation=True)
    enc = {k: v.to(model.device) for k, v in enc.items()}
    with torch.no_grad():
        out = model.generate(
            **enc,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    # Strip prompt echo if present.
    if text.startswith(prompt):
        text = text[len(prompt) :]
    return sanitize_model_output(text.strip())


def main() -> None:
    p = argparse.ArgumentParser(description="Generate Assembly-SWE predictions from a CodeForge checkpoint")
    p.add_argument("--tasks", required=True, help="Task JSONL")
    p.add_argument("--checkpoint-dir", required=True, help="Local checkpoint dir (e.g., checkpoints/iter_30)")
    p.add_argument("--out", required=True, help="Output predictions JSONL")
    p.add_argument("--base-model", default="mistralai/Ministral-8B-Instruct-2410")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--load-in-4bit", action="store_true", default=False)
    args = p.parse_args()

    tasks_path = Path(args.tasks)
    ckpt = Path(args.checkpoint_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tasks = read_jsonl(tasks_path)
    model, tok = load_model(args.base_model, ckpt, load_in_4bit=args.load_in_4bit)

    with out_path.open("w", encoding="utf-8") as f:
        for i, task in enumerate(tasks):
            asm = generate_one(
                model=model,
                tokenizer=tok,
                instruction=str(task["instruction"]),
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            row = {
                "task_id": str(task["task_id"]),
                "candidate_id": f"c{i}",
                "candidate_rank": 0,
                "asm": asm,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"wrote {len(tasks)} predictions -> {out_path}")


if __name__ == "__main__":
    main()
