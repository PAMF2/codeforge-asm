from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.prompt_engine import PromptEngine
from src.reward import RewardPipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="prompts/dataset.json")
    parser.add_argument("--predictions", type=str, required=True, help="JSON list with {prompt_id, asm} rows")
    args = parser.parse_args()

    engine = PromptEngine(args.dataset)
    prompt_by_id = {p.id: p for p in engine.all_items()}
    rows = json.loads(Path(args.predictions).read_text(encoding="utf-8"))

    reward = RewardPipeline("artifacts/eval", timeout_seconds=5)
    scored = []
    for idx, row in enumerate(rows):
        prompt = prompt_by_id[row["prompt_id"]]
        result = reward.evaluate(prompt, row["asm"], sample_id=f"eval-{idx}")
        scored.append({"prompt_id": prompt.id, "reward": result.reward, "correct": result.correct})

    avg_reward = sum(x["reward"] for x in scored) / max(1, len(scored))
    acc = sum(1 for x in scored if x["correct"]) / max(1, len(scored))

    print(json.dumps({"count": len(scored), "avg_reward": avg_reward, "accuracy": acc}, indent=2))


if __name__ == "__main__":
    main()
