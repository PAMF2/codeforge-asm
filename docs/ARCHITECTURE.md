# Architecture Notes

## Hackathon Strategy
- Phase 1 (must-have): Best-of-N + Reward Pipeline + W&B logging.
- Phase 2: Plug real GRPO updates with LoRA + 4-bit.
- Phase 3 (upgrade): MCTS line-by-line for tier 3/4 prompts.

## Runtime Loop
1. `PromptEngine` samples tasks.
2. `BestOfN` generates N full candidates.
3. `RewardPipeline` scores each candidate (assemble/link/run/correctness).
4. Aggregated trajectories are logged to artifacts and W&B.
5. GRPO update step consumes `(prompt, asm, reward)` tuples.

## Reward Contract
- Assemble success: `+0.25`
- Link success: `+0.25`
- Run success (no fatal signal): `+0.20`
- Correct output/exit match: `+0.30`

## Design Choices
- Deterministic prompt sampling early on for reproducible hackathon demos.
- `DummyGenerator` in `trainer.py` keeps the training loop verifiable before model integration.
- MCTS kept isolated in `src/mcts.py` to avoid blocking MVP delivery.
