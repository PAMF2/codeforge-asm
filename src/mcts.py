from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MCTSConfig:
    simulations: int = 32
    max_lines: int = 30
    branch_factor: int = 4


class MCTSLineSearch:
    """Upgrade path for tier 3/4 prompts after early GRPO stabilization."""

    def __init__(self, cfg: MCTSConfig) -> None:
        self.cfg = cfg

    def generate_candidates(self, _: str) -> list[str]:
        # Placeholder intentionally kept explicit for hackathon staging.
        return []
