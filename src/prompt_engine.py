from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class PromptItem:
    id: str
    tier: int
    instruction: str
    expected_stdout: str | None = None
    expected_exit_code: int | None = None


class PromptEngine:
    def __init__(self, dataset_path: str | Path) -> None:
        self.dataset_path = Path(dataset_path)
        self._items = self._load()

    def _load(self) -> list[PromptItem]:
        raw = json.loads(self.dataset_path.read_text(encoding="utf-8"))
        return [PromptItem(**item) for item in raw]

    def sample(self, n: int) -> list[PromptItem]:
        # Deterministic start for hackathon reliability.
        return self._items[:n]

    def all_items(self) -> list[PromptItem]:
        return list(self._items)

    @staticmethod
    def from_mistral_generation(_: dict[str, Any]) -> list[dict[str, Any]]:
        """Placeholder for automatic prompt generation via Mistral API."""
        raise NotImplementedError("Mistral prompt generation will be added after MVP loop is stable.")
