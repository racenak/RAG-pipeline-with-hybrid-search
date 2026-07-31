"""Dataset management — loading, filtering, and validating evaluation datasets."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from evaluation.dataset_schema import (
    DifficultyLevel,
    EvalCase,
    EvalDataset,
    QueryCategory,
)


class EvalDatasetManager:
    """Load and manage evaluation datasets."""

    def __init__(self, dataset_path: str | Path) -> None:
        self._path = Path(dataset_path)
        self._dataset: EvalDataset | None = None

    def load(self) -> EvalDataset:
        """Load from JSON, validate against schema."""
        with self._path.open() as f:
            data = json.load(f)
        self._dataset = EvalDataset(**data)
        return self._dataset

    @property
    def dataset(self) -> EvalDataset:
        """Return the loaded dataset, loading if necessary."""
        if self._dataset is None:
            self._dataset = self.load()
        return self._dataset

    def filter_by_category(self, category: QueryCategory) -> list[EvalCase]:
        """Filter cases by category."""
        return [c for c in self.dataset.cases if c.category == category]

    def filter_by_difficulty(self, difficulty: DifficultyLevel) -> list[EvalCase]:
        """Filter cases by difficulty."""
        return [c for c in self.dataset.cases if c.difficulty == difficulty]

    def filter_by_ids(self, ids: list[str]) -> list[EvalCase]:
        """Get specific cases by ID."""
        id_set = set(ids)
        return [c for c in self.dataset.cases if c.id in id_set]

    def get_statistics(self) -> dict:
        """Return dataset stats: count per category, difficulty."""
        category_counts = Counter(c.category.value for c in self.dataset.cases)
        difficulty_counts = Counter(c.difficulty.value for c in self.dataset.cases)
        return {
            "total_cases": len(self.dataset.cases),
            "categories": dict(category_counts),
            "difficulties": dict(difficulty_counts),
        }

    def validate(self) -> list[str]:
        """Validate all cases, return list of errors (empty if valid)."""
        errors: list[str] = []
        seen_ids: set[str] = set()

        for i, case in enumerate(self.dataset.cases):
            if case.id in seen_ids:
                errors.append(f"Duplicate case ID at index {i}: {case.id}")
            seen_ids.add(case.id)

            if not case.query and case.category.value != "edge_case":
                errors.append(f"Case {case.id}: empty query in non-edge_case category")

            if not case.expected_answer:
                errors.append(f"Case {case.id}: empty expected_answer")

        return errors
