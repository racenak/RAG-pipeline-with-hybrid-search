"""Tests for the evaluation dataset module."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from evaluation.dataset import EvalDatasetManager
from evaluation.dataset_schema import (
    DifficultyLevel,
    EvalCase,
    EvalDataset,
    QueryCategory,
)
from pydantic import ValidationError

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

VALID_CASE = EvalCase(
    id="test-001",
    query="What is the default embedding dimension?",
    expected_answer="1024",
    expected_documents=["config.py"],
    category=QueryCategory.FACTUAL,
    difficulty=DifficultyLevel.EASY,
)

FULL_DATASET = EvalDataset(
    version="1.0.0",
    description="Test dataset",
    cases=[
        VALID_CASE,
        EvalCase(
            id="test-002",
            query="Compare BM25 and vector search.",
            expected_answer="BM25 uses term frequency, vector uses embeddings.",
            category=QueryCategory.COMPARISON,
            difficulty=DifficultyLevel.MEDIUM,
        ),
        EvalCase(
            id="test-003",
            query="Summarize the ingestion pipeline.",
            expected_answer="Parse, clean, chunk, embed, index.",
            category=QueryCategory.SUMMARIZATION,
            difficulty=DifficultyLevel.EASY,
        ),
        EvalCase(
            id="test-004",
            query="How does RRF combine results?",
            expected_answer="Score = 1 / (k + rank)",
            category=QueryCategory.MULTI_HOP,
            difficulty=DifficultyLevel.HARD,
        ),
        EvalCase(
            id="test-005",
            query="",
            expected_answer="Empty query should return no results.",
            category=QueryCategory.EDGE_CASE,
            difficulty=DifficultyLevel.MEDIUM,
        ),
    ],
)


def _write_dataset(data: EvalDataset, tmp_dir: Path) -> Path:
    """Write dataset to a temp JSON file and return the path."""
    path = tmp_dir / "dataset.json"
    with path.open("w") as f:
        json.dump(data.model_dump(), f)
    return path


# ---------------------------------------------------------------------------
# EvalCase validation tests
# ---------------------------------------------------------------------------


class TestEvalCase:
    def test_valid_case(self):
        case = EvalCase(
            id="c1",
            query="test query",
            expected_answer="test answer",
            category=QueryCategory.FACTUAL,
            difficulty=DifficultyLevel.EASY,
        )
        assert case.id == "c1"
        assert case.category == QueryCategory.FACTUAL
        assert case.expected_documents == []
        assert case.metadata == {}

    def test_case_with_all_fields(self):
        case = EvalCase(
            id="c2",
            query="query",
            expected_answer="answer",
            expected_documents=["doc1.py", "doc2.py"],
            category=QueryCategory.MULTI_HOP,
            difficulty=DifficultyLevel.HARD,
            metadata={"source": "test"},
        )
        assert len(case.expected_documents) == 2
        assert case.metadata["source"] == "test"

    def test_invalid_category(self):
        with pytest.raises(ValidationError):
            EvalCase(
                id="c3",
                query="q",
                expected_answer="a",
                category="invalid_category",
                difficulty=DifficultyLevel.EASY,
            )

    def test_invalid_difficulty(self):
        with pytest.raises(ValidationError):
            EvalCase(
                id="c4",
                query="q",
                expected_answer="a",
                category=QueryCategory.FACTUAL,
                difficulty="invalid_difficulty",
            )

    def test_missing_required_fields(self):
        with pytest.raises(ValidationError):
            EvalCase()  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# EvalDataset loading tests
# ---------------------------------------------------------------------------


class TestEvalDataset:
    def test_load_from_dict(self):
        data = FULL_DATASET.model_dump()
        dataset = EvalDataset(**data)
        assert len(dataset.cases) == 5
        assert dataset.version == "1.0.0"

    def test_roundtrip_json(self, tmp_path: Path):
        path = _write_dataset(FULL_DATASET, tmp_path)
        with path.open() as f:
            loaded = json.load(f)
        dataset = EvalDataset(**loaded)
        assert len(dataset.cases) == 5
        assert dataset.cases[0].id == "test-001"

    def test_categories_covered(self):
        categories = {c.category for c in FULL_DATASET.cases}
        assert QueryCategory.FACTUAL in categories
        assert QueryCategory.COMPARISON in categories
        assert QueryCategory.SUMMARIZATION in categories
        assert QueryCategory.MULTI_HOP in categories
        assert QueryCategory.EDGE_CASE in categories


# ---------------------------------------------------------------------------
# EvalDatasetManager tests
# ---------------------------------------------------------------------------


class TestEvalDatasetManager:
    def test_load(self, tmp_path: Path):
        path = _write_dataset(FULL_DATASET, tmp_path)
        manager = EvalDatasetManager(path)
        dataset = manager.load()
        assert len(dataset.cases) == 5

    def test_filter_by_category(self, tmp_path: Path):
        path = _write_dataset(FULL_DATASET, tmp_path)
        manager = EvalDatasetManager(path)
        factual = manager.filter_by_category(QueryCategory.FACTUAL)
        assert len(factual) == 1
        assert factual[0].category == QueryCategory.FACTUAL

    def test_filter_by_difficulty(self, tmp_path: Path):
        path = _write_dataset(FULL_DATASET, tmp_path)
        manager = EvalDatasetManager(path)
        easy = manager.filter_by_difficulty(DifficultyLevel.EASY)
        assert all(c.difficulty == DifficultyLevel.EASY for c in easy)
        assert len(easy) == 2

    def test_filter_by_ids(self, tmp_path: Path):
        path = _write_dataset(FULL_DATASET, tmp_path)
        manager = EvalDatasetManager(path)
        results = manager.filter_by_ids(["test-001", "test-003"])
        assert len(results) == 2
        ids = {r.id for r in results}
        assert ids == {"test-001", "test-003"}

    def test_get_statistics(self, tmp_path: Path):
        path = _write_dataset(FULL_DATASET, tmp_path)
        manager = EvalDatasetManager(path)
        stats = manager.get_statistics()
        assert stats["total_cases"] == 5
        assert stats["categories"]["factual"] == 1
        assert stats["categories"]["comparison"] == 1
        assert stats["difficulties"]["easy"] == 2

    def test_validate_valid_dataset(self, tmp_path: Path):
        path = _write_dataset(FULL_DATASET, tmp_path)
        manager = EvalDatasetManager(path)
        manager.load()
        errors = manager.validate()
        assert errors == []

    def test_validate_duplicate_ids(self, tmp_path: Path):
        data = EvalDataset(
            version="1.0.0",
            description="Duplicate IDs",
            cases=[
                EvalCase(
                    id="dup",
                    query="q1",
                    expected_answer="a1",
                    category=QueryCategory.FACTUAL,
                    difficulty=DifficultyLevel.EASY,
                ),
                EvalCase(
                    id="dup",
                    query="q2",
                    expected_answer="a2",
                    category=QueryCategory.FACTUAL,
                    difficulty=DifficultyLevel.EASY,
                ),
            ],
        )
        path = _write_dataset(data, tmp_path)
        manager = EvalDatasetManager(path)
        manager.load()
        errors = manager.validate()
        assert len(errors) == 1
        assert "Duplicate case ID" in errors[0]


# ---------------------------------------------------------------------------
# Golden dataset tests
# ---------------------------------------------------------------------------


class TestGoldenDataset:
    GOLDEN_PATH = Path(__file__).resolve().parent.parent.parent / "evaluation" / "golden_dataset.json"

    def test_golden_dataset_loads(self):
        if not self.GOLDEN_PATH.exists():
            pytest.skip("Golden dataset not found")
        manager = EvalDatasetManager(self.GOLDEN_PATH)
        dataset = manager.load()
        assert len(dataset.cases) >= 50

    def test_golden_dataset_categories(self):
        if not self.GOLDEN_PATH.exists():
            pytest.skip("Golden dataset not found")
        manager = EvalDatasetManager(self.GOLDEN_PATH)
        manager.load()
        stats = manager.get_statistics()
        assert stats["categories"].get("factual", 0) >= 20
        assert stats["categories"].get("multi_hop", 0) >= 10
        assert stats["categories"].get("summarization", 0) >= 10
        assert stats["categories"].get("comparison", 0) >= 10
        assert stats["categories"].get("edge_case", 0) >= 5

    def test_golden_dataset_unique_ids(self):
        if not self.GOLDEN_PATH.exists():
            pytest.skip("Golden dataset not found")
        manager = EvalDatasetManager(self.GOLDEN_PATH)
        manager.load()
        errors = manager.validate()
        assert not any("Duplicate" in e for e in errors)

    def test_golden_dataset_validates(self):
        if not self.GOLDEN_PATH.exists():
            pytest.skip("Golden dataset not found")
        manager = EvalDatasetManager(self.GOLDEN_PATH)
        manager.load()
        errors = manager.validate()
        assert errors == []
