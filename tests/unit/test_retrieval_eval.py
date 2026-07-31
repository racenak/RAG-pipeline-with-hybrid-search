"""Tests for retrieval evaluation metrics, comparison, and pipeline."""

from rag_pipeline.evaluation.compare import EvalComparator
from rag_pipeline.evaluation.pipeline import RetrievalEvaluator, _derive_category
from rag_pipeline.evaluation.retrieval import (
    RetrievalMetrics,
    QueryResult,
    evaluate_dataset,
    evaluate_query,
    hit_rate,
    mean_average_precision,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


# ---- precision_at_k ----------------------------------------------------------


class TestPrecisionAtK:
    def test_all_relevant(self):
        assert precision_at_k(["a", "b", "c"], ["a", "b", "c"], k=3) == 1.0

    def test_none_relevant(self):
        assert precision_at_k(["x", "y", "z"], ["a", "b"], k=3) == 0.0

    def test_partial_relevant(self):
        # 2 of 4 relevant in top-4
        assert precision_at_k(["a", "x", "b", "y"], ["a", "b", "c"], k=4) == 0.5

    def test_k_larger_than_retrieved(self):
        assert precision_at_k(["a"], ["a", "b"], k=5) == 1.0

    def test_k_zero(self):
        assert precision_at_k(["a"], ["a"], k=0) == 0.0

    def test_empty_retrieved(self):
        assert precision_at_k([], ["a"], k=5) == 0.0


# ---- recall_at_k -------------------------------------------------------------


class TestRecallAtK:
    def test_all_found(self):
        assert recall_at_k(["a", "b", "c", "d"], ["a", "b"], k=4) == 1.0

    def test_none_found(self):
        assert recall_at_k(["x", "y"], ["a", "b"], k=2) == 0.0

    def test_partial_found(self):
        assert recall_at_k(["a", "x", "y"], ["a", "b", "c"], k=3) == 1 / 3

    def test_k_limits_recall(self):
        # Only "a" in top-1, but "b" exists at rank 2
        assert recall_at_k(["a", "b"], ["a", "b"], k=1) == 0.5

    def test_empty_expected(self):
        assert recall_at_k(["a"], [], k=5) == 0.0


# ---- mean_reciprocal_rank ---------------------------------------------------


class TestMRR:
    def test_first_relevant(self):
        assert mean_reciprocal_rank(["a", "b", "c"], ["a"]) == 1.0

    def test_second_relevant(self):
        assert mean_reciprocal_rank(["x", "a", "c"], ["a"]) == 0.5

    def test_none_relevant(self):
        assert mean_reciprocal_rank(["x", "y", "z"], ["a"]) == 0.0

    def test_multiple_expected_uses_first(self):
        # First relevant at rank 2
        assert mean_reciprocal_rank(["x", "b", "a"], ["a", "b"]) == 0.5


# ---- ndcg_at_k ---------------------------------------------------------------


class TestNDCGAtK:
    def test_perfect_ranking(self):
        # All relevant docs at top positions
        score = ndcg_at_k(["a", "b"], ["a", "b"], k=2)
        assert score == 1.0

    def test_no_relevant_docs(self):
        assert ndcg_at_k(["x", "y"], ["a", "b"], k=2) == 0.0

    def test_partial_relevant(self):
        # "a" at rank 1, "x" at rank 2 — only "a" relevant
        score = ndcg_at_k(["a", "x"], ["a", "b"], k=2)
        assert 0.0 < score < 1.0

    def test_empty_expected(self):
        assert ndcg_at_k(["a"], [], k=1) == 0.0

    def test_known_value(self):
        # k=3, expected=["a","b"], retrieved=["x","a","b"]
        # DCG = 0/log2(2) + 1/log2(3) + 1/log2(4) = 0 + 0.6310 + 0.5 = 1.1310
        # IDCG = 1/log2(2) + 1/log2(3) = 1.0 + 0.6310 = 1.6310
        # NDCG = 1.1310 / 1.6310 ≈ 0.6934
        score = ndcg_at_k(["x", "a", "b"], ["a", "b"], k=3)
        assert abs(score - 0.6934) < 0.001


# ---- hit_rate ----------------------------------------------------------------


class TestHitRate:
    def test_hit(self):
        assert hit_rate(["x", "a", "y"], ["a"]) == 1.0

    def test_no_hit(self):
        assert hit_rate(["x", "y"], ["a"]) == 0.0

    def test_empty_retrieved(self):
        assert hit_rate([], ["a"]) == 0.0


# ---- mean_average_precision -------------------------------------------------


class TestMAP:
    def test_perfect(self):
        # All relevant at top positions: AP = (1/1 + 2/2) / 2 = 1.0
        assert mean_average_precision(["a", "b"], ["a", "b"]) == 1.0

    def test_none_relevant(self):
        assert mean_average_precision(["x", "y"], ["a"]) == 0.0

    def test_partial(self):
        # retrieved=["a","x","b"], expected=["a","b"]
        # a at rank 1: 1/1, b at rank 3: 2/3 -> AP = (1 + 0.6667)/2 = 0.8333
        score = mean_average_precision(["a", "x", "b"], ["a", "b"])
        assert abs(score - 5 / 6) < 1e-6

    def test_empty_expected(self):
        assert mean_average_precision(["a"], []) == 0.0


# ---- evaluate_query ----------------------------------------------------------


class TestEvaluateQuery:
    def test_computes_all_metrics(self):
        result = evaluate_query(
            query_id="q1",
            query="test query",
            retrieved_doc_ids=["a", "b", "c", "d"],
            expected_doc_ids=["a", "b"],
            k_values=[1, 3, 5],
        )
        assert result.query_id == "q1"
        assert result.query == "test query"
        assert result.retrieved_doc_ids == ["a", "b", "c", "d"]
        assert result.expected_doc_ids == ["a", "b"]
        assert isinstance(result.metrics, RetrievalMetrics)
        assert 1 in result.metrics.precision_at_k
        assert 3 in result.metrics.recall_at_k
        assert 5 in result.metrics.ndcg_at_k
        assert result.metrics.mrr > 0
        assert result.metrics.hit_rate == 1.0
        assert result.metrics.map_score > 0

    def test_no_expected_docs(self):
        result = evaluate_query(
            query_id="q2",
            query="empty",
            retrieved_doc_ids=["a", "b"],
            expected_doc_ids=[],
        )
        assert result.metrics.mrr == 0.0
        assert result.metrics.hit_rate == 0.0
        assert result.metrics.map_score == 0.0


# ---- evaluate_dataset --------------------------------------------------------


class TestEvaluateDataset:
    def test_aggregates_mean(self):
        r1 = evaluate_query("q1", "query1", ["a", "b"], ["a"], k_values=[1, 5])
        r2 = evaluate_query("q2", "query2", ["x", "y"], ["x"], k_values=[1, 5])
        agg = evaluate_dataset([r1, r2])
        # Both have MRR=1.0 (first result relevant)
        assert agg.mrr == 1.0
        assert agg.hit_rate == 1.0
        assert 1 in agg.precision_at_k
        assert 5 in agg.precision_at_k

    def test_empty_list(self):
        agg = evaluate_dataset([])
        assert agg.mrr == 0.0
        assert agg.precision_at_k == {}


# ---- EvalComparator ----------------------------------------------------------


class TestEvalComparator:
    def test_detects_regression(self):
        baseline = RetrievalMetrics(mrr=0.8, hit_rate=0.9, map_score=0.7)
        current = RetrievalMetrics(mrr=0.7, hit_rate=0.85, map_score=0.65)
        comp = EvalComparator(regression_threshold=0.05)
        results = comp.compare(baseline, current)
        regressions = [r for r in results if r.regression]
        # mrr dropped 0.1 (12.5% of 0.8) -> regression
        assert len(regressions) >= 1
        assert any(r.metric == "mrr" for r in regressions)

    def test_no_regression(self):
        baseline = RetrievalMetrics(mrr=0.8, hit_rate=0.9)
        current = RetrievalMetrics(mrr=0.85, hit_rate=0.92)
        comp = EvalComparator(regression_threshold=0.05)
        results = comp.compare(baseline, current)
        assert not any(r.regression for r in results)

    def test_to_markdown_format(self):
        baseline = RetrievalMetrics(mrr=0.5)
        current = RetrievalMetrics(mrr=0.6)
        comp = EvalComparator()
        results = comp.compare(baseline, current)
        md = comp.to_markdown(results)
        assert "| Metric |" in md
        assert "| mrr |" in md
        assert "+0.1000" in md

    def test_to_json_format(self):
        baseline = RetrievalMetrics(mrr=0.5, hit_rate=0.7)
        current = RetrievalMetrics(mrr=0.6, hit_rate=0.8)
        comp = EvalComparator()
        results = comp.compare(baseline, current)
        report = comp.to_json(results)
        assert "threshold" in report
        assert "total_metrics" in report
        assert "regressions" in report
        assert "metrics" in report
        assert len(report["metrics"]) > 0
        assert report["metrics"][0]["name"] == "mrr"


# ---- RetrievalEvaluator -----------------------------------------------------


class TestRetrievalEvaluator:
    def test_evaluate_with_mock_search(self):
        def mock_search(query: str, top_k: int) -> list[str]:
            if "foo" in query:
                return ["doc_a", "doc_b", "doc_c"]
            return ["doc_x", "doc_y"]

        # Create minimal eval cases (not using EvalCase to avoid import dependency)
        class FakeCase:
            def __init__(self, id: str, query: str, expected_docs: list[str]):
                self.id = id
                self.query = query
                self.expected_documents = expected_docs

        cases = [
            FakeCase("q1", "foo query", ["doc_a"]),
            FakeCase("q2", "bar query", ["doc_x", "doc_y"]),
        ]

        evaluator = RetrievalEvaluator()
        report = evaluator.evaluate(cases, search_fn=mock_search)

        assert report.total_queries == 2
        assert report.latency_ms >= 0
        assert len(report.per_query) == 2
        assert report.metrics.mrr > 0
        assert len(report.category_breakdown) > 0

    def test_generate_report(self):
        r1 = evaluate_query("q1", "query1", ["a", "b"], ["a"])
        r2 = evaluate_query("q2", "query2", ["c", "d"], ["c"])
        evaluator = RetrievalEvaluator()
        report = evaluator.generate_report([r1, r2], latency_ms=123.45)
        assert report.total_queries == 2
        assert report.latency_ms == 123.45
        assert report.metrics.mrr == 1.0


# ---- _derive_category --------------------------------------------------------


class TestDeriveCategory:
    def test_question_word(self):
        assert _derive_category("What is RAG?") == "q_what"

    def test_how_question(self):
        assert _derive_category("How does search work?") == "q_how"

    def test_non_question(self):
        assert _derive_category("RAG pipeline overview") == "other"

    def test_empty_query(self):
        assert _derive_category("") == "unknown"
