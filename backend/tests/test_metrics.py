"""Unit tests for RAGMetrics."""

import pytest

from backend.evaluation.metrics import RAGMetrics


@pytest.fixture
def metrics():
    return RAGMetrics(llm=None)  # Test deterministic heuristic fallbacks


def test_retrieval_recall_at_k(metrics):
    retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
    relevant = ["doc2", "doc5", "doc7"]
    score = metrics.retrieval_recall_at_k(retrieved, relevant)
    assert score == round(2 / 3, 4)


def test_mrr(metrics):
    retrieved = ["doc1", "doc2", "doc3"]
    relevant = ["doc2"]
    # doc2 is at rank 2
    assert metrics.mrr(retrieved, relevant) == 0.5


def test_mrr_first_rank(metrics):
    retrieved = ["doc1", "doc2"]
    relevant = ["doc1"]
    assert metrics.mrr(retrieved, relevant) == 1.0


def test_mrr_not_found(metrics):
    retrieved = ["doc1", "doc2"]
    relevant = ["doc3"]
    assert metrics.mrr(retrieved, relevant) == 0.0


def test_faithfulness_heuristic(metrics):
    contexts = ["The Transformer model was introduced in 2017 by Vaswani et al."]
    answer = "The Transformer was introduced in 2017."
    score = metrics.faithfulness(answer, contexts)
    assert score > 0.5


def test_hallucination_rate(metrics):
    contexts = ["The Transformer model was introduced in 2017."]
    answer = "The Transformer model was introduced in 2017."
    hallucination = metrics.hallucination_rate(answer, contexts)
    assert hallucination < 0.5
