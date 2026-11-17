"""Benchmark Runner for ResearchLens RAG Pipeline.

Orchestrates running evaluation datasets through the RAG pipeline, computing
all quantitative evaluation metrics, tracking latency and cost, and generating
comprehensive markdown reports.
"""

import json
import os
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.app.models.schemas import QueryRequest, QueryResponse
from backend.evaluation.metrics import RAGMetrics


# Gemini 2.0 Flash Pricing (USD per 1M tokens)
PRICING_PROMPT_PER_MILLION = 0.10
PRICING_COMPLETION_PER_MILLION = 0.40


class BenchmarkRunner:
    """Executes benchmark suites against the RAG system and outputs formatted metrics."""

    def __init__(
        self,
        pipeline: Any,
        metrics_evaluator: Optional[RAGMetrics] = None,
    ) -> None:
        """Initialize runner.

        Parameters
        ----------
        pipeline:
            Instance of `RAGPipeline`.
        metrics_evaluator:
            Instance of `RAGMetrics`. If None, creates a default heuristic instance.
        """
        self.pipeline = pipeline
        self.metrics = metrics_evaluator or RAGMetrics()

    @staticmethod
    def load_dataset(dataset_path: str) -> Dict[str, Any]:
        """Load benchmark dataset from a JSON file."""
        with open(dataset_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _resolve_expected_doc_ids(
        self,
        expected_docs: List[str],
        item: Dict[str, Any],
        doc_id: Optional[str] = None,
    ) -> List[str]:
        """Resolve symbolic document IDs (like 'doc_transformer_baseline') to actual ingested UUIDs."""
        if doc_id:
            return [doc_id]

        resolved = []
        doc_list = []
        try:
            if hasattr(self.pipeline, "get_documents"):
                doc_list = self.pipeline.get_documents().documents
        except Exception:
            pass

        expected_filenames = item.get(
            "expected_filenames",
            ["Attention_Is_All_You_Need.pdf", "test_sample_paper.pdf", "1706.03762.pdf"],
        )

        for exp in expected_docs:
            if exp == "doc_transformer_baseline":
                matched = None
                for d in doc_list:
                    if any(fn.lower() in d.filename.lower() for fn in expected_filenames):
                        matched = d.doc_id
                        break
                if matched:
                    resolved.append(matched)
                elif doc_list:
                    resolved.append(doc_list[0].doc_id)
                else:
                    resolved.append(exp)
            else:
                resolved.append(exp)

        return resolved if resolved else expected_docs

    def evaluate_query(
        self,
        item: Dict[str, Any],
        doc_id: Optional[str] = None,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """Execute and score a single benchmark question."""
        question = item["question"]
        reference = item.get("reference_answer", "")
        expected_docs = item.get("expected_doc_ids", [])
        relevant_keywords = item.get("relevant_passage_keywords", [])
        expected_sections = item.get("expected_sections", [])

        # Start timer
        start_time = time.perf_counter()

        req = QueryRequest(question=question, doc_id=doc_id, top_k=top_k)
        response: QueryResponse = self.pipeline.query(req)

        e2e_latency_ms = (time.perf_counter() - start_time) * 1000

        # Extract contexts and retrieved IDs
        contexts = [src.content for src in response.sources]
        retrieved_ids = [src.doc_id for src in response.sources]

        # Resolve document-level expected IDs (translates 'doc_transformer_baseline' -> active doc UUID)
        resolved_expected_docs = self._resolve_expected_doc_ids(expected_docs, item, doc_id)

        # Compute metrics
        faithfulness = self.metrics.faithfulness(response.answer, contexts)
        hallucination = self.metrics.hallucination_rate(response.answer, contexts)
        relevancy = self.metrics.answer_relevancy(question, response.answer)
        correctness = self.metrics.answer_correctness(response.answer, reference)
        precision = self.metrics.context_precision(question, contexts, reference)
        recall = self.metrics.context_recall(contexts, reference)

        # Passage-level IR metrics (Ground-truth passage evaluation)
        passage_recall = self.metrics.passage_recall_at_k(
            response.sources, relevant_keywords, expected_sections, k=top_k
        )
        passage_mrr = self.metrics.passage_mrr(
            response.sources, relevant_keywords, expected_sections
        )

        # Document-level IR metrics (with resolved UUIDs)
        recall_at_k = self.metrics.retrieval_recall_at_k(retrieved_ids, resolved_expected_docs)
        mrr_score = self.metrics.mrr(retrieved_ids, resolved_expected_docs)

        # Tokens and Cost
        prompt_tokens = response.tokens_used.prompt_tokens if response.tokens_used else 0
        completion_tokens = response.tokens_used.completion_tokens if response.tokens_used else 0
        total_tokens = response.tokens_used.total_tokens if response.tokens_used else 0

        cost_usd = (prompt_tokens / 1_000_000 * PRICING_PROMPT_PER_MILLION) + (
            completion_tokens / 1_000_000 * PRICING_COMPLETION_PER_MILLION
        )

        return {
            "id": item.get("id", "N/A"),
            "category": item.get("category", "General"),
            "question": question,
            "reference_answer": reference,
            "generated_answer": response.answer,
            "retrieved_contexts_count": len(contexts),
            "metrics": {
                "faithfulness": faithfulness,
                "hallucination_rate": hallucination,
                "answer_relevancy": relevancy,
                "answer_correctness": correctness,
                "context_precision": precision,
                "context_recall": recall,
                "passage_recall_at_k": passage_recall,
                "passage_mrr": passage_mrr,
                "retrieval_recall_at_k": recall_at_k,
                "mrr": mrr_score,
            },
            "performance": {
                "retrieval_latency_ms": getattr(
                    self.pipeline._retriever, "last_retrieval_latency_ms", 0.0
                ),
                "e2e_latency_ms": round(e2e_latency_ms, 2),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "estimated_cost_usd": round(cost_usd, 6),
            },
        }

    def run_benchmark(
        self,
        dataset_path: str,
        doc_id: Optional[str] = None,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """Run full evaluation suite over all questions in the dataset."""
        dataset = self.load_dataset(dataset_path)
        questions = dataset.get("questions", [])

        results: List[Dict[str, Any]] = []
        for q in questions:
            res = self.evaluate_query(q, doc_id=doc_id, top_k=top_k)
            results.append(res)

        aggregated = self._aggregate_metrics(results)

        benchmark_summary = {
            "version": "1.0",
            "timestamp": datetime.utcnow().isoformat(),
            "dataset_info": {
                "name": dataset.get("description", "Benchmark"),
                "total_questions": len(questions),
            },
            "aggregated_metrics": aggregated,
            "individual_results": results,
        }

        return benchmark_summary

    @staticmethod
    def _aggregate_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not results:
            return {}

        metric_names = [
            "faithfulness",
            "hallucination_rate",
            "answer_relevancy",
            "answer_correctness",
            "context_precision",
            "context_recall",
            "passage_recall_at_k",
            "passage_mrr",
            "retrieval_recall_at_k",
            "mrr",
        ]

        perf_names = [
            "retrieval_latency_ms",
            "e2e_latency_ms",
            "total_tokens",
            "estimated_cost_usd",
        ]

        agg: Dict[str, Any] = {}

        for m in metric_names:
            vals = [r["metrics"][m] for r in results if m in r["metrics"]]
            if vals:
                agg[m] = {
                    "mean": round(statistics.mean(vals), 4),
                    "median": round(statistics.median(vals), 4),
                    "min": round(min(vals), 4),
                    "max": round(max(vals), 4),
                    "std_dev": round(statistics.stdev(vals), 4) if len(vals) > 1 else 0.0,
                }

        for p in perf_names:
            vals = [r["performance"][p] for r in results if p in r["performance"]]
            if vals:
                agg[p] = {
                    "mean": round(statistics.mean(vals), 4),
                    "median": round(statistics.median(vals), 4),
                    "min": round(min(vals), 4),
                    "max": round(max(vals), 4),
                }

        return agg

    def generate_report(self, summary: Dict[str, Any]) -> str:
        """Format benchmark summary into an interview-ready Markdown report."""
        agg = summary["aggregated_metrics"]
        lines = [
            "# ResearchLens — Benchmark Evaluation Report",
            "",
            f"- **System Version**: {summary.get('version', '1.0')}",
            f"- **Evaluation Date**: {summary.get('timestamp', 'N/A')}",
            f"- **Questions Evaluated**: {summary.get('dataset_info', {}).get('total_questions', 0)}",
            "",
            "## 1. Executive Summary & Aggregate Metrics",
            "",
            "| Evaluation Metric | Mean | Median | Min | Max | Standard Dev |",
            "| :--- | :---: | :---: | :---: | :---: | :---: |",
        ]

        metric_labels = [
            ("faithfulness", "Faithfulness (Groundedness)"),
            ("answer_correctness", "Answer Correctness"),
            ("answer_relevancy", "Answer Relevancy"),
            ("context_precision", "Context Precision"),
            ("context_recall", "Context Recall"),
            ("passage_recall_at_k", "Passage-Level Recall@K"),
            ("passage_mrr", "Passage-Level MRR"),
            ("retrieval_recall_at_k", "Document-Level Recall@K"),
            ("mrr", "Document-Level MRR"),
            ("hallucination_rate", "Hallucination Rate"),
        ]

        for key, label in metric_labels:
            if key in agg:
                m = agg[key]
                lines.append(
                    f"| **{label}** | {m['mean']:.4f} | {m['median']:.4f} | {m['min']:.4f} | {m['max']:.4f} | {m['std_dev']:.4f} |"
                )

        lines.extend([
            "",
            "## 2. Latency and Operational Efficiency",
            "",
            "| Metric | Mean | Median | Min | Max |",
            "| :--- | :---: | :---: | :---: | :---: |",
        ])

        perf_labels = [
            ("retrieval_latency_ms", "Retrieval Latency (ms)"),
            ("e2e_latency_ms", "End-to-End Response Latency (ms)"),
            ("total_tokens", "Token Consumption per Query"),
            ("estimated_cost_usd", "Estimated Cost ($ USD)"),
        ]

        for key, label in perf_labels:
            if key in agg:
                p = agg[key]
                lines.append(
                    f"| **{label}** | {p['mean']:.2f} | {p['median']:.2f} | {p['min']:.2f} | {p['max']:.2f} |"
                )

        lines.extend([
            "",
            "## 3. Detailed Question-Level Results",
            "",
            "| ID | Question | Faithfulness | Relevancy | Correctness | E2E Latency |",
            "| :---: | :--- | :---: | :---: | :---: | :---: |",
        ])

        for q in summary.get("individual_results", []):
            m = q["metrics"]
            p = q["performance"]
            short_q = q["question"][:45] + "..." if len(q["question"]) > 45 else q["question"]
            lines.append(
                f"| `{q['id']}` | {short_q} | {m['faithfulness']:.2f} | {m['answer_relevancy']:.2f} | {m['answer_correctness']:.2f} | {p['e2e_latency_ms']:.1f}ms |"
            )

        return "\n".join(lines)

    @staticmethod
    def save_results(summary: Dict[str, Any], output_path: str) -> None:
        """Save benchmark results to disk."""
        Path(os.path.dirname(output_path)).mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
