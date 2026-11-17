"""RAG Evaluation Metrics — ResearchLens V1.0.

Provides both automated and LLM-as-a-judge metric evaluators for:
- Faithfulness (groundedness)
- Answer Correctness (semantic agreement with reference)
- Answer Relevancy (alignment with question)
- Context Precision (signal-to-noise ratio in retrieved context)
- Context Recall (retrieval coverage against reference)
- Retrieval Recall@K (standard IR metric)
- Mean Reciprocal Rank / MRR (standard ranking metric)
- Hallucination Rate (1 - faithfulness)
"""

import json
import re
from typing import Any, Dict, List, Optional

from langchain_core.prompts import ChatPromptTemplate


class RAGMetrics:
    """Computes RAG evaluation metrics using LLM-as-judge and deterministic rules."""

    def __init__(self, llm: Optional[Any] = None) -> None:
        """Initialize metrics evaluator.

        Parameters
        ----------
        llm:
            Optional LangChain ChatModel (e.g. ChatGoogleGenerativeAI).
            When omitted, fallback heuristic scorers are used.
        """
        self.llm = llm

    # ── Deterministic IR Metrics ─────────────────────────────────────────────

    @staticmethod
    def retrieval_recall_at_k(
        retrieved_ids: List[str],
        relevant_ids: List[str],
    ) -> float:
        """Standard Information Retrieval Recall@K.

        Formula:
            Recall@K = |retrieved ∩ relevant| / |relevant|
        """
        if not relevant_ids:
            return 1.0
        if not retrieved_ids:
            return 0.0

        hits = set(retrieved_ids).intersection(set(relevant_ids))
        return round(len(hits) / len(relevant_ids), 4)

    @staticmethod
    def mrr(
        retrieved_ids: List[str],
        relevant_ids: List[str],
    ) -> float:
        """Mean Reciprocal Rank (MRR) for a single query.

        Formula:
            RR = 1 / rank_of_first_relevant_item (or 0 if none found)
        """
        if not relevant_ids or not retrieved_ids:
            return 0.0

        rel_set = set(relevant_ids)
        for rank, doc_id in enumerate(retrieved_ids, start=1):
            if doc_id in rel_set:
                return round(1.0 / rank, 4)
        return 0.0

    @staticmethod
    def is_passage_relevant(
        source_content: str,
        relevant_keywords: List[str],
        expected_sections: Optional[List[str]] = None,
        source_section: Optional[str] = None,
        min_keyword_ratio: float = 0.35,
    ) -> bool:
        """Determine whether a retrieved passage is genuinely relevant to the benchmark item.

        Evaluates passage-level ground truth:
        1. Section hierarchy match against expected canonical sections.
        2. Presence of key semantic terms/phrases characteristic of the target passage.
        """
        if not source_content:
            return False

        content_lower = source_content.lower()

        # Check section match if both provided
        if expected_sections and source_section:
            for sec in expected_sections:
                if sec.lower() in source_section.lower():
                    return True

        # Check keyword / key phrase presence
        if relevant_keywords:
            matches = sum(1 for kw in relevant_keywords if kw.lower() in content_lower)
            if matches / len(relevant_keywords) >= min_keyword_ratio or matches >= 2:
                return True

        return False

    @classmethod
    def passage_recall_at_k(
        cls,
        retrieved_sources: List[Any],
        relevant_keywords: List[str],
        expected_sections: Optional[List[str]] = None,
        k: Optional[int] = None,
    ) -> float:
        """Measure whether at least one retrieved chunk in top-K genuinely contains the relevant passage.

        Returns 1.0 if a genuine relevant passage is retrieved within top-K, 0.0 otherwise.
        """
        sources = retrieved_sources[:k] if k is not None else retrieved_sources
        if not sources:
            return 0.0

        for src in sources:
            content = getattr(src, "content", "") if not isinstance(src, str) else src
            section = (
                getattr(src, "section", None)
                or getattr(src, "hierarchy", None)
                or (getattr(src, "metadata", {}).get("section") if hasattr(src, "metadata") else None)
            )
            if cls.is_passage_relevant(content, relevant_keywords, expected_sections, section):
                return 1.0
        return 0.0

    @classmethod
    def passage_mrr(
        cls,
        retrieved_sources: List[Any],
        relevant_keywords: List[str],
        expected_sections: Optional[List[str]] = None,
    ) -> float:
        """Compute MRR based on the rank of the first genuinely relevant passage chunk.

        Returns 1.0 / rank of first relevant passage, or 0.0 if not found in retrieved list.
        """
        for rank, src in enumerate(retrieved_sources, start=1):
            content = getattr(src, "content", "") if not isinstance(src, str) else src
            section = (
                getattr(src, "section", None)
                or getattr(src, "hierarchy", None)
                or (getattr(src, "metadata", {}).get("section") if hasattr(src, "metadata") else None)
            )
            if cls.is_passage_relevant(content, relevant_keywords, expected_sections, section):
                return round(1.0 / rank, 4)
        return 0.0

    # ── LLM-as-a-Judge & Heuristic Metrics ───────────────────────────────────

    def faithfulness(self, answer: str, contexts: List[str]) -> float:
        """Measure what fraction of claims made in the answer are supported by contexts.

        Returns float between 0.0 and 1.0.
        """
        if not answer.strip():
            return 0.0
        if not contexts:
            return 0.0

        if not self.llm:
            return self._heuristic_faithfulness(answer, contexts)

        prompt = ChatPromptTemplate.from_template(
            "You are an expert factual consistency evaluator. "
            "Analyze the following answer and determine whether every claim made in it is "
            "directly supported by the provided context.\n\n"
            "Context:\n{context}\n\n"
            "Answer:\n{answer}\n\n"
            "Task:\n"
            "1. Break down the answer into atomic claims.\n"
            "2. For each claim, check if it is directly supported by the context.\n"
            "3. Output a single JSON object with the format:\n"
            "{{\"total_claims\": <int>, \"supported_claims\": <int>, \"faithfulness\": <float between 0 and 1>}}\n"
            "Respond ONLY with the raw JSON object."
        )

        try:
            combined_context = "\n---\n".join(contexts)
            chain = prompt | self.llm
            res = chain.invoke({"context": combined_context, "answer": answer})
            raw_text = res.content if hasattr(res, "content") else str(res)
            parsed = self._extract_json(raw_text)
            return float(parsed.get("faithfulness", 0.0))
        except Exception:
            return self._heuristic_faithfulness(answer, contexts)

    def hallucination_rate(self, answer: str, contexts: List[str]) -> float:
        """Hallucination rate is the complement of faithfulness.

        Formula:
            Hallucination Rate = 1.0 - Faithfulness
        """
        return round(1.0 - self.faithfulness(answer, contexts), 4)

    def answer_relevancy(self, question: str, answer: str) -> float:
        """Measure how directly and comprehensively the answer addresses the question.

        Returns float between 0.0 and 1.0.
        """
        if not answer.strip() or not question.strip():
            return 0.0

        if not self.llm:
            return self._heuristic_relevancy(question, answer)

        prompt = ChatPromptTemplate.from_template(
            "You are an evaluator of answer quality. "
            "Score how relevant and complete the answer is with respect to the user's question.\n\n"
            "Question: {question}\n"
            "Answer: {answer}\n\n"
            "Instructions:\n"
            "- Score from 0.0 (completely irrelevant) to 1.0 (perfectly relevant and directly addresses prompt).\n"
            "- Penalize answers that state 'I don't know' only if context would have allowed an answer.\n"
            "- Output ONLY a JSON object: {{\"relevancy\": <float 0.0 to 1.0>}}"
        )

        try:
            chain = prompt | self.llm
            res = chain.invoke({"question": question, "answer": answer})
            raw_text = res.content if hasattr(res, "content") else str(res)
            parsed = self._extract_json(raw_text)
            return float(parsed.get("relevancy", 0.0))
        except Exception:
            return self._heuristic_relevancy(question, answer)

    def answer_correctness(self, answer: str, reference: str) -> float:
        """Measure semantic correctness of answer compared against reference answer.

        Returns float between 0.0 and 1.0.
        """
        if not answer.strip() or not reference.strip():
            return 0.0

        if not self.llm:
            return self._heuristic_jaccard(answer, reference)

        prompt = ChatPromptTemplate.from_template(
            "You are an expert benchmark grader. "
            "Compare the candidate answer to the ground-truth reference answer and assess semantic correctness.\n\n"
            "Ground-Truth Reference Answer:\n{reference}\n\n"
            "Candidate Answer:\n{answer}\n\n"
            "Instructions:\n"
            "- Assess factual agreement and key takeaways.\n"
            "- Score from 0.0 (completely wrong or contradictory) to 1.0 (factually identical).\n"
            "- Output ONLY a JSON object: {{\"correctness\": <float 0.0 to 1.0>}}"
        )

        try:
            chain = prompt | self.llm
            res = chain.invoke({"reference": reference, "answer": answer})
            raw_text = res.content if hasattr(res, "content") else str(res)
            parsed = self._extract_json(raw_text)
            return float(parsed.get("correctness", 0.0))
        except Exception:
            return self._heuristic_jaccard(answer, reference)

    def context_precision(
        self,
        question: str,
        contexts: List[str],
        reference: str,
    ) -> float:
        """Proportion of retrieved contexts that contain information relevant to the question.

        Returns float between 0.0 and 1.0.
        """
        if not contexts:
            return 0.0

        if not self.llm:
            return 0.5

        prompt = ChatPromptTemplate.from_template(
            "You are evaluating context retrieval quality.\n"
            "Question: {question}\n"
            "Reference Answer: {reference}\n\n"
            "Evaluate each of the following retrieved contexts. For each chunk, determine if it "
            "contains relevant information to answer the question.\n\n"
            "Contexts:\n{contexts}\n\n"
            "Output ONLY a JSON object: {{\"relevant_chunks\": <int>, \"total_chunks\": <int>, \"precision\": <float 0.0 to 1.0>}}"
        )

        try:
            formatted_contexts = "\n\n".join(
                [f"[{i+1}] {ctx}" for i, ctx in enumerate(contexts)]
            )
            chain = prompt | self.llm
            res = chain.invoke(
                {
                    "question": question,
                    "reference": reference,
                    "contexts": formatted_contexts,
                }
            )
            raw_text = res.content if hasattr(res, "content") else str(res)
            parsed = self._extract_json(raw_text)
            return float(parsed.get("precision", 0.0))
        except Exception:
            return 0.5

    def context_recall(self, contexts: List[str], reference: str) -> float:
        """Measure how much of the ground truth reference answer is covered by retrieved context.

        Returns float between 0.0 and 1.0.
        """
        if not contexts or not reference.strip():
            return 0.0

        if not self.llm:
            return self._heuristic_recall(contexts, reference)

        prompt = ChatPromptTemplate.from_template(
            "You are evaluating retrieval completeness (context recall).\n"
            "Reference Answer: {reference}\n\n"
            "Retrieved Contexts:\n{contexts}\n\n"
            "Task:\n"
            "Determine what fraction of the core facts in the Reference Answer can be found or attributed "
            "to the retrieved contexts.\n"
            "Output ONLY a JSON object: {{\"recall\": <float 0.0 to 1.0>}}"
        )

        try:
            combined = "\n\n".join(contexts)
            chain = prompt | self.llm
            res = chain.invoke({"reference": reference, "contexts": combined})
            raw_text = res.content if hasattr(res, "content") else str(res)
            parsed = self._extract_json(raw_text)
            return float(parsed.get("recall", 0.0))
        except Exception:
            return self._heuristic_recall(contexts, reference)

    # ── Fallback Scorer Helpers ──────────────────────────────────────────────

    @staticmethod
    def _extract_json(text: str) -> Dict[str, Any]:
        match = re.search(r"\{.*?\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return json.loads(text)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return [w.lower() for w in re.findall(r"\b\w{3,}\b", text)]

    def _heuristic_jaccard(self, a: str, b: str) -> float:
        tokens_a = set(self._tokenize(a))
        tokens_b = set(self._tokenize(b))
        if not tokens_a or not tokens_b:
            return 0.0
        intersection = tokens_a.intersection(tokens_b)
        union = tokens_a.union(tokens_b)
        return round(len(intersection) / len(union), 4)

    def _heuristic_relevancy(self, question: str, answer: str) -> float:
        q_tokens = set(self._tokenize(question))
        a_tokens = set(self._tokenize(answer))
        if not q_tokens:
            return 0.0
        overlap = q_tokens.intersection(a_tokens)
        return round(min(1.0, len(overlap) / max(1, len(q_tokens) * 0.5)), 4)

    def _heuristic_faithfulness(self, answer: str, contexts: List[str]) -> float:
        answer_tokens = set(self._tokenize(answer))
        if not answer_tokens:
            return 0.0
        context_tokens = set()
        for ctx in contexts:
            context_tokens.update(self._tokenize(ctx))
        overlap = answer_tokens.intersection(context_tokens)
        return round(min(1.0, len(overlap) / len(answer_tokens)), 4)

    def _heuristic_recall(self, contexts: List[str], reference: str) -> float:
        ref_tokens = set(self._tokenize(reference))
        if not ref_tokens:
            return 1.0
        context_tokens = set()
        for ctx in contexts:
            context_tokens.update(self._tokenize(ctx))
        overlap = ref_tokens.intersection(context_tokens)
        return round(len(overlap) / len(ref_tokens), 4)
