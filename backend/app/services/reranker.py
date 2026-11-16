"""Two-stage retrieval cross-encoder reranking service using Google Gemini.

Re-scores candidate chunks retrieved in Stage 1 (Hybrid RRF / Dense / BM25)
by evaluating joint query-document semantic relevance using Gemini Flash
at temperature=0.0 to improve context precision and reduce prompt tokens.
"""

import json
import re
import time
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

try:
    from langsmith import traceable
except ImportError:
    def traceable(name=None, run_type=None, **kwargs):
        def decorator(fn):
            return fn
        return decorator

from backend.app.config import Settings
from backend.app.models.schemas import SourceDocument
from backend.app.utils.logger import setup_logger

logger = setup_logger(__name__)


class RerankerService:
    """Rerank retrieved candidate chunks using an LLM cross-encoder."""

    def __init__(self, settings: Settings) -> None:
        """Initialize the reranker service.

        Parameters
        ----------
        settings:
            Application settings with reranker model name and Google API key.
        """
        self._settings = settings
        self._model_name = getattr(settings, "reranker_model", "gemini-3.5-flash-lite")
        self._top_k = getattr(settings, "reranker_top_k", 3)
        self.last_rerank_latency_ms: float = 0.0

        self._llm = ChatGoogleGenerativeAI(
            model=self._model_name,
            temperature=0.0,
            google_api_key=settings.google_api_key,
        )
        logger.info(
            "RerankerService initialized (model=%s, default_top_k=%d)",
            self._model_name,
            self._top_k,
        )

    @property
    def settings(self) -> Settings:
        """Return the configuration settings."""
        return self._settings

    @property
    def model_name(self) -> str:
        """Return the model name."""
        return self._model_name

    @property
    def top_k(self) -> int:
        """Return default top_k."""
        return self._top_k

    @property
    def llm(self) -> Any:
        """Return the underlying LLM instance."""
        return self._llm

    @llm.setter
    def llm(self, value: Any) -> None:
        """Set the underlying LLM instance (useful for testing and dependency injection)."""
        self._llm = value

    def _build_scoring_prompt(self, query: str, candidates: List[SourceDocument]) -> str:
        """Format query and candidate chunks into a scoring prompt."""
        candidate_blocks = []
        for idx, doc in enumerate(candidates):
            candidate_blocks.append(
                f"[Candidate {idx}]\n"
                f"Source: {doc.filename} (Page {doc.page_number})\n"
                f"Content: {doc.content.strip()}"
            )
        candidates_str = "\n\n".join(candidate_blocks)

        prompt = (
            "You are an expert relevance evaluator for an academic research retrieval system.\n"
            "Given a user query and a list of candidate document chunks, evaluate the semantic "
            "relevance of each candidate chunk to the query.\n\n"
            "Assign a relevance score between 0.0 and 1.0 to each candidate:\n"
            "- 1.0: Directly and completely answers the query with high factual relevance.\n"
            "- 0.7 - 0.9: Highly relevant; contains key concepts or strong partial answers.\n"
            "- 0.4 - 0.6: Moderately relevant; related topic or background context.\n"
            "- 0.1 - 0.3: Marginally relevant; mentions keywords in passing.\n"
            "- 0.0: Completely irrelevant or off-topic.\n\n"
            f"Query: {query}\n\n"
            f"Candidates:\n{candidates_str}\n\n"
            "Output your evaluations as a strict JSON array of objects with 'index' and 'score' keys, like:\n"
            "[\n"
            "  {\"index\": 0, \"score\": 0.85},\n"
            "  {\"index\": 1, \"score\": 0.20}\n"
            "]\n"
            "Respond ONLY with the JSON array, with no markdown code blocks, prefixes, or explanation."
        )
        return prompt

    def _parse_scores(self, output: Any, expected_count: int) -> Dict[int, float]:
        """Extract candidate index-to-score mappings from LLM response.

        Handles JSON arrays, JSON objects, raw text, and mock objects.
        """
        scores: Dict[int, float] = {}

        # 1. If output has .content (LangChain BaseMessage / AIMessage)
        if hasattr(output, "content"):
            content = output.content
        else:
            content = output

        # Handle LangChain content block lists, e.g. [{'type': 'text', 'text': '...'}]
        if isinstance(content, list):
            if content and any(isinstance(p, dict) and "text" in p for p in content):
                text_parts = []
                for p in content:
                    if isinstance(p, dict) and "text" in p:
                        text_parts.append(p["text"])
                    elif isinstance(p, str):
                        text_parts.append(p)
                content = "".join(text_parts)
            elif content and all(isinstance(p, str) for p in content):
                content = "".join(content)

        # 2. If content is already a list or dict of candidate scores
        if isinstance(content, list):
            for i, item in enumerate(content):
                if isinstance(item, dict):
                    idx = item.get("index", item.get("id", i))
                    sc = item.get("score", item.get("relevance", 0.0))
                    try:
                        scores[int(idx)] = float(sc)
                    except (ValueError, TypeError):
                        pass
                elif isinstance(item, (int, float)):
                    scores[i] = float(item)
            return scores

        if isinstance(content, dict):
            items = content.get("scores", content.get("candidates", []))
            if isinstance(items, list):
                for i, item in enumerate(items):
                    if isinstance(item, dict):
                        idx = item.get("index", item.get("id", i))
                        sc = item.get("score", item.get("relevance", 0.0))
                        try:
                            scores[int(idx)] = float(sc)
                        except (ValueError, TypeError):
                            pass
                    elif isinstance(item, (int, float)):
                        scores[i] = float(item)
                return scores
            for k, v in content.items():
                try:
                    scores[int(k)] = float(v)
                except (ValueError, TypeError):
                    pass
            if scores:
                return scores

        # 3. String parsing
        text = str(content).strip()

        # Remove markdown code fences if present
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
            text = re.sub(r"\s*```$", "", text)
            text = text.strip()

        # Attempt standard JSON decode
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                for i, item in enumerate(parsed):
                    if isinstance(item, dict):
                        idx = item.get("index", item.get("id", i))
                        sc = item.get("score", item.get("relevance", 0.0))
                        try:
                            scores[int(idx)] = float(sc)
                        except (ValueError, TypeError):
                            pass
                    elif isinstance(item, (int, float)):
                        scores[i] = float(item)
                return scores
            elif isinstance(parsed, dict):
                items = parsed.get("scores", parsed.get("candidates", []))
                if isinstance(items, list):
                    for i, item in enumerate(items):
                        if isinstance(item, dict):
                            idx = item.get("index", item.get("id", i))
                            sc = item.get("score", item.get("relevance", 0.0))
                            try:
                                scores[int(idx)] = float(sc)
                            except (ValueError, TypeError):
                                pass
                        elif isinstance(item, (int, float)):
                            scores[i] = float(item)
                    return scores
                for k, v in parsed.items():
                    try:
                        scores[int(k)] = float(v)
                    except (ValueError, TypeError):
                        pass
                if scores:
                    return scores
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

        # 4. Regex fallback: search for {"index": X, "score": Y}
        dict_matches = re.findall(
            r'\{[^{}]*"(?:index|id)"\s*:\s*(\d+)[^{}]*"score"\s*:\s*([0-9]*\.?[0-9]+)[^{}]*\}',
            text,
        )
        if dict_matches:
            for idx_str, sc_str in dict_matches:
                scores[int(idx_str)] = float(sc_str)
            return scores

        # Regex fallback: "Candidate 0: 0.85" or "0: 0.85"
        line_matches = re.findall(
            r'(?:\[?(?:candidate|chunk)?\s*(\d+)\]?)\s*[:\-=\s]\s*([0-9]*\.?[0-9]+)',
            text,
            flags=re.IGNORECASE,
        )
        if line_matches:
            for idx_str, sc_str in line_matches:
                scores[int(idx_str)] = float(sc_str)
            return scores

        # Regex fallback: list of floats [0.85, 0.2]
        float_matches = re.findall(r'\b[0-1]\.[0-9]+\b', text)
        if float_matches:
            for i, sc_str in enumerate(float_matches[:expected_count]):
                scores[i] = float(sc_str)
            return scores

        logger.warning("Could not parse reranker LLM output into structured scores: %s", text[:100])
        return scores

    def _score_candidates(self, query: str, candidates: List[SourceDocument]) -> Dict[int, float]:
        """Invoke LLM cross-encoder and parse relevance scores."""
        prompt = self._build_scoring_prompt(query, candidates)
        messages = [
            SystemMessage(
                content="You are an expert AI relevance scoring engine. You evaluate query-document relevance and output strict JSON."
            ),
            HumanMessage(content=prompt),
        ]
        response = self._llm.invoke(messages)
        return self._parse_scores(response, expected_count=len(candidates))

    @traceable(
        name="cross_encoder_rerank",
        run_type="chain",
        metadata={"agent_id": "researchpilot", "agent_environment": "local"},
        tags=["researchpilot", "local", "reranker"],
    )
    def rerank(
        self,
        query: str,
        candidates: List[SourceDocument],
        top_k: Optional[int] = None,
    ) -> List[SourceDocument]:
        """Rerank candidate chunks using Gemini cross-encoder scoring.

        Parameters
        ----------
        query:
            The research question or search query.
        candidates:
            List of candidate SourceDocuments from Stage 1 retrieval.
        top_k:
            Optional number of top documents to return. If None, defaults to
            settings.reranker_top_k.

        Returns
        -------
        List[SourceDocument]
            Top-K reranked documents with rerank_score and updated relevance_score.
        """
        if not candidates:
            logger.debug("Reranker received empty candidate list; returning empty list")
            return []

        k = top_k if top_k is not None else self._top_k
        k = max(1, k)
        start = time.perf_counter()

        try:
            scores_map = self._score_candidates(query, candidates)

            reranked: List[SourceDocument] = []
            for idx, doc in enumerate(candidates):
                raw_score = scores_map.get(idx, doc.relevance_score)
                score = round(max(0.0, min(1.0, float(raw_score))), 4)

                reranked_doc = SourceDocument(
                    content=doc.content,
                    page_number=doc.page_number,
                    chunk_index=doc.chunk_index,
                    relevance_score=score,
                    doc_id=doc.doc_id,
                    filename=doc.filename,
                    retrieval_method=doc.retrieval_method,
                    rrf_score=doc.rrf_score,
                    rerank_score=score,
                )
                reranked.append(reranked_doc)

            # Sort descending by rerank_score
            reranked.sort(
                key=lambda d: (d.rerank_score if d.rerank_score is not None else -1.0),
                reverse=True,
            )

            final_docs = reranked[:k]
            elapsed_ms = (time.perf_counter() - start) * 1000
            self.last_rerank_latency_ms = elapsed_ms

            logger.info(
                "Reranked %d candidates to top %d in %.1f ms",
                len(candidates),
                len(final_docs),
                elapsed_ms,
            )
            return final_docs

        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000
            self.last_rerank_latency_ms = elapsed_ms
            logger.warning(
                "Reranking failed (%s); falling back to original candidate order",
                exc,
            )
            return candidates[:k]
