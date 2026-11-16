"""Answer generation service using Google Gemini via LangChain.

Takes retrieved source chunks, formats them into a carefully-engineered
prompt with conversational context, and invokes ``ChatGoogleGenerativeAI``
to produce cited answers via either standard or real-time streaming interfaces.
"""

from typing import Dict, Iterator, List, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

try:
    from langsmith import traceable
except ImportError:
    def traceable(name=None, run_type=None, **kwargs):
        def decorator(fn):
            return fn
        return decorator

from backend.app.config import Settings
from backend.app.models.schemas import ChatMessage, SourceDocument
from backend.app.utils.logger import setup_logger

logger = setup_logger(__name__)


class GeneratorService:
    """Generate cited answers from retrieved research paper chunks."""

    RAG_PROMPT_TEMPLATE: str = (
        "You are a rigorous research assistant specializing in academic paper analysis. "
        "Your task is to answer the user's question using the provided source context, "
        "taking into account previous conversation turns when relevant.\n\n"
        "## Guidelines\n"
        "1. Base your answer exclusively on the source context below.\n"
        "2. Cite every claim using [Source N] notation (e.g., [Source 1], [Source 2]).\n"
        "3. If the context does not contain enough information to answer fully, "
        "state: \"I don't have enough information in the provided sources to answer this completely.\"\n"
        "4. Do NOT fabricate information or use knowledge outside the provided context.\n"
        "5. Structure your answer clearly: use paragraphs, bullet points, or numbered lists where appropriate.\n"
        "6. When multiple sources support the same point, cite all of them (e.g., [Source 1][Source 3]).\n"
        "7. Maintain conversational continuity when referring to entities mentioned in the conversation history.\n\n"
        "## Conversation History\n"
        "{chat_history}\n\n"
        "## Source Context\n"
        "{context}\n\n"
        "## Question\n"
        "{question}\n\n"
        "## Answer"
    )

    def __init__(self, settings: Settings) -> None:
        """Initialise the LLM.

        Parameters
        ----------
        settings:
            Application settings with model name, temperature, max tokens,
            and API key.
        """
        self._llm = ChatGoogleGenerativeAI(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            max_output_tokens=settings.llm_max_tokens,
            google_api_key=settings.google_api_key,
        )
        self._prompt = ChatPromptTemplate.from_template(self.RAG_PROMPT_TEMPLATE)
        logger.info(
            "GeneratorService initialised (model=%s, temp=%.2f)",
            settings.llm_model,
            settings.llm_temperature,
        )

    # ── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _format_context(sources: List[SourceDocument]) -> str:
        """Format source chunks into a labelled context block.

        Each chunk is prefixed with ``[Source N]`` and annotated with its
        page number and filename for traceability.
        """
        parts: List[str] = []
        for idx, src in enumerate(sources, start=1):
            header = (
                f"[Source {idx}] "
                f"(File: {src.filename}, Page: {src.page_number}, "
                f"Chunk: {src.chunk_index})"
            )
            parts.append(f"{header}\n{src.content}")
        return "\n\n".join(parts)

    @staticmethod
    def _format_chat_history(chat_history: Optional[List[ChatMessage]]) -> str:
        """Format chat message history into structured conversational turns."""
        if not chat_history:
            return "None (this is the start of the conversation)"
        turns: List[str] = []
        for msg in chat_history:
            role_label = (
                "User"
                if msg.role == "user"
                else "Assistant"
                if msg.role == "assistant"
                else "System"
            )
            turns.append(f"{role_label}: {msg.content}")
        return "\n".join(turns)

    # ── Generation ───────────────────────────────────────────────────────

    @traceable(
        name="answer_generation",
        run_type="chain",
        metadata={"agent_id": "researchpilot", "agent_environment": "local"},
        tags=["researchpilot", "local", "generation"],
    )
    def generate(
        self,
        question: str,
        sources: List[SourceDocument],
        chat_history: Optional[List[ChatMessage]] = None,
    ) -> Dict:
        """Generate an answer with citations from retrieved sources and conversation history.

        Parameters
        ----------
        question:
            The user's research question.
        sources:
            Retrieved and ranked source chunks.
        chat_history:
            Optional previous turns in the conversation.

        Returns
        -------
        dict
            ``{"answer": str, "tokens_used": {"prompt_tokens": int,
            "completion_tokens": int, "total_tokens": int}}``
        """
        if not sources:
            logger.warning("generate() called with no sources")
            return {
                "answer": (
                    "I don't have enough information in the provided "
                    "sources to answer this question. Please ingest "
                    "relevant documents first."
                ),
                "tokens_used": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
            }

        context = self._format_context(sources)
        formatted_history = self._format_chat_history(chat_history)
        logger.info(
            "Generating answer for question (len=%d) with %d sources and %d history turns",
            len(question),
            len(sources),
            len(chat_history) if chat_history else 0,
        )

        chain = self._prompt | self._llm
        response = chain.invoke(
            {
                "context": context,
                "question": question,
                "chat_history": formatted_history,
            }
        )

        # Extract token usage from the response metadata
        usage_metadata = getattr(response, "usage_metadata", None) or {}
        tokens_used = {
            "prompt_tokens": usage_metadata.get("input_tokens", 0),
            "completion_tokens": usage_metadata.get("output_tokens", 0),
            "total_tokens": usage_metadata.get("total_tokens", 0),
        }

        raw_content = getattr(response, "content", response)
        if isinstance(raw_content, list):
            text_parts = []
            for part in raw_content:
                if isinstance(part, dict) and "text" in part:
                    text_parts.append(part["text"])
                elif isinstance(part, str):
                    text_parts.append(part)
                else:
                    text_parts.append(str(part))
            answer = "".join(text_parts)
        else:
            answer = str(raw_content)
        logger.info(
            "Generation complete (tokens: %s)", tokens_used
        )

        return {
            "answer": answer,
            "tokens_used": tokens_used,
        }

    @traceable(
        name="answer_generation_stream",
        run_type="chain",
        metadata={"agent_id": "researchpilot", "agent_environment": "local"},
        tags=["researchpilot", "local", "generation", "streaming"],
    )
    def generate_stream(
        self,
        question: str,
        sources: List[SourceDocument],
        chat_history: Optional[List[ChatMessage]] = None,
    ) -> Iterator[str]:
        """Stream generated answer tokens in real-time.

        Parameters
        ----------
        question:
            The user's research question.
        sources:
            Retrieved and ranked source chunks.
        chat_history:
            Optional previous turns in the conversation.

        Yields
        ------
        str
            Token fragments / text deltas as they are generated by the model.
        """
        if not sources:
            logger.warning("generate_stream() called with no sources")
            yield (
                "I don't have enough information in the provided "
                "sources to answer this question. Please ingest "
                "relevant documents first."
            )
            return

        context = self._format_context(sources)
        formatted_history = self._format_chat_history(chat_history)
        logger.info(
            "Streaming answer for question (len=%d) with %d sources and %d history turns",
            len(question),
            len(sources),
            len(chat_history) if chat_history else 0,
        )

        chain = self._prompt | self._llm
        for chunk in chain.stream(
            {
                "context": context,
                "question": question,
                "chat_history": formatted_history,
            }
        ):
            content = getattr(chunk, "content", chunk)
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and "text" in part:
                        yield part["text"]
                    elif isinstance(part, str):
                        yield part
                    else:
                        yield str(part)
            elif isinstance(content, str) and content:
                yield content
            elif hasattr(content, "text"):
                yield content.text
