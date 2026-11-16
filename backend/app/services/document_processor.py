"""PDF text extraction and chunking service.

Handles the ingestion half of the RAG pipeline: reads a PDF with PyMuPDF,
splits the text into overlapping chunks using LangChain's
``RecursiveCharacterTextSplitter``, and attaches rich metadata to each chunk.
"""

import uuid
from typing import Dict, List, Tuple

import fitz  # PyMuPDF
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from langsmith import traceable
except ImportError:
    def traceable(name=None, run_type=None, **kwargs):
        def decorator(fn):
            return fn
        return decorator

from backend.app.config import Settings
from backend.app.utils.logger import setup_logger

logger = setup_logger(__name__)


class DocumentProcessor:
    """Extract text from PDFs and split into retrieval-ready chunks."""

    def __init__(self, settings: Settings) -> None:
        """Initialise the processor with chunking parameters from *settings*.

        Parameters
        ----------
        settings:
            Application settings containing ``chunk_size`` and
            ``chunk_overlap``.
        """
        self.chunk_size = settings.chunk_size
        self.chunk_overlap = settings.chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )
        logger.info(
            "DocumentProcessor initialised (chunk_size=%d, overlap=%d)",
            self.chunk_size,
            self.chunk_overlap,
        )

    # ── PDF extraction ───────────────────────────────────────────────────

    @traceable(
        name="pdf_text_extraction",
        run_type="parser",
        metadata={"agent_id": "researchpilot", "agent_environment": "local"},
        tags=["researchpilot", "local", "extraction"],
    )
    def extract_text_from_pdf(self, file_path: str) -> List[Dict]:
        """Extract text from every page of a PDF file.

        Parameters
        ----------
        file_path:
            Absolute or relative path to the PDF.

        Returns
        -------
        list[dict]
            Each element has ``page_number`` (1-indexed), ``text``, and
            ``char_count``.

        Raises
        ------
        FileNotFoundError
            If *file_path* does not exist.
        ValueError
            If the PDF contains no extractable text.
        """
        logger.info("Extracting text from PDF: %s", file_path)

        try:
            doc = fitz.open(file_path)
        except Exception as exc:
            logger.error("Failed to open PDF %s: %s", file_path, exc)
            raise

        pages: List[Dict] = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")
            if text and text.strip():
                pages.append(
                    {
                        "page_number": page_num + 1,  # 1-indexed
                        "text": text.strip(),
                        "char_count": len(text.strip()),
                    }
                )

        doc.close()

        if not pages:
            raise ValueError(
                f"No extractable text found in PDF: {file_path}"
            )

        logger.info(
            "Extracted %d pages with text from %s (total chars: %d)",
            len(pages),
            file_path,
            sum(p["char_count"] for p in pages),
        )
        return pages

    # ── Chunking ─────────────────────────────────────────────────────────

    @traceable(
        name="document_chunking",
        run_type="parser",
        metadata={"agent_id": "researchpilot", "agent_environment": "local"},
        tags=["researchpilot", "local", "chunking"],
    )
    def create_chunks(
        self,
        pages: List[Dict],
        doc_id: str,
        filename: str,
    ) -> List[Document]:
        """Split page texts into overlapping chunks with metadata.

        Parameters
        ----------
        pages:
            Output of :meth:`extract_text_from_pdf`.
        doc_id:
            Unique document identifier.
        filename:
            Original filename for metadata.

        Returns
        -------
        list[Document]
            LangChain ``Document`` objects ready for embedding.
        """
        if not pages:
            raise ValueError(
                f"Cannot create chunks from empty pages list for document '{filename}'"
            )

        total_pages = len(pages)
        all_chunks: List[Document] = []

        for page in pages:
            page_chunks = self.splitter.split_text(page["text"])
            for chunk_idx, chunk_text in enumerate(page_chunks):
                doc = Document(
                    page_content=chunk_text,
                    metadata={
                        "doc_id": doc_id,
                        "filename": filename,
                        "page_number": page["page_number"],
                        "chunk_index": chunk_idx,
                        "total_pages": total_pages,
                    },
                )
                all_chunks.append(doc)

        logger.info(
            "Created %d chunks for document '%s' (doc_id=%s)",
            len(all_chunks),
            filename,
            doc_id,
        )
        return all_chunks

    # ── Orchestrator ─────────────────────────────────────────────────────

    @traceable(
        name="pdf_processing_pipeline",
        run_type="chain",
        metadata={"agent_id": "researchpilot", "agent_environment": "local"},
        tags=["researchpilot", "local", "document_processing"],
    )
    def process_pdf(
        self,
        file_path: str,
        filename: str,
    ) -> Tuple[str, List[Document]]:
        """Full PDF processing pipeline: extract → chunk → return.

        Parameters
        ----------
        file_path:
            Path to the PDF on disk.
        filename:
            Original upload filename.

        Returns
        -------
        tuple[str, list[Document]]
            A ``(doc_id, chunks)`` pair.

        Raises
        ------
        ValueError
            If no text can be extracted or no chunks are produced.
        """
        doc_id = str(uuid.uuid4())
        logger.info(
            "Processing PDF '%s' → doc_id=%s", filename, doc_id
        )

        pages = self.extract_text_from_pdf(file_path)
        chunks = self.create_chunks(pages, doc_id, filename)

        if not chunks:
            raise ValueError(
                f"Chunking produced 0 chunks for document '{filename}'"
            )

        logger.info(
            "PDF processing complete: %d pages → %d chunks",
            len(pages),
            len(chunks),
        )
        return doc_id, chunks
