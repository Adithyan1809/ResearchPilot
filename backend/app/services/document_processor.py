"""PDF text extraction and structure-aware hierarchical chunking service.

Handles the ingestion half of the RAG pipeline: reads a PDF with PyMuPDF,
parses academic document structure (headings, sections, equations, tables),
injects hierarchical breadcrumbs, preserves atomic blocks, and attaches rich
metadata to retrieval-ready chunks.
"""

import re
import uuid
from typing import Dict, List, Optional, Tuple

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

# ── Academic Structure & Heading Regex Patterns ──────────────────────────────

NUMBERED_HEADING_PATTERN = re.compile(
    r"^(?:(\d+(?:\.\d+)*)\.?\s+([A-Z][\w\s\-\:\,\(\)\/]+)|([IVXLCDM]+)\.\s+([A-Z][\w\s\-\:\,\(\)\/]+))$"
)

STANDARD_HEADING_NAMES = (
    r"Abstract|Introduction|Background|Related Work|Methodology|Methods|"
    r"Architecture|Model Architecture|Model|Experiments|Experimental Setup|"
    r"Results|Evaluation|Discussion|Conclusions?|Acknowledgements?|"
    r"References|Bibliography|Appendix(?:\s+[A-Z0-9]+)?"
)

STANDARD_HEADING_PATTERN = re.compile(
    rf"^({STANDARD_HEADING_NAMES})[\:\.\-]?$",
    re.IGNORECASE,
)

INLINE_ABSTRACT_PATTERN = re.compile(
    r"^(Abstract)\s*[:\.\-]\s*(.*)$",
    re.IGNORECASE,
)

EQ_TABLE_PATTERNS = [
    re.compile(
        r"\\begin\{(?:equation|align|gather|multline|matrix|bmatrix|pmatrix)\*?\}(?:[\s\S]*?)\\end\{(?:equation|align|gather|multline|matrix|bmatrix|pmatrix)\*?\}"
    ),
    re.compile(r"(?<!\\)\$\$[\s\S]*?\$\$"),
    re.compile(r"\\\[[\s\S]*?\\\]"),
    re.compile(
        r"(?:^|\n)([ \t]*\|[^\n]+\|\r?\n[ \t]*\|(?:\s*:?-+:?\s*\|)+\r?\n(?:[ \t]*\|[^\n]+\|\r?\n?)+)"
    ),
    re.compile(
        r"(?:^|\n)([ \t]*\+[-+=]+\+[^\n]*\r?\n(?:[ \t]*[|+].*[|+]\r?\n?)+)"
    ),
]


class DocumentProcessor:
    """Extract text from PDFs and split into structure-aware, retrieval-ready chunks."""

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

    # ── Structure Detection & Parsing Helpers ─────────────────────────────

    def _find_atomic_spans(self, text: str) -> List[Tuple[int, int]]:
        """Find non-overlapping character index spans for equations and tables."""
        spans: List[Tuple[int, int]] = []
        for pat in EQ_TABLE_PATTERNS:
            for m in pat.finditer(text):
                spans.append((m.start(), m.end()))
        spans.sort(key=lambda s: (s[0], -s[1]))

        merged: List[Tuple[int, int]] = []
        for s, e in spans:
            if not merged:
                merged.append((s, e))
            else:
                last_s, last_e = merged[-1]
                if s < last_e:
                    merged[-1] = (last_s, max(last_e, e))
                else:
                    merged.append((s, e))
        return merged

    def _check_heading(self, line: str) -> Optional[Dict]:
        """Check if a line represents an academic heading using regex and layout heuristics."""
        s = line.strip()
        if not s or len(s) > 100 or len(s.split()) > 10:
            return None
        # Exclude math or table characters
        if any(ch in s for ch in ["$", "=", "\\", "|", "+", "{", "}"]):
            return None

        # Check inline abstract
        m_abs = INLINE_ABSTRACT_PATTERN.match(s)
        if m_abs:
            return {
                "title": "Abstract",
                "level": 1,
                "inline_content": m_abs.group(2).strip(),
            }

        # Check standard unnumbered headings
        m_std = STANDARD_HEADING_PATTERN.match(s)
        if m_std:
            matched = m_std.group(1)
            normalized = matched.title()
            if normalized.lower() == "references":
                normalized = "References"
            return {
                "title": normalized,
                "level": 1,
                "inline_content": "",
            }

        # Check numbered headings
        m_num = NUMBERED_HEADING_PATTERN.match(s)
        if m_num:
            num_str = m_num.group(1) or m_num.group(3)
            title = (m_num.group(2) or m_num.group(4)).strip()
            words = set(title.lower().split())
            # Heuristic: reject conversational full sentences in lists
            if s.endswith(".") and words.intersection(
                {"we", "our", "is", "are", "that", "with", "were", "have", "from", "this"}
            ):
                return None
            level = len(num_str.split(".")) if m_num.group(1) else 1
            heading_text = s.rstrip(":")
            return {
                "title": heading_text,
                "level": level,
                "inline_content": "",
            }

        return None

    @traceable(
        name="section_parsing",
        run_type="parser",
        metadata={"agent_id": "researchpilot", "agent_environment": "local"},
        tags=["researchpilot", "local", "section_parsing"],
    )
    def parse_sections(
        self,
        text: str,
        section_stack: Optional[List[Tuple[int, str]]] = None,
    ) -> List[Dict]:
        """Parse academic sections and headings from document text.

        Parameters
        ----------
        text:
            Extracted text content from a PDF page or section.
        section_stack:
            Optional running hierarchy stack of ``(level, title)`` tuples
            maintained across pages.

        Returns
        -------
        list[dict]
            Each dict has ``heading``, ``level``, ``content``, and ``hierarchy``.
        """
        if section_stack is None:
            section_stack = []

        atomic_spans = self._find_atomic_spans(text)

        lines_with_offsets: List[Tuple[str, int, int]] = []
        cur = 0
        for line in text.splitlines(keepends=True):
            nxt = cur + len(line)
            lines_with_offsets.append((line, cur, nxt))
            cur = nxt

        found_headings: List[Tuple[Dict, int, int]] = []
        for line, s, e in lines_with_offsets:
            in_atomic = any(span_s <= s and e <= span_e for span_s, span_e in atomic_spans)
            if in_atomic:
                continue
            hd = self._check_heading(line)
            if hd:
                found_headings.append((hd, s, e))

        if not found_headings:
            cur_hier = " > ".join(t for _, t in section_stack) if section_stack else "Document"
            return [{
                "heading": "",
                "level": 1,
                "content": text.strip(),
                "hierarchy": cur_hier,
            }]

        sections: List[Dict] = []
        first_hd, first_s, _ = found_headings[0]
        if first_s > 0:
            preamble = text[:first_s].strip()
            if preamble:
                cur_hier = " > ".join(t for _, t in section_stack) if section_stack else "Document"
                sections.append({
                    "heading": "",
                    "level": 1,
                    "content": preamble,
                    "hierarchy": cur_hier,
                })

        for i, (hd, s, e) in enumerate(found_headings):
            nxt_s = found_headings[i + 1][1] if i + 1 < len(found_headings) else len(text)
            raw_content = text[e:nxt_s].strip()
            if hd["inline_content"]:
                content = hd["inline_content"] + ("\n\n" + raw_content if raw_content else "")
            else:
                content = raw_content

            lvl = hd["level"]
            while section_stack and section_stack[-1][0] >= lvl:
                section_stack.pop()
            section_stack.append((lvl, hd["title"]))
            hierarchy = " > ".join(t for _, t in section_stack)
            sections.append({
                "heading": hd["title"],
                "level": lvl,
                "content": content,
                "hierarchy": hierarchy,
            })

        return sections

    def _extract_blocks(self, text: str) -> List[Tuple[str, bool]]:
        """Segment section text into atomic blocks (equations/tables) and paragraph blocks.

        Returns
        -------
        list[tuple[str, bool]]
            Tuples of ``(block_text, is_atomic)``.
        """
        atomic_spans = self._find_atomic_spans(text)
        if not atomic_spans:
            paras = text.split("\n\n")
            return [(p.strip(), False) for p in paras if p.strip()]

        blocks: List[Tuple[str, bool]] = []
        last_idx = 0
        for s, e in atomic_spans:
            if s > last_idx:
                inter = text[last_idx:s].strip()
                if inter:
                    for p in inter.split("\n\n"):
                        if p.strip():
                            blocks.append((p.strip(), False))
            blocks.append((text[s:e].strip(), True))
            last_idx = e

        if last_idx < len(text):
            inter = text[last_idx:].strip()
            if inter:
                for p in inter.split("\n\n"):
                    if p.strip():
                        blocks.append((p.strip(), False))

        return blocks

    # ── Chunking ─────────────────────────────────────────────────────────

    @traceable(
        name="structure_aware_chunking",
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
        """Split page texts into structure-aware hierarchical chunks with breadcrumbs.

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
            LangChain ``Document`` objects with breadcrumb headers and rich metadata.
        """
        if not pages:
            raise ValueError(
                f"Cannot create chunks from empty pages list for document '{filename}'"
            )

        total_pages = len(pages)
        all_chunks: List[Document] = []
        section_stack: List[Tuple[int, str]] = []

        for page in pages:
            page_num = page["page_number"]
            text = page["text"]
            sections = self.parse_sections(text, section_stack)
            page_chunk_idx = 0

            for sec in sections:
                content = sec["content"].strip()
                if not content:
                    continue
                hierarchy = sec["hierarchy"]
                breadcrumb = f"[Document: {filename} | Section: {hierarchy}]"
                max_body = max(200, self.chunk_size - len(breadcrumb) - 2)

                blocks = self._extract_blocks(content)
                cur_blocks: List[str] = []
                cur_len = 0

                def emit_chunk(body_text: str) -> None:
                    nonlocal page_chunk_idx
                    chunk_content = f"{breadcrumb}\n\n{body_text}"
                    doc = Document(
                        page_content=chunk_content,
                        metadata={
                            "doc_id": doc_id,
                            "filename": filename,
                            "page_number": page_num,
                            "chunk_index": page_chunk_idx,
                            "total_pages": total_pages,
                            "section": hierarchy,
                            "hierarchy": hierarchy,
                            "section_hierarchy": hierarchy,
                            "breadcrumb": breadcrumb,
                            "header": breadcrumb,
                        },
                    )
                    all_chunks.append(doc)
                    page_chunk_idx += 1

                for b_text, is_atomic in blocks:
                    if is_atomic:
                        if cur_blocks:
                            if cur_len + 2 + len(b_text) <= max_body:
                                cur_blocks.append(b_text)
                                cur_len += 2 + len(b_text)
                            else:
                                emit_chunk("\n\n".join(cur_blocks))
                                cur_blocks = []
                                cur_len = 0
                                emit_chunk(b_text)
                        else:
                            emit_chunk(b_text)
                    else:
                        if len(b_text) <= max_body:
                            if cur_blocks and (cur_len + 2 + len(b_text) > max_body):
                                emit_chunk("\n\n".join(cur_blocks))
                                cur_blocks = [b_text]
                                cur_len = len(b_text)
                            else:
                                cur_blocks.append(b_text)
                                cur_len += (2 if cur_len > 0 else 0) + len(b_text)
                        else:
                            if cur_blocks:
                                emit_chunk("\n\n".join(cur_blocks))
                                cur_blocks = []
                                cur_len = 0
                            # Sub-split large paragraph using fallback character splitter
                            sub_parts = self.splitter.split_text(b_text)
                            for sp in sub_parts:
                                emit_chunk(sp)

                if cur_blocks:
                    emit_chunk("\n\n".join(cur_blocks))

        logger.info(
            "Created %d structure-aware chunks for document '%s' (doc_id=%s)",
            len(all_chunks),
            filename,
            doc_id,
        )
        return all_chunks

    # Backward-compatible method alias
    structure_aware_chunking = create_chunks

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
