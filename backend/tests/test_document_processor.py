"""Unit tests for DocumentProcessor with structure-aware hierarchical chunking."""

import os
from unittest.mock import MagicMock, patch

import pytest

from backend.app.config import Settings
from backend.app.services.document_processor import DocumentProcessor


@pytest.fixture
def test_settings():
    return Settings(
        google_api_key="test-api-key",
        chunk_size=500,
        chunk_overlap=100,
    )


@pytest.fixture
def processor(test_settings):
    return DocumentProcessor(test_settings)


# ── Existing Baseline Unit Tests (Maintained 100% Intact) ─────────────────────

def test_processor_initialization(processor, test_settings):
    assert processor.chunk_size == 500
    assert processor.chunk_overlap == 100
    assert processor.splitter is not None


def test_create_chunks_produces_correct_metadata(processor):
    pages = [
        {"page_number": 1, "text": "This is page one of a research paper on attention mechanisms. " * 10, "char_count": 620},
        {"page_number": 2, "text": "This is page two presenting empirical results and BLEU benchmarks. " * 8, "char_count": 520},
    ]

    doc_id = "test-doc-123"
    filename = "paper.pdf"

    chunks = processor.create_chunks(pages, doc_id, filename)

    assert len(chunks) >= 2
    for chunk in chunks:
        assert chunk.metadata["doc_id"] == doc_id
        assert chunk.metadata["filename"] == filename
        assert chunk.metadata["total_pages"] == 2
        assert chunk.metadata["page_number"] in [1, 2]
        assert "chunk_index" in chunk.metadata
        assert len(chunk.page_content) > 0


def test_empty_pages_raises_error(processor):
    with pytest.raises(ValueError, match="empty pages list"):
        processor.create_chunks([], "test-doc", "empty.pdf")


@patch("fitz.open")
def test_extract_text_from_pdf_mocked(mock_fitz_open, processor):
    # Mock PyMuPDF Document
    mock_doc = MagicMock()
    mock_page1 = MagicMock()
    mock_page1.get_text.return_value = "Abstract: We propose the Transformer architecture."
    mock_page2 = MagicMock()
    mock_page2.get_text.return_value = "Results: SOTA BLEU score on WMT 2014 translation."

    mock_doc.__len__.return_value = 2
    mock_doc.__getitem__.side_effect = [mock_page1, mock_page2]
    mock_fitz_open.return_value = mock_doc

    pages = processor.extract_text_from_pdf("dummy.pdf")

    assert len(pages) == 2
    assert pages[0]["page_number"] == 1
    assert "Transformer" in pages[0]["text"]
    assert pages[1]["page_number"] == 2
    assert "BLEU" in pages[1]["text"]


@patch("fitz.open")
def test_extract_text_empty_pdf_raises_error(mock_fitz_open, processor):
    mock_doc = MagicMock()
    mock_page = MagicMock()
    mock_page.get_text.return_value = "   "  # whitespace only
    mock_doc.__len__.return_value = 1
    mock_doc.__getitem__.return_value = mock_page
    mock_fitz_open.return_value = mock_doc

    with pytest.raises(ValueError, match="No extractable text"):
        processor.extract_text_from_pdf("empty.pdf")


# ── Synthetic Academic Paper Test Fixtures ───────────────────────────────────

@pytest.fixture
def synthetic_academic_paper_pages():
    """Multi-page synthetic academic paper with headings, equations, tables, and paragraphs."""
    page1 = {
        "page_number": 1,
        "char_count": 850,
        "text": (
            "Deep Residual Learning for Vision and Language\n"
            "Author One, Author Two - AI Research Lab\n\n"
            "Abstract\n"
            "We present a unified deep residual architecture for multi-task vision and language processing. "
            "Our formulation models cross-modal interactions through hierarchical attention maps.\n\n"
            "1. Introduction\n"
            "Neural representations have transformed natural language and computer vision processing over the past decade. "
            "Early approaches relied on separate unimodal representations that were concatenated prior to classification.\n\n"
            "In this work, we propose a joint attention framework that operates over tokenized text and visual patches. "
            "We demonstrate substantial improvements across diverse downstream evaluation benchmarks."
        ),
    }

    page2 = {
        "page_number": 2,
        "char_count": 920,
        "text": (
            "2. Related Work\n"
            "Previous work on multimodal fusion explored early fusion and late fusion paradigms. "
            "Recent architectures adopt transformer blocks with self-attention across multimodal streams.\n\n"
            "3. Architecture\n"
            "The model consists of a visual patch encoder, a textual tokenizer, and stacked cross-attention blocks.\n\n"
            "3.1 Patch Projection\n"
            "Visual inputs are projected into d-dimensional embedding vectors using 2D convolutions.\n\n"
            "3.2 Multi-Head Cross-Attention\n"
            "Attention functions map a query and key-value pairs to an output representation:\n"
            "\\begin{equation}\n"
            "\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V\n"
            "\\end{equation}\n\n"
            "The multi-head mechanism extends single attention via linear projections:\n"
            "$$\n"
            "\\text{MultiHead}(Q, K, V) = \\text{Concat}(\\text{head}_1, \\dots, \\text{head}_h)W^O\n"
            "$$\n"
            "where each head computes scaled dot-product attention independently."
        ),
    }

    page3 = {
        "page_number": 3,
        "char_count": 890,
        "text": (
            "4. Experiments\n"
            "We evaluate our proposed architecture across standard multimodal benchmarks.\n\n"
            "| Model | VQA Acc (%) | Image Captioning CIDEr | FLOPs (1e18) |\n"
            "|---|---|---|---|\n"
            "| Baseline ResNet | 68.2 | 102.5 | 4.2 |\n"
            "| Transformer-Base | 72.8 | 114.1 | 5.8 |\n"
            "| Ours (Proposed) | 76.4 | 123.9 | 6.1 |\n\n"
            "5. Conclusion\n"
            "We introduced a unified cross-modal transformer that achieves state-of-the-art results.\n\n"
            "References\n"
            "[1] Vaswani et al., Attention Is All You Need, NeurIPS 2017.\n"
            "[2] He et al., Deep Residual Learning for Image Recognition, CVPR 2016."
        ),
    }
    return [page1, page2, page3]


@pytest.fixture
def synthetic_long_section_pages():
    """Synthetic section exceeding chunk_size with multiple distinct paragraph boundaries."""
    paragraph1 = "First introductory paragraph outlining the historical background and motivation of transformers. " * 3
    paragraph2 = "Second paragraph describing the limitations of recurrent neural networks in parallelization. " * 3
    paragraph3 = "Third paragraph detailing how self-attention resolves sequential bottlenecks across long contexts. " * 3
    paragraph4 = "Fourth paragraph summarizing the structural contributions and experimental validation roadmap. " * 3

    return [
        {
            "page_number": 1,
            "char_count": len(paragraph1) + len(paragraph2) + len(paragraph3) + len(paragraph4) + 6,
            "text": f"1. Introduction\n\n{paragraph1}\n\n{paragraph2}\n\n{paragraph3}\n\n{paragraph4}",
        }
    ]


@pytest.fixture
def synthetic_atomic_blocks_pages():
    """Synthetic page containing large equation blocks and markdown tables."""
    return [
        {
            "page_number": 1,
            "char_count": 800,
            "text": (
                "3. Mathematical Foundations\n\n"
                "The loss function optimizes cross-entropy over masked token positions:\n"
                "\\begin{equation}\n"
                "\\mathcal{L}_{MLM}(\\theta) = -\\sum_{i \\in M} \\log P(x_i \\mid x_{\\setminus M}; \\theta)\n"
                "\\end{equation}\n\n"
                "Table 1: Hyperparameter configuration across model variants.\n\n"
                "| Parameter | Small | Base | Large |\n"
                "|---|---|---|---|\n"
                "| Layers | 6 | 12 | 24 |\n"
                "| Hidden Size | 512 | 768 | 1024 |\n"
                "| Attention Heads | 8 | 12 | 16 |\n"
            ),
        }
    ]


# ── Comprehensive Structure-Aware Unit Tests ─────────────────────────────────

def test_academic_headings_parsed_correctly(processor, synthetic_academic_paper_pages):
    """Verify that academic sections across all levels are recognized and hierarchically parsed."""
    chunks = processor.create_chunks(synthetic_academic_paper_pages, "doc_1", "paper.pdf")
    sections = {c.metadata.get("section") for c in chunks}

    assert "Abstract" in sections
    assert "1. Introduction" in sections
    assert "3. Architecture > 3.2 Multi-Head Cross-Attention" in sections
    assert "4. Experiments" in sections
    assert "References" in sections


def test_hierarchical_breadcrumb_injection(processor, synthetic_academic_paper_pages):
    """Verify that every chunk has breadcrumb headers in both text and metadata."""
    chunks = processor.create_chunks(synthetic_academic_paper_pages, "doc_1", "paper.pdf")

    for chunk in chunks:
        assert chunk.page_content.startswith("[Document: paper.pdf | Section:")
        assert "section" in chunk.metadata
        assert "breadcrumb" in chunk.metadata
        assert chunk.metadata["breadcrumb"] == f"[Document: paper.pdf | Section: {chunk.metadata['section']}]"

    # Verify subsection specific breadcrumb formatting
    sub_chunks = [c for c in chunks if "3.2 Multi-Head Cross-Attention" in c.metadata.get("section", "")]
    assert len(sub_chunks) > 0
    assert sub_chunks[0].metadata["section"] == "3. Architecture > 3.2 Multi-Head Cross-Attention"
    assert sub_chunks[0].page_content.startswith(
        "[Document: paper.pdf | Section: 3. Architecture > 3.2 Multi-Head Cross-Attention]\n\n"
    )


def test_atomic_equation_blocks_preserved(processor, synthetic_atomic_blocks_pages):
    """Verify that LaTeX equation blocks remain atomic and are not sliced across chunks."""
    chunks = processor.create_chunks(synthetic_atomic_blocks_pages, "doc_1", "paper.pdf")

    eq_chunks = [c for c in chunks if "\\mathcal{L}_{MLM}" in c.page_content]
    assert len(eq_chunks) == 1
    assert "\\begin{equation}" in eq_chunks[0].page_content
    assert "\\end{equation}" in eq_chunks[0].page_content
    assert eq_chunks[0].page_content.count("\\begin{equation}") == eq_chunks[0].page_content.count("\\end{equation}")


def test_atomic_table_blocks_preserved(processor, synthetic_atomic_blocks_pages):
    """Verify that markdown tables remain atomic and are not split across chunks."""
    chunks = processor.create_chunks(synthetic_atomic_blocks_pages, "doc_1", "paper.pdf")

    table_chunks = [c for c in chunks if "| Hidden Size |" in c.page_content]
    assert len(table_chunks) == 1
    assert "| Parameter | Small | Base | Large |" in table_chunks[0].page_content
    assert "|---|---|---|---|" in table_chunks[0].page_content
    assert "| Attention Heads | 8 | 12 | 16 |" in table_chunks[0].page_content


def test_sub_split_on_paragraph_boundaries(processor, synthetic_long_section_pages):
    """Verify that sections exceeding chunk_size are split cleanly at paragraph boundaries."""
    chunks = processor.create_chunks(synthetic_long_section_pages, "doc_1", "long.pdf")

    assert len(chunks) > 1
    for c in chunks:
        assert c.metadata["section"] == "1. Introduction"
        assert c.page_content.startswith("[Document: long.pdf | Section: 1. Introduction]\n\n")

    assert "First introductory paragraph" in chunks[0].page_content


def test_section_hierarchy_reset_across_sibling_sections(processor, synthetic_academic_paper_pages):
    """Verify that encountering a new top-level section clears subsection hierarchy from the stack."""
    chunks = processor.create_chunks(synthetic_academic_paper_pages, "doc_1", "paper.pdf")

    exp_chunks = [c for c in chunks if "4. Experiments" in c.metadata.get("section", "")]
    assert len(exp_chunks) > 0
    for c in exp_chunks:
        assert "3. Architecture" not in c.metadata["section"]
        assert c.metadata["section"] == "4. Experiments"


def test_multi_page_section_hierarchy_continuity(processor):
    """Verify that section hierarchy propagates across page boundaries when text continues."""
    pages = [
        {
            "page_number": 1,
            "text": "1. Introduction\nInitial introduction text spanning the bottom of the page.",
            "char_count": 75,
        },
        {
            "page_number": 2,
            "text": "Continuation of introduction text at the top of the next page before any new heading.",
            "char_count": 86,
        },
    ]

    chunks = processor.create_chunks(pages, "doc_cont", "continuation.pdf")
    assert len(chunks) >= 2
    # Both page 1 and page 2 chunks should belong to 1. Introduction
    p2_chunks = [c for c in chunks if c.metadata["page_number"] == 2]
    assert len(p2_chunks) > 0
    assert p2_chunks[0].metadata["section"] == "1. Introduction"
    assert p2_chunks[0].page_content.startswith("[Document: continuation.pdf | Section: 1. Introduction]\n\n")


def test_traceable_decorator_configuration(processor):
    """Verify that DocumentProcessor methods are configured with the required @traceable metadata."""
    targets = [
        ("pdf_text_extraction", processor.extract_text_from_pdf),
        ("structure_aware_chunking", processor.create_chunks),
        ("section_parsing", processor.parse_sections),
    ]

    for expected_name, method in targets:
        config = getattr(method, "__traceable_config__", None)
        if config is not None:
            metadata = config.get("metadata", {})
            assert metadata.get("agent_id") == "researchpilot"
            assert metadata.get("agent_environment") == "local"


def test_parse_sections_isolated(processor):
    """Test parse_sections directly on inline abstract and numbered sections."""
    sample_text = (
        "Abstract: A novel deep residual framework for language models.\n\n"
        "1. Introduction\n"
        "Background details on sequence transduction.\n\n"
        "2. Methods\n"
        "Algorithmic formulation."
    )
    sections = processor.parse_sections(sample_text)
    assert len(sections) == 3
    assert sections[0]["heading"] == "Abstract"
    assert sections[0]["hierarchy"] == "Abstract"
    assert "novel deep residual framework" in sections[0]["content"]

    assert sections[1]["heading"] == "1. Introduction"
    assert sections[1]["hierarchy"] == "1. Introduction"

    assert sections[2]["heading"] == "2. Methods"
    assert sections[2]["hierarchy"] == "2. Methods"
