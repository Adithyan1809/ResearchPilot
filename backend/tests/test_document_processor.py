"""Unit tests for DocumentProcessor."""

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
