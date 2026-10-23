"""
Unit tests for PDF Ingestion module.
"""

import pytest
from pathlib import Path
from config import PDF_PATH, CHROMA_DIR, CHROMA_COLLECTION_NAME
from ingest import clean_page_text, load_and_process_pdf
from rag_pipeline import get_vectorstore


def test_pdf_exists():
    """Verify that the primary PDF exists and is non-empty."""
    assert PDF_PATH.exists(), f"PDF does not exist at {PDF_PATH}"
    assert PDF_PATH.stat().st_size > 100_000, "PDF file is suspiciously small"


def test_clean_page_text():
    """Verify that headers, footers, and page numbers are properly handled."""
    sample_text = (
        "Doc Name : GER-3620     Rev P     Released Date : 2021/03/15 Page 25 of 60\n"
        "Heavy-Duty Gas Turbine Operating and Maintenance Considerations\n"
        "21\n"
        "Maintenance Inspections\n"
        "Maintenance inspection types may be broadly classified as standby, running, and disassembly inspections.\n"
        "Non-Public- Class II (Non Critical) - NLR\n"
    )
    cleaned, doc_page = clean_page_text(sample_text, 25)
    assert doc_page == 21, f"Expected doc_page 21, got {doc_page}"
    assert "Doc Name : GER-3620" not in cleaned
    assert "Non-Public" not in cleaned
    assert "Maintenance inspection types" in cleaned


def test_chunking_metadata():
    """Verify that chunked documents contain required metadata fields."""
    docs = load_and_process_pdf(PDF_PATH)
    assert len(docs) > 50, "Expected at least 50 chunks for 60-page PDF"

    first_chunk = docs[0]
    required_keys = {"source", "page", "doc_page", "section", "chunk_id", "chunk_index"}
    assert required_keys.issubset(first_chunk.metadata.keys())
    assert first_chunk.metadata["source"] == "GER-3620-P.pdf"
    assert first_chunk.metadata["page"] >= 1


def test_chroma_store_populated():
    """Verify that ChromaDB vector store is populated and accessible."""
    assert CHROMA_DIR.exists(), "ChromaDB directory does not exist"
    store = get_vectorstore()
    data = store.get()
    assert len(data["ids"]) > 0, "ChromaDB collection is empty"
