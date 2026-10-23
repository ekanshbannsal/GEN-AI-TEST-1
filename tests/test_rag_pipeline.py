"""
Unit tests for RAG Pipeline execution and grounded technical answers.
"""

import pytest
from rag_pipeline import (
    get_embedding_model,
    retrieve_relevant_chunks,
    query_rag,
)
from config import NOT_FOUND_RESPONSE


def test_embedding_model():
    """Verify embedding model generates vectors of expected dimension."""
    emb = get_embedding_model()
    vec = emb.embed_query("Gas turbine combustion inspection")
    assert len(vec) == 384, f"Expected 384 dims, got {len(vec)}"


def test_retrieval_returns_chunks():
    """Verify retrieval returns non-empty list of documents with metadata."""
    chunks = retrieve_relevant_chunks("What are the three main types of inspection?", top_k=4)
    assert len(chunks) > 0, "No chunks retrieved"
    first_doc, dist = chunks[0]
    assert hasattr(first_doc, "page_content")
    assert "page" in first_doc.metadata


def test_answer_q1_three_inspection_types():
    """Q1: What are the three main types of inspection?"""
    res = query_rag("What are the three main types of inspection?")
    assert res["found"] is True
    answer = res["answer"].lower()
    assert "standby" in answer
    assert "running" in answer
    assert "disassembly" in answer
    assert len(res["sources"]) > 0


def test_answer_q2_maintenance_factor_of_two():
    """Q2: What does a maintenance factor of two mean?"""
    res = query_rag("What does a maintenance factor of two mean?")
    assert res["found"] is True
    answer = res["answer"].lower()
    assert "half" in answer
    assert "baseline" in answer
    assert len(res["sources"]) > 0


def test_answer_q5_combustion_inspection_interval():
    """Q5: What is the recommended combustion-inspection interval in hours?"""
    res = query_rag("What is the recommended combustion-inspection interval in hours?")
    assert res["found"] is True
    answer = res["answer"].lower()
    # Check that key intervals are mentioned (e.g. 8,000, 12,000, 24,000, or 32,000)
    assert any(h in answer for h in ["8,000", "12,000", "24,000", "32,000", "hours"])
    assert len(res["sources"]) > 0
