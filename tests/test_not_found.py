"""
Unit tests verifying strict 'Not found' behavior for out-of-domain / unsupported questions.
"""

import pytest
from rag_pipeline import query_rag
from config import NOT_FOUND_RESPONSE


def test_q3_purchase_price_not_found():
    """Q3: What is the purchase price of this gas turbine? Must return exactly 'Not found'."""
    res = query_rag("What is the purchase price of this gas turbine?")
    assert res["found"] is False
    assert res["answer"] == NOT_FOUND_RESPONSE
    # Strict formatting: exactly "Not found", no trailing punctuation or conversational filler
    assert res["answer"].strip() == "Not found"


def test_q4_melting_point_titanium_not_found():
    """Q4: What is the melting point of titanium? Must return exactly 'Not found'."""
    res = query_rag("What is the melting point of titanium?")
    assert res["found"] is False
    assert res["answer"] == NOT_FOUND_RESPONSE
    assert res["answer"].strip() == "Not found"


def test_completely_irrelevant_query_not_found():
    """Verify completely out-of-domain questions return 'Not found'."""
    queries = [
        "What is the current GST rate in India?",
        "What is the capital of Australia?",
        "How many days of maternity leave are given?",
        "Who is the current Controller General of Patents?",
    ]
    for q in queries:
        res = query_rag(q)
        assert res["found"] is False, f"Query '{q}' should not be found in GER-3620"
        assert res["answer"] == NOT_FOUND_RESPONSE
