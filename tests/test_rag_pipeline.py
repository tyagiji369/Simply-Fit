import pytest

from src.rag_pipeline import rag_engine


def test_corpus_loaded():
    assert rag_engine.guidelines, "guideline corpus should be indexed"
    assert rag_engine.tfidf_matrix is not None


def test_hypertension_query_retrieves_hypertension():
    matches = rag_engine.search_guidelines("sodium and blood pressure", condition="Hypertension", top_k=2)
    assert matches
    assert matches[0]["condition"] == "Hypertension"
    assert matches[0]["relevance_score"] > 0


def test_top_k_respected():
    assert len(rag_engine.search_guidelines("diet", top_k=2)) == 2
    assert len(rag_engine.search_guidelines("diet", top_k=3)) == 3


def test_format_context_includes_source():
    ctx = rag_engine.format_retrieved_context("protein intake", condition="CKD", top_k=1)
    assert "Clinical Guideline 1" in ctx
    assert "Recommendation:" in ctx
