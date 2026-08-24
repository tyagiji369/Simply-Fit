from src.rag_pipeline import ClinicalRAGPipeline


def test_condition_matched_retrieval():
    rag = ClinicalRAGPipeline()
    results = rag.search_guidelines("protein and diet", condition="Chronic Kidney Disease (CKD)", top_k=2)
    assert results, "should retrieve something"
    assert any("CKD" in r["condition"] or "Kidney" in r["condition"] for r in results), (
        "condition-matched guidelines must be preferred"
    )


def test_diabetes_matching():
    rag = ClinicalRAGPipeline()
    results = rag.search_guidelines("what should I eat for my blood sugar",
                                    condition="Type 2 Diabetes", top_k=1)
    assert results
    assert "Diabetes" in results[0]["condition"]


def test_no_condition_falls_back_to_corpus():
    rag = ClinicalRAGPipeline()
    results = rag.search_guidelines("weight plateau diet", condition=None, top_k=2)
    assert len(results) > 0
