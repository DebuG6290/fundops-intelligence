from src.memory.cases import CaseMemory, SemanticMemory, seed_historical_cases


def test_seed_cases_exist():
    memory = CaseMemory(seed_historical_cases())
    assert len(memory.search("NAV price mismatch", "NAV_DISCREPANCY")) >= 1


def test_price_query_retrieves_price_case():
    memory = CaseMemory(seed_historical_cases())
    results = memory.search(
        "NAV variance price vendor discrepancy",
        exception_type="NAV_DISCREPANCY",
    )

    assert results
    assert results[0].root_cause == "STALE_PRICE"


def test_exception_type_filter():
    memory = CaseMemory(seed_historical_cases())
    results = memory.search(
        "pending settlement",
        exception_type="TRANSACTION_MISMATCH",
    )

    assert results
    assert all(case.exception_type == "TRANSACTION_MISMATCH" for case in results)


def test_semantic_memory_uses_injected_embeddings_and_returns_analogies():
    # Small deterministic test encoder; no external model/network is used.
    def embedding(text: str):
        lowered = text.lower()
        return [float("price" in lowered), float("settlement" in lowered), float("nav" in lowered)]

    memory = SemanticMemory(seed_historical_cases(), embedding_fn=embedding)
    results = memory.search("price nav variance", "NAV_DISCREPANCY")
    assert results
    assert results[0].case_id == "CASE_001"
