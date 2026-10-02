from src.memory.cases import CaseMemory, HistoricalCase, SemanticMemory, seed_historical_cases
from src.tools.memory_tools import search_historical_cases_tool


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


def test_validated_search_filters_before_ranking_and_top_k():
    unvalidated = HistoricalCase(
        case_id="CASE_U", exception_type="NAV_DISCREPANCY",
        title="NAV price vendor break", symptoms=("price vendor",),
        root_cause="PRICE_EXCEPTION", resolution="Review prices.",
    )
    validated = HistoricalCase(
        case_id="CASE_V", exception_type="NAV_DISCREPANCY",
        title="NAV investigation", symptoms=("price",),
        root_cause="STALE_PRICE", resolution="Review source records.",
        human_validated=True,
    )
    memory = CaseMemory([unvalidated, validated])

    results = memory.search("NAV price vendor", "NAV_DISCREPANCY", top_k=1, validated_only=True)
    assert [case.case_id for case in results] == ["CASE_V"]
    tool_results = search_historical_cases_tool(memory, "NAV price vendor", "NAV_DISCREPANCY", 1)
    assert [case["case_id"] for case in tool_results] == ["CASE_V"]
    assert tool_results[0]["root_cause"] == "STALE_PRICE"  # Retained for audit after retrieval.


def test_keyword_relevance_ignores_root_cause_and_resolution():
    answer_only = HistoricalCase(
        case_id="CASE_ANSWER", exception_type="NAV_DISCREPANCY",
        title="NAV variance caused by vendor pricing", symptoms=("cash",),
        root_cause="VENDOR_EXCEPTION", resolution="Review vendor pricing.",
        human_validated=True,
    )
    observable = HistoricalCase(
        case_id="CASE_OBSERVABLE", exception_type="NAV_DISCREPANCY",
        title="NAV record", symptoms=("vendor disagreement",),
        root_cause="UNKNOWN", resolution="Obtain more records.",
        human_validated=True,
    )
    memory = CaseMemory([answer_only, observable])

    results = memory.search("vendor", validated_only=True)
    assert [case.case_id for case in results] == ["CASE_OBSERVABLE"]


def test_semantic_guidance_excludes_unvalidated_and_answer_fields():
    def embedding(text: str):
        return [float("vendor" in text.lower())]

    answer_only = HistoricalCase(
        case_id="CASE_ANSWER", exception_type="NAV_DISCREPANCY",
        title="NAV variance caused by vendor pricing", symptoms=("cash",),
        root_cause="VENDOR_EXCEPTION", resolution="Review vendor pricing.",
        human_validated=True,
    )
    unvalidated = HistoricalCase(
        case_id="CASE_U", exception_type="NAV_DISCREPANCY",
        title="Vendor disagreement", symptoms=("vendor",),
        root_cause="UNKNOWN", resolution="Review records.",
    )
    observable = HistoricalCase(
        case_id="CASE_OBSERVABLE", exception_type="NAV_DISCREPANCY",
        title="NAV record", symptoms=("vendor disagreement",),
        root_cause="UNKNOWN", resolution="Review records.",
        human_validated=True,
    )
    memory = SemanticMemory([answer_only, unvalidated, observable], embedding_fn=embedding)

    results = memory.search("vendor", validated_only=True, top_k=1)
    assert [case.case_id for case in results] == ["CASE_OBSERVABLE"]
