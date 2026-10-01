from src.memory.cases import CaseMemory, seed_historical_cases


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
