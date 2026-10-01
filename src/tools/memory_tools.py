from __future__ import annotations

from typing import Any

from src.memory.cases import CaseMemory, HistoricalCase


def search_historical_cases_tool(
    memory: CaseMemory,
    query: str,
    exception_type: str | None = None,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    return [
        case.to_dict()
        for case in memory.search(
            query=query,
            exception_type=exception_type,
            top_k=top_k,
        )
    ]


def build_default_memory() -> CaseMemory:
    from src.memory.cases import seed_historical_cases

    return CaseMemory(seed_historical_cases())
