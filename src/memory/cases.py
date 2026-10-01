from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Iterable


@dataclass(frozen=True)
class Evidence:
    evidence_id: str
    source_type: str
    source_name: str
    claim: str
    supports: bool = True


@dataclass(frozen=True)
class HistoricalCase:
    case_id: str
    exception_type: str
    title: str
    symptoms: tuple[str, ...]
    root_cause: str
    resolution: str
    evidence: tuple[Evidence, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        return asdict(self)


def seed_historical_cases() -> list[HistoricalCase]:
    return [
        HistoricalCase(
            case_id="CASE_001",
            exception_type="NAV_DISCREPANCY",
            title="Large NAV variance caused by stale security price",
            symptoms=("nav variance", "price mismatch", "vendor discrepancy"),
            root_cause="STALE_PRICE",
            resolution="Validate primary and secondary price sources; refresh the stale price after operations approval.",
            evidence=(
                Evidence("E001", "price_feed", "Primary vendor", "Primary and secondary prices differ materially."),
                Evidence("E002", "historical_case", "CASE_001", "Similar NAV variance was caused by a stale price."),
            ),
        ),
        HistoricalCase(
            case_id="CASE_002",
            exception_type="NAV_DISCREPANCY",
            title="NAV variance caused by missing transaction",
            symptoms=("nav variance", "position mismatch", "transaction"),
            root_cause="MISSING_TRANSACTION",
            resolution="Reconcile recent trades and verify settlement status before rerunning NAV.",
            evidence=(
                Evidence("E003", "transaction_system", "Trade blotter", "Expected transaction is absent from the position set."),
            ),
        ),
        HistoricalCase(
            case_id="CASE_003",
            exception_type="NAV_DISCREPANCY",
            title="NAV variance caused by corporate action",
            symptoms=("nav variance", "corporate action", "security event"),
            root_cause="CORPORATE_ACTION",
            resolution="Review effective corporate-action terms and verify position adjustment.",
            evidence=(
                Evidence("E004", "corporate_actions", "CA feed", "A security event occurred on the valuation date."),
            ),
        ),
        HistoricalCase(
            case_id="CASE_004",
            exception_type="TRANSACTION_MISMATCH",
            title="Trade mismatch caused by pending settlement",
            symptoms=("transaction mismatch", "pending", "settlement"),
            root_cause="PENDING_SETTLEMENT",
            resolution="Check settlement status and counterparty confirmation.",
            evidence=(
                Evidence("E005", "settlement_system", "Settlement queue", "Trade remains pending settlement."),
            ),
        ),
    ]


class CaseMemory:
    """Simple transparent retrieval baseline; vector retrieval can replace it later."""

    def __init__(self, cases: Iterable[HistoricalCase] = ()) -> None:
        self._cases = list(cases)

    def add(self, case: HistoricalCase) -> None:
        self._cases.append(case)

    def search(
        self,
        query: str,
        exception_type: str | None = None,
        top_k: int = 3,
    ) -> list[HistoricalCase]:
        tokens = {
            token.strip(".,:;!?").lower()
            for token in query.split()
            if len(token.strip(".,:;!?")) >= 3
        }

        candidates = [
            case for case in self._cases
            if exception_type is None or case.exception_type == exception_type
        ]

        scored = []
        for case in candidates:
            text = " ".join(
                (case.title, *case.symptoms, case.root_cause, case.resolution)
            ).lower()
            score = sum(1 for token in tokens if token in text)
            scored.append((score, case))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [case for score, case in scored[:top_k] if score > 0]
