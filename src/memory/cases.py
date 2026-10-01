from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from src.models.evidence import EvidenceItem, EvidenceSourceType


# Transitional import alias for callers that used the old memory-local model.
Evidence = EvidenceItem


@dataclass(frozen=True)
class HistoricalCase:
    case_id: str
    exception_type: str
    title: str
    symptoms: tuple[str, ...]
    root_cause: str
    resolution: str
    evidence: tuple[EvidenceItem, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "exception_type": self.exception_type,
            "title": self.title,
            "symptoms": list(self.symptoms),
            "root_cause": self.root_cause,
            "resolution": self.resolution,
            "evidence": [item.model_dump(mode="json") for item in self.evidence],
        }


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
                EvidenceItem(
                    evidence_id="E001",
                    source_type=EvidenceSourceType.HISTORICAL_CASE,
                    source_name="CASE_001",
                    claim="Primary and secondary prices differ materially.",
                    metadata={"evidence_role": "analogy", "original_source": "Primary vendor"},
                ),
                EvidenceItem(
                    evidence_id="E002",
                    source_type=EvidenceSourceType.HISTORICAL_CASE,
                    source_name="CASE_001",
                    claim="Similar NAV variance was caused by a stale price.",
                    metadata={"evidence_role": "analogy"},
                ),
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
                EvidenceItem(
                    evidence_id="E003",
                    source_type=EvidenceSourceType.HISTORICAL_CASE,
                    source_name="CASE_002",
                    claim="Expected transaction is absent from the position set.",
                    metadata={"evidence_role": "analogy", "original_source": "Trade blotter"},
                ),
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
                EvidenceItem(
                    evidence_id="E004",
                    source_type=EvidenceSourceType.HISTORICAL_CASE,
                    source_name="CASE_003",
                    claim="A security event occurred on the valuation date.",
                    metadata={"evidence_role": "analogy", "original_source": "CA feed"},
                ),
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
                EvidenceItem(
                    evidence_id="E005",
                    source_type=EvidenceSourceType.HISTORICAL_CASE,
                    source_name="CASE_004",
                    claim="Trade remains pending settlement.",
                    metadata={"evidence_role": "analogy", "original_source": "Settlement queue"},
                ),
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
