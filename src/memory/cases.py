from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Protocol, Sequence, runtime_checkable

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
    human_validated: bool = False
    review_metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "exception_type": self.exception_type,
            "title": self.title,
            "symptoms": list(self.symptoms),
            "root_cause": self.root_cause,
            "resolution": self.resolution,
            "evidence": [item.model_dump(mode="json") for item in self.evidence],
            "human_validated": self.human_validated,
            "review_metadata": dict(self.review_metadata),
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


@runtime_checkable
class CaseMemoryBackend(Protocol):
    """Replaceable memory boundary; current implementation is keyword-only."""

    def add(self, case: HistoricalCase) -> None: ...

    def next_case_id(self) -> str: ...

    def search(
        self, query: str, exception_type: str | None = None, top_k: int = 3
    ) -> list[HistoricalCase]: ...


class CaseMemory:
    """Transparent keyword retrieval baseline behind ``CaseMemoryBackend``."""

    def __init__(self, cases: Iterable[HistoricalCase] = ()) -> None:
        self._cases = list(cases)

    def add(self, case: HistoricalCase) -> None:
        if any(existing.case_id == case.case_id for existing in self._cases):
            raise ValueError(f"Case already exists: {case.case_id}")
        self._cases.append(case)

    def next_case_id(self) -> str:
        numbers = [
            int(case.case_id.removeprefix("CASE_"))
            for case in self._cases
            if case.case_id.startswith("CASE_")
            and case.case_id.removeprefix("CASE_").isdigit()
        ]
        return f"CASE_{max(numbers, default=0) + 1:03d}"

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


class KeywordMemory(CaseMemory):
    """Named baseline implementation retained behind the memory protocol."""


class SemanticMemory(CaseMemory):
    """Embedding-backed retrieval boundary with injectable embedding function.

    No external embedding model is bundled or called by default. This lets a
    future governed embedding service be introduced without changing review
    or workflow callers. Historical cases remain retrieval analogies only.
    """

    def __init__(
        self,
        cases: Iterable[HistoricalCase] = (),
        *,
        embedding_fn: Callable[[str], Sequence[float]],
    ) -> None:
        super().__init__(cases)
        self._embedding_fn = embedding_fn

    def search(
        self, query: str, exception_type: str | None = None, top_k: int = 3
    ) -> list[HistoricalCase]:
        query_vector = self._embedding_fn(query)
        if not query_vector:
            return []
        candidates = [
            case for case in self._cases
            if exception_type is None or case.exception_type == exception_type
        ]
        ranked: list[tuple[float, HistoricalCase]] = []
        for case in candidates:
            # Exclude root_cause: retrieval is based on symptoms/context, not
            # an answer-key match. Retrieved cases are still explicitly analogies.
            text = " ".join((case.title, *case.symptoms, case.resolution))
            score = _cosine_similarity(query_vector, self._embedding_fn(text))
            ranked.append((score, case))
        ranked.sort(key=lambda item: item[0], reverse=True)
        return [case for score, case in ranked[:top_k] if score > 0]


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    left_norm = sum(float(item) ** 2 for item in left) ** 0.5
    right_norm = sum(float(item) ** 2 for item in right) ** 0.5
    if not left_norm or not right_norm:
        return 0.0
    return sum(float(a) * float(b) for a, b in zip(left, right)) / (left_norm * right_norm)
