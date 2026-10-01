from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.models.evidence import EvidenceItem


@dataclass
class InvestigationState:
    exception: dict[str, Any]
    observations: list[dict[str, Any]] = field(default_factory=list)
    hypotheses: list[dict[str, Any]] = field(default_factory=list)
    evidence: list[EvidenceItem] = field(default_factory=list)
    recommended_action: str | None = None
    confidence: float = 0.0
    status: str = "OPEN"

    def add_evidence(self, item: EvidenceItem) -> None:
        self.evidence.append(item)

    def add_observation(self, name: str, value: Any, source: str) -> None:
        self.observations.append({
            "name": name,
            "value": value,
            "source": source,
        })

    def add_hypothesis(
        self,
        root_cause: str,
        rationale: str,
        confidence: float,
    ) -> None:
        self.hypotheses.append({
            "root_cause": root_cause,
            "rationale": rationale,
            "confidence": confidence,
        })
