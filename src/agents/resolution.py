from __future__ import annotations

from dataclasses import dataclass

from src.agents.state import InvestigationState


@dataclass(frozen=True)
class ResolutionRecommendation:
    decision: str
    rationale: str
    confidence: float
    requires_human_approval: bool


class ResolutionAgent:
    """Convert an investigated state into a human-review recommendation."""

    def resolve(self, state: InvestigationState) -> ResolutionRecommendation:
        if state.status == "ESCALATE" or state.confidence < 0.70:
            return ResolutionRecommendation(
                decision="INVESTIGATE_FURTHER",
                rationale=state.recommended_action
                or "Evidence is insufficient for a reliable recommendation.",
                confidence=state.confidence,
                requires_human_approval=True,
            )

        return ResolutionRecommendation(
            decision="REVIEW_RECOMMENDATION",
            rationale=state.recommended_action
            or "Review the leading hypothesis and supporting evidence.",
            confidence=state.confidence,
            requires_human_approval=True,
        )
