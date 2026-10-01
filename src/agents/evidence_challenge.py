from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.agents.state import InvestigationState


@dataclass(frozen=True)
class ChallengeResult:
    challenged: bool
    contradiction_found: bool
    final_confidence: float
    recommendation: str


class EvidenceChallengeAgent:
    """
    Deterministic challenge layer used as a safety/evaluation baseline.

    It checks whether the leading hypothesis is supported by the observed
    price-source evidence and whether there is an obvious contradiction.
    """

    def review(self, state: InvestigationState) -> ChallengeResult:
        if not state.hypotheses:
            return ChallengeResult(
                challenged=True,
                contradiction_found=False,
                final_confidence=0.0,
                recommendation="Escalate: no hypothesis available.",
            )

        leading = state.hypotheses[0]
        price_check = next(
            (
                item["value"]
                for item in state.observations
                if item["name"] == "price_source_check"
            ),
            None,
        )

        contradiction = False

        if leading["root_cause"] == "PRICE_EXCEPTION":
            if not price_check or not price_check.get("found"):
                contradiction = True
            elif abs(price_check.get("difference_pct", 0)) < 10:
                contradiction = True

        if contradiction:
            return ChallengeResult(
                challenged=True,
                contradiction_found=True,
                final_confidence=min(leading["confidence"], 0.35),
                recommendation="Escalate: leading hypothesis lacks sufficient supporting evidence.",
            )

        return ChallengeResult(
            challenged=True,
            contradiction_found=False,
            final_confidence=leading["confidence"],
            recommendation=state.recommended_action or "Proceed to human review.",
        )


def apply_challenge(
    state: InvestigationState,
    challenge: ChallengeResult,
) -> InvestigationState:
    state.confidence = challenge.final_confidence

    if challenge.contradiction_found:
        state.status = "ESCALATE"
        state.recommended_action = challenge.recommendation
    else:
        state.status = "READY_FOR_HUMAN_REVIEW"

    return state
