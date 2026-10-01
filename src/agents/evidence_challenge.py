from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.agents.state import InvestigationState


@dataclass(frozen=True)
class ChallengeResult:
    challenged: bool
    contradiction_found: bool
    ambiguity_found: bool
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
                ambiguity_found=False,
                final_confidence=0.0,
                recommendation="Escalate: no hypothesis available.",
            )

        leading = state.hypotheses[0]
        specialist_report_without_evidence = (
            state.exception.get("exception_type")
            in {"TRANSACTION_MISMATCH", "CORPORATE_ACTION"}
            and not state.evidence
        )
        if specialist_report_without_evidence:
            return ChallengeResult(
                challenged=True,
                contradiction_found=False,
                ambiguity_found=False,
                final_confidence=min(float(leading["confidence"]), 0.35),
                recommendation="Escalate: no supporting evidence available.",
            )

        price_check = next(
            (
                item["value"]
                for item in state.observations
                if item["name"] == "price_source_check"
            ),
            None,
        )

        contradiction = False
        ambiguity = False

        # Escalate when multiple plausible hypotheses are too close to call.
        if len(state.hypotheses) >= 2:
            second = state.hypotheses[1]
            if abs(float(leading["confidence"]) - float(second["confidence"])) < 0.15:
                ambiguity = True

        # Explicit counter-evidence from an investigator or specialist agent
        # overrides an otherwise confident recommendation.
        for item in state.observations:
            if item["name"] == "counter_evidence":
                value = item.get("value", {})
                if isinstance(value, dict) and value.get("contradicts"):
                    contradiction = True

        if leading["root_cause"] == "PRICE_EXCEPTION":
            if not price_check or not price_check.get("found"):
                contradiction = True
            elif abs(price_check.get("difference_pct", 0)) < 10:
                contradiction = True

        if contradiction or ambiguity:
            reasons = []
            if contradiction:
                reasons.append("conflicting evidence")
            if ambiguity:
                reasons.append("multiple plausible hypotheses")
            reason_text = " and ".join(reasons)
            return ChallengeResult(
                challenged=True,
                contradiction_found=contradiction,
                ambiguity_found=ambiguity,
                final_confidence=min(leading["confidence"], 0.35),
                recommendation=(
                    f"Escalate: {reason_text} prevents a reliable conclusion."
                ),
            )

        return ChallengeResult(
            challenged=True,
            contradiction_found=False,
            ambiguity_found=False,
            final_confidence=leading["confidence"],
            recommendation=state.recommended_action or "Proceed to human review.",
        )


def apply_challenge(
    state: InvestigationState,
    challenge: ChallengeResult,
) -> InvestigationState:
    state.confidence = challenge.final_confidence

    if (
        not state.hypotheses
        or (
            state.exception.get("exception_type")
            in {"TRANSACTION_MISMATCH", "CORPORATE_ACTION"}
            and not state.evidence
        )
        or challenge.contradiction_found
        or challenge.ambiguity_found
    ):
        state.status = "ESCALATE"
        state.recommended_action = challenge.recommendation
    else:
        state.status = "READY_FOR_HUMAN_REVIEW"

    return state
