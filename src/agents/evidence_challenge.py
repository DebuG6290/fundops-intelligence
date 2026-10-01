from __future__ import annotations

from dataclasses import dataclass

from src.agents.state import InvestigationState
from src.models.evidence import EvidenceItem, EvidenceSourceType


def _is_primary_evidence(item: EvidenceItem) -> bool:
    return item.source_type not in {
        EvidenceSourceType.HISTORICAL_CASE,
        EvidenceSourceType.SPECIALIST_OBSERVATION,
    }


@dataclass(frozen=True)
class ChallengeResult:
    challenged: bool
    contradiction_found: bool
    ambiguity_found: bool
    final_confidence: float
    recommendation: str
    insufficient_evidence: bool = False


class EvidenceChallengeAgent:
    """
    Deterministic challenge layer used as a safety/evaluation baseline.

    It checks structured evidence relationships, report counter-claims, and
    the NAV-specific price-source threshold.
    """

    def review(self, state: InvestigationState) -> ChallengeResult:
        if not state.hypotheses:
            return ChallengeResult(
                challenged=True,
                contradiction_found=False,
                ambiguity_found=False,
                final_confidence=0.0,
                recommendation="Escalate: no hypothesis available.",
                insufficient_evidence=True,
            )

        leading = state.hypotheses[0]
        primary_evidence = [item for item in state.evidence if _is_primary_evidence(item)]
        if not primary_evidence:
            return ChallengeResult(
                challenged=True,
                contradiction_found=False,
                ambiguity_found=False,
                final_confidence=min(float(leading["confidence"]), 0.35),
                recommendation="Escalate: no primary evidence is available.",
                insufficient_evidence=True,
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
        leading_id = leading.get("hypothesis_id")
        insufficient = not any(
            item.supports_hypothesis == leading_id
            or (not item.supports_hypothesis and item.supports == leading["root_cause"])
            for item in primary_evidence
        )

        # Escalate when multiple plausible hypotheses are too close to call.
        if len(state.hypotheses) >= 2:
            second = state.hypotheses[1]
            if abs(float(leading["confidence"]) - float(second["confidence"])) < 0.15:
                ambiguity = True

        # Explicit counter-evidence from an investigator or specialist agent
        # overrides an otherwise confident recommendation.
        for item in state.observations:
            if item["name"] in {"counter_evidence", "reported_counter_evidence"}:
                value = item.get("value", {})
                if item["name"] == "reported_counter_evidence" or (
                    isinstance(value, dict) and value.get("contradicts")
                ):
                    contradiction = True
        if any(
            item.contradicts_hypothesis == leading_id
            or (not item.contradicts_hypothesis and item.contradicts == leading["root_cause"])
            for item in state.evidence
        ):
            contradiction = True

        if leading["root_cause"] == "PRICE_EXCEPTION":
            if not price_check or not price_check.get("found"):
                contradiction = True
            elif abs(price_check.get("difference_pct", 0)) < 10:
                contradiction = True

        if contradiction or ambiguity or insufficient:
            reasons = []
            if contradiction:
                reasons.append("conflicting evidence")
            if ambiguity:
                reasons.append("multiple plausible hypotheses")
            if insufficient:
                reasons.append("no primary evidence supports the leading hypothesis")
            reason_text = " and ".join(reasons)
            return ChallengeResult(
                challenged=True,
                contradiction_found=contradiction,
                ambiguity_found=ambiguity,
                final_confidence=min(leading["confidence"], 0.35),
                recommendation=(
                    f"Escalate: {reason_text} prevents a reliable conclusion."
                ),
                insufficient_evidence=insufficient,
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
        or not any(_is_primary_evidence(item) for item in state.evidence)
        or challenge.insufficient_evidence
        or challenge.contradiction_found
        or challenge.ambiguity_found
    ):
        state.status = "ESCALATE"
        state.recommended_action = challenge.recommendation
    else:
        state.status = "READY_FOR_HUMAN_REVIEW"

    return state
