from __future__ import annotations

from dataclasses import dataclass

from src.agents.state import InvestigationState
from src.models.evidence import EvidenceItem, EvidenceSourceType
from src.models.root_cause import evidence_targets_hypothesis, root_cause_code


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
    supporting_evidence_ids: tuple[str, ...] = ()
    contradictory_evidence_ids: tuple[str, ...] = ()
    missing_evidence: tuple[str, ...] = ()


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
                missing_evidence=("A testable current-case hypothesis",),
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
                missing_evidence=("Primary current-case evidence",),
            )

        price_checks = [item["value"] for item in state.observations
                        if item["name"] == "price_source_check" and isinstance(item.get("value"), dict)]

        contradiction = False
        ambiguity = False
        leading_id = leading.get("hypothesis_id")
        supporting_ids = tuple(item.evidence_id for item in primary_evidence if (
            item.supports_hypothesis == leading_id
            or (not item.supports_hypothesis and evidence_targets_hypothesis(
                item.supports, leading["root_cause"], item.metadata))
        ))
        contradictory_ids = tuple(item.evidence_id for item in primary_evidence if (
            item.contradicts_hypothesis == leading_id
            or (not item.contradicts_hypothesis and evidence_targets_hypothesis(
                item.contradicts, leading["root_cause"], item.metadata))
        ))
        missing = list(dict.fromkeys(
            requirement for requirement in leading.get("required_evidence", [])
            if not any(requirement.lower() in f"{item.source_type.value} {item.source_name} {item.claim}".lower()
                       for item in primary_evidence)
        ))
        insufficient = not supporting_ids or bool(missing)
        if not supporting_ids:
            missing.append("Primary evidence supporting the leading hypothesis")

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
            or (not item.contradicts_hypothesis and evidence_targets_hypothesis(
                item.contradicts, leading["root_cause"], item.metadata))
            for item in state.evidence
        ):
            contradiction = True

        try:
            pricing_hypothesis = root_cause_code(leading["root_cause"]) == "PRICE_EXCEPTION"
        except ValueError:
            pricing_hypothesis = False  # legacy specialist labels retain their existing challenge path
        if pricing_hypothesis:
            security_id = leading["root_cause"].partition(":")[2]
            price_records = [item for item in primary_evidence
                             if item.source_type == EvidenceSourceType.PRICE_SOURCE
                             and (not security_id or item.metadata.get("security_id") == security_id)]
            if price_records:
                # Structured current-case records survive the specialist-to-
                # workflow handoff; a transient tool observation need not.
                if not any(item.evidence_id in supporting_ids for item in price_records):
                    contradiction = True
            else:
                price_check = next((item for item in reversed(price_checks)
                                    if not security_id or item.get("security_id") == security_id), None)
                if not price_check or not price_check.get("found") or abs(price_check.get("difference_pct", 0)) < 10:
                    contradiction = True

        if contradiction or ambiguity or insufficient:
            reasons = []
            if contradiction:
                reasons.append("conflicting evidence")
            if ambiguity:
                reasons.append("multiple plausible hypotheses")
            if insufficient:
                reasons.append("required current-case evidence is missing" if supporting_ids
                               else "no primary evidence supports the leading hypothesis")
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
                supporting_evidence_ids=supporting_ids,
                contradictory_evidence_ids=contradictory_ids,
                missing_evidence=tuple(missing),
            )

        return ChallengeResult(
            challenged=True,
            contradiction_found=False,
            ambiguity_found=False,
            final_confidence=leading["confidence"],
            recommendation=state.recommended_action or "Proceed to human review.",
            supporting_evidence_ids=supporting_ids,
            contradictory_evidence_ids=contradictory_ids,
            missing_evidence=tuple(missing),
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

