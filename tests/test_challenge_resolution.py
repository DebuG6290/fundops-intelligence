from src.agents.evidence_challenge import EvidenceChallengeAgent, apply_challenge
from src.agents.resolution import ResolutionAgent
from src.agents.rule_based_investigator import RuleBasedInvestigator
from src.agents.state import InvestigationState
from src.data.scenarios import create_price_exception_scenario
from src.memory.cases import CaseMemory, seed_historical_cases
from src.models.evidence import EvidenceItem, EvidenceSourceType


def test_challenge_preserves_supported_price_hypothesis():
    scenario = create_price_exception_scenario()
    state = RuleBasedInvestigator(
        CaseMemory(seed_historical_cases())
    ).investigate(scenario)

    challenge = EvidenceChallengeAgent().review(state)
    state = apply_challenge(state, challenge)

    assert challenge.contradiction_found is False
    assert state.status == "READY_FOR_HUMAN_REVIEW"


def test_resolution_keeps_human_in_loop():
    scenario = create_price_exception_scenario()
    state = RuleBasedInvestigator(
        CaseMemory(seed_historical_cases())
    ).investigate(scenario)

    recommendation = ResolutionAgent().resolve(state)

    assert recommendation.requires_human_approval is True


def test_challenge_escalates_ambiguous_hypotheses():
    scenario = create_price_exception_scenario()
    state = RuleBasedInvestigator(
        CaseMemory(seed_historical_cases())
    ).investigate(scenario)
    state.add_hypothesis(
        root_cause="TRANSACTION_MISMATCH",
        rationale="A transaction issue could also explain the observed variance.",
        confidence=0.86,
    )

    challenge = EvidenceChallengeAgent().review(state)
    state = apply_challenge(state, challenge)

    assert challenge.ambiguity_found is True
    assert challenge.contradiction_found is False
    assert state.status == "ESCALATE"
    assert state.confidence <= 0.35
    assert "multiple plausible hypotheses" in state.recommended_action


def test_challenge_escalates_explicit_conflicting_evidence():
    scenario = create_price_exception_scenario()
    state = RuleBasedInvestigator(
        CaseMemory(seed_historical_cases())
    ).investigate(scenario)
    state.add_observation(
        name="counter_evidence",
        value={
            "claim": "Primary and secondary price sources agree",
            "contradicts": True,
        },
        source="PRICE_SOURCE_REVIEW",
    )

    challenge = EvidenceChallengeAgent().review(state)
    state = apply_challenge(state, challenge)

    assert challenge.contradiction_found is True
    assert challenge.ambiguity_found is False
    assert state.status == "ESCALATE"
    assert state.confidence <= 0.35
    assert "conflicting evidence" in state.recommended_action


def test_resolution_escalates_ambiguous_case_to_investigate_further():
    scenario = create_price_exception_scenario()
    state = RuleBasedInvestigator(
        CaseMemory(seed_historical_cases())
    ).investigate(scenario)
    state.add_hypothesis(
        root_cause="TRANSACTION_MISMATCH",
        rationale="Alternative operational cause.",
        confidence=0.86,
    )

    challenge = EvidenceChallengeAgent().review(state)
    apply_challenge(state, challenge)
    recommendation = ResolutionAgent().resolve(state)

    assert recommendation.decision == "INVESTIGATE_FURTHER"
    assert recommendation.requires_human_approval is True


def test_structured_counter_evidence_still_caps_confidence_and_escalates():
    state = InvestigationState(exception={"exception_type": "NAV_DISCREPANCY"})
    state.add_hypothesis("PRICE_EXCEPTION:SEC-1", "Price variance hypothesis.", 0.95)
    state.add_evidence(EvidenceItem(
        source_type=EvidenceSourceType.PRICE_SOURCE,
        source_name="SEC-1",
        claim="Source records challenge the price exception hypothesis.",
        contradicts="PRICE_EXCEPTION:SEC-1",
    ))

    challenge = EvidenceChallengeAgent().review(state)
    apply_challenge(state, challenge)

    assert challenge.contradiction_found is True
    assert challenge.final_confidence == 0.35
    assert state.status == "ESCALATE"
