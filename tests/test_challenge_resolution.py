from src.agents.evidence_challenge import EvidenceChallengeAgent, apply_challenge
from src.agents.resolution import ResolutionAgent
from src.agents.rule_based_investigator import RuleBasedInvestigator
from src.data.scenarios import create_price_exception_scenario
from src.memory.cases import CaseMemory, seed_historical_cases


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
