from src.agents.rule_based_investigator import RuleBasedInvestigator
from src.data.scenarios import create_price_exception_scenario
from src.memory.cases import CaseMemory, seed_historical_cases


def test_rule_based_investigator_finds_price_exception():
    memory = CaseMemory(seed_historical_cases())
    investigator = RuleBasedInvestigator(memory)

    state = investigator.investigate(create_price_exception_scenario())

    assert state.status == "READY_FOR_HUMAN_REVIEW"
    assert state.hypotheses[0]["root_cause"] == "PRICE_EXCEPTION"
    assert state.confidence >= 0.8
    assert state.recommended_action
