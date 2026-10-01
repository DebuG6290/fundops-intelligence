from pprint import pprint

from src.agents.rule_based_investigator import RuleBasedInvestigator
from src.data.scenarios import create_price_exception_scenario
from src.memory.cases import CaseMemory, seed_historical_cases


def main() -> None:
    scenario = create_price_exception_scenario()
    investigator = RuleBasedInvestigator(
        CaseMemory(seed_historical_cases())
    )

    state = investigator.investigate(scenario)

    print("\n=== INVESTIGATION RESULT ===")
    pprint(state.exception)

    print("\n=== OBSERVATIONS ===")
    for observation in state.observations:
        pprint(observation)

    print("\n=== HYPOTHESES ===")
    for hypothesis in state.hypotheses:
        pprint(hypothesis)

    print("\n=== RECOMMENDATION ===")
    print(state.recommended_action)
    print(f"Confidence: {state.confidence:.0%}")
    print(f"Status: {state.status}")


if __name__ == "__main__":
    main()
