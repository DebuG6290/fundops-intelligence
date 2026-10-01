from __future__ import annotations

from src.agents.rule_based_investigator import RuleBasedInvestigator
from src.data.scenarios import create_price_exception_scenario
from src.memory.cases import CaseMemory, seed_historical_cases
from evaluation.metrics import calculate_accuracy


def run_price_baseline(case_count: int = 20):
    memory = CaseMemory(seed_historical_cases())
    investigator = RuleBasedInvestigator(memory)

    predictions = []
    actuals = []

    for seed in range(case_count):
        scenario = create_price_exception_scenario(seed=seed)
        state = investigator.investigate(scenario)

        predictions.append(
            state.hypotheses[0]["root_cause"]
            if state.hypotheses
            else "UNKNOWN"
        )
        actuals.append("PRICE_EXCEPTION")

    return calculate_accuracy(predictions, actuals)


if __name__ == "__main__":
    result = run_price_baseline()
    print(f"Cases: {result.total_cases}")
    print(f"Correct: {result.correct_root_causes}")
    print(f"Accuracy: {result.root_cause_accuracy:.1%}")
