from __future__ import annotations

from typing import Any

from src.agents.workflow import InvestigationWorkflow
from src.data.scenarios import create_price_exception_scenario
from src.data.scenarios_extra import (
    create_corporate_action_scenario,
    create_transaction_mismatch_scenario,
)
from src.memory.cases import CaseMemory, HistoricalCase, seed_historical_cases
from src.review.models import HumanDecision, HumanReviewRecord


class DemoWorkbench:
    """Reproducible app-facing orchestration; all calculations stay in ``src``."""

    NAV = "NAV Discrepancy"
    TRANSACTION = "Transaction Mismatch"
    CORPORATE_ACTION = "Corporate Action"
    INSUFFICIENT = "Insufficient Evidence (NAV)"

    def __init__(self, memory: CaseMemory | None = None) -> None:
        self.memory = memory or CaseMemory(seed_historical_cases())
        self.workflow = InvestigationWorkflow(self.memory)

    def investigate(self, scenario_name: str, provider: Any | None = None) -> dict[str, Any]:
        if scenario_name in (self.NAV, self.INSUFFICIENT):
            scenario = create_price_exception_scenario(seed=42)
            if scenario_name == self.INSUFFICIENT:
                scenario.calculated_prices = scenario.expected_prices.copy(deep=True)
            return self.workflow.run_nav(scenario)
        if scenario_name == self.TRANSACTION:
            return self.workflow.run_transaction(create_transaction_mismatch_scenario(seed=42), provider)
        if scenario_name == self.CORPORATE_ACTION:
            return self.workflow.run_corporate_action(create_corporate_action_scenario(seed=42), provider)
        raise ValueError(f"Unsupported demo scenario: {scenario_name}")

    def submit_review(
        self, result: dict[str, Any], decision: HumanDecision | str, reason: str
    ) -> HumanReviewRecord:
        return self.workflow.submit_human_review(result, decision, reason)

    def accepted_case(self, review_id: str) -> HistoricalCase | None:
        return self.workflow.accepted_cases_by_review_id.get(review_id)

    def search_memory(self, query: str, exception_type: str | None = None) -> list[HistoricalCase]:
        return self.memory.search(query, exception_type=exception_type, top_k=5)
