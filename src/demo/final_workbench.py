from __future__ import annotations

from typing import Any

from src.agents.workflow import InvestigationWorkflow
from src.data.demo_scenarios import create_final_demo_scenarios
from src.memory.cases import CaseMemory, HistoricalCase, seed_historical_cases
from src.review.models import HumanDecision, HumanReviewRecord


class FinalDemoWorkbench:
    """Interactive, deterministic-first final demo over controlled NAV cases."""

    def __init__(self, memory: CaseMemory | None = None, seed: int = 42) -> None:
        self.memory = memory or CaseMemory(seed_historical_cases())
        self.workflow = InvestigationWorkflow(self.memory)
        self.scenarios = create_final_demo_scenarios(seed)

    def list_scenarios(self) -> list[dict[str, str]]:
        return [{"scenario_id": item.scenario_id, "title": item.title, "description": item.description}
                for item in self.scenarios.values()]

    def investigate(self, scenario_id: str, provider: Any | None = None) -> dict[str, Any]:
        item = self.scenarios.get(scenario_id)
        if item is None:
            raise ValueError(f"Unsupported final demo scenario: {scenario_id}")
        # Retrieve relevant validated analogies inside the workflow for this case.
        # Do not use whichever case happened to be accepted most recently.
        result = self.workflow.run_nav(item.scenario, provider=provider)
        result.update({"scenario_id": item.scenario_id, "scenario_title": item.title,
                       "scenario_description": item.description})
        # expected_root_cause and difficulty deliberately stay evaluation-only.
        return result

    def submit_review(self, result: dict[str, Any], decision: HumanDecision | str, reason: str) -> HumanReviewRecord:
        return self.workflow.submit_human_review(result, decision, reason)

    def accepted_case(self, review_id: str) -> HistoricalCase | None:
        return self.workflow.accepted_cases_by_review_id.get(review_id)

    def search_memory(self, query: str, exception_type: str | None = None) -> list[HistoricalCase]:
        return self.memory.search(query, exception_type=exception_type, top_k=5)

    def get_memory_preview(self, result: dict[str, Any]) -> dict[str, Any]:
        return result.get("memory_context", {"retrieved_cases": [], "retrieved_case_count": 0,
                                               "prior_investigation_paths": [], "memory_influence": None})

