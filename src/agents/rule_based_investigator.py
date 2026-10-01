from __future__ import annotations

from src.agents.state import InvestigationState
from src.data.scenarios import InvestigationScenario
from src.memory.cases import CaseMemory
from src.tools.memory_tools import search_historical_cases_tool
from src.tools.nav_tools import (
    calculate_nav_variance_tool,
    compare_price_sources_tool,
    identify_top_contributors_tool,
)


class RuleBasedInvestigator:
    """
    Transparent baseline for the capstone.

    It is deliberately deterministic. The future LLM/agentic investigator
    must be evaluated against this baseline rather than assumed to be better.
    """

    def __init__(self, memory: CaseMemory) -> None:
        self.memory = memory

    def investigate(self, scenario: InvestigationScenario) -> InvestigationState:
        state = InvestigationState(
            exception=calculate_nav_variance_tool(scenario)
        )

        contributors = identify_top_contributors_tool(scenario, top_n=3)
        state.add_observation(
            "top_contributors",
            contributors,
            "deterministic_contribution_analysis",
        )

        if not contributors:
            state.status = "ESCALATE"
            state.recommended_action = "Investigate further; no contribution evidence available."
            return state

        top_security = contributors[0]["security_id"]
        price_check = compare_price_sources_tool(scenario, top_security)
        state.add_observation(
            "price_source_check",
            price_check,
            "price_comparison_tool",
        )

        historical = search_historical_cases_tool(
            self.memory,
            query="NAV variance price vendor discrepancy",
            exception_type="NAV_DISCREPANCY",
            top_k=3,
        )
        state.add_observation(
            "historical_cases",
            historical,
            "case_memory",
        )

        if (
            price_check.get("found")
            and abs(price_check["difference_pct"]) >= 10
        ):
            state.add_hypothesis(
                root_cause="PRICE_EXCEPTION",
                rationale=(
                    f"{top_security} is the largest NAV contributor and its "
                    "calculated price differs materially from the reference price."
                ),
                confidence=0.90,
            )
            state.recommended_action = (
                f"Validate the price for {top_security} against the primary "
                "vendor and confirm whether the secondary-feed price is stale or erroneous."
            )
            state.confidence = 0.90
            state.status = "READY_FOR_HUMAN_REVIEW"
        else:
            state.add_hypothesis(
                root_cause="UNKNOWN",
                rationale="Available deterministic signals do not isolate a material price issue.",
                confidence=0.35,
            )
            state.recommended_action = "Escalate for additional transaction and corporate-action investigation."
            state.confidence = 0.35
            state.status = "ESCALATE"

        return state
