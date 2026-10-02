from __future__ import annotations

from typing import Any

from src.agents.agent_loop import AgentRun, InvestigationAgentLoop
from src.llm.sarvam_provider import SarvamProvider
from src.agents.prompts import SYSTEM_PROMPT
from src.agents.tool_registry import ToolDefinition, ToolRegistry
from src.data.scenarios import InvestigationScenario
from src.memory.cases import CaseMemory
from src.tools.memory_tools import search_historical_cases_tool
from src.tools.nav_tools import (
    calculate_nav_variance_tool,
    compare_price_sources_tool,
    identify_top_contributors_tool,
)
from src.agents.evidence import evidence_from_tool_result
from src.agents.schemas import ToolTrace
from src.tools.investigation_tools import (
    check_transaction_activity_tool, check_corporate_actions_tool,
    check_security_mapping_tool, check_fx_context_tool,
)


def build_nav_tool_registry(
    scenario: InvestigationScenario,
    memory: CaseMemory,
) -> ToolRegistry:
    return ToolRegistry([
        ToolDefinition(
            name="identify_top_contributors",
            description="Identify securities contributing most to the NAV variance.",
            parameters={
                "type": "object",
                "properties": {"top_n": {"type": "integer", "minimum": 1, "maximum": 10}},
                "required": ["top_n"],
                "additionalProperties": False,
            },
            function=lambda top_n: identify_top_contributors_tool(scenario, top_n),
        ),
        ToolDefinition(
            name="compare_price_sources",
            description="Compare the expected reference price and calculated fund price for a security.",
            parameters={
                "type": "object",
                "properties": {"security_id": {"type": "string"}},
                "required": ["security_id"],
                "additionalProperties": False,
            },
            function=lambda security_id: compare_price_sources_tool(
                scenario, security_id
            ),
        ),
        ToolDefinition(
            name="check_transaction_activity",
            description="Inspect expected and actual transaction records for unexplained transaction activity relevant to a security.",
            parameters={"type": "object", "properties": {"security_id": {"type": ["string", "null"]}}, "required": ["security_id"], "additionalProperties": False},
            function=lambda security_id: check_transaction_activity_tool(scenario, security_id),
        ),
        ToolDefinition(
            name="check_corporate_actions",
            description="Inspect effective corporate actions relevant to a security.",
            parameters={"type": "object", "properties": {"security_id": {"type": ["string", "null"]}}, "required": ["security_id"], "additionalProperties": False},
            function=lambda security_id: check_corporate_actions_tool(scenario, security_id),
        ),
        ToolDefinition(
            name="check_security_mapping",
            description="Verify a security identifier against the current fund security dataset.",
            parameters={"type": "object", "properties": {"security_id": {"type": "string"}}, "required": ["security_id"], "additionalProperties": False},
            function=lambda security_id: check_security_mapping_tool(scenario, security_id),
        ),
        ToolDefinition(
            name="check_fx_context",
            description="Inspect current scenario FX observations relevant to a security.",
            parameters={"type": "object", "properties": {"security_id": {"type": ["string", "null"]}}, "required": ["security_id"], "additionalProperties": False},
            function=lambda security_id: check_fx_context_tool(scenario, security_id),
        ),
        ToolDefinition(
            name="search_historical_cases",
            description="Search historical resolved fund-operation cases for similar symptoms.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "exception_type": {"type": ["string", "null"]},
                },
                "required": ["query", "exception_type"],
                "additionalProperties": False,
            },
            function=lambda query, exception_type: search_historical_cases_tool(
                memory,
                query=query,
                exception_type=exception_type,
                top_k=3,
            ),
        ),
    ])


class NavInvestigationAgent:
    def __init__(
        self,
        memory: CaseMemory,
        provider: Any | None = None,
    ) -> None:
        self.memory = memory
        self.provider = provider or SarvamProvider()

    def investigate(self, scenario: InvestigationScenario) -> AgentRun:
        exception = calculate_nav_variance_tool(scenario)
        historical_cases = search_historical_cases_tool(
            self.memory, "NAV variance price transaction corporate action",
            "NAV_DISCREPANCY", top_k=5,
        )
        context = {
            "exception": exception,
            "current_observations": ["NAV variance and valuation fields are deterministic outputs; inspect current source records before attributing cause."],
            "historical_cases": historical_cases,
            "historical_cases_are_analogies_not_proof": True,
            "fund_id": str(scenario.dataset.funds.iloc[0]["fund_id"]),
            "known_exception_type": "NAV_DISCREPANCY",
            "investigation_objective": (
                "Identify the most probable operational root cause and "
                "recommend the next human investigation step."
                " Start with memory when useful, test alternative causes, and do not treat historical cases as proof."
            ),
            "historical_cases_are_analogies": True,
            "primary_evidence_and_human_review_required": True,
        }

        loop = InvestigationAgentLoop(
            provider=self.provider,
            tools=build_nav_tool_registry(scenario, self.memory),
            max_steps=8,
        )
        run = loop.run(
            system_instructions=SYSTEM_PROMPT,
            context=context,
        )
        run.evidence.extend(evidence_from_tool_result("search_historical_cases", historical_cases, scenario.exception_id))
        return run

