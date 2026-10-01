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
        context = {
            "exception": exception,
            "known_exception_type": "NAV_DISCREPANCY",
            "investigation_objective": (
                "Identify the most probable operational root cause and "
                "recommend the next human investigation step."
            ),
        }

        loop = InvestigationAgentLoop(
            provider=self.provider,
            tools=build_nav_tool_registry(scenario, self.memory),
            max_steps=6,
        )
        return loop.run(
            system_instructions=SYSTEM_PROMPT,
            context=context,
        )
