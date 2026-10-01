from __future__ import annotations

from typing import Any

from src.agents.agent_loop import AgentRun, InvestigationAgentLoop
from src.agents.openai_provider import OpenAIResponsesProvider
from src.agents.prompts import SYSTEM_PROMPT
from src.agents.tool_registry import ToolDefinition, ToolRegistry
from src.data.scenarios_extra import CorporateActionScenario
from src.memory.cases import CaseMemory
from src.tools.exception_tools import find_corporate_actions_tool
from src.tools.memory_tools import search_historical_cases_tool


def build_corporate_action_tool_registry(
    scenario: CorporateActionScenario,
    memory: CaseMemory,
) -> ToolRegistry:
    return ToolRegistry([
        ToolDefinition(
            name="find_effective_corporate_actions",
            description="Find corporate actions effective for the exception date and return their details.",
            parameters={
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
            function=lambda: find_corporate_actions_tool(scenario),
        ),
        ToolDefinition(
            name="search_historical_cases",
            description="Search historical corporate-action exception cases for similar symptoms.",
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


class CorporateActionInvestigationAgent:
    """Agentic specialist for corporate-action-related discrepancies."""

    def __init__(
        self,
        memory: CaseMemory,
        provider: Any | None = None,
    ) -> None:
        self.memory = memory
        self.provider = provider or OpenAIResponsesProvider()

    def investigate(self, scenario: CorporateActionScenario) -> AgentRun:
        context = {
            "exception_id": scenario.exception_id,
            "known_exception_type": "CORPORATE_ACTION",
            "investigation_objective": (
                "Identify whether an effective corporate action plausibly explains "
                "the operational discrepancy and recommend the next human "
                "investigation step. Do not infer accounting impact beyond evidence."
            ),
        }
        loop = InvestigationAgentLoop(
            provider=self.provider,
            tools=build_corporate_action_tool_registry(scenario, self.memory),
            max_steps=5,
        )
        return loop.run(system_instructions=SYSTEM_PROMPT, context=context)
