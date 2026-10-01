from __future__ import annotations

from typing import Any

from src.agents.agent_loop import AgentRun, InvestigationAgentLoop
from src.agents.openai_provider import OpenAIResponsesProvider
from src.agents.prompts import SYSTEM_PROMPT
from src.agents.tool_registry import ToolDefinition, ToolRegistry
from src.data.scenarios_extra import TransactionMismatchScenario
from src.memory.cases import CaseMemory
from src.tools.exception_tools import find_transaction_mismatches_tool
from src.tools.memory_tools import search_historical_cases_tool


def build_transaction_tool_registry(
    scenario: TransactionMismatchScenario,
    memory: CaseMemory,
) -> ToolRegistry:
    return ToolRegistry([
        ToolDefinition(
            name="find_transaction_mismatches",
            description="Compare expected and actual transactions and identify missing, extra, or changed fields.",
            parameters={
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
            function=lambda: find_transaction_mismatches_tool(scenario),
        ),
        ToolDefinition(
            name="search_historical_cases",
            description="Search historical transaction exception cases for similar symptoms.",
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


class TransactionInvestigationAgent:
    """Agentic specialist for transaction mismatch investigation."""

    def __init__(
        self,
        memory: CaseMemory,
        provider: Any | None = None,
    ) -> None:
        self.memory = memory
        self.provider = provider or OpenAIResponsesProvider()

    def investigate(self, scenario: TransactionMismatchScenario) -> AgentRun:
        context = {
            "exception_id": scenario.exception_id,
            "known_exception_type": "TRANSACTION_MISMATCH",
            "investigation_objective": (
                "Identify the most probable transaction-level root cause, "
                "distinguish quantity/type/missing-transaction issues, and "
                "recommend the next human investigation step."
            ),
        }
        loop = InvestigationAgentLoop(
            provider=self.provider,
            tools=build_transaction_tool_registry(scenario, self.memory),
            max_steps=5,
        )
        return loop.run(system_instructions=SYSTEM_PROMPT, context=context)
