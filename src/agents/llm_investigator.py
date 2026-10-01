from __future__ import annotations

import json
import os
from typing import Any, Callable

from src.data.scenarios import InvestigationScenario
from src.memory.cases import CaseMemory
from src.tools.memory_tools import search_historical_cases_tool
from src.tools.nav_tools import (
    calculate_nav_variance_tool,
    compare_price_sources_tool,
    identify_top_contributors_tool,
)


SYSTEM_INSTRUCTIONS = """
You are a financial-operations investigation agent.

Your job is to investigate a fund-operation exception, not to make an investment
decision and not to modify accounting records.

Rules:
1. Never perform financial arithmetic yourself when a deterministic tool can do it.
2. Use tools to gather evidence before forming a conclusion.
3. Consider at least two plausible root-cause hypotheses when the evidence permits.
4. Historical cases are supporting evidence, not proof.
5. Distinguish observations from hypotheses.
6. If evidence is insufficient or contradictory, recommend escalation.
7. Return a concise investigation report with:
   - probable_root_cause
   - confidence
   - observations
   - supporting_evidence
   - counter_evidence
   - recommended_next_step
   - human_review_required
"""


class LLMInvestigator:
    """
    Model-agnostic boundary around an LLM.

    The actual provider is injected so the rest of the project does not depend
    on one model vendor.
    """

    def __init__(
        self,
        provider: Any,
        model: str,
        memory: CaseMemory,
    ) -> None:
        self.provider = provider
        self.model = model
        self.memory = memory

    def investigate(
        self,
        scenario: InvestigationScenario,
    ) -> dict[str, Any]:
        context = {
            "exception": calculate_nav_variance_tool(scenario),
            "top_contributors": identify_top_contributors_tool(scenario, top_n=5),
            "fund_snapshot": {
                "fund_id": scenario.dataset.funds.iloc[0]["fund_id"],
                "fund_name": scenario.dataset.funds.iloc[0]["fund_name"],
            },
        }

        tools = self._tools(scenario)
        return self.provider.run(
            model=self.model,
            system_instructions=SYSTEM_INSTRUCTIONS,
            context=context,
            tools=tools,
        )

    def _tools(self, scenario: InvestigationScenario) -> dict[str, Callable[..., Any]]:
        return {
            "compare_price_sources": lambda security_id: compare_price_sources_tool(
                scenario, security_id
            ),
            "search_historical_cases": lambda query, exception_type=None: (
                search_historical_cases_tool(
                    self.memory,
                    query=query,
                    exception_type=exception_type,
                    top_k=3,
                )
            ),
        }


class OpenAIResponsesProvider:
    """Minimal Responses API provider with an explicit function-call loop."""

    def __init__(self, client: Any | None = None) -> None:
        if client is None:
            from openai import OpenAI
            client = OpenAI()
        self.client = client

    def run(
        self,
        model: str,
        system_instructions: str,
        context: dict[str, Any],
        tools: dict[str, Callable[..., Any]],
    ) -> dict[str, Any]:
        function_tools = [
            {
                "type": "function",
                "name": "compare_price_sources",
                "description": "Compare expected and calculated prices for a security.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "security_id": {"type": "string"}
                    },
                    "required": ["security_id"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
            {
                "type": "function",
                "name": "search_historical_cases",
                "description": "Search resolved historical fund-operation cases.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "exception_type": {
                            "type": ["string", "null"]
                        },
                    },
                    "required": ["query", "exception_type"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        ]

        response = self.client.responses.create(
            model=model,
            instructions=system_instructions,
            input=json.dumps(context, default=str),
            tools=function_tools,
            tool_choice="auto",
        )

        # Keep the raw output available for later tracing/evaluation.
        # A production version will iterate until no function calls remain.
        function_calls = [
            item for item in response.output
            if getattr(item, "type", None) == "function_call"
        ]

        if not function_calls:
            return {
                "output_text": response.output_text,
                "tool_calls": [],
            }

        tool_outputs = []
        for call in function_calls:
            arguments = json.loads(call.arguments)
            fn = tools.get(call.name)
            if fn is None:
                raise ValueError(f"Unknown tool requested by model: {call.name}")

            result = fn(**arguments)
            tool_outputs.append({
                "type": "function_call_output",
                "call_id": call.call_id,
                "output": json.dumps(result, default=str),
            })

        follow_up = self.client.responses.create(
            model=model,
            instructions=system_instructions,
            previous_response_id=response.id,
            input=tool_outputs,
        )

        return {
            "output_text": follow_up.output_text,
            "tool_calls": [
                {
                    "name": call.name,
                    "arguments": json.loads(call.arguments),
                }
                for call in function_calls
            ],
        }


def build_openai_investigator(memory: CaseMemory) -> LLMInvestigator:
    model = os.getenv("LLM_MODEL", "gpt-5.6-luna")
    return LLMInvestigator(
        provider=OpenAIResponsesProvider(),
        model=model,
        memory=memory,
    )
