from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from src.agents.schemas import InvestigationReport, ToolTrace
from src.agents.tool_registry import ToolRegistry
from src.agents.telemetry import AgentTelemetry


@dataclass
class AgentRun:
    report: InvestigationReport | None = None
    trace: list[ToolTrace] = field(default_factory=list)
    raw_outputs: list[str] = field(default_factory=list)
    telemetry: AgentTelemetry = field(default_factory=AgentTelemetry)


class InvestigationAgentLoop:
    """
    Provider-agnostic investigation loop.

    The provider must expose:
      create_response(...)
      continue_response(...)

    The loop repeatedly executes requested tools until the model produces
    a final structured investigation report or the step limit is reached.
    """

    def __init__(
        self,
        provider: Any,
        tools: ToolRegistry,
        max_steps: int = 6,
    ) -> None:
        self.provider = provider
        self.tools = tools
        self.max_steps = max_steps

    def run(
        self,
        system_instructions: str,
        context: dict[str, Any],
    ) -> AgentRun:
        run = AgentRun()

        try:
            response = self.provider.create_response(
                system_instructions=system_instructions,
                context=context,
                tools=self.tools.definitions(),
            )

            for _ in range(self.max_steps):
                run.raw_outputs.append(getattr(response, "output_text", ""))

                calls = [
                item for item in response.output
                if getattr(item, "type", None) == "function_call"
                ]

                if not calls:
                    run.report = self._parse_report(response.output_text)
                    return run

                outputs = []

                for call in calls:
                    arguments = json.loads(call.arguments)
                    result = self.tools.execute(call.name, arguments)
                    run.telemetry.record_tool_call()

                    run.trace.append(
                        ToolTrace(
                            tool_name=call.name,
                            arguments=arguments,
                            result=result,
                        )
                    )

                    outputs.append({
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": json.dumps(result, default=str),
                    })

                response = self.provider.continue_response(
                    previous_response_id=response.id,
                    tool_outputs=outputs,
                    system_instructions=system_instructions,
                )

            raise RuntimeError(
                f"Investigation agent exceeded max_steps={self.max_steps}"
            )
        finally:
            run.telemetry.finish()

    @staticmethod
    def _parse_report(output_text: str) -> InvestigationReport:
        try:
            payload = json.loads(output_text)
            return InvestigationReport.model_validate(payload)
        except Exception as exc:
            raise ValueError(
                "Agent did not return valid InvestigationReport JSON"
            ) from exc
