from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from src.agents.evidence import evidence_from_tool_result
from src.agents.schemas import InvestigationReport, ToolTrace
from src.agents.telemetry import AgentTelemetry
from src.agents.tool_registry import ToolRegistry
from src.models.evidence import EvidenceItem


@dataclass
class AgentRun:
    report: InvestigationReport | None = None
    trace: list[ToolTrace] = field(default_factory=list)
    evidence: list[EvidenceItem] = field(default_factory=list)
    raw_outputs: list[str] = field(default_factory=list)
    telemetry: AgentTelemetry = field(default_factory=AgentTelemetry)
    timeline: list[dict[str, Any]] = field(default_factory=list)


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
            self._record_provider_usage(run)

            for _ in range(self.max_steps):
                run.raw_outputs.append(getattr(response, "output_text", ""))

                calls = [
                item for item in response.output
                if getattr(item, "type", None) == "function_call"
                ]

                if not calls:
                    try:
                        run.report = self._parse_report(response.output_text)
                    except ValueError as validation_error:
                        repair = getattr(self.provider, "repair_structured_response", None)
                        if repair is None:
                            raise
                        response = repair(
                            invalid_output=getattr(response, "output_text", "") or "",
                            validation_error=str(validation_error),
                            system_instructions=system_instructions,
                        )
                        self._record_provider_usage(run)
                        run.raw_outputs.append(getattr(response, "output_text", "") or "")
                        run.report = self._parse_report(getattr(response, "output_text", "") or "")
                    return run

                outputs = []

                for call in calls:
                    if not getattr(call, "name", None) or not getattr(call, "call_id", None):
                        raise ValueError("Provider returned an invalid tool call (missing name or call ID)")
                    try:
                        arguments = json.loads(call.arguments)
                    except (TypeError, json.JSONDecodeError):
                        raise ValueError(f"Provider returned invalid JSON arguments for tool {call.name}") from None
                    if not isinstance(arguments, dict):
                        raise ValueError(f"Provider tool arguments for {call.name} must be a JSON object")
                    result = self.tools.execute(call.name, arguments)
                    run.telemetry.record_tool_call()

                    run.trace.append(
                        ToolTrace(
                            tool_name=call.name,
                            arguments=arguments,
                            result=result,
                        )
                    )
                    exception = context.get("exception")
                    exception_id = context.get("exception_id") or (
                        exception.get("exception_id", "unknown")
                        if isinstance(exception, dict)
                        else "unknown"
                    )
                    run.evidence.extend(
                        evidence_from_tool_result(
                            call.name,
                            result,
                            exception_id=exception_id,
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
                self._record_provider_usage(run)

            raise RuntimeError(
                f"Investigation agent exceeded max_steps={self.max_steps}"
            )
        finally:
            run.telemetry.finish()

    def _record_provider_usage(self, run: AgentRun) -> None:
        usage = getattr(self.provider, "last_call", None) or {}
        run.telemetry.provider = usage.get("provider", run.telemetry.provider)
        run.telemetry.model = usage.get("model", run.telemetry.model)
        request_id = usage.get("request_id")
        if request_id:
            run.telemetry.request_ids.append(str(request_id))
        for target, key in (("input_tokens", "input_tokens"), ("output_tokens", "output_tokens")):
            value = usage.get(key)
            if value is not None:
                current = getattr(run.telemetry, target) or 0
                setattr(run.telemetry, target, current + int(value))

    @staticmethod
    def _parse_report(output_text: str) -> InvestigationReport:
        try:
            payload = json.loads(output_text)
            if not isinstance(payload, dict):
                raise ValueError("Investigation report must be a JSON object")
            # Enforce the human-in-the-loop boundary at the parser even if a
            # model incorrectly emits false; never let model text waive review.
            payload["human_review_required"] = True
            return InvestigationReport.model_validate(payload)
        except Exception as exc:
            raise ValueError(
                "Agent did not return valid InvestigationReport JSON"
            ) from exc

