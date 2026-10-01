import json
from types import SimpleNamespace

from src.agents.agent_loop import InvestigationAgentLoop
from src.agents.tool_registry import ToolDefinition, ToolRegistry


class Provider:
    def create_response(self, **kwargs):
        return SimpleNamespace(
            id="r1",
            output=[
                SimpleNamespace(
                    type="function_call",
                    name="ping",
                    arguments=json.dumps({}),
                    call_id="c1",
                )
            ],
            output_text="",
        )

    def continue_response(self, **kwargs):
        return SimpleNamespace(
            id="r2",
            output=[],
            output_text=json.dumps({
                "probable_root_cause": "TEST",
                "confidence": 1,
                "observations": [],
                "supporting_evidence": [],
                "counter_evidence": [],
                "recommended_next_step": "Review",
                "human_review_required": True,
            }),
        )


def test_agent_records_telemetry():
    registry = ToolRegistry([
        ToolDefinition(
            name="ping",
            description="Ping.",
            parameters={
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
            function=lambda: {"ok": True},
        )
    ])

    run = InvestigationAgentLoop(Provider(), registry).run("test", {})

    assert run.telemetry.tool_calls == 1
    assert run.telemetry.latency_seconds is not None
