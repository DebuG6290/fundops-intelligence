import json
from types import SimpleNamespace

from src.agents.agent_loop import InvestigationAgentLoop
from src.agents.tool_registry import ToolDefinition, ToolRegistry


class FakeProvider:
    def __init__(self):
        self.calls = 0

    def create_response(self, **kwargs):
        self.calls += 1
        return SimpleNamespace(
            id="r1",
            output=[
                SimpleNamespace(
                    type="function_call",
                    name="get_price",
                    arguments=json.dumps({"security_id": "SEC_001"}),
                    call_id="call_1",
                )
            ],
            output_text="",
        )

    def continue_response(self, **kwargs):
        self.calls += 1
        return SimpleNamespace(
            id="r2",
            output=[],
            output_text=json.dumps({
                "probable_root_cause": "PRICE_EXCEPTION",
                "confidence": 0.91,
                "observations": ["Price source differs materially."],
                "supporting_evidence": ["Reference price is 100; calculated price is 72."],
                "counter_evidence": [],
                "recommended_next_step": "Validate the primary vendor price.",
                "human_review_required": True,
            }),
        )


def test_agent_loop_executes_tool_and_returns_structured_report():
    registry = ToolRegistry([
        ToolDefinition(
            name="get_price",
            description="Get a price.",
            parameters={
                "type": "object",
                "properties": {"security_id": {"type": "string"}},
                "required": ["security_id"],
                "additionalProperties": False,
            },
            function=lambda security_id: {
                "security_id": security_id,
                "expected": 100,
                "calculated": 72,
            },
        )
    ])

    run = InvestigationAgentLoop(
        provider=FakeProvider(),
        tools=registry,
    ).run(
        system_instructions="test",
        context={"exception": "NAV_DISCREPANCY"},
    )

    assert run.report is not None
    assert run.report.probable_root_cause == "PRICE_EXCEPTION"
    assert run.report.confidence == 0.91
    assert len(run.trace) == 1
