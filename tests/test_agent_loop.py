import json
from types import SimpleNamespace

import pytest

from src.agents.agent_loop import InvestigationAgentLoop
from src.agents.schemas import InvestigationReport
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


def test_report_accepts_ranked_multi_hypothesis_output():
    report = InvestigationReport.model_validate({
        "probable_root_cause": "STALE_PRICE",
        "confidence": 0.84,
        "observations": [],
        "supporting_evidence": [],
        "counter_evidence": [],
        "recommended_next_step": "Verify source timestamp.",
        "human_review_required": True,
        "hypotheses": [
            {"hypothesis_id": "HYP-001", "root_cause": "STALE_PRICE", "rationale": "Price is old.", "confidence": 0.84,
             "required_evidence": ["vendor timestamp"], "uncertainty": "Independent feed not checked."},
            {"hypothesis_id": "HYP-002", "root_cause": "FX_MISMATCH", "rationale": "Currency conversion may differ.", "confidence": 0.31},
        ],
    })
    assert [item.hypothesis_id for item in report.hypotheses] == ["HYP-001", "HYP-002"]
    assert report.hypotheses[1].required_evidence == []


def test_report_rejects_out_of_range_hypothesis_confidence():
    payload = {
        "probable_root_cause": "STALE_PRICE", "confidence": 0.84,
        "observations": [], "supporting_evidence": [], "counter_evidence": [],
        "recommended_next_step": "Check price.",
        "hypotheses": [{"hypothesis_id": "HYP-1", "root_cause": "STALE_PRICE",
                        "rationale": "old", "confidence": 1.2}],
    }
    with pytest.raises(Exception):
        InvestigationReport.model_validate(payload)
