from __future__ import annotations

import json
from types import SimpleNamespace

from evaluation.baselines import (
    AgenticLLMBaseline, BaselineResult, ObservableCase, RuleBaseline,
    SingleLLMBaseline,
)


def _report():
    return json.dumps({
        "probable_root_cause": "STALE_PRICE",
        "confidence": .82,
        "observations": [], "supporting_evidence": [], "counter_evidence": [],
        "recommended_next_step": "Verify the vendor timestamp.",
        "human_review_required": True,
        "hypotheses": [{"hypothesis_id": "HYP-001", "root_cause": "STALE_PRICE", "rationale": "A current record shows a material price difference.", "confidence": .82}],
        "cited_evidence_ids": ["EV-1"],
        "recommendation_category": "REVIEW_RECOMMENDATION",
        "escalation_required": False,
        "evidence_assessments": [{"evidence_id": "EV-1", "hypothesis_id": "HYP-001", "relationship": "SUPPORTS"}],
    })


class _SingleProvider:
    model = "sarvam-105b"
    last_call = {"provider": "sarvam", "model": "sarvam-105b", "request_id": "req-1", "input_tokens": 12, "output_tokens": 8}

    def create_response(self, **kwargs):
        self.context = kwargs["context"]
        self.last_call = {"provider": "sarvam", "model": self.model, "request_id": "req-1", "input_tokens": 12, "output_tokens": 8}
        return SimpleNamespace(output_text=_report())


class _AgenticProvider(_SingleProvider):
    def __init__(self):
        self.continuations = 0

    def create_response(self, **kwargs):
        self.context = kwargs["context"]
        return SimpleNamespace(
            id="r1", output=[SimpleNamespace(
                type="function_call", name="retrieve_current_evidence",
                arguments=json.dumps({"evidence_ids": ["EV-1"]}), call_id="c1",
            )], output_text="",
        )

    def continue_response(self, **kwargs):
        self.continuations += 1
        return SimpleNamespace(id="r2", output=[], output_text=_report())


def _case():
    return ObservableCase(
        exception_id="EXC-1", features={"price_variance_pct": 14.0},
        evidence=({"evidence_id": "EV-1", "claim": "Vendor A price differs from Vendor B."},),
    )


def test_rules_single_llm_and_agentic_share_result_contract_without_ground_truth():
    case = _case()
    rules = RuleBaseline().predict(case)
    single_provider = _SingleProvider()
    single = SingleLLMBaseline(single_provider).predict(case)
    agentic_provider = _AgenticProvider()
    agentic = AgenticLLMBaseline(agentic_provider).predict(case)

    assert all(isinstance(result, BaselineResult) for result in (rules, single, agentic))
    for result in (rules, single, agentic):
        assert result.approach
        assert result.latency_seconds is not None
        assert result.error is None
    assert single_provider.context == case.to_context()
    assert agentic_provider.continuations == 1
    assert single.cited_evidence_ids == ["EV-1"]
    assert single.request_ids == ["req-1"]
    assert agentic.cited_evidence_ids == ["EV-1"]
    assert agentic.escalation_required is False
    assert "ground_truth" not in single_provider.context


def test_single_llm_retries_schema_failure_once_and_surfaces_provider_timeout():
    class RepairingProvider(_SingleProvider):
        def __init__(self):
            self.repairs = 0

        def create_response(self, **kwargs):
            return SimpleNamespace(output_text="")

        def repair_structured_response(self, **kwargs):
            self.repairs += 1
            return SimpleNamespace(output_text=_report())

    provider = RepairingProvider()
    result = SingleLLMBaseline(provider).predict(_case())
    assert result.error is None
    assert provider.repairs == 1

    class TimeoutProvider:
        model = "sarvam-105b"

        def create_response(self, **kwargs):
            raise TimeoutError("request timed out")

    failed = SingleLLMBaseline(TimeoutProvider()).predict(_case())
    assert failed.error is not None and "TimeoutError" in failed.error
