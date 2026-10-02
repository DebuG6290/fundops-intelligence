from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from src.agents.evidence import evidence_from_tool_result
from src.agents.evidence_challenge import EvidenceChallengeAgent
from src.agents.state import InvestigationState
from src.agents.stateful_investigation import InvestigationTurn, StatefulInvestigationLoop
from src.agents.tool_registry import ToolDefinition, ToolRegistry
from src.data.benchmark import CAUSES as EVALUATION_CAUSES
from src.memory.cases import seed_historical_cases
from src.models.root_cause import canonical_root_cause


def _turn(root: str, *, hypothesis_id: str = "HYP-001", confidence: float = .5,
          rationale: str = "Check current records.", requirements: list[str] | None = None,
          action: str | None = None, arguments: dict | None = None, stop: bool = False) -> dict:
    return {
        "hypotheses": [{"hypothesis_id": hypothesis_id, "root_cause": root,
                        "confidence": confidence, "rationale": rationale,
                        "required_evidence": requirements or [], "uncertainty": "Other records may matter."}],
        "next_action": action, "action_arguments": arguments or {},
        "decision_rationale": rationale, "stop": stop,
        "stopping_rationale": "The current record is sufficient." if stop else "",
        "recommended_next_step": "Have an operator review the cited source records.",
    }


class SequencedProvider:
    def __init__(self, turns: list[dict]):
        self.turns = iter(turns)
        self.contexts: list[dict] = []

    def create_response(self, *, context, **_):
        self.contexts.append(context)
        return SimpleNamespace(output_text=json.dumps(next(self.turns)))


def _price_row(security: str, difference_pct: float) -> dict:
    return {"security_id": security, "found": True, "expected_source": "primary",
            "calculated_source": "secondary", "expected_price": 100.0,
            "calculated_price": 100.0 + difference_pct, "difference_pct": difference_pct}


def _price_registry(security: str, difference_pct: float) -> ToolRegistry:
    return ToolRegistry([ToolDefinition(
        "compare_price_sources", "Compare current prices", {"type": "object",
            "properties": {"security_id": {"type": "string"}}, "required": ["security_id"]},
        lambda security_id: _price_row(security_id, difference_pct),
    )])


def test_canonical_taxonomy_accepts_memory_and_evaluation_aliases_but_rejects_invented_causes():
    assert all(canonical_root_cause(cause) for cause in EVALUATION_CAUSES)
    assert all(canonical_root_cause(case.root_cause) for case in seed_historical_cases())
    assert canonical_root_cause("STALE_PRICE:SEC-1") == "PRICE_EXCEPTION:SEC-1"
    assert canonical_root_cause("WRONG_QUANTITY:TX-1") == "TRANSACTION_QUANTITY_MISMATCH:TX-1"
    with pytest.raises(ValueError, match="Unsupported root cause"):
        InvestigationTurn.model_validate(_turn("CAUSE:NAV_CALC_ERROR"))


def test_current_price_record_links_to_scoped_pricing_hypothesis_and_can_support_it():
    state = InvestigationState(exception={"exception_id": "EXC-PRICE"})
    state.add_hypothesis("PRICE_EXCEPTION:SEC-1", "A current price differs.", .82)
    row = _price_row("SEC-1", 28.0)
    state.add_observation("price_source_check", row, "compare_price_sources")
    state.evidence.extend(evidence_from_tool_result("compare_price_sources", row, "EXC-PRICE"))
    result = EvidenceChallengeAgent().review(state)
    assert result.supporting_evidence_ids == (state.evidence[0].evidence_id,)
    assert result.contradictory_evidence_ids == ()
    assert result.insufficient_evidence is False
    assert result.contradiction_found is False


def test_current_price_counter_evidence_blocks_pricing_conclusion():
    state = InvestigationState(exception={"exception_id": "EXC-PRICE"})
    state.add_hypothesis("PRICE_EXCEPTION:SEC-1", "The source may differ.", .84)
    row = _price_row("SEC-1", .2)
    state.add_observation("price_source_check", row, "compare_price_sources")
    state.evidence.extend(evidence_from_tool_result("compare_price_sources", row, "EXC-PRICE"))
    result = EvidenceChallengeAgent().review(state)
    assert result.contradiction_found is True
    assert result.contradictory_evidence_ids == (state.evidence[0].evidence_id,)
    assert result.supporting_evidence_ids == ()
    assert result.final_confidence <= .35


def test_pricing_run_updates_stable_hypothesis_and_stops_on_primary_evidence():
    provider = SequencedProvider([
        _turn("PRICE_EXCEPTION", confidence=.48, requirements=["top_n", "PRICE_SOURCE"],
              action="compare_price_sources", arguments={"security_id": "SEC-1"}),
        _turn("STALE_PRICE:SEC-1", hypothesis_id="HYP-999", confidence=.88,
              rationale="The current vendor prices differ materially.", requirements=["PRICE_SOURCE"],
              action="compare_price_sources", arguments={"security_id": "SEC-1"}),
    ])
    run = StatefulInvestigationLoop(provider, _price_registry("SEC-1", 28.0), max_turns=2).run(
        {"exception": {"exception_id": "EXC-PRICE"}}, exception_id="EXC-PRICE",
    )
    hypothesis = run.report.hypotheses[0]
    assert hypothesis.hypothesis_id == "HYP-001"
    assert hypothesis.root_cause == "PRICE_EXCEPTION:SEC-1"
    assert hypothesis.confidence == .88
    assert hypothesis.rationale == "The current vendor prices differ materially."
    assert run.report.confidence == .88
    assert run.report.cited_evidence_ids
    assert run.telemetry.tool_calls == 1
    assert all("top_n" not in item for event in run.timeline if event["stage"] == "challenge_assessment"
               for item in event["details"].get("missing_evidence", []))
    assert any(event["stage"] == "stopping_decision" and event["details"]["accepted"]
               for event in run.timeline)


def test_identical_completed_action_replans_without_spending_investigation_turn():
    registry = _price_registry("SEC-1", 28.0)
    registry.register(ToolDefinition(
        "check_security_mapping", "Check identifier", {"type": "object",
            "properties": {"security_id": {"type": "string"}}, "required": ["security_id"]},
        lambda security_id: {"security_id": security_id, "valid": True},
    ))
    provider = SequencedProvider([
        _turn("PRICE_EXCEPTION:SEC-1", action="check_security_mapping", arguments={"security_id": "SEC-1"}),
        _turn("PRICE_EXCEPTION:SEC-1", action="check_security_mapping", arguments={"security_id": "SEC-1"}),
        _turn("PRICE_EXCEPTION:SEC-1", action="compare_price_sources", arguments={"security_id": "SEC-1"}),
        _turn("PRICE_EXCEPTION:SEC-1", confidence=.85, rationale="Current price records confirm the discrepancy.", stop=True),
    ])
    run = StatefulInvestigationLoop(provider, registry, max_turns=2).run(
        {"exception": {"exception_id": "EXC-DUP"}}, exception_id="EXC-DUP",
    )
    assert [step.tool_name for step in run.trace] == ["check_security_mapping", "compare_price_sources"]
    duplicate = next(event for event in run.timeline if event["details"].get("duplicate_action"))
    assert duplicate["details"]["investigation_turn_consumed"] is False
    assert duplicate["details"]["turn"] == 2
    assert provider.contexts[2]["blocked_duplicate_action"]["tool_name"] == "check_security_mapping"
    assert run.report.confidence == .85


def test_missing_price_records_escalate_without_reexecuting_the_same_check():
    registry = ToolRegistry([ToolDefinition(
        "compare_price_sources", "Compare current prices", {"type": "object",
            "properties": {"security_id": {"type": "string"}}, "required": ["security_id"]},
        lambda security_id: {"security_id": security_id, "found": False},
    )])
    provider = SequencedProvider([
        _turn("PRICE_EXCEPTION:SEC-1", action="compare_price_sources", arguments={"security_id": "SEC-1"}),
        *[_turn("PRICE_EXCEPTION:SEC-1", action="compare_price_sources",
                arguments={"security_id": "SEC-1"}) for _ in range(3)],
    ])
    run = StatefulInvestigationLoop(provider, registry, max_turns=3).run(
        {"exception": {"exception_id": "EXC-MISSING"}}, exception_id="EXC-MISSING",
    )
    assert run.telemetry.tool_calls == 1
    assert run.report.confidence <= .35
    assert run.report.human_review_required is True
    assert not run.report.cited_evidence_ids
    assert any(event["stage"] == "challenge_assessment" and event["details"]["escalation_required"]
               for event in run.timeline)
    assert any(event["stage"] == "stopping_decision" and not event["details"]["accepted"]
               for event in run.timeline)
