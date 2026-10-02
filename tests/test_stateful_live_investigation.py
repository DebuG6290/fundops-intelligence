from __future__ import annotations

import json
import re
from types import SimpleNamespace
from pathlib import Path

from src.agents.stateful_investigation import StatefulInvestigationLoop
from src.agents.evidence_challenge import EvidenceChallengeAgent
from src.agents.state import InvestigationState
from src.agents.tool_registry import ToolRegistry
from src.data.demo_scenarios import get_final_demo_scenario
from src.demo.final_workbench import FinalDemoWorkbench
from src.demo.timeline_view import render_investigation_timeline
from src.llm.sarvam_provider import SarvamProvider
from src.memory.cases import CaseMemory, HistoricalCase
from src.models.evidence import EvidenceItem, EvidenceSourceType
from src.review.models import HumanDecision
from src.agents.nav_agent import build_nav_tool_registry


def _plan(root, action=None, args=None, *, stop=False, rationale="Current records determine the next check."):
    return {
        "hypotheses": [{"hypothesis_id": "HYP-001", "root_cause": root,
                        "rationale": "Test this cause against current records.", "confidence": .88,
                        "required_evidence": [], "uncertainty": "Other causes need checking."}],
        "next_action": action, "action_arguments": args or {}, "decision_rationale": rationale,
        "stop": stop, "stopping_rationale": "Current primary records support the leader." if stop else "",
        "recommended_next_step": "Review the cited operational records with a human.",
    }


class EvidenceDrivenProvider:
    """Fake planner chooses its path from returned evidence, never scenario labels."""

    def __init__(self):
        self.contexts = []
        self.last_call = {}

    def create_response(self, *, context, **_):
        self.contexts.append(context)
        evidence = context["current_evidence"]
        price = next((item for item in evidence if item["source_type"] == "PRICE_SOURCE"), None)
        transaction = next((item for item in evidence if item["source_type"] == "TRANSACTION_RECORD"), None)
        contributor = next((item for item in evidence if item["source_name"] == "identify_top_contributors"), None)
        if transaction and transaction["supports"]:
            payload = _plan(transaction["supports"], stop=True,
                            rationale="The current transaction reconciliation supports this cause.")
        elif price and price["supports"]:
            security = price["source_name"]
            payload = _plan(f"PRICE_EXCEPTION:{security}", stop=True,
                            rationale="The current price-source records support this cause.")
        elif price and price["contradicts"]:
            security = price["source_name"]
            payload = _plan("PRICE_EXCEPTION", "check_transaction_activity", {"security_id": security},
                            rationale="The price check contradicts pricing, so test transaction records.")
        elif contributor:
            security = re.search(r"Security (\S+)", contributor["claim"]).group(1)
            payload = _plan("PRICE_EXCEPTION", "compare_price_sources", {"security_id": security},
                            rationale="Check the largest contributor against current price sources.")
        else:
            payload = _plan("PRICE_EXCEPTION", "identify_top_contributors", {"top_n": 3},
                            rationale="Locate the material NAV contributor before testing causes.")
        self.last_call = {"provider": "fake-sarvam", "model": "test-planner", "request_id": f"req-{len(self.contexts)}",
                          "input_tokens": 10, "output_tokens": 6}
        return SimpleNamespace(output_text=json.dumps(payload), output=[], id=f"req-{len(self.contexts)}")


def test_live_tool_path_changes_with_current_evidence_and_replans_on_contradiction():
    price_provider = EvidenceDrivenProvider()
    txn_provider = EvidenceDrivenProvider()
    price_result = FinalDemoWorkbench().investigate("NAV_PRICE", price_provider)
    txn_result = FinalDemoWorkbench().investigate("NAV_TRANSACTION", txn_provider)
    price_tools = [step["tool_name"] for step in price_result["trace"]]
    txn_tools = [step["tool_name"] for step in txn_result["trace"]]
    assert price_tools == ["identify_top_contributors", "compare_price_sources"]
    assert txn_tools == ["identify_top_contributors", "compare_price_sources", "check_transaction_activity"]
    assert price_result["probable_root_cause"].startswith("PRICE_EXCEPTION:")
    assert txn_result["probable_root_cause"].startswith("MISSING_TRANSACTION:")
    assert any(event["stage"] == "replan" and "conflicting evidence" in event["rationale"].lower()
               for event in txn_result["timeline"])
    assert any(event["stage"] == "challenge_assessment" and event["details"].get("contradiction_found")
               for event in txn_result["timeline"])
    stages = [event["stage"] for event in price_result["timeline"]]
    assert stages.index("challenge_assessment") < stages.index("recommendation")
    assert price_result["resolution"]["requires_human_approval"] is True
    assert price_result["telemetry"]["tool_calls"] == 2
    assert all("known_root_cause" not in json.dumps(context["initial_context"])
               and "culprit_security_id" not in json.dumps(context["initial_context"])
               for context in price_provider.contexts + txn_provider.contexts)


class MemoryFirstProvider(EvidenceDrivenProvider):
    def create_response(self, *, context, **kwargs):
        self.contexts.append(context)
        evidence = context["current_evidence"]
        analogies = context["historical_analogies"]
        transaction = next((item for item in evidence if item["source_type"] == "TRANSACTION_RECORD"), None)
        if transaction and transaction["supports"]:
            payload = _plan(transaction["supports"], stop=True,
                            rationale="The current transaction record, not memory, supports the cause.")
        elif analogies:
            payload = _plan("MISSING_TRANSACTION", "check_transaction_activity", {"security_id": None},
                            rationale="A validated prior path prioritizes transaction reconciliation.")
        else:
            payload = _plan("MISSING_TRANSACTION", "search_historical_cases",
                            {"query": "transaction position records", "exception_type": "NAV_DISCREPANCY"},
                            rationale="Retrieve a validated prior investigation path as an analogy.")
        self.last_call = {"provider": "fake-sarvam", "model": "test-planner"}
        return SimpleNamespace(output_text=json.dumps(payload), output=[], id="memory-step")


def test_memory_changes_priority_without_becoming_current_case_evidence():
    historical = HistoricalCase(
        case_id="CASE_900", exception_type="NAV_DISCREPANCY", title="Neutral prior case",
        symptoms=("transaction position records",), root_cause="STALE_PRICE",
        resolution="Prior reviewer checked records.", human_validated=True,
        investigation_path=("check_transaction_activity",), useful_evidence=("transaction reconciliation",),
    )
    unvalidated = HistoricalCase(
        case_id="CASE_901", exception_type="NAV_DISCREPANCY", title="Unvalidated decoy",
        symptoms=("transaction position records",), root_cause="MISSING_TRANSACTION",
        resolution="Not reviewed.", human_validated=False,
        investigation_path=("compare_price_sources",),
    )
    provider = MemoryFirstProvider()
    result = FinalDemoWorkbench(CaseMemory([unvalidated, historical])).investigate("NAV_TRANSACTION", provider)
    assert [step["tool_name"] for step in result["trace"]] == ["search_historical_cases", "check_transaction_activity"]
    assert result["probable_root_cause"].startswith("MISSING_TRANSACTION:")
    assert all("root_cause" not in json.dumps(context["historical_analogies"])
               for context in provider.contexts)
    assert any(event["stage"] == "memory_retrieval" for event in result["timeline"])
    assert {case["case_id"] for context in provider.contexts
            for case in context["historical_analogies"]} == {"CASE_900"}
    assert all(evidence["source_type"] != "HISTORICAL_CASE"
               for context in provider.contexts for evidence in context["current_evidence"])
    assert result["challenge"]["supporting_evidence_ids"]


class StopWithoutEvidence:
    def __init__(self):
        self.calls = 0

    def create_response(self, **_):
        self.calls += 1
        return SimpleNamespace(output_text=json.dumps(_plan(
            "PRICE_EXCEPTION", stop=True, rationale="Try to stop without current evidence."
        )), output=[], id=str(self.calls))


def test_bounded_loop_escalates_when_model_repeatedly_stops_without_evidence():
    provider = StopWithoutEvidence()
    run = StatefulInvestigationLoop(provider, ToolRegistry(), max_turns=3).run(
        {"exception": {"exception_id": "EXC-EMPTY"}}, exception_id="EXC-EMPTY"
    )
    assert provider.calls == 3
    assert run.report.confidence <= .35
    assert run.report.human_review_required is True
    assert any(event["stage"] == "stopping_decision" and event["details"]["accepted"]
               for event in run.timeline)


def test_only_accept_writes_live_result_to_validated_memory():
    for decision in (HumanDecision.REJECT, HumanDecision.INVESTIGATE_FURTHER, HumanDecision.ACCEPT):
        workbench = FinalDemoWorkbench()
        original_holdings = workbench.scenarios["NAV_PRICE"].scenario.dataset.holdings.copy(deep=True)
        result = workbench.investigate("NAV_PRICE", EvidenceDrivenProvider())
        record = workbench.submit_review(result, decision, "Reviewed current source evidence.")
        assert workbench.scenarios["NAV_PRICE"].scenario.dataset.holdings.equals(original_holdings)
        stages = [event["stage"] for event in result["timeline"]]
        assert "human_decision" in stages
        assert result["human_review_required"] is True
        if decision is HumanDecision.ACCEPT:
            assert workbench.accepted_case(record.review_id).human_validated
            assert "memory_write_back" in stages
        else:
            assert workbench.accepted_case(record.review_id) is None
            assert "memory_write_back" not in stages


def test_timeline_view_shows_hypothesis_evolution_and_full_event_details():
    class FakeUI:
        def __init__(self):
            self.text = []
            self.details = []

        def markdown(self, value): self.text.append(value)
        def write(self, value): self.text.append(value)
        def caption(self, value): self.text.append(value)
        def json(self, value): self.details.append(value)
        def expander(self, _): return self
        def __enter__(self): return self
        def __exit__(self, *_): return False

    result = FinalDemoWorkbench().investigate("NAV_TRANSACTION", EvidenceDrivenProvider())
    ui = FakeUI()
    render_investigation_timeline(ui, result["timeline"], str)
    displayed = "\n".join(ui.text)
    assert "Hypotheses updated" in displayed
    assert "Investigation re-planned" in displayed
    assert "Evidence challenged" in displayed
    assert "Supporting:" in displayed and "Contradicting:" in displayed and "Missing:" in displayed
    assert len(ui.details) == len(result["timeline"])
    assert all("raw_outputs" not in detail for detail in ui.details)


def test_challenger_names_missing_required_evidence_even_with_one_supporting_record():
    state = InvestigationState(exception={"exception_id": "EXC-1"})
    state.add_hypothesis("PRICE_EXCEPTION", "Current price differs.", .9,
                         required_evidence=["independent vendor timestamp"])
    state.add_evidence(EvidenceItem(
        source_type=EvidenceSourceType.PRICE_SOURCE, source_name="primary_vendor",
        claim="Primary vendor price differs.", supports="PRICE_EXCEPTION",
    ))
    result = EvidenceChallengeAgent().review(state)
    assert result.supporting_evidence_ids == (state.evidence[0].evidence_id,)
    assert "independent vendor timestamp" in result.missing_evidence
    assert result.insufficient_evidence is True


def test_stateful_path_works_with_mocked_sarvam_v1_adapter():
    scenario = get_final_demo_scenario("NAV_PRICE").scenario
    security = scenario.culprit_security_id
    responses = iter([
        SimpleNamespace(id="req-1", choices=[SimpleNamespace(message=SimpleNamespace(
            content=json.dumps(_plan(f"PRICE_EXCEPTION:{security}", "compare_price_sources",
                                     {"security_id": security})), tool_calls=[]))],
            usage=SimpleNamespace(prompt_tokens=12, completion_tokens=8)),
        SimpleNamespace(id="req-2", choices=[SimpleNamespace(message=SimpleNamespace(
            content=json.dumps(_plan(f"PRICE_EXCEPTION:{security}", stop=True)), tool_calls=[]))],
            usage=SimpleNamespace(prompt_tokens=14, completion_tokens=7)),
    ])
    requests = []

    def complete(**kwargs):
        requests.append(kwargs)
        return next(responses)

    client = SimpleNamespace(chat=SimpleNamespace(completions=complete))
    provider = SarvamProvider(client=client)
    run = StatefulInvestigationLoop(provider, build_nav_tool_registry(scenario, CaseMemory()), max_turns=3).run(
        {"exception": {"exception_id": scenario.exception_id}}, exception_id=scenario.exception_id
    )
    assert run.report.probable_root_cause == f"PRICE_EXCEPTION:{security}"
    assert run.telemetry.input_tokens == 26 and run.telemetry.output_tokens == 15
    assert run.telemetry.request_ids == ["req-1", "req-2"]
    assert all(request["response_format"] == {"type": "json_object"} for request in requests)


def test_streamlit_live_mode_renders_stateful_timeline(monkeypatch):
    from streamlit.testing.v1 import AppTest
    import src.llm.sarvam_provider as provider_module

    monkeypatch.setenv("SARVAM_API_KEY", "unit-test-only")
    monkeypatch.setattr(provider_module, "SarvamProvider", EvidenceDrivenProvider)
    app_path = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"
    app = AppTest.from_file(str(app_path)).run(timeout=30)
    assert not app.exception
    app.selectbox[1].set_value("Live Sarvam").run(timeout=30)
    app.button[0].click().run(timeout=30)
    assert not app.exception
    assert app.session_state["investigation_result"]["investigator_mode"] == "specialist_agent"
    text = "\n".join(item.value for item in app.markdown)
    assert "Initial hypotheses" in text and "Evidence challenged" in text

