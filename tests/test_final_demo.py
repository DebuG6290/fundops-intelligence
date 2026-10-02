import copy
import json

import pytest

from src.data.demo_scenarios import create_final_demo_scenarios, get_final_demo_scenario
from src.demo.final_workbench import FinalDemoWorkbench
from src.agents.nav_agent import build_nav_tool_registry
from src.memory.cases import CaseMemory, HistoricalCase, seed_historical_cases
from src.review.models import HumanDecision


@pytest.mark.parametrize("scenario_id", ["NAV_PRICE", "NAV_TRANSACTION", "NAV_CORPORATE_ACTION", "NAV_INSUFFICIENT"])
def test_scenarios_are_reproducible_nav_exceptions(scenario_id):
    item = get_final_demo_scenario(scenario_id, seed=42)
    assert item.scenario_id == scenario_id
    assert item.scenario.exception_id
    assert item.expected_root_cause


def test_all_scenarios_have_material_nav_exception():
    workbench = FinalDemoWorkbench()
    for scenario_id in workbench.scenarios:
        assert workbench.investigate(scenario_id)["exception"]["exception_type"] == "NAV_DISCREPANCY"
        assert abs(workbench.investigate(scenario_id)["exception"]["variance_bps"]) >= 10


@pytest.mark.parametrize("scenario_id", ["NAV_PRICE", "NAV_TRANSACTION", "NAV_CORPORATE_ACTION", "NAV_INSUFFICIENT"])
def test_hidden_expected_cause_is_not_in_observable_result(scenario_id):
    result = FinalDemoWorkbench().investigate(scenario_id)
    assert "expected_root_cause" not in result
    assert "known_root_cause" not in result
    assert "culprit_security_id" not in json.dumps(result, default=str)


def test_cause_scenarios_have_corresponding_evidence_and_trace():
    workbench = FinalDemoWorkbench()
    expected = {
        "NAV_PRICE": "PRICE_SOURCE",
        "NAV_TRANSACTION": "TRANSACTION_RECORD",
        "NAV_CORPORATE_ACTION": "CORPORATE_ACTION_RECORD",
    }
    for scenario_id, source_type in expected.items():
        result = workbench.investigate(scenario_id)
        assert any(item["source_type"] == source_type for item in result["evidence"])
        assert {"search_historical_cases", "identify_top_contributors"}.issubset(
            {step["tool_name"] for step in result["trace"]}
        )


def test_insufficient_evidence_escalates_and_memory_is_analogy():
    result = FinalDemoWorkbench().investigate("NAV_INSUFFICIENT")
    assert result["status"] == "ESCALATE"
    assert result["resolution"]["decision"] == "INVESTIGATE_FURTHER"
    assert all(item["metadata"].get("evidence_role") == "analogy"
               for item in result["evidence"] if item["source_type"] == "HISTORICAL_CASE")


def test_accept_stores_path_evidence_and_becomes_retrievable():
    workbench = FinalDemoWorkbench()
    result = workbench.investigate("NAV_PRICE")
    record = workbench.submit_review(result, HumanDecision.ACCEPT, "Reviewed the price records.")
    case = workbench.accepted_case(record.review_id)
    assert case and case.human_validated
    assert case.investigation_path == tuple(step["tool_name"] for step in result["trace"])
    assert "price-source comparison" in case.useful_evidence
    assert case in workbench.search_memory("nav discrepancy price vendor corporate", "NAV_DISCREPANCY")
    follow_up = workbench.investigate("NAV_TRANSACTION")
    assert all(item["metadata"].get("case_id") != case.case_id for item in follow_up["evidence"]
               if item["source_type"] == "HISTORICAL_CASE")
    assert follow_up["memory_context"]["influencing_case_ids"] == []
    # The next case is determined by its own current transaction evidence.
    assert follow_up["probable_root_cause"].startswith("MISSING_TRANSACTION:")


@pytest.mark.parametrize("decision", [HumanDecision.REJECT, HumanDecision.INVESTIGATE_FURTHER])
def test_non_accept_decisions_never_promote_memory(decision):
    workbench = FinalDemoWorkbench()
    result = workbench.investigate("NAV_PRICE")
    before = len(workbench.memory.search("human-accepted nav", top_k=20))
    record = workbench.submit_review(result, decision, "Do not validate this case.")
    assert workbench.accepted_case(record.review_id) is None
    assert len(workbench.memory.search("human-accepted nav", top_k=20)) == before


def test_human_review_does_not_mutate_synthetic_financial_data():
    workbench = FinalDemoWorkbench()
    scenario = workbench.scenarios["NAV_PRICE"].scenario
    snapshot = copy.deepcopy((scenario.dataset.holdings, scenario.expected_prices, scenario.calculated_prices))
    result = workbench.investigate("NAV_PRICE")
    workbench.submit_review(result, HumanDecision.ACCEPT, "Reviewed.")
    for current, original in zip((scenario.dataset.holdings, scenario.expected_prices, scenario.calculated_prices), snapshot):
        assert current.equals(original)


def test_relevant_validated_memory_path_prioritizes_alternate_check():
    prior_case = HistoricalCase(
        case_id="CASE_005", exception_type="NAV_DISCREPANCY",
        title="Prior transaction investigation", symptoms=("transaction position break",),
        root_cause="MISSING_TRANSACTION", resolution="Review the transaction record.",
        human_validated=True,
        investigation_path=("check_transaction_activity", "check_corporate_actions"),
        useful_evidence=("transaction reconciliation",),
    )
    workbench = FinalDemoWorkbench(CaseMemory([*seed_historical_cases(), prior_case]))
    second = workbench.investigate("NAV_TRANSACTION")
    names = [step["tool_name"] for step in second["trace"]]
    assert names.index("check_transaction_activity") < names.index("check_corporate_actions")
    assert second["memory_context"]["influencing_case_ids"] == ["CASE_005"]


def test_nav_agent_registry_exposes_current_domain_investigation_tools():
    workbench = FinalDemoWorkbench()
    scenario = workbench.scenarios["NAV_TRANSACTION"].scenario
    names = {item["name"] for item in build_nav_tool_registry(scenario, workbench.memory).definitions()}
    assert names == {
        "identify_top_contributors", "compare_price_sources", "check_transaction_activity",
        "check_corporate_actions", "check_security_mapping", "check_fx_context",
        "search_historical_cases",
    }


def test_demo_workbench_scenarios_do_not_leak_expected_labels_into_memory():
    scenarios = create_final_demo_scenarios()
    assert all(item.expected_root_cause for item in scenarios.values())
    workbench = FinalDemoWorkbench(CaseMemory(seed_historical_cases()))
    result = workbench.investigate("NAV_PRICE")
    assert "expected_root_cause" not in result

