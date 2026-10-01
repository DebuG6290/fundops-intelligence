import json

import pandas as pd
import pytest
from pydantic import ValidationError

from src.agents.evidence_challenge import EvidenceChallengeAgent
from src.agents.evidence import evidence_from_tool_result
from src.agents.state import InvestigationState
from src.agents.workflow import InvestigationWorkflow
from src.data.scenarios import create_price_exception_scenario
from src.data.scenarios_extra import (
    create_corporate_action_scenario,
    create_transaction_mismatch_scenario,
)
from src.memory.cases import CaseMemory, seed_historical_cases
from src.models.evidence import EvidenceItem, EvidenceSourceType
from src.review.models import (
    HumanDecision,
    HumanReviewRecord,
    HumanReviewSubmission,
)
from src.review.service import HumanReviewService


class ToolProvider:
    def __init__(self, tool_name, root_cause="SPECIALIST_ROOT_CAUSE"):
        self.tool_name = tool_name
        self.root_cause = root_cause

    def create_response(self, system_instructions, context, tools):
        return type("Response", (), {
            "id": "response-1",
            "output": [type("Call", (), {
                "type": "function_call",
                "name": self.tool_name,
                "arguments": "{}",
                "call_id": "call-1",
            })()],
        })()

    def continue_response(self, previous_response_id, tool_outputs, system_instructions):
        return type("Response", (), {
            "id": "response-2",
            "output": [],
            "output_text": json.dumps({
                "probable_root_cause": self.root_cause,
                "confidence": 0.9,
                "observations": ["Model observation only."],
                "supporting_evidence": ["Model generated assertion."],
                "counter_evidence": [],
                "recommended_next_step": "Human review.",
                "human_review_required": False,
            }),
        })()


def test_evidence_item_has_typed_source_relationship_and_serialization():
    item = EvidenceItem(
        evidence_id="EVIDENCE-1",
        source_type=EvidenceSourceType.TRANSACTION_RECORD,
        source_name="TX-104",
        claim="Expected quantity 1000; actual quantity 650.",
        supports="exception:EXC_TXN_1",
        metadata={"expected_quantity": 1000, "actual_quantity": 650},
    )

    serialized = item.model_dump(mode="json")
    assert item.direction == "SUPPORTS"
    assert serialized["source_type"] == "TRANSACTION_RECORD"
    assert serialized["source_name"] == "TX-104"
    assert serialized["supports"] == "exception:EXC_TXN_1"
    assert serialized["metadata"]["actual_quantity"] == 650


def test_counter_evidence_is_explicit_and_not_also_supporting():
    item = EvidenceItem(
        source_type=EvidenceSourceType.PRICE_SOURCE,
        source_name="SEC-1",
        claim="Price feeds agree within threshold.",
        contradicts="PRICE_EXCEPTION:SEC-1",
    )
    assert item.direction == "CONTRADICTS"
    with pytest.raises(ValidationError):
        EvidenceItem(
            source_type=EvidenceSourceType.PRICE_SOURCE,
            source_name="SEC-1",
            claim="Ambiguous relationship.",
            supports="PRICE_EXCEPTION:SEC-1",
            contradicts="PRICE_EXCEPTION:SEC-1",
        )


def test_evidence_rejects_missing_required_fields_and_bad_reliability():
    with pytest.raises(ValidationError):
        EvidenceItem(source_type=EvidenceSourceType.TOOL_RESULT, source_name="tool")
    with pytest.raises(ValidationError):
        EvidenceItem(
            source_type=EvidenceSourceType.TOOL_RESULT,
            source_name="tool",
            claim="Returned output.",
            reliability=1.2,
        )


def test_nav_workflow_evidence_uses_actual_deterministic_outputs():
    scenario = create_price_exception_scenario(seed=42)
    result = InvestigationWorkflow(
        CaseMemory(seed_historical_cases())
    ).run_nav(scenario)
    evidence = [EvidenceItem.model_validate(item) for item in result["evidence"]]

    variance = next(item for item in evidence if item.source_name == "calculate_nav_variance")
    contributor = next(
        item for item in evidence if item.source_name == "identify_top_contributors"
    )
    price = next(item for item in evidence if item.source_type == EvidenceSourceType.PRICE_SOURCE)
    price_observation = next(
        item["value"] for item in result["observations"]
        if item["name"] == "price_source_check"
    )
    assert result["human_review_required"] is True
    assert variance.metadata["expected_nav"] == result["exception"]["expected_nav"]
    assert contributor.metadata["security_id"] == scenario.culprit_security_id
    assert contributor.metadata["contribution_pct"] > 0
    assert price.metadata["difference_pct"] == price_observation["difference_pct"]
    assert price.supports == "PRICE_EXCEPTION"
    assert any(item.source_name == "get_fund_snapshot" for item in evidence)


def test_transaction_workflow_evidence_contains_actual_mismatch_not_model_claim():
    scenario = create_transaction_mismatch_scenario(seed=42)
    transaction_id = scenario.actual_transactions.loc[
        scenario.actual_transactions["status"].eq("MISMATCH"), "transaction_id"
    ].iloc[0]
    workflow = InvestigationWorkflow(CaseMemory(seed_historical_cases()))
    result = workflow.run_transaction(
        scenario,
        ToolProvider(
            "find_transaction_mismatches",
            f"TRANSACTION_QUANTITY_MISMATCH:{transaction_id}",
        ),
    )
    evidence = [EvidenceItem.model_validate(item) for item in result["evidence"]]
    mismatch = next(
        item for item in evidence
        if item.source_type == EvidenceSourceType.TRANSACTION_RECORD
    )
    actual_row = result["trace"][0]["result"][0]

    assert mismatch.metadata["expected_quantity"] == actual_row["expected_quantity"]
    assert mismatch.metadata["actual_quantity"] == actual_row["actual_quantity"]
    assert mismatch.metadata["quantity_difference"] == actual_row["quantity_difference"]
    assert mismatch.supports == f"TRANSACTION_QUANTITY_MISMATCH:{actual_row['transaction_id']}"
    assert "Model generated assertion." not in [item.claim for item in evidence]
    assert result["report"]["human_review_required"] is True
    assert result["report_evidence_is_narrative"] is True


def test_corporate_action_workflow_evidence_contains_returned_record():
    scenario = create_corporate_action_scenario(seed=42)
    result = InvestigationWorkflow(
        CaseMemory(seed_historical_cases())
    ).run_corporate_action(
        scenario,
        ToolProvider(
            "find_effective_corporate_actions",
            f"CORPORATE_ACTION:{scenario.corporate_actions.iloc[0]['security_id']}",
        ),
    )
    evidence = [EvidenceItem.model_validate(item) for item in result["evidence"]]
    action = next(
        item for item in evidence
        if item.source_type == EvidenceSourceType.CORPORATE_ACTION_RECORD
    )

    assert action.metadata["security_id"] == scenario.corporate_actions.iloc[0]["security_id"]
    assert action.metadata["action_type"] == "STOCK_SPLIT"
    assert action.metadata["ratio"] == 2.0
    assert action.supports == f"CORPORATE_ACTION:{action.metadata['security_id']}"


def test_primary_evidence_relevant_to_exception_but_not_hypothesis_escalates():
    scenario = create_transaction_mismatch_scenario(seed=42)
    result = InvestigationWorkflow(
        CaseMemory(seed_historical_cases())
    ).run_transaction(
        scenario,
        ToolProvider("find_transaction_mismatches", "UNSUPPORTED_SETTLEMENT_CAUSE"),
    )

    assert result["evidence"]
    assert result["challenge"]["insufficient_evidence"] is True
    assert result["status"] == "ESCALATE"
    assert result["resolution"]["decision"] == "INVESTIGATE_FURTHER"


def test_historical_case_records_are_analogies_not_primary_evidence():
    retrieved = [case.to_dict() for case in seed_historical_cases()[:1]]
    evidence = evidence_from_tool_result(
        "search_historical_cases", retrieved, "EXC_NEW"
    )

    assert evidence
    assert all(item.source_type == EvidenceSourceType.HISTORICAL_CASE for item in evidence)
    assert all(item.metadata["evidence_role"] == "analogy" for item in evidence)
    assert all(item.supports is None for item in evidence)
    assert all(item.contradicts is None for item in evidence)


def test_historical_analogy_alone_does_not_satisfy_primary_evidence_check():
    state = InvestigationState(
        exception={"exception_type": "TRANSACTION_MISMATCH"}
    )
    state.add_hypothesis("PENDING_SETTLEMENT", "", 0.9)
    state.add_evidence(EvidenceItem(
        source_type=EvidenceSourceType.HISTORICAL_CASE,
        source_name="CASE_004",
        claim="A historical transaction had pending settlement.",
        metadata={"evidence_role": "analogy"},
    ))

    challenge = EvidenceChallengeAgent().review(state)
    assert challenge.insufficient_evidence is True
    assert challenge.final_confidence <= 0.35


def _review_evidence():
    return [
        EvidenceItem(
            evidence_id="SUP-1",
            source_type=EvidenceSourceType.TRANSACTION_RECORD,
            source_name="TX-1",
            claim="Mismatch exists.",
            supports="exception:EXC-1",
        ),
        EvidenceItem(
            evidence_id="CTR-1",
            source_type=EvidenceSourceType.TOOL_RESULT,
            source_name="reconciliation",
            claim="A source record challenges the hypothesis.",
            contradicts="ROOT-1",
        ),
    ]


def _submission(decision):
    return HumanReviewSubmission(
        exception_id="EXC-1",
        agent_root_cause="ROOT-1",
        agent_confidence=0.35,
        agent_recommendation="INVESTIGATE_FURTHER: verify source records.",
        supporting_evidence_references=("SUP-1",),
        counter_evidence_references=("CTR-1",),
        human_decision=decision,
        reviewer_reason="Reviewed the listed evidence.",
    )


@pytest.mark.parametrize(
    "decision",
    [HumanDecision.ACCEPT, HumanDecision.REJECT, HumanDecision.INVESTIGATE_FURTHER],
)
def test_each_human_decision_creates_required_audit_record(decision):
    service = HumanReviewService()
    record = service.submit_review(_submission(decision), _review_evidence())

    assert record.human_decision is decision
    assert record.human_review_required is True
    assert record.agent_root_cause == "ROOT-1"
    assert record.agent_recommendation == "INVESTIGATE_FURTHER: verify source records."
    assert record.supporting_evidence_references == ("SUP-1",)
    assert record.counter_evidence_references == ("CTR-1",)
    assert record.timestamp.tzinfo is not None
    assert service.get_review(record.review_id) == record


def test_human_review_rejects_invalid_decision_or_missing_fields():
    payload = _submission(HumanDecision.ACCEPT).model_dump()
    payload["human_decision"] = "MAYBE"
    with pytest.raises(ValidationError):
        HumanReviewSubmission.model_validate(payload)
    with pytest.raises(ValidationError):
        HumanReviewSubmission(
            exception_id=" ",
            agent_root_cause="ROOT-1",
            agent_confidence=0.8,
            human_decision=HumanDecision.ACCEPT,
            reviewer_reason="Reviewed.",
        )
    payload = _submission(HumanDecision.ACCEPT).model_dump()
    del payload["human_decision"]
    with pytest.raises(ValidationError):
        HumanReviewSubmission.model_validate(payload)
    payload = _submission(HumanDecision.ACCEPT).model_dump()
    payload["human_review_required"] = False
    with pytest.raises(ValidationError):
        HumanReviewSubmission.model_validate(payload)
    payload = _submission(HumanDecision.ACCEPT).model_dump()
    payload["agent_root_cause"] = " "
    with pytest.raises(ValidationError):
        HumanReviewSubmission.model_validate(payload)


def test_review_references_must_exist_and_match_evidence_direction():
    service = HumanReviewService()
    payload = _submission(HumanDecision.ACCEPT).model_dump()
    payload["supporting_evidence_references"] = ("NOT-FOUND",)

    with pytest.raises(ValueError, match="Unknown or non-supporting"):
        service.submit_review(payload, _review_evidence())


def test_review_history_is_append_only_and_ordered():
    service = HumanReviewService()
    first = service.submit_review(_submission(HumanDecision.ACCEPT), _review_evidence())
    second = service.submit_review(
        _submission(HumanDecision.INVESTIGATE_FURTHER), _review_evidence()
    )

    history = service.get_review_history("EXC-1")
    assert history == (first, second)
    assert isinstance(history, tuple)
    assert service.get_review_history("UNKNOWN") == ()
    with pytest.raises(ValidationError):
        first.human_decision = HumanDecision.REJECT
    record_payload = first.model_dump()
    record_payload["timestamp"] = record_payload["timestamp"].replace(tzinfo=None)
    with pytest.raises(ValidationError):
        HumanReviewRecord.model_validate(record_payload)


def test_human_review_service_does_not_mutate_financial_records():
    scenario = create_transaction_mismatch_scenario(seed=42)
    expected_before = scenario.expected_transactions.copy(deep=True)
    actual_before = scenario.actual_transactions.copy(deep=True)
    HumanReviewService().submit_review(_submission(HumanDecision.REJECT), _review_evidence())

    pd.testing.assert_frame_equal(scenario.expected_transactions, expected_before)
    pd.testing.assert_frame_equal(scenario.actual_transactions, actual_before)


def test_workflow_records_explicit_human_decision_without_auto_acceptance():
    workflow = InvestigationWorkflow(CaseMemory(seed_historical_cases()))
    scenario = create_transaction_mismatch_scenario(seed=42)
    transaction_id = scenario.actual_transactions.loc[
        scenario.actual_transactions["status"].eq("MISMATCH"), "transaction_id"
    ].iloc[0]
    result = workflow.run_transaction(
        scenario,
        ToolProvider(
            "find_transaction_mismatches",
            f"TRANSACTION_QUANTITY_MISMATCH:{transaction_id}",
        ),
    )
    assert "human_decision" not in result

    record = workflow.submit_human_review(
        result, HumanDecision.INVESTIGATE_FURTHER, "Need settlement confirmation."
    )
    assert record.human_decision is HumanDecision.INVESTIGATE_FURTHER
    assert record.agent_recommendation.startswith("REVIEW_RECOMMENDATION:")
    assert workflow.review_service.get_review_history(record.exception_id) == (record,)
