from src.agents.router import InvestigationRouter
from src.agents.evidence_challenge import EvidenceChallengeAgent, apply_challenge
from src.agents.schemas import InvestigationReport
from src.agents.state import InvestigationState
from src.agents.workflow import InvestigationWorkflow
from src.agents.resolution import ResolutionAgent
from src.agents.workflow import _state_from_specialist_report
from src.data.scenarios import create_price_exception_scenario
from src.data.scenarios_extra import (
    create_corporate_action_scenario,
    create_transaction_mismatch_scenario,
)
from src.memory.cases import CaseMemory, seed_historical_cases


class FakeProvider:
    def __init__(self, tool_name):
        self.tool_name = tool_name

    def create_response(self, system_instructions, context, tools):
        import json

        return type("Response", (), {
            "id": "resp-1",
            "output": [type("Call", (), {
                "type": "function_call",
                "name": self.tool_name,
                "arguments": json.dumps({}),
                "call_id": "call-1",
            })()],
        })()

    def continue_response(self, previous_response_id, tool_outputs, system_instructions):
        import json

        return type("Response", (), {
            "id": "resp-2",
            "output": [],
            "output_text": json.dumps({
                "probable_root_cause": "TEST_ROOT_CAUSE",
                "confidence": 0.9,
                "observations": ["Tool evidence inspected."],
                "supporting_evidence": ["Specialist tool output."],
                "counter_evidence": [],
                "recommended_next_step": "Human review.",
                "human_review_required": True,
            }),
        })()


def test_router_covers_all_supported_exception_types():
    router = InvestigationRouter()
    assert router.route("NAV_DISCREPANCY") == "NAV_INVESTIGATOR"
    assert router.route("TRANSACTION_MISMATCH") == "TRANSACTION_INVESTIGATOR"
    assert router.route("CORPORATE_ACTION") == "CORPORATE_ACTION_INVESTIGATOR"


def test_transaction_route_runs_specialist():
    workflow = InvestigationWorkflow(CaseMemory(seed_historical_cases()))
    result = workflow.run_transaction(
        create_transaction_mismatch_scenario(),
        FakeProvider("find_transaction_mismatches"),
    )

    assert result["route"] == "TRANSACTION_INVESTIGATOR"
    assert result["report"]["human_review_required"] is True
    assert result["trace"][0]["tool_name"] == "find_transaction_mismatches"
    assert result["known_root_cause"] == create_transaction_mismatch_scenario(
        seed=42
    ).known_root_cause
    assert result["challenge"]["challenged"] is True
    assert result["resolution"]["requires_human_approval"] is True
    assert result["status"] == "READY_FOR_HUMAN_REVIEW"


def test_corporate_action_route_runs_specialist():
    workflow = InvestigationWorkflow(CaseMemory(seed_historical_cases()))
    result = workflow.run_corporate_action(
        create_corporate_action_scenario(),
        FakeProvider("find_effective_corporate_actions"),
    )

    assert result["route"] == "CORPORATE_ACTION_INVESTIGATOR"
    assert result["report"]["human_review_required"] is True
    assert result["trace"][0]["tool_name"] == "find_effective_corporate_actions"
    assert result["known_root_cause"] == create_corporate_action_scenario(
        seed=42
    ).known_root_cause
    assert result["challenge"]["challenged"] is True
    assert result["resolution"]["requires_human_approval"] is True
    assert result["status"] == "READY_FOR_HUMAN_REVIEW"


def _specialist_state(specialist_type, counter_evidence=None):
    if specialist_type == "transaction":
        scenario = create_transaction_mismatch_scenario(seed=42)
    else:
        scenario = create_corporate_action_scenario(seed=42)
    report = InvestigationReport(
        probable_root_cause="SPECIALIST_ROOT_CAUSE",
        confidence=0.9,
        observations=["Investigated the exception."],
        supporting_evidence=["Specialist evidence."],
        counter_evidence=counter_evidence or [],
        recommended_next_step="Human review.",
    )
    return _state_from_specialist_report(scenario, report)


def test_transaction_ambiguous_report_escalates():
    state = _specialist_state("transaction")
    state.add_hypothesis("ALTERNATIVE", "Also plausible.", 0.82)
    challenge = EvidenceChallengeAgent().review(state)
    apply_challenge(state, challenge)
    resolution = ResolutionAgent().resolve(state)

    assert challenge.ambiguity_found is True
    assert state.status == "ESCALATE"
    assert resolution.decision == "INVESTIGATE_FURTHER"
    assert resolution.requires_human_approval is True


def test_corporate_action_ambiguous_report_escalates():
    state = _specialist_state("corporate_action")
    state.add_hypothesis("ALTERNATIVE", "Also plausible.", 0.82)
    challenge = EvidenceChallengeAgent().review(state)
    apply_challenge(state, challenge)
    resolution = ResolutionAgent().resolve(state)

    assert challenge.ambiguity_found is True
    assert state.status == "ESCALATE"
    assert resolution.decision == "INVESTIGATE_FURTHER"
    assert resolution.requires_human_approval is True


def test_transaction_conflicting_evidence_escalates():
    state = _specialist_state("transaction", ["Contradiction"])
    challenge = EvidenceChallengeAgent().review(state)
    apply_challenge(state, challenge)
    resolution = ResolutionAgent().resolve(state)

    assert challenge.contradiction_found is True
    assert state.status == "ESCALATE"
    assert resolution.decision == "INVESTIGATE_FURTHER"
    assert resolution.requires_human_approval is True


def test_corporate_action_conflicting_evidence_escalates():
    state = _specialist_state("corporate_action", ["Contradiction"])
    challenge = EvidenceChallengeAgent().review(state)
    apply_challenge(state, challenge)
    resolution = ResolutionAgent().resolve(state)

    assert challenge.contradiction_found is True
    assert state.status == "ESCALATE"
    assert resolution.decision == "INVESTIGATE_FURTHER"
    assert resolution.requires_human_approval is True


def test_missing_specialist_supporting_evidence_escalates():
    state = _specialist_state("transaction")
    state.evidence.clear()
    challenge = EvidenceChallengeAgent().review(state)
    apply_challenge(state, challenge)

    assert state.status == "ESCALATE"
    assert "no supporting evidence" in challenge.recommendation
    assert ResolutionAgent().resolve(state).decision == "INVESTIGATE_FURTHER"


def test_missing_specialist_hypothesis_escalates():
    state = InvestigationState(exception={"exception_id": "EXC_EMPTY"})
    challenge = EvidenceChallengeAgent().review(state)
    apply_challenge(state, challenge)

    assert state.status == "ESCALATE"
    assert ResolutionAgent().resolve(state).decision == "INVESTIGATE_FURTHER"


def test_nav_baseline_route_remains_available():
    result = InvestigationWorkflow(
        CaseMemory(seed_historical_cases())
    ).run_nav(create_price_exception_scenario())

    assert result["route"] == "NAV_INVESTIGATOR"
    assert result["resolution"]["requires_human_approval"] is True
