from src.agents.corporate_action_agent import CorporateActionInvestigationAgent
from src.agents.schemas import InvestigationReport
from src.agents.transaction_agent import TransactionInvestigationAgent
from src.data.scenarios_extra import (
    create_corporate_action_scenario,
    create_transaction_mismatch_scenario,
)
from src.memory.cases import CaseMemory, seed_historical_cases


class FakeSpecialistProvider:
    def __init__(self, tool_name: str, arguments: dict):
        self.tool_name = tool_name
        self.arguments = arguments
        self.calls = 0

    def create_response(self, system_instructions, context, tools):
        self.calls += 1
        return type("Response", (), {
            "id": "resp-1",
            "output": [type("Call", (), {
                "type": "function_call",
                "name": self.tool_name,
                "arguments": __import__("json").dumps(self.arguments),
                "call_id": "call-1",
            })()],
        })()

    def continue_response(self, previous_response_id, tool_outputs, system_instructions):
        return type("Response", (), {
            "id": "resp-2",
            "output_text": __import__("json").dumps({
                "probable_root_cause": "SPECIALIST_ROOT_CAUSE",
                "confidence": 0.91,
                "observations": ["Specialist tool executed."],
                "supporting_evidence": ["Tool result returned matching evidence."],
                "counter_evidence": [],
                "recommended_next_step": "Human review of the operational record.",
                "human_review_required": True,
            }),
            "output": [],
        })()


def test_transaction_specialist_executes_transaction_tool():
    scenario = create_transaction_mismatch_scenario()
    provider = FakeSpecialistProvider("find_transaction_mismatches", {})
    run = TransactionInvestigationAgent(
        CaseMemory(seed_historical_cases()), provider=provider
    ).investigate(scenario)

    assert provider.calls == 1
    assert isinstance(run.report, InvestigationReport)
    assert run.trace[0].tool_name == "find_transaction_mismatches"


def test_corporate_action_specialist_executes_action_tool():
    scenario = create_corporate_action_scenario()
    provider = FakeSpecialistProvider("find_effective_corporate_actions", {})
    run = CorporateActionInvestigationAgent(
        CaseMemory(seed_historical_cases()), provider=provider
    ).investigate(scenario)

    assert provider.calls == 1
    assert isinstance(run.report, InvestigationReport)
    assert run.trace[0].tool_name == "find_effective_corporate_actions"
