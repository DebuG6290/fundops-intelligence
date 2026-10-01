import json
from types import SimpleNamespace

from src.agents.nav_agent import NavInvestigationAgent
from src.data.scenarios import create_price_exception_scenario
from src.memory.cases import CaseMemory, seed_historical_cases


class FakeNavProvider:
    def create_response(self, **kwargs):
        return SimpleNamespace(
            id="r1",
            output=[
                SimpleNamespace(
                    type="function_call",
                    name="identify_top_contributors",
                    arguments=json.dumps({"top_n": 3}),
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
                "probable_root_cause": "PRICE_EXCEPTION",
                "confidence": 0.94,
                "observations": ["A single security dominates the NAV variance."],
                "supporting_evidence": ["The security is the largest contributor."],
                "counter_evidence": [],
                "recommended_next_step": "Compare price sources for the top contributor.",
                "human_review_required": True,
            }),
        )


def test_nav_agent_has_real_domain_tools():
    scenario = create_price_exception_scenario()
    agent = NavInvestigationAgent(
        memory=CaseMemory(seed_historical_cases()),
        provider=FakeNavProvider(),
    )

    run = agent.investigate(scenario)

    assert run.report.probable_root_cause == "PRICE_EXCEPTION"
    assert run.trace[0].tool_name == "identify_top_contributors"
