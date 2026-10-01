from src.agents.router import InvestigationRouter
from src.agents.workflow import InvestigationWorkflow
from src.data.scenarios import create_price_exception_scenario
from src.memory.cases import CaseMemory, seed_historical_cases


def test_router_selects_nav_specialist():
    assert InvestigationRouter().route("NAV_DISCREPANCY") == "NAV_INVESTIGATOR"


def test_nav_workflow_runs_end_to_end():
    result = InvestigationWorkflow(
        CaseMemory(seed_historical_cases())
    ).run_nav(create_price_exception_scenario())

    assert result["route"] == "NAV_INVESTIGATOR"
    assert result["resolution"]["requires_human_approval"] is True
    assert result["challenge"]["challenged"] is True
