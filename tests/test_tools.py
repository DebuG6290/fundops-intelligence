from src.data.scenarios import create_price_exception_scenario
from src.tools.nav_tools import (
    calculate_nav_variance_tool,
    compare_price_sources_tool,
    get_fund_snapshot_tool,
    identify_top_contributors_tool,
)


def test_nav_tool_returns_exception():
    scenario = create_price_exception_scenario()
    result = calculate_nav_variance_tool(scenario)

    assert result["exception_detected"] is True
    assert result["exception_type"] == "NAV_DISCREPANCY"


def test_contributor_tool_returns_records():
    scenario = create_price_exception_scenario()
    records = identify_top_contributors_tool(scenario, top_n=3)

    assert len(records) == 3
    assert records[0]["security_id"] == scenario.culprit_security_id


def test_price_comparison_tool():
    scenario = create_price_exception_scenario()
    result = compare_price_sources_tool(
        scenario,
        scenario.culprit_security_id,
    )

    assert result["found"] is True
    assert result["difference_pct"] < -20


def test_fund_snapshot_tool():
    scenario = create_price_exception_scenario()
    result = get_fund_snapshot_tool(scenario)

    assert result["fund_id"] == "FUND_DEMO_001"
    assert result["positions"] == len(scenario.dataset.holdings)
