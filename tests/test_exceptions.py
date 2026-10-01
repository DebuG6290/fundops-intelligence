from src.analytics.exceptions import detect_nav_exception, decompose_nav_exception
from src.data.scenarios import create_price_exception_scenario


def test_controlled_scenario_triggers_nav_exception():
    scenario = create_price_exception_scenario()
    exception = detect_nav_exception(
        exception_id=scenario.exception_id,
        fund_id=scenario.dataset.funds.iloc[0]["fund_id"],
        holdings=scenario.dataset.holdings,
        expected_prices=scenario.expected_prices,
        calculated_prices=scenario.calculated_prices,
        shares_outstanding=scenario.shares_outstanding,
    )

    assert exception is not None
    assert exception.exception_type == "NAV_DISCREPANCY"
    assert abs(exception.variance_bps) > exception.threshold_bps


def test_decomposition_puts_culprit_first():
    scenario = create_price_exception_scenario()
    result = decompose_nav_exception(
        scenario.dataset.holdings,
        scenario.expected_prices,
        scenario.calculated_prices,
    )

    assert result.iloc[0]["security_id"] == scenario.culprit_security_id
    assert result.iloc[0]["contribution_pct"] > 50
