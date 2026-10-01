import pytest

from src.analytics.fund_valuation import calculate_fund_value, calculate_nav
from src.data.scenarios import create_price_exception_scenario


def test_fund_value_is_positive():
    scenario = create_price_exception_scenario()
    value = calculate_fund_value(
        scenario.dataset.holdings,
        scenario.calculated_prices,
    )
    assert value > 0


def test_nav_uses_shares_outstanding():
    scenario = create_price_exception_scenario()
    nav = calculate_nav(
        scenario.dataset.holdings,
        scenario.expected_prices,
        scenario.shares_outstanding,
    )
    assert nav > 0


def test_invalid_shares_rejected():
    scenario = create_price_exception_scenario()
    with pytest.raises(ValueError):
        calculate_nav(
            scenario.dataset.holdings,
            scenario.expected_prices,
            0,
        )
