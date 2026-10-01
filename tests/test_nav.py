import pytest

from src.analytics.nav import (
    calculate_nav_variance,
    calculate_security_contribution,
)
from src.data.synthetic import generate_nav_exception


def test_nav_variance():
    result = calculate_nav_variance(100.0, 99.5)
    assert result.absolute_difference == pytest.approx(-0.5)
    assert result.percentage_difference == pytest.approx(-0.005)
    assert result.basis_point_difference == pytest.approx(-50.0)


def test_nav_variance_rejects_non_positive_expected_nav():
    with pytest.raises(ValueError):
        calculate_nav_variance(0, 100)


def test_security_contribution_identifies_injected_exception():
    scenario = generate_nav_exception(seed=42)
    result = calculate_security_contribution(
        scenario.holdings,
        scenario.expected_prices,
        scenario.calculated_prices,
    )
    culprit = scenario.known_root_cause.split(":")[1]
    assert result.iloc[0]["security_id"] == culprit
    assert result.iloc[0]["contribution_pct"] > 50


def test_synthetic_scenario_has_ground_truth():
    scenario = generate_nav_exception(seed=42)
    assert scenario.exception_type == "NAV_DISCREPANCY"
    assert scenario.known_root_cause.startswith("PRICE_EXCEPTION:")
