from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any
import pandas as pd

from src.analytics.fund_valuation import calculate_nav
from src.analytics.nav import calculate_nav_variance, calculate_security_contribution


@dataclass(frozen=True)
class NavException:
    exception_id: str
    fund_id: str
    exception_type: str
    expected_nav: float
    calculated_nav: float
    variance_bps: float
    threshold_bps: float
    status: str = "OPEN"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def detect_nav_exception(
    exception_id: str,
    fund_id: str,
    holdings: pd.DataFrame,
    expected_prices: pd.DataFrame,
    calculated_prices: pd.DataFrame,
    shares_outstanding: float,
    threshold_bps: float = 10.0,
    expected_holdings: pd.DataFrame | None = None,
) -> NavException | None:
    expected_nav = calculate_nav(expected_holdings if expected_holdings is not None else holdings, expected_prices, shares_outstanding)
    calculated_nav = calculate_nav(holdings, calculated_prices, shares_outstanding)
    variance = calculate_nav_variance(expected_nav, calculated_nav)

    if abs(variance.basis_point_difference) < threshold_bps:
        return None

    return NavException(
        exception_id=exception_id,
        fund_id=fund_id,
        exception_type="NAV_DISCREPANCY",
        expected_nav=expected_nav,
        calculated_nav=calculated_nav,
        variance_bps=variance.basis_point_difference,
        threshold_bps=threshold_bps,
    )


def decompose_nav_exception(
    holdings: pd.DataFrame,
    expected_prices: pd.DataFrame,
    calculated_prices: pd.DataFrame,
) -> pd.DataFrame:
    return calculate_security_contribution(
        holdings=holdings,
        prices_expected=expected_prices,
        prices_calculated=calculated_prices,
    )

