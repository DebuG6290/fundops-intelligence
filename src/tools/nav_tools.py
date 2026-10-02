from __future__ import annotations

from typing import Any

from src.analytics.exceptions import (
    decompose_nav_exception,
    detect_nav_exception,
)
from src.analytics.fund_valuation import calculate_fund_value
from src.data.scenarios import InvestigationScenario


def calculate_nav_variance_tool(scenario: InvestigationScenario) -> dict[str, Any]:
    exception = detect_nav_exception(
        exception_id=scenario.exception_id,
        fund_id=scenario.dataset.funds.iloc[0]["fund_id"],
        holdings=scenario.dataset.holdings,
        expected_prices=scenario.expected_prices,
        calculated_prices=scenario.calculated_prices,
        shares_outstanding=scenario.shares_outstanding,
        expected_holdings=scenario.expected_holdings,
    )

    if exception is None:
        return {"exception_detected": False}

    return {"exception_detected": True, **exception.to_dict()}


def identify_top_contributors_tool(
    scenario: InvestigationScenario,
    top_n: int = 5,
) -> list[dict[str, Any]]:
    result = decompose_nav_exception(
        scenario.expected_holdings if scenario.expected_holdings is not None else scenario.dataset.holdings,
        scenario.expected_prices,
        scenario.calculated_prices,
    )
    if scenario.expected_holdings is not None:
        expected = scenario.expected_holdings.set_index("security_id")["quantity"]
        actual = scenario.dataset.holdings.set_index("security_id")["quantity"]
        result["quantity_difference"] = result["security_id"].map(actual).fillna(0) - result["security_id"].map(expected).fillna(0)
        result["holding_value_difference"] = result["quantity_difference"] * result["calculated_price"]
        if result["holding_value_difference"].abs().sum() > 0:
            result["value_difference"] = result["holding_value_difference"]
            result["contribution_pct"] = result["value_difference"].abs() / result["value_difference"].abs().sum() * 100
            result = result.sort_values("contribution_pct", ascending=False)
    return result.head(top_n).to_dict(orient="records")


def compare_price_sources_tool(
    scenario: InvestigationScenario,
    security_id: str,
) -> dict[str, Any]:
    expected = scenario.expected_prices[
        scenario.expected_prices.security_id.eq(security_id)
    ]
    calculated = scenario.calculated_prices[
        scenario.calculated_prices.security_id.eq(security_id)
    ]

    if expected.empty or calculated.empty:
        return {
            "security_id": security_id,
            "found": False,
            "message": "Security not found in both price sets.",
        }

    expected_row = expected.iloc[0]
    calculated_row = calculated.iloc[0]

    expected_price = float(expected_row["price"])
    calculated_price = float(calculated_row["price"])

    return {
        "security_id": security_id,
        "found": True,
        "expected_price": expected_price,
        "calculated_price": calculated_price,
        "difference": round(calculated_price - expected_price, 4),
        "difference_pct": round(
            (calculated_price - expected_price) / expected_price * 100, 4
        ),
        "expected_source": expected_row["source"],
        "calculated_source": calculated_row["source"],
    }


def get_fund_snapshot_tool(scenario: InvestigationScenario) -> dict[str, Any]:
    value = calculate_fund_value(
        scenario.dataset.holdings,
        scenario.calculated_prices,
    )
    return {
        "fund_id": scenario.dataset.funds.iloc[0]["fund_id"],
        "fund_name": scenario.dataset.funds.iloc[0]["fund_name"],
        "base_currency": scenario.dataset.funds.iloc[0]["base_currency"],
        "gross_value": round(value, 2),
        "positions": len(scenario.dataset.holdings),
    }

