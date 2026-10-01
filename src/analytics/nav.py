from __future__ import annotations

from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True)
class NavVariance:
    expected_nav: float
    calculated_nav: float
    absolute_difference: float
    percentage_difference: float
    basis_point_difference: float


def calculate_nav_variance(expected_nav: float, calculated_nav: float) -> NavVariance:
    if expected_nav <= 0:
        raise ValueError("expected_nav must be positive")

    absolute_difference = calculated_nav - expected_nav
    percentage_difference = absolute_difference / expected_nav
    basis_point_difference = percentage_difference * 10_000

    return NavVariance(
        expected_nav=expected_nav,
        calculated_nav=calculated_nav,
        absolute_difference=absolute_difference,
        percentage_difference=percentage_difference,
        basis_point_difference=basis_point_difference,
    )


def calculate_security_contribution(
    holdings: pd.DataFrame,
    prices_expected: pd.DataFrame,
    prices_calculated: pd.DataFrame,
) -> pd.DataFrame:
    required = {
        "holdings": {"security_id", "quantity"},
        "prices_expected": {"security_id", "price"},
        "prices_calculated": {"security_id", "price"},
    }

    for name, columns in required.items():
        frame = locals()[name]
        missing = columns - set(frame.columns)
        if missing:
            raise ValueError(f"{name} missing columns: {sorted(missing)}")

    result = holdings.merge(
        prices_expected.rename(columns={"price": "expected_price"}),
        on="security_id",
        how="left",
    ).merge(
        prices_calculated.rename(columns={"price": "calculated_price"}),
        on="security_id",
        how="left",
    )

    if result[["expected_price", "calculated_price"]].isna().any().any():
        raise ValueError("Missing price for one or more held securities")

    result["expected_value"] = result["quantity"] * result["expected_price"]
    result["calculated_value"] = result["quantity"] * result["calculated_price"]
    result["value_difference"] = result["calculated_value"] - result["expected_value"]

    total_abs = result["value_difference"].abs().sum()
    result["contribution_pct"] = (
        0.0 if total_abs == 0
        else result["value_difference"].abs() / total_abs * 100
    )

    return result.sort_values(
        "contribution_pct", ascending=False
    ).reset_index(drop=True)
