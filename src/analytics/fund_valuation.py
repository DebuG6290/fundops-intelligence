from __future__ import annotations

import pandas as pd


def calculate_fund_value(
    holdings: pd.DataFrame,
    prices: pd.DataFrame,
) -> float:
    """Calculate gross fund value using deterministic arithmetic."""
    required_holdings = {"security_id", "quantity"}
    required_prices = {"security_id", "price"}

    missing_h = required_holdings - set(holdings.columns)
    missing_p = required_prices - set(prices.columns)

    if missing_h:
        raise ValueError(f"holdings missing columns: {sorted(missing_h)}")
    if missing_p:
        raise ValueError(f"prices missing columns: {sorted(missing_p)}")

    merged = holdings.merge(prices, on="security_id", how="left")

    if merged["price"].isna().any():
        missing = merged.loc[merged["price"].isna(), "security_id"].tolist()
        raise ValueError(f"Missing prices for securities: {missing}")

    return float((merged["quantity"] * merged["price"]).sum())


def calculate_nav(
    holdings: pd.DataFrame,
    prices: pd.DataFrame,
    shares_outstanding: float,
) -> float:
    """Calculate NAV per share."""
    if shares_outstanding <= 0:
        raise ValueError("shares_outstanding must be positive")

    return calculate_fund_value(holdings, prices) / shares_outstanding
