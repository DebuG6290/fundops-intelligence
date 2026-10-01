from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.data.generator import FundDataset, generate_fund_dataset


@dataclass
class InvestigationScenario:
    dataset: FundDataset
    expected_prices: pd.DataFrame
    calculated_prices: pd.DataFrame
    shares_outstanding: float
    exception_id: str
    known_root_cause: str
    culprit_security_id: str


def create_price_exception_scenario(seed: int = 42) -> InvestigationScenario:
    """Create a controlled scenario for end-to-end investigation testing."""
    dataset = generate_fund_dataset(seed=seed)
    expected_prices = dataset.prices.copy(deep=True)
    calculated_prices = dataset.prices.copy(deep=True)

    values = (
        dataset.holdings
        .merge(expected_prices[["security_id", "price"]], on="security_id")
        .assign(position_value=lambda x: x.quantity * x.price)
    )
    culprit = values.sort_values("position_value", ascending=False).iloc[0]["security_id"]

    mask = calculated_prices["security_id"].eq(culprit)
    calculated_prices.loc[mask, "price"] = (
        calculated_prices.loc[mask, "price"] * 0.72
    ).round(2)
    calculated_prices.loc[mask, "source"] = "SECONDARY_FEED"

    return InvestigationScenario(
        dataset=dataset,
        expected_prices=expected_prices,
        calculated_prices=calculated_prices,
        shares_outstanding=1_000_000.0,
        exception_id="EXC_NAV_0001",
        known_root_cause=f"PRICE_EXCEPTION:{culprit}",
        culprit_security_id=culprit,
    )
