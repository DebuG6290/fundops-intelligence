from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import numpy as np
import pandas as pd


@dataclass
class SyntheticScenario:
    fund_id: str
    holdings: pd.DataFrame
    expected_prices: pd.DataFrame
    calculated_prices: pd.DataFrame
    exception_type: str
    known_root_cause: str


def generate_nav_exception(
    seed: int = 42,
    security_count: int = 8,
) -> SyntheticScenario:
    """Generate a controlled NAV discrepancy with known ground truth."""
    if security_count < 3:
        raise ValueError("security_count must be at least 3")

    rng = np.random.default_rng(seed)
    securities = [f"SEC_{i:03d}" for i in range(1, security_count + 1)]
    quantities = rng.integers(500, 5000, size=security_count)
    expected = np.round(rng.uniform(50, 500, size=security_count), 2)

    holdings = pd.DataFrame({
        "security_id": securities,
        "quantity": quantities.astype(float),
    })

    expected_prices = pd.DataFrame({
        "security_id": securities,
        "price": expected,
        "price_date": date.today(),
        "source": "expected_reference",
    })

    calculated = expected.copy()
    expected_values = quantities * expected
    culprit_idx = int(np.argmax(expected_values))
    calculated[culprit_idx] = round(expected[culprit_idx] * 0.72, 2)

    calculated_prices = pd.DataFrame({
        "security_id": securities,
        "price": calculated,
        "price_date": date.today(),
        "source": "fund_price_feed",
    })

    return SyntheticScenario(
        fund_id="FUND_DEMO_001",
        holdings=holdings,
        expected_prices=expected_prices,
        calculated_prices=calculated_prices,
        exception_type="NAV_DISCREPANCY",
        known_root_cause=f"PRICE_EXCEPTION:{securities[culprit_idx]}",
    )
