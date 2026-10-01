from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
import numpy as np
import pandas as pd


@dataclass
class FundDataset:
    funds: pd.DataFrame
    securities: pd.DataFrame
    holdings: pd.DataFrame
    prices: pd.DataFrame
    transactions: pd.DataFrame


def generate_fund_dataset(seed: int = 42, security_count: int = 12) -> FundDataset:
    """Generate a small but realistic synthetic fund-operations dataset."""
    if security_count < 5:
        raise ValueError("security_count must be at least 5")

    rng = np.random.default_rng(seed)
    as_of = date.today()
    security_ids = [f"SEC_{i:03d}" for i in range(1, security_count + 1)]

    securities = pd.DataFrame({
        "security_id": security_ids,
        "name": [f"Security {i:03d}" for i in range(1, security_count + 1)],
        "currency": rng.choice(["INR", "USD", "EUR"], size=security_count),
        "asset_class": rng.choice(["EQUITY", "BOND", "ETF"], size=security_count),
    })

    funds = pd.DataFrame({
        "fund_id": ["FUND_DEMO_001"],
        "fund_name": ["Demo Global Opportunities Fund"],
        "base_currency": ["USD"],
    })

    holdings = pd.DataFrame({
        "fund_id": "FUND_DEMO_001",
        "security_id": security_ids,
        "quantity": rng.integers(500, 5000, size=security_count).astype(float),
    })

    prices = pd.DataFrame({
        "fund_id": "FUND_DEMO_001",
        "security_id": security_ids,
        "price_date": as_of,
        "price": np.round(rng.uniform(50, 500, size=security_count), 2),
        "source": "PRIMARY_VENDOR",
    })

    transaction_dates = [
        as_of - timedelta(days=int(x)) for x in rng.integers(0, 10, size=security_count)
    ]
    transactions = pd.DataFrame({
        "transaction_id": [f"TXN_{i:04d}" for i in range(1, security_count + 1)],
        "fund_id": "FUND_DEMO_001",
        "security_id": security_ids,
        "trade_date": transaction_dates,
        "transaction_type": rng.choice(["BUY", "SELL"], size=security_count),
        "quantity": rng.integers(50, 500, size=security_count).astype(float),
        "status": rng.choice(["MATCHED", "MATCHED", "PENDING"], size=security_count),
    })

    return FundDataset(
        funds=funds,
        securities=securities,
        holdings=holdings,
        prices=prices,
        transactions=transactions,
    )
