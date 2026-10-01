from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from src.data.generator import generate_fund_dataset


@dataclass
class TransactionMismatchScenario:
    expected_transactions: pd.DataFrame
    actual_transactions: pd.DataFrame
    exception_id: str
    known_root_cause: str


def create_transaction_mismatch_scenario(seed: int = 42) -> TransactionMismatchScenario:
    dataset = generate_fund_dataset(seed=seed)
    expected = dataset.transactions.copy(deep=True)
    actual = expected.copy(deep=True)

    culprit = actual.iloc[0]["transaction_id"]
    actual.loc[actual.transaction_id.eq(culprit), "quantity"] *= 1.35
    actual.loc[actual.transaction_id.eq(culprit), "quantity"] = (
        actual.loc[actual.transaction_id.eq(culprit), "quantity"].round(2)
    )
    actual.loc[actual.transaction_id.eq(culprit), "status"] = "MISMATCH"

    return TransactionMismatchScenario(
        expected_transactions=expected,
        actual_transactions=actual,
        exception_id="EXC_TXN_0001",
        known_root_cause=f"TRANSACTION_QUANTITY_MISMATCH:{culprit}",
    )


@dataclass
class CorporateActionScenario:
    corporate_actions: pd.DataFrame
    exception_id: str
    known_root_cause: str


def create_corporate_action_scenario(seed: int = 42) -> CorporateActionScenario:
    dataset = generate_fund_dataset(seed=seed)
    security_id = dataset.securities.iloc[1]["security_id"]

    actions = pd.DataFrame({
        "corporate_action_id": ["CA_0001"],
        "security_id": [security_id],
        "action_type": ["STOCK_SPLIT"],
        "ratio": [2.0],
        "effective_date": [date.today()],
        "status": ["EFFECTIVE"],
    })

    return CorporateActionScenario(
        corporate_actions=actions,
        exception_id="EXC_CA_0001",
        known_root_cause=f"CORPORATE_ACTION:{security_id}",
    )
