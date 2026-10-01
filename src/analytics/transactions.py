from __future__ import annotations

import pandas as pd


def find_transaction_mismatches(
    expected_transactions: pd.DataFrame,
    actual_transactions: pd.DataFrame,
) -> pd.DataFrame:
    """Identify transactions whose expected and actual records disagree."""
    key = "transaction_id"
    required = {
        key,
        "security_id",
        "quantity",
        "transaction_type",
    }

    for name, frame in {
        "expected_transactions": expected_transactions,
        "actual_transactions": actual_transactions,
    }.items():
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{name} missing columns: {sorted(missing)}")

    expected = expected_transactions[list(required)].rename(
        columns={
            "security_id": "expected_security_id",
            "quantity": "expected_quantity",
            "transaction_type": "expected_type",
        }
    )
    actual = actual_transactions[list(required)].rename(
        columns={
            "security_id": "actual_security_id",
            "quantity": "actual_quantity",
            "transaction_type": "actual_type",
        }
    )

    result = expected.merge(actual, on=key, how="outer", indicator=True)
    result["quantity_difference"] = (
        result["actual_quantity"].fillna(0)
        - result["expected_quantity"].fillna(0)
    )
    result["type_mismatch"] = (
        result["expected_type"].fillna("") != result["actual_type"].fillna("")
    )

    return result[
        (result["_merge"] != "both")
        | (result["quantity_difference"] != 0)
        | result["type_mismatch"]
    ].reset_index(drop=True)
