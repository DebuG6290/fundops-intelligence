from __future__ import annotations

import pandas as pd


def find_effective_corporate_actions(
    corporate_actions: pd.DataFrame,
    as_of_date,
) -> pd.DataFrame:
    required = {"security_id", "action_type", "effective_date"}
    missing = required - set(corporate_actions.columns)
    if missing:
        raise ValueError(f"corporate_actions missing columns: {sorted(missing)}")

    actions = corporate_actions.copy()
    actions["effective_date"] = pd.to_datetime(actions["effective_date"]).dt.date
    return actions[actions["effective_date"] == as_of_date].reset_index(drop=True)
