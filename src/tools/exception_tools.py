from __future__ import annotations

from typing import Any

from src.analytics.corporate_actions import find_effective_corporate_actions
from src.analytics.transactions import find_transaction_mismatches
from src.data.scenarios_extra import (
    CorporateActionScenario,
    TransactionMismatchScenario,
)


def find_transaction_mismatches_tool(
    scenario: TransactionMismatchScenario,
) -> list[dict[str, Any]]:
    return find_transaction_mismatches(
        scenario.expected_transactions,
        scenario.actual_transactions,
    ).to_dict(orient="records")


def find_corporate_actions_tool(
    scenario: CorporateActionScenario,
) -> list[dict[str, Any]]:
    result = find_effective_corporate_actions(
        scenario.corporate_actions,
        scenario.corporate_actions.iloc[0]["effective_date"],
    )
    return result.to_dict(orient="records")
