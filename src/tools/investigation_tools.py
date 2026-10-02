from __future__ import annotations

from typing import Any

import pandas as pd

from src.analytics.corporate_actions import find_effective_corporate_actions
from src.analytics.transactions import find_transaction_mismatches
from src.data.scenarios import InvestigationScenario


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [{str(k): (v.isoformat() if hasattr(v, "isoformat") else v.item() if hasattr(v, "item") else v) for k, v in row.items()} for row in frame.to_dict(orient="records")]


def check_transaction_activity_tool(scenario: InvestigationScenario, security_id: str | None = None) -> dict[str, Any]:
    if scenario.expected_transactions is None or scenario.actual_transactions is None:
        return {"available": False, "message": "No expected/actual transaction pair is available in this scenario.", "records": []}
    rows = find_transaction_mismatches(scenario.expected_transactions, scenario.actual_transactions)
    if security_id:
        rows = rows[(rows.expected_security_id == security_id) | (rows.actual_security_id == security_id)]
    records = _records(rows)
    return {"available": True, "mismatch_count": len(records), "records": records,
            "message": "Unexplained transaction mismatches found." if records else "No unexplained transaction mismatch."}


def check_corporate_actions_tool(scenario: InvestigationScenario, security_id: str | None = None) -> dict[str, Any]:
    actions = scenario.corporate_actions
    if actions is None:
        return {"available": False, "action_count": 0, "records": [], "message": "No corporate-action records are available."}
    as_of = pd.to_datetime(scenario.dataset.prices.iloc[0].price_date).date()
    found = find_effective_corporate_actions(actions, as_of)
    if security_id:
        found = found[found.security_id == security_id]
    records = _records(found)
    return {"available": True, "action_count": len(records), "records": records,
            "message": "Effective corporate-action records found." if records else "No effective corporate action found."}


def check_security_mapping_tool(scenario: InvestigationScenario, security_id: str) -> dict[str, Any]:
    rows = scenario.dataset.securities[scenario.dataset.securities.security_id == security_id]
    if rows.empty:
        return {"security_id": security_id, "valid": False, "message": "Security identifier is not present in the current dataset."}
    row = rows.iloc[0]
    return {"security_id": str(row.security_id), "valid": True, "name": str(row["name"]), "asset_class": str(row.asset_class), "currency": str(row.currency)}


def check_fx_context_tool(scenario: InvestigationScenario, security_id: str | None = None) -> dict[str, Any]:
    fx = getattr(scenario.dataset, "fx_rates", None)
    if fx is None or fx.empty:
        return {"available": False, "records": [], "message": "No material FX evidence available in this scenario."}
    records = _records(fx)
    return {"available": True, "records": records, "message": f"{len(records)} FX observations available."}

