from __future__ import annotations

from datetime import date, datetime
import math
from typing import Any

from src.models.evidence import EvidenceItem, EvidenceSourceType


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if hasattr(value, "item"):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def evidence_from_tool_result(
    tool_name: str,
    result: Any,
    exception_id: str,
) -> list[EvidenceItem]:
    """Create evidence only from records returned by a deterministic tool."""
    if tool_name == "search_historical_cases":
        return [
            EvidenceItem(
                exception_id=exception_id,
                source_type=EvidenceSourceType.HISTORICAL_CASE,
                source_name=str(case.get("case_id", "unknown-case")),
                claim=(
                    f"Historical case: {case.get('title', 'untitled')}; "
                    f"recorded root cause: {case.get('root_cause', 'unknown')}."
                ),
                metadata={
                    "evidence_role": "analogy",
                    "case_id": case.get("case_id"),
                    "exception_type": case.get("exception_type"),
                    "symptoms": _json_safe(case.get("symptoms", [])),
                    "historical_root_cause": case.get("root_cause"),
                    "historical_resolution": case.get("resolution"),
                    "case_evidence": _json_safe(case.get("evidence", [])),
                },
            )
            for case in result or []
        ]

    if isinstance(result, dict):
        rows = [result]
    elif isinstance(result, list):
        rows = result
    else:
        return []

    items: list[EvidenceItem] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        record = _json_safe(row)
        supports = f"exception:{exception_id}"

        if tool_name == "find_transaction_mismatches":
            source_type = EvidenceSourceType.TRANSACTION_RECORD
            source_name = str(row.get("transaction_id", tool_name))
            quantity_difference = row.get("quantity_difference")
            if row.get("_merge") == "left_only":
                supports = f"MISSING_TRANSACTION:{source_name}"
            elif row.get("_merge") == "right_only":
                supports = f"EXTRA_TRANSACTION:{source_name}"
            elif quantity_difference not in (None, 0, 0.0):
                supports = f"TRANSACTION_QUANTITY_MISMATCH:{source_name}"
            elif row.get("type_mismatch"):
                supports = f"TRANSACTION_TYPE_MISMATCH:{source_name}"
            claim = (
                f"Transaction {source_name}: expected quantity "
                f"{row.get('expected_quantity')}, actual quantity "
                f"{row.get('actual_quantity')}, quantity difference "
                f"{row.get('quantity_difference')}, expected type "
                f"{row.get('expected_type')}, actual type "
                f"{row.get('actual_type')}."
            )
        elif tool_name == "find_effective_corporate_actions":
            source_type = EvidenceSourceType.CORPORATE_ACTION_RECORD
            source_name = str(
                row.get("corporate_action_id") or row.get("security_id") or tool_name
            )
            supports = f"CORPORATE_ACTION:{row.get('security_id')}"
            claim = (
                f"Security {row.get('security_id')} has effective action "
                f"{row.get('action_type')} with ratio {row.get('ratio')} "
                f"on {row.get('effective_date')}."
            )
        elif tool_name == "compare_price_sources":
            source_type = EvidenceSourceType.PRICE_SOURCE
            source_name = str(row.get("security_id", tool_name))
            claim = (
                f"Security {source_name}: {row.get('expected_source')} price "
                f"{row.get('expected_price')} versus {row.get('calculated_source')} "
                f"price {row.get('calculated_price')}; difference "
                f"{row.get('difference_pct')}%."
            )
            price_hypothesis = "PRICE_EXCEPTION"
            supports = price_hypothesis if abs(row.get("difference_pct", 0)) >= 10 else None
            contradicts = price_hypothesis if supports is None else None
        elif tool_name == "identify_top_contributors":
            source_type = EvidenceSourceType.DETERMINISTIC_ANALYTICS
            source_name = tool_name
            claim = (
                f"Security {row.get('security_id')} contributes "
                f"{row.get('contribution_pct')}% of absolute NAV variance "
                f"with value difference {row.get('value_difference')}."
            )
        elif tool_name in {"calculate_nav_variance", "get_fund_snapshot"}:
            source_type = EvidenceSourceType.DETERMINISTIC_ANALYTICS
            source_name = tool_name
            claim = f"Deterministic output from {tool_name}: {record}."
            supports = (
                f"exception:{exception_id}"
                if tool_name == "get_fund_snapshot"
                or record.get("exception_detected")
                else None
            )
        else:
            source_type = EvidenceSourceType.TOOL_RESULT
            source_name = tool_name
            claim = f"Tool {tool_name} returned record: {record}."

        if tool_name == "compare_price_sources":
            items.append(
                EvidenceItem(
                    exception_id=exception_id,
                    source_type=source_type,
                    source_name=source_name,
                    claim=claim,
                    supports=supports,
                    contradicts=contradicts,
                    metadata=record,
                )
            )
        else:
            items.append(
                EvidenceItem(
                    exception_id=exception_id,
                    source_type=source_type,
                    source_name=source_name,
                    claim=claim,
                    supports=supports,
                    metadata=record,
                )
            )
    return items
