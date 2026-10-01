"""Scalable, reproducible synthetic benchmark with isolated ground truth.

Only ``observable`` should be passed to investigators. ``ground_truth`` is a
separate evaluation artifact and is intentionally not nested in observable
rows or included in model context.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


CAUSES = (
    "STALE_PRICE", "MISSING_TRANSACTION", "CORPORATE_ACTION", "FX_MISMATCH",
    "PENDING_SETTLEMENT", "DUPLICATE_TRANSACTION", "WRONG_QUANTITY",
    "WRONG_SECURITY_MAPPING", "VENDOR_DISCREPANCY", "MULTI_CAUSE_EXCEPTION",
    "INSUFFICIENT_EVIDENCE", "CONFLICTING_EVIDENCE",
)


@dataclass(frozen=True)
class BenchmarkDataset:
    observable: dict[str, pd.DataFrame]
    ground_truth: pd.DataFrame


def generate_benchmark_dataset(
    *,
    seed: int = 42,
    fund_count: int = 100,
    security_count: int = 5_000,
    position_count: int = 200_000,
    price_count: int = 500_000,
    transaction_count: int = 300_000,
    corporate_action_count: int = 20_000,
    fx_count: int = 50_000,
    exception_count: int = 10_000,
    historical_case_count: int = 5_000,
) -> BenchmarkDataset:
    """Generate target-scale synthetic fund records and labeled exceptions.

    Entity rows are generated with valid fund/security foreign keys. The
    small defaults are intended for unit tests; passing the documented target
    sizes creates a substantial workload without committing generated blobs.
    """
    counts = {
        "fund_count": fund_count, "security_count": security_count,
        "position_count": position_count, "price_count": price_count,
        "transaction_count": transaction_count,
        "corporate_action_count": corporate_action_count,
        "fx_count": fx_count, "exception_count": exception_count,
        "historical_case_count": historical_case_count,
    }
    if any(value <= 0 for value in counts.values()):
        raise ValueError("All benchmark dataset counts must be positive")
    if fund_count < 1 or security_count < 1:
        raise ValueError("At least one fund and security are required")

    rng = np.random.default_rng(seed)
    fund_ids = np.array([f"FUND_{i:05d}" for i in range(fund_count)])
    security_ids = np.array([f"SEC_{i:06d}" for i in range(security_count)])
    funds = pd.DataFrame({
        "fund_id": fund_ids,
        "base_currency": rng.choice(["USD", "INR", "EUR", "GBP"], fund_count),
    })
    securities = pd.DataFrame({
        "security_id": security_ids,
        "currency": rng.choice(["USD", "INR", "EUR", "GBP", "JPY"], security_count),
        "asset_class": rng.choice(["EQUITY", "BOND", "ETF"], security_count),
    })

    positions = _entity_rows(rng, position_count, fund_ids, security_ids)
    positions["quantity"] = np.round(rng.lognormal(7.0, 1.0, position_count), 3)
    positions["unit_price"] = np.round(rng.lognormal(3.5, 0.7, position_count), 4)
    positions["market_value"] = np.round(positions["quantity"] * positions["unit_price"], 2)
    prices = _entity_rows(rng, price_count, fund_ids, security_ids)
    prices["price_date"] = pd.Timestamp("2025-01-01") + pd.to_timedelta(
        rng.integers(0, 365, price_count), unit="D"
    )
    prices["price"] = np.round(rng.lognormal(3.5, 0.7, price_count), 4)
    prices["vendor"] = rng.choice(["VENDOR_A", "VENDOR_B", "VENDOR_C"], price_count)
    transactions = _entity_rows(rng, transaction_count, fund_ids, security_ids)
    transactions["transaction_id"] = [f"TX_{i:08d}" for i in range(transaction_count)]
    transactions["quantity"] = np.round(rng.normal(0, 1000, transaction_count), 3)
    transactions["status"] = rng.choice(
        ["MATCHED", "PENDING", "SETTLED"], transaction_count, p=[0.7, 0.1, 0.2]
    )
    settlement_records = pd.DataFrame({
        "transaction_id": transactions["transaction_id"],
        "settlement_status": transactions["status"],
        "settlement_date": pd.Timestamp("2025-01-01") + pd.to_timedelta(
            rng.integers(0, 30, transaction_count), unit="D"
        ),
    })
    corporate_actions = _entity_rows(rng, corporate_action_count, fund_ids, security_ids)
    corporate_actions["action_id"] = [f"CA_{i:07d}" for i in range(corporate_action_count)]
    corporate_actions["action_type"] = rng.choice(
        ["STOCK_SPLIT", "DIVIDEND", "MERGER"], corporate_action_count
    )
    corporate_actions["effective"] = rng.choice([True, False], corporate_action_count)
    fx = pd.DataFrame({
        "currency": rng.choice(["INR", "EUR", "GBP", "JPY"], fx_count),
        "rate_date": pd.Timestamp("2025-01-01") + pd.to_timedelta(rng.integers(0, 365, fx_count), unit="D"),
        "rate": np.round(rng.lognormal(0, 0.5, fx_count), 6),
        "source": rng.choice(["FX_A", "FX_B"], fx_count),
    })
    vendor_price_feeds = prices[["fund_id", "security_id", "price_date", "price", "vendor"]].copy()
    nav_calculations = (
        positions.groupby("fund_id", as_index=False)["market_value"]
        .sum()
        .rename(columns={"market_value": "calculated_net_assets"})
    )
    nav_calculations["valuation_date"] = pd.Timestamp("2025-01-01")

    cause = rng.choice(CAUSES, exception_count)
    difficulty = rng.choice(
        ["easy", "medium", "hard", "conflicting", "insufficient", "multi_cause"],
        exception_count,
        p=[0.30, 0.25, 0.15, 0.12, 0.10, 0.08],
    )
    # Include a useful feature for every injection mechanism; these are noisy
    # operational observations rather than labels or hidden truth.
    observable = pd.DataFrame({
        "exception_id": [f"EXC_{i:08d}" for i in range(exception_count)],
        "exception_type": rng.choice(["NAV_DISCREPANCY", "TRANSACTION_MISMATCH", "CORPORATE_ACTION"], exception_count),
        "fund_id": rng.choice(fund_ids, exception_count),
        "security_id": rng.choice(security_ids, exception_count),
        "price_variance_pct": np.round(rng.normal(0, 3, exception_count), 3),
        "quantity_variance_pct": np.round(rng.normal(0, 2, exception_count), 3),
        "nav_contribution_pct": np.round(rng.uniform(0, 100, exception_count), 3),
        "vendor_disagreement_pct": np.round(np.abs(rng.normal(0, 1, exception_count)), 3),
        "transaction_age_days": rng.integers(0, 15, exception_count),
        "settlement_pending": rng.choice([False, True], exception_count),
        "corporate_action_flag": rng.choice([False, True], exception_count),
        "fx_deviation_pct": np.round(rng.normal(0, 0.5, exception_count), 3),
        "source_conflict_count": rng.integers(0, 3, exception_count),
        "record_delay_hours": rng.integers(0, 48, exception_count),
        "severity": rng.choice(["LOW", "MEDIUM", "HIGH", "CRITICAL"], exception_count),
        "financial_impact": np.round(rng.lognormal(8.0, 1.2, exception_count), 2),
        "difficulty": difficulty,
        "evidence_ids": [f"EV_{i:08d}" for i in range(exception_count)],
    })

    # Inject cause-specific observable signals, without exposing the cause.
    for index, mechanism in enumerate(cause):
        if mechanism == "STALE_PRICE": observable.at[index, "price_variance_pct"] = 18.0 + rng.random() * 20
        elif mechanism == "MISSING_TRANSACTION": observable.at[index, "quantity_variance_pct"] = 10.0 + rng.random() * 40
        elif mechanism == "CORPORATE_ACTION": observable.at[index, "corporate_action_flag"] = True
        elif mechanism == "FX_MISMATCH": observable.at[index, "fx_deviation_pct"] = 4.0 + rng.random() * 10
        elif mechanism == "PENDING_SETTLEMENT": observable.at[index, "settlement_pending"] = True
        elif mechanism == "DUPLICATE_TRANSACTION": observable.at[index, "transaction_age_days"] = 0
        elif mechanism == "WRONG_QUANTITY": observable.at[index, "quantity_variance_pct"] = 25.0 + rng.random() * 50
        elif mechanism == "WRONG_SECURITY_MAPPING": observable.at[index, "source_conflict_count"] = 2
        elif mechanism == "VENDOR_DISCREPANCY": observable.at[index, "vendor_disagreement_pct"] = 8.0 + rng.random() * 20
        elif mechanism == "MULTI_CAUSE_EXCEPTION":
            observable.at[index, "price_variance_pct"] = 16.0
            observable.at[index, "fx_deviation_pct"] = 5.0
            observable.at[index, "difficulty"] = "multi_cause"
        elif mechanism == "INSUFFICIENT_EVIDENCE":
            observable.at[index, "evidence_ids"] = ""
            observable.at[index, "difficulty"] = "insufficient"
        elif mechanism == "CONFLICTING_EVIDENCE":
            observable.at[index, "source_conflict_count"] = max(
                2, int(observable.at[index, "source_conflict_count"])
            )
            observable.at[index, "difficulty"] = "conflicting"

    gt = pd.DataFrame({
        "exception_id": observable["exception_id"].copy(),
        "root_cause": ["STALE_PRICE" if item == "MULTI_CAUSE_EXCEPTION" else item for item in cause],
        "injection_mechanism": cause,
        "secondary_causes": [
            ["FX_MISMATCH"] if item == "MULTI_CAUSE_EXCEPTION" else [] for item in cause
        ],
        "relevant_evidence_ids": [
            [] if item == "INSUFFICIENT_EVIDENCE" else [observable.iloc[i]["evidence_ids"]]
            for i, item in enumerate(cause)
        ],
        "expected_escalation": np.isin(cause, ["INSUFFICIENT_EVIDENCE", "CONFLICTING_EVIDENCE"]),
        "expected_recommendation_category": np.where(
            np.isin(cause, ["INSUFFICIENT_EVIDENCE", "CONFLICTING_EVIDENCE"]),
            "INVESTIGATE_FURTHER", "REVIEW_RECOMMENDATION",
        ),
        "expected_investigation_category": observable["exception_type"].copy(),
        "difficulty": observable["difficulty"].copy(),
    })
    evidence_rows = observable.loc[observable["evidence_ids"].ne(""), [
        "exception_id", "evidence_ids", "price_variance_pct", "quantity_variance_pct",
        "fx_deviation_pct", "source_conflict_count",
    ]].copy()
    evidence_records = pd.DataFrame({
        "exception_id": evidence_rows["exception_id"],
        "evidence_id": evidence_rows["evidence_ids"],
        "source_type": "DETERMINISTIC_ANALYTICS",
        "source_name": "synthetic_reconciliation_signals",
        "claim": [
            "Observed signals: price variance " + str(row.price_variance_pct)
            + "%; quantity variance " + str(row.quantity_variance_pct)
            + "%; FX deviation " + str(row.fx_deviation_pct)
            + "%; source conflicts " + str(int(row.source_conflict_count)) + "."
            for row in evidence_rows.itertuples(index=False)
        ],
    })
    historical_cases = pd.DataFrame({
        "case_id": [f"CASE_{i:08d}" for i in range(historical_case_count)],
        "exception_type": rng.choice(["NAV_DISCREPANCY", "TRANSACTION_MISMATCH", "CORPORATE_ACTION"], historical_case_count),
        "root_cause": rng.choice(CAUSES, historical_case_count),
        "evidence_role": "analogy",
    })
    operational_notes = pd.DataFrame({
        "exception_id": observable["exception_id"].copy(),
        "note": rng.choice([
            "Awaiting source-system confirmation.",
            "Operations review requested.",
            "Record received after valuation cutoff.",
            "No additional operator note supplied.",
        ], exception_count),
    })
    return BenchmarkDataset(
        observable={
            "funds": funds, "securities": securities, "positions": positions,
            "prices": prices, "vendor_price_feeds": vendor_price_feeds,
            "transactions": transactions, "settlement_records": settlement_records,
            "corporate_actions": corporate_actions, "fx_rates": fx,
            "nav_calculations": nav_calculations,
            "exceptions": observable, "historical_cases": historical_cases,
            "operational_notes": operational_notes,
            "evidence_records": evidence_records,
        },
        ground_truth=gt,
    )


def _entity_rows(
    rng: np.random.Generator,
    count: int,
    funds: np.ndarray,
    securities: np.ndarray,
) -> pd.DataFrame:
    return pd.DataFrame({
        "fund_id": rng.choice(funds, count),
        "security_id": rng.choice(securities, count),
    })
