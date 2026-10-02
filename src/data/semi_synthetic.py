"""Controlled exceptions over filed N-PORT holdings; labels stay evaluator-only.

The SEC provides portfolio holdings, not internal transaction/CA ledgers. All
operational discrepancies and their labels generated here are synthetic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data.benchmark import BenchmarkDataset
from src.data.nport import NPortSnapshot
from src.memory.cases import HistoricalCase


CAUSE_COUNTS = {
    "STALE_PRICE": 25, "WRONG_QUANTITY": 25, "CORPORATE_ACTION": 20,
    "WRONG_SECURITY_MAPPING": 10, "FX_MISMATCH": 10,
    "INSUFFICIENT_EVIDENCE": 5, "CONFLICTING_EVIDENCE": 5,
}
CAUSES = tuple(CAUSE_COUNTS)
EXCEPTION_TYPES = {
    "STALE_PRICE": "NAV_DISCREPANCY", "WRONG_QUANTITY": "TRANSACTION_MISMATCH",
    "CORPORATE_ACTION": "CORPORATE_ACTION", "WRONG_SECURITY_MAPPING": "NAV_DISCREPANCY",
    "FX_MISMATCH": "NAV_DISCREPANCY", "INSUFFICIENT_EVIDENCE": "NAV_DISCREPANCY",
    "CONFLICTING_EVIDENCE": "NAV_DISCREPANCY",
}
CHECKS = {
    "STALE_PRICE": "compare_price_sources", "WRONG_QUANTITY": "find_transaction_mismatches",
    "CORPORATE_ACTION": "find_effective_corporate_actions",
    "WRONG_SECURITY_MAPPING": "verify_security_mapping", "FX_MISMATCH": "compare_fx_sources",
    "INSUFFICIENT_EVIDENCE": "request_missing_source_records",
    "CONFLICTING_EVIDENCE": "reconcile_conflicting_sources",
}


@dataclass(frozen=True)
class SemiSyntheticSuite:
    dataset: BenchmarkDataset
    memory_cases: tuple[HistoricalCase, ...]
    source_sha256: dict[str, str]
    memory_holding_keys: frozenset[str]
    evaluation_holding_keys: frozenset[str]


def build_semi_synthetic_suite(snapshot: NPortSnapshot, *, seed: int = 42, evaluation_count: int = 120) -> SemiSyntheticSuite:
    """Build a 100-case synthetic-label memory corpus and disjoint holdout.

    Source holding keys are partitioned *before* sampling. Sampling with
    replacement permits a small public-data PoC, but repeated source holdings
    within either partition must be disclosed in the methodology.
    """
    if evaluation_count < 1:
        raise ValueError("evaluation_count must be positive")
    eligible = snapshot.holdings.loc[
        snapshot.holdings.reported_balance.notna() & snapshot.holdings.reported_value.notna()
        & snapshot.holdings.reported_balance.ne(0) & snapshot.holdings.reported_value.gt(0)
    ].copy()
    eligible["source_key"] = eligible.accession_number.astype(str) + ":" + eligible.holding_id.astype(str)
    eligible = eligible.drop_duplicates("source_key").sort_values("source_key").reset_index(drop=True)
    if len(eligible) < 2:
        raise ValueError("At least two distinct usable public holdings are required for an isolated split")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(eligible))
    cutoff = max(1, min(len(eligible) - 1, len(eligible) // 2))
    memory_pool = eligible.iloc[order[:cutoff]].reset_index(drop=True)
    evaluation_pool = eligible.iloc[order[cutoff:]].reset_index(drop=True)
    memory_labels = [cause for cause, count in CAUSE_COUNTS.items() for _ in range(count)]
    rng.shuffle(memory_labels)
    evaluation_labels = list(rng.choice(CAUSES, size=evaluation_count))
    memory_rows = _make_rows(memory_pool, memory_labels, rng, "MEM")
    evaluation_rows = _make_rows(evaluation_pool, evaluation_labels, rng, "EVAL")
    memory_cases = tuple(_to_history(row) for row in memory_rows)
    exceptions = pd.DataFrame([row["observable"] for row in evaluation_rows])
    evidence = pd.DataFrame([item for row in evaluation_rows for item in row["evidence"]], columns=[
        "exception_id", "evidence_id", "source_type", "source_name", "claim", "reliability", "provenance",
    ])
    truth = pd.DataFrame([row["truth"] for row in evaluation_rows])
    return SemiSyntheticSuite(
        BenchmarkDataset({"exceptions": exceptions, "evidence_records": evidence,
                          "operational_notes": pd.DataFrame(columns=["exception_id", "note"])}, truth),
        memory_cases, dict(snapshot.source_sha256),
        frozenset(memory_pool.source_key), frozenset(evaluation_pool.source_key),
    )


def _make_rows(pool: pd.DataFrame, labels: list[str], rng: np.random.Generator, prefix: str) -> list[dict]:
    result = []
    for index, cause in enumerate(labels):
        source = pool.iloc[int(rng.integers(len(pool)))]
        exception_id = f"SEMI_{prefix}_{index:04d}"
        # Filing amounts remain unchanged. These are explicitly injected
        # operational observations, with distractors and signal overlap.
        signals = np.abs(rng.normal([4, 4, 2, 2], [3, 3, 1.5, 1.5]))
        target = {"STALE_PRICE": 0, "WRONG_QUANTITY": 1, "FX_MISMATCH": 2,
                  "WRONG_SECURITY_MAPPING": 3}.get(cause)
        if target is not None:
            signals[target] += rng.uniform(3, 10)
        if cause == "CORPORATE_ACTION":
            signals[1] += rng.uniform(2, 7)
        if rng.random() < .35:
            signals[int(rng.integers(4))] += rng.uniform(5, 14)
        # A small true-cause signal can be weaker than an irrelevant one.
        difficulty = "hard" if rng.random() < .3 else "medium"
        if difficulty == "hard" and target is not None:
            signals[target] *= .45
        conflict = 2 if cause == "CONFLICTING_EVIDENCE" else int(rng.random() < .14)
        if cause == "INSUFFICIENT_EVIDENCE":
            signals *= .15
        evidence_id = f"SEMI_EV_{prefix}_{index:04d}"
        filing_evidence_id = f"SEMI_SEC_{prefix}_{index:04d}"
        source_key = str(source.source_key)
        observable = {
            "exception_id": exception_id, "exception_type": EXCEPTION_TYPES[cause],
            "fund_id": str(source.accession_number), "security_id": str(source.holding_id),
            "price_variance_pct": round(float(signals[0]), 3),
            "quantity_variance_pct": round(float(signals[1]), 3),
            "nav_contribution_pct": round(float(rng.uniform(1, 35)), 3),
            "vendor_disagreement_pct": round(float(signals[0] * rng.uniform(.3, 1.2)), 3),
            "transaction_age_days": int(rng.integers(0, 8)),
            "settlement_pending": bool(rng.random() < .2),
            "corporate_action_flag": bool(cause == "CORPORATE_ACTION" or rng.random() < .18),
            "fx_deviation_pct": round(float(signals[2]), 3),
            "source_conflict_count": conflict,
            "record_delay_hours": int(rng.integers(0, 72)),
            "severity": "MEDIUM", "financial_impact": round(float(source.reported_value) * .01, 2),
            "evidence_ids": "" if cause == "INSUFFICIENT_EVIDENCE" else evidence_id,
            "source_accession": str(source.accession_number), "source_holding_id": str(source.holding_id),
            "source_reported_balance": float(source.reported_balance),
            "source_reported_value": float(source.reported_value),
            "source_currency": str(source.currency),
            "injected_comparison_balance": round(float(source.reported_balance) * (1 + float(signals[1]) / 100), 6),
            "injected_comparison_unit_value": round(float(source.implied_unit_value) * (1 + float(signals[0]) / 100), 6),
        }
        evidence = [] if cause == "INSUFFICIENT_EVIDENCE" else [
            {"exception_id": exception_id, "evidence_id": filing_evidence_id,
             "source_type": "PUBLIC_FILING", "source_name": "SEC_N_PORT_FUND_REPORTED_HOLDING",
             "claim": f"As filed for holding {source_key}: balance {source.reported_balance}, value {source.reported_value} {source.currency}.",
             "reliability": None, "provenance": source_key},
            {"exception_id": exception_id, "evidence_id": evidence_id,
             "source_type": "SYNTHETIC_RECONCILIATION", "source_name": "controlled_exception_injector",
             "claim": f"Controlled comparison for holding {source_key}: balance {observable['injected_comparison_balance']}, implied unit value {observable['injected_comparison_unit_value']}; injected price variance {observable['price_variance_pct']}%, quantity variance {observable['quantity_variance_pct']}%, FX deviation {observable['fx_deviation_pct']}%.",
             "reliability": None, "provenance": source_key},
        ]
        truth = {
            "exception_id": exception_id, "root_cause": cause, "injection_mechanism": cause,
            "secondary_causes": [], "relevant_evidence_ids": [evidence_id] if evidence else [],
            "expected_escalation": cause in {"INSUFFICIENT_EVIDENCE", "CONFLICTING_EVIDENCE"},
            "expected_contradiction": cause == "CONFLICTING_EVIDENCE",
            "expected_recommendation_category": "INVESTIGATE_FURTHER" if cause in {"INSUFFICIENT_EVIDENCE", "CONFLICTING_EVIDENCE"} else "REVIEW_RECOMMENDATION",
            "expected_investigation_category": EXCEPTION_TYPES[cause], "difficulty": difficulty,
            "source_key": source_key,
        }
        result.append({"observable": observable, "evidence": evidence, "truth": truth})
    return result


def _to_history(row: dict) -> HistoricalCase:
    observable, truth = row["observable"], row["truth"]
    # Only observable, neutral signal descriptions are indexed for retrieval.
    symptoms = tuple(_symptoms(observable))
    return HistoricalCase(
        case_id=observable["exception_id"], exception_type=observable["exception_type"],
        title="Controlled historical exception", symptoms=symptoms,
        root_cause=truth["root_cause"], resolution="Review source records with a human operator.",
        human_validated=False,
        review_metadata={"validation_source": "synthetic_ground_truth", "eligible_for_operational_guidance": False,
                         "source_key": truth["source_key"]},
        investigation_path=(CHECKS[truth["root_cause"]],), useful_evidence=symptoms,
    )


def _symptoms(row: dict) -> list[str]:
    symptoms = [str(row["exception_type"]).lower().replace("_", " ")]
    for field, term in (("price_variance_pct", "price variance"), ("quantity_variance_pct", "quantity variance"),
                        ("fx_deviation_pct", "FX deviation"), ("source_conflict_count", "source conflict")):
        if float(row[field]) >= (1 if field == "source_conflict_count" else 5):
            symptoms.append(term)
    if row["corporate_action_flag"]:
        symptoms.append("corporate action flag")
    if not row["evidence_ids"]:
        symptoms.append("missing source evidence")
    if row["source_conflict_count"] >= 2:
        symptoms.append("conflicting source records")
    return symptoms


def observable_symptom_query(row: dict) -> str:
    return " ".join(_symptoms(row))

