"""Leakage-conscious benchmark runners for synthetic exception features."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.data.benchmark import BenchmarkDataset


FEATURE_COLUMNS = (
    "price_variance_pct", "quantity_variance_pct", "nav_contribution_pct",
    "vendor_disagreement_pct", "transaction_age_days", "settlement_pending",
    "corporate_action_flag", "fx_deviation_pct", "source_conflict_count",
    "record_delay_hours",
)


def rule_prediction(row: pd.Series) -> str:
    """Small transparent, explicitly heuristic baseline."""
    if not row["evidence_ids"]:
        return "INSUFFICIENT_EVIDENCE"
    if row["source_conflict_count"] >= 2:
        return "CONFLICTING_EVIDENCE"
    signals = {
        "STALE_PRICE": abs(row["price_variance_pct"]) / 10,
        "MISSING_TRANSACTION": abs(row["quantity_variance_pct"]) / 15,
        "WRONG_QUANTITY": abs(row["quantity_variance_pct"]) / 25,
        "CORPORATE_ACTION": 1.5 if row["corporate_action_flag"] else 0,
        "FX_MISMATCH": abs(row["fx_deviation_pct"]) / 4,
        "PENDING_SETTLEMENT": 1.5 if row["settlement_pending"] else 0,
        "VENDOR_DISCREPANCY": row["vendor_disagreement_pct"] / 8,
    }
    return max(signals, key=signals.get)


def evaluate_rules_and_ml(dataset: BenchmarkDataset, *, seed: int = 42) -> dict[str, Any]:
    """Evaluate rules and Random Forest on identical held-out exception IDs.

    Ground truth is joined after predictions are produced. IDs, difficulty,
    evidence identifiers, injection mechanism, and all target fields are kept
    out of the feature matrix.
    """
    observable = dataset.observable["exceptions"].copy()
    truth = dataset.ground_truth.set_index("exception_id").loc[observable["exception_id"]]
    X = observable.loc[:, FEATURE_COLUMNS].copy()
    X["settlement_pending"] = X["settlement_pending"].astype(int)
    X["corporate_action_flag"] = X["corporate_action_flag"].astype(int)
    y = truth["root_cause"].astype(str).to_numpy()
    indices = np.arange(len(observable))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.25, random_state=seed, stratify=y
    )
    labels = sorted(set(y))
    encoder = LabelEncoder().fit(labels)
    model = RandomForestClassifier(
        n_estimators=160, min_samples_leaf=2, class_weight="balanced",
        random_state=seed, n_jobs=-1,
    )
    model.fit(X.iloc[train_idx], encoder.transform(y[train_idx]))
    probabilities = model.predict_proba(X.iloc[test_idx])
    ml_labels = encoder.inverse_transform(model.classes_[probabilities.argmax(axis=1)])
    rules_labels = np.array([
        rule_prediction(observable.iloc[index]) for index in test_idx
    ])
    actual = y[test_idx]
    expected_escalation = truth["expected_escalation"].to_numpy()[test_idx].astype(bool)
    rule_escalation = np.array([
        label in {"INSUFFICIENT_EVIDENCE", "CONFLICTING_EVIDENCE"}
        for label in rules_labels
    ])
    ml_escalation = np.array([
        label in {"INSUFFICIENT_EVIDENCE", "CONFLICTING_EVIDENCE"}
        for label in ml_labels
    ])
    ml_top3 = np.argsort(probabilities, axis=1)[:, -3:]
    encoded_actual = encoder.transform(actual)
    top3_recall = float(np.mean([label in row for label, row in zip(encoded_actual, ml_top3)]))

    result = {
        "evaluation_split": {
            "train_cases": int(len(train_idx)), "test_cases": int(len(test_idx)),
            "seed": seed, "stratified": True,
        },
        "features": list(FEATURE_COLUMNS),
        "models": {
            "rules": {
                "root_cause_accuracy": float(np.mean(rules_labels == actual)),
                "escalation_accuracy": float(np.mean(rule_escalation == expected_escalation)),
                "top3_hypothesis_recall": "Not evaluated (single-label rule output)",
                "evidence_precision": "Not evaluated",
                "evidence_recall": "Not evaluated",
                "recommendation_accuracy": "Not evaluated",
            },
            "classical_ml_random_forest": {
                "root_cause_accuracy": float(np.mean(ml_labels == actual)),
                "top3_hypothesis_recall": top3_recall,
                "escalation_accuracy": float(np.mean(ml_escalation == expected_escalation)),
                "evidence_precision": "Not evaluated",
                "evidence_recall": "Not evaluated",
                "recommendation_accuracy": "Not evaluated",
            },
            "single_llm": "Not evaluated (requires configured Sarvam credentials and run)",
            "agentic_llm": "Not evaluated (requires configured Sarvam credentials and run)",
        },
        "by_difficulty": _metrics_by_difficulty(
            observable.iloc[test_idx].reset_index(drop=True), actual,
            rules_labels, ml_labels,
        ),
        "latency_seconds": "Not evaluated by this offline runner",
        "token_usage": "Not available for non-LLM baselines",
        "estimated_api_cost": "Not evaluated; pricing not configured",
    }
    return result


def _metrics_by_difficulty(
    cases: pd.DataFrame, actual: np.ndarray,
    rules: np.ndarray, ml: np.ndarray,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for difficulty in sorted(cases["difficulty"].unique()):
        mask = cases["difficulty"].to_numpy() == difficulty
        results[str(difficulty)] = {
            "cases": int(mask.sum()),
            "rules_root_cause_accuracy": float(np.mean(rules[mask] == actual[mask])),
            "classical_ml_root_cause_accuracy": float(np.mean(ml[mask] == actual[mask])),
        }
    return results
