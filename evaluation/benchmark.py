"""Leakage-conscious common evaluator for rules, ML, and Sarvam approaches."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from evaluation.baselines import (
    FEATURE_COLUMNS, AgenticLLMBaseline, Baseline, BaselineResult,
    ClassicalMLBaseline, ObservableCase, RuleBaseline, SingleLLMBaseline,
    observable_cases,
)
from src.data.benchmark import BenchmarkDataset


def rule_prediction(row: pd.Series) -> str:
    """Compatibility helper for users of the original rule function."""
    case = ObservableCase(
        exception_id="compat", features={key: row[key] for key in FEATURE_COLUMNS if key in row},
        evidence=(() if not row.get("evidence_ids") else ({"evidence_id": row["evidence_ids"]},)),
    )
    return str(RuleBaseline().predict(case).predicted_root_cause)


def evaluate_rules_and_ml(
    dataset: BenchmarkDataset, *, seed: int = 42,
    single_llm_provider: Any | None = None,
    agentic_llm_provider: Any | None = None,
    max_llm_cases: int | None = None,
) -> dict[str, Any]:
    """Evaluate all configured adapters on one identical stratified holdout.

    Sarvam is not called unless a provider is explicitly supplied. Optional
    ``max_llm_cases`` bounds live calls; deterministic baselines still use the
    complete common test set.
    """
    observable = dataset.observable["exceptions"].reset_index(drop=True)
    truth = dataset.ground_truth.set_index("exception_id").loc[observable["exception_id"]].reset_index()
    labels = truth["root_cause"].astype(str).to_numpy()
    indices = np.arange(len(observable))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.25, random_state=seed, stratify=labels
    )
    live_configured = single_llm_provider is not None or agentic_llm_provider is not None
    if max_llm_cases is not None and max_llm_cases < 1:
        raise ValueError("max_llm_cases must be positive when supplied")
    scored_count = min(len(test_idx), max_llm_cases) if live_configured and max_llm_cases is not None else len(test_idx)
    scored_test_idx = test_idx[:scored_count]
    ml = ClassicalMLBaseline.fit(observable, labels, train_idx, seed=seed)
    cases = observable_cases(dataset)
    adapters: list[Baseline] = [RuleBaseline(), ml]
    if single_llm_provider is not None:
        adapters.append(SingleLLMBaseline(single_llm_provider))
    if agentic_llm_provider is not None:
        adapters.append(AgenticLLMBaseline(agentic_llm_provider))

    outputs: dict[str, list[BaselineResult | None]] = {adapter.name: [None] * scored_count for adapter in adapters}
    for test_pos, source_idx in enumerate(scored_test_idx):
        for adapter in adapters:
            outputs[adapter.name][test_pos] = adapter.predict(cases[int(source_idx)])

    models = {
        name: _summarize(name, results, observable.iloc[scored_test_idx].reset_index(drop=True),
                         truth.iloc[scored_test_idx].reset_index(drop=True))
        for name, results in outputs.items()
    }
    for name in ("single_llm", "agentic_llm"):
        if name not in models:
            models[name] = "Not evaluated (Sarvam provider not supplied)"
    return {
        "evaluation_split": {
            "train_cases": int(len(train_idx)), "test_cases": int(len(test_idx)),
            "scored_cases": int(scored_count), "seed": seed, "stratified": True,
        },
        "features": list(FEATURE_COLUMNS),
        "approaches": list(models),
        "models": models,
        "by_difficulty": _by_difficulty(outputs, scored_test_idx, observable, truth),
        "ground_truth_joined_after_prediction": True,
        "metrics_note": "Unavailable measures are null/Not evaluated, never zero-filled.",
    }


def _summarize(name: str, rows: list[BaselineResult | None], cases: pd.DataFrame, truth: pd.DataFrame) -> dict[str, Any]:
    available = [(row, idx) for idx, row in enumerate(rows) if row is not None]
    if not available:
        return {"status": "Not evaluated (no predictions were run)"}
    valid = [(row, idx) for row, idx in available if not row.failed]
    failed = len(available) - len(valid)
    actual = truth["root_cause"].astype(str).tolist()
    if not valid:
        return {"cases": 0, "failures": failed, "failure_rate": 1.0, "status": "All predictions failed"}
    predicted = [row.predicted_root_cause for row, _ in valid]
    expected_by_idx = truth.set_index("exception_id")
    case_by_idx = cases.reset_index(drop=True)
    truth_by_id = expected_by_idx
    correct = []
    top3 = []
    recommendation = []
    escalation = []
    evidence_precision, evidence_recall = [], []
    contradiction = []
    latency, input_tokens, output_tokens = [], [], []
    token_usage_seen = False
    for row, local_idx in valid:
        exception_id = str(case_by_idx.iloc[local_idx]["exception_id"])
        expected = truth_by_id.loc[exception_id]
        cause = str(expected["root_cause"])
        correct.append(row.predicted_root_cause == cause)
        ranked = [item.get("root_cause") for item in row.ranked_hypotheses]
        expected_causes = {cause, *(expected["secondary_causes"] or [])}
        top3.append(
            len(expected_causes & set(ranked[:3])) / len(expected_causes)
            if row.ranked_hypotheses else None
        )
        recommendation.append(None if row.recommendation_category is None else row.recommendation_category == expected["expected_recommendation_category"])
        escalation.append(None if row.escalation_required is None else row.escalation_required == bool(expected["expected_escalation"]))
        if row.contradiction_detected is not None:
            contradiction.append(row.contradiction_detected == bool(expected["expected_contradiction"]))
        relevant = set(expected["relevant_evidence_ids"] or [])
        cited = set(row.cited_evidence_ids)
        if row.cited_evidence_ids:
            evidence_precision.append(len(cited & relevant) / len(cited))
        if relevant:
            evidence_recall.append(len(cited & relevant) / len(relevant))
        if row.latency_seconds is not None:
            latency.append(row.latency_seconds)
        if row.input_tokens is not None:
            input_tokens.append(row.input_tokens); token_usage_seen = True
        if row.output_tokens is not None:
            output_tokens.append(row.output_tokens); token_usage_seen = True
    return {
        "cases": len(available), "successful_cases": len(valid), "failures": failed,
        "failure_rate": failed / len(available),
        "root_cause_accuracy": float(np.mean(correct)),
        "top3_hypothesis_recall": _mean_available(top3),
        "recommendation_accuracy": _mean_available(recommendation),
        "escalation_accuracy": _mean_available(escalation),
        "evidence_precision": _mean_available(evidence_precision),
        "evidence_recall": _mean_available(evidence_recall),
        "contradiction_detection": _mean_available(contradiction),
        "mean_latency_seconds": float(np.mean(latency)) if latency else None,
        "input_tokens": int(sum(input_tokens)) if token_usage_seen and input_tokens else None,
        "output_tokens": int(sum(output_tokens)) if token_usage_seen and output_tokens else None,
        "estimated_cost": None,
        "provider": next((row.provider for row, _ in valid if row.provider), None),
        "model": next((row.model for row, _ in valid if row.model), None),
        "request_ids": [request_id for row, _ in available for request_id in row.request_ids],
        "failure_details": [
            {"error": row.error}
            for row, _ in available if row.error
        ],
        "metric_availability": {
            "root_cause_accuracy": "measured", "evidence_precision": "measured when citations exist; otherwise unavailable",
            "evidence_recall": "measured when ground-truth evidence is defined; otherwise unavailable",
            "contradiction_detection": "measured only when the adapter emits a contradiction assessment",
            "estimated_cost": "Not evaluated (provider pricing not configured)",
        },
    }


def _mean_available(values: list[Any]) -> float | None:
    numeric = [value for value in values if value is not None]
    return float(np.mean(numeric)) if numeric else None


def _by_difficulty(outputs: dict[str, list[BaselineResult | None]], test_idx: np.ndarray, observable: pd.DataFrame, truth: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {}
    heldout_truth = truth.iloc[test_idx].reset_index(drop=True)
    for difficulty, positions in heldout_truth.groupby("difficulty").groups.items():
        local_positions = list(positions)
        section = {"cases": len(local_positions)}
        for name, predictions in outputs.items():
            evaluated = []
            for local in local_positions:
                row = predictions[local]
                if row is not None and not row.failed:
                    expected = heldout_truth.iloc[local]["root_cause"]
                    evaluated.append(row.predicted_root_cause == expected)
            section[name + "_root_cause_accuracy"] = _mean_available(evaluated)
            top3_values = []
            escalation_values = []
            recommendation_values = []
            for local in local_positions:
                row = predictions[local]
                if row is None or row.failed:
                    continue
                expected = heldout_truth.iloc[local]
                target_causes = {str(expected["root_cause"]), *(expected["secondary_causes"] or [])}
                ranked = {item.get("root_cause") for item in row.ranked_hypotheses[:3]}
                if ranked:
                    top3_values.append(len(target_causes & ranked) / len(target_causes))
                if row.escalation_required is not None:
                    escalation_values.append(row.escalation_required == bool(expected["expected_escalation"]))
                if row.recommendation_category is not None:
                    recommendation_values.append(row.recommendation_category == expected["expected_recommendation_category"])
            section[name + "_top3_hypothesis_recall"] = _mean_available(top3_values)
            section[name + "_escalation_accuracy"] = _mean_available(escalation_values)
            section[name + "_recommendation_accuracy"] = _mean_available(recommendation_values)
        result[str(difficulty)] = section
    return result
