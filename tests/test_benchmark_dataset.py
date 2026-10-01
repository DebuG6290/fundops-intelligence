from __future__ import annotations

import pandas as pd
import pytest

from evaluation.benchmark import FEATURE_COLUMNS, evaluate_rules_and_ml
from src.data.benchmark import CAUSES, generate_benchmark_dataset


def _small_dataset(seed=17):
    return generate_benchmark_dataset(
        seed=seed, fund_count=5, security_count=30, position_count=100,
        price_count=150, transaction_count=120, corporate_action_count=40,
        fx_count=50, exception_count=480, historical_case_count=100,
    )


def test_benchmark_generation_is_seeded_and_ground_truth_separate():
    first = _small_dataset()
    second = _small_dataset()
    pd.testing.assert_frame_equal(first.observable["exceptions"], second.observable["exceptions"])
    pd.testing.assert_frame_equal(first.ground_truth, second.ground_truth)
    assert "root_cause" not in first.observable["exceptions"].columns
    assert "root_cause" in first.ground_truth.columns
    assert set(first.ground_truth["injection_mechanism"]).issubset(set(CAUSES))


def test_benchmark_has_injected_mechanisms_and_real_entity_counts():
    dataset = _small_dataset()
    assert len(dataset.observable["funds"]) == 5
    assert len(dataset.observable["positions"]) == 100
    assert len(dataset.observable["exceptions"]) == 480
    assert len(dataset.ground_truth) == 480
    assert len(dataset.observable["evidence_records"]) > 0
    assert "settlement_records" in dataset.observable
    assert "vendor_price_feeds" in dataset.observable
    assert "nav_calculations" in dataset.observable
    assert "operational_notes" in dataset.observable
    assert set(dataset.observable["exceptions"]["difficulty"]).issubset(
        {"easy", "medium", "hard", "conflicting", "insufficient", "multi_cause"}
    )


def test_evaluation_uses_features_without_ground_truth_leakage_and_returns_measured_metrics():
    dataset = _small_dataset()
    results = evaluate_rules_and_ml(dataset)
    assert results["features"] == list(FEATURE_COLUMNS)
    assert results["evaluation_split"]["train_cases"] + results["evaluation_split"]["test_cases"] == 480
    ml = results["models"]["classical_ml_random_forest"]
    assert 0 <= ml["root_cause_accuracy"] <= 1
    assert 0 <= ml["top3_hypothesis_recall"] <= 1
    assert results["models"]["single_llm"].startswith("Not evaluated")


def test_benchmark_rejects_nonpositive_sizes():
    with pytest.raises(ValueError):
        generate_benchmark_dataset(fund_count=0)
