from __future__ import annotations

import json

import pandas as pd
import pytest

from evaluation.baselines import observable_cases
from evaluation.memory_experiment import evaluate_memory_experiment
from src.data.nport import load_nport_tables
from src.data.semi_synthetic import build_semi_synthetic_suite
from src.memory.cases import CaseMemory


def _public_format_snapshot(tmp_path):
    funds = pd.DataFrame([
        {"ACCESSION_NUMBER": "0000000000-25-000001", "SERIES_NAME": "Public Fund A", "NET_ASSETS": "1000000"},
        {"ACCESSION_NUMBER": "0000000000-25-000002", "SERIES_NAME": "Public Fund B", "NET_ASSETS": "2000000"},
    ])
    holdings = pd.DataFrame([
        {"ACCESSION_NUMBER": f"0000000000-25-{(i % 2) + 1:06d}", "HOLDING_ID": str(i),
         "ISSUER_TITLE": f"Issuer {i}", "ISSUER_CUSIP": f"{i:09d}",
         "BALANCE": str(100 + i), "UNIT": "NS", "CURRENCY_CODE": "USD",
         "CURRENCY_VALUE": str((100 + i) * 10), "ASSET_CAT": "EC"}
        for i in range(12)
    ])
    funds.to_csv(tmp_path / "FUND_REPORTED_INFO.tsv", sep="\t", index=False)
    holdings.to_csv(tmp_path / "FUND_REPORTED_HOLDING.tsv", sep="\t", index=False)
    return load_nport_tables(tmp_path)


def test_nport_loader_preserves_filed_provenance_and_does_not_invent_price(tmp_path):
    snapshot = _public_format_snapshot(tmp_path)
    assert len(snapshot.holdings) == 12
    assert snapshot.holdings.iloc[0].implied_unit_value == 10
    assert snapshot.holdings.iloc[0].reported_value == 1000
    assert len(snapshot.source_sha256["FUND_REPORTED_HOLDING"]) == 64
    assert "transaction_id" not in snapshot.holdings


def test_nport_loader_rejects_missing_fields_and_unmatched_filing(tmp_path):
    _public_format_snapshot(tmp_path)
    path = tmp_path / "FUND_REPORTED_HOLDING.tsv"
    rows = pd.read_csv(path, sep="\t", dtype=str)
    rows.loc[0, "ACCESSION_NUMBER"] = "unknown"
    rows.to_csv(path, sep="\t", index=False)
    with pytest.raises(ValueError, match="missing fund filings"):
        load_nport_tables(tmp_path)


def test_semi_synthetic_split_is_seeded_and_source_holdings_do_not_overlap(tmp_path):
    snapshot = _public_format_snapshot(tmp_path)
    first = build_semi_synthetic_suite(snapshot, seed=7)
    second = build_semi_synthetic_suite(snapshot, seed=7)
    assert len(first.memory_cases) == 100
    assert len(first.dataset.ground_truth) == 120
    assert first.memory_holding_keys.isdisjoint(first.evaluation_holding_keys)
    pd.testing.assert_frame_equal(first.dataset.observable["exceptions"], second.dataset.observable["exceptions"])
    assert all(not case.human_validated for case in first.memory_cases)
    assert all(case.review_metadata["validation_source"] == "synthetic_ground_truth" for case in first.memory_cases)
    assert all(not case.review_metadata["eligible_for_operational_guidance"] for case in first.memory_cases)
    sample_id = first.dataset.observable["exceptions"].loc[
        first.dataset.observable["exceptions"].evidence_ids.ne(""), "exception_id"
    ].iloc[0]
    records = first.dataset.observable["evidence_records"]
    source_and_injection = records.loc[records.exception_id.eq(sample_id)]
    assert set(source_and_injection.source_type) == {"PUBLIC_FILING", "SYNTHETIC_RECONCILIATION"}
    assert "As filed" in source_and_injection.loc[source_and_injection.source_type.eq("PUBLIC_FILING"), "claim"].iloc[0]
    assert "Controlled comparison" in source_and_injection.loc[source_and_injection.source_type.eq("SYNTHETIC_RECONCILIATION"), "claim"].iloc[0]


def test_labels_do_not_leak_to_investigator_context_or_memory_relevance(tmp_path):
    suite = build_semi_synthetic_suite(_public_format_snapshot(tmp_path), seed=9)
    for case in observable_cases(suite.dataset):
        text = json.dumps(case.to_context())
        for hidden in ("root_cause", "injection_mechanism", "difficulty", "secondary_causes"):
            assert hidden not in text
    memory = CaseMemory(suite.memory_cases)
    assert memory.search("price variance", validated_only=True) == []
    candidate = memory.search("price variance", validated_only=False)
    assert candidate
    original = candidate[0]
    # Changing an answer-key field cannot change retrieval order.
    changed = type(original)(**{**original.__dict__, "root_cause": "COMPLETELY_DIFFERENT"})
    other = [changed if case.case_id == original.case_id else case for case in suite.memory_cases]
    assert [item.case_id for item in CaseMemory(other).search("price variance")] == [
        item.case_id for item in candidate
    ]


def test_memory_experiment_is_paired_measured_and_no_llm_claims(tmp_path):
    suite = build_semi_synthetic_suite(_public_format_snapshot(tmp_path), seed=11)
    result = evaluate_memory_experiment(suite)
    assert result["evaluation_cases"] == 120
    assert result["source_holding_overlap"] == 0
    assert result["memory_off"]["cases"] == result["memory_on"]["cases"] == 120
    for arm in ("memory_off", "memory_on"):
        assert 0 <= result[arm]["root_cause_accuracy"] <= 1
        assert 0 <= result[arm]["first_check_accuracy"] <= 1
        assert result[arm]["token_usage"] is None
    assert result["llm_results"] == "Not evaluated"

