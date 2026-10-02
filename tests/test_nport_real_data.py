from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from evaluation.baselines import observable_cases
from evaluation.build_nport_benchmark import build_manifest, validate_suite
from evaluation.profile_nport import profile_nport
from src.data.nport import load_nport_tables
from src.data.nport_selection import select_disjoint_holdings
from src.data.semi_synthetic import CAUSES, build_real_nport_suite


def _sec_bundle(tmp_path):
    info, submission, holdings = [], [], []
    for index in range(2 * len(CAUSES)):
        accession = f"0000000000-25-{index + 1:06d}"
        period = "30-SEP-2025" if index < len(CAUSES) else "31-OCT-2025"
        info.append({"ACCESSION_NUMBER": accession, "SERIES_ID": f"S{index:04d}",
                     "SERIES_NAME": f"Fund {index}", "NET_ASSETS": "1000000"})
        submission.append({"ACCESSION_NUMBER": accession, "REPORT_DATE": period})
        holdings.append({"ACCESSION_NUMBER": accession, "HOLDING_ID": f"H{index:04d}",
                         "ISSUER_NAME": f"Issuer {index}", "ISSUER_TITLE": f"Equity {index}",
                         "ISSUER_CUSIP": f"{index:09d}", "BALANCE": str(100 + index),
                         "CURRENCY_VALUE": str((100 + index) * 20), "CURRENCY_CODE": "USD",
                         "UNIT": "NS", "ASSET_CAT": "EC"})
    for name, rows in (("FUND_REPORTED_INFO", info), ("SUBMISSION", submission),
                       ("FUND_REPORTED_HOLDING", holdings)):
        pd.DataFrame(rows).to_csv(tmp_path / f"{name}.tsv", sep="\t", index=False)
    return tmp_path


def test_real_data_profile_and_loader_preserve_filed_record(tmp_path):
    source = _sec_bundle(tmp_path)
    profile = profile_nport(source, quarter="2025q4")
    assert profile["filings"]["unique_accessions"] == 14
    assert profile["funds"]["unique_series_ids"] == 14
    assert profile["holdings"]["total"] == profile["holdings"]["usable_nonzero_balance_positive_value"] == 14
    assert profile["holdings"]["duplicate_accession_holding_keys"] == 0
    assert profile["holdings"]["reported_balance"]["median"] == 106.5
    assert profile["holdings"]["missing_fields"]["ISSUER_CUSIP"] == 0
    assert len(profile["files"]["FUND_REPORTED_HOLDING"]["sha256"]) == 64
    selected = select_disjoint_holdings(source, seed=42, memory_period="30-SEP-2025",
                                        evaluation_period="31-OCT-2025", per_partition=7)
    snapshot = load_nport_tables(source, accessions=list(selected.memory_accessions + selected.evaluation_accessions),
                                 holding_keys=selected.memory_keys | selected.evaluation_keys)
    assert len(snapshot.holdings) == 14
    assert snapshot.holdings.iloc[0].filed_balance_raw == "100"
    assert snapshot.holdings.iloc[0].filed_value_raw == "2000"
    assert set(snapshot.holdings.report_date) == {"30-SEP-2025", "31-OCT-2025"}
    assert snapshot.source_files["FUND_REPORTED_HOLDING"] == "FUND_REPORTED_HOLDING.tsv"


def test_real_benchmark_is_balanced_reproducible_and_leakage_checked(tmp_path):
    source = _sec_bundle(tmp_path)
    manifest, first = build_manifest(source, quarter="2025q4", seed=42, per_family=1)
    _, second = build_manifest(source, quarter="2025q4", seed=42, per_family=1)
    assert manifest["case_counts"] == {"historical_synthetic_label_cases": 7,
                                        "held_out_evaluation_cases": 7, "per_class_per_partition": 1}
    assert manifest["class_distribution"] == {cause: 1 for cause in CAUSES}
    assert manifest["split"]["overlap_counts"] == {"holding": 0, "issuer_cusip": 0, "issuer_name": 0,
                                                      "fund_series": 0, "filing_period": 0}
    pd.testing.assert_frame_equal(first.dataset.observable["exceptions"], second.dataset.observable["exceptions"])
    pd.testing.assert_frame_equal(first.dataset.ground_truth, second.dataset.ground_truth)
    exceptions = first.dataset.observable["exceptions"]
    assert exceptions.source_accession.nunique() == len(exceptions)
    assert (exceptions.injected_comparison_balance > 0).all()
    assert (exceptions.injected_comparison_unit_value > 0).all()
    assert all(case.human_validated is False for case in first.memory_cases)
    for case in observable_cases(first.dataset):
        context = json.dumps(case.to_context())
        for key in ("root_cause", "injection_mechanism", "difficulty", "perturbation_applied"):
            assert key not in context
    evidence = first.dataset.observable["evidence_records"]
    assert set(evidence.source_type) == {"PUBLIC_FILING", "SYNTHETIC_RECONCILIATION"}
    assert evidence.loc[evidence.source_type.eq("PUBLIC_FILING"), "source_sha256"].notna().all()
    assert evidence.loc[evidence.source_type.eq("SYNTHETIC_RECONCILIATION"), "source_sha256"].isna().all()


def test_real_benchmark_rejects_corrupt_perturbation_and_source_overlap(tmp_path):
    source = _sec_bundle(tmp_path)
    selected = select_disjoint_holdings(source, seed=42, memory_period="30-SEP-2025",
                                        evaluation_period="31-OCT-2025", per_partition=7)
    snapshot = load_nport_tables(source, accessions=list(selected.memory_accessions + selected.evaluation_accessions),
                                 holding_keys=selected.memory_keys | selected.evaluation_keys)
    original = snapshot.holdings.copy(deep=True)
    suite = build_real_nport_suite(snapshot, memory_keys=selected.memory_keys,
                                   evaluation_keys=selected.evaluation_keys, seed=42, per_family=1)
    pd.testing.assert_frame_equal(original, snapshot.holdings)
    suite.dataset.observable["exceptions"].loc[0, "injected_comparison_balance"] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        validate_suite(suite, snapshot, per_family=1)
    with pytest.raises(ValueError, match="disjoint"):
        build_real_nport_suite(snapshot, memory_keys=selected.memory_keys,
                               evaluation_keys=selected.memory_keys, seed=42, per_family=1)
