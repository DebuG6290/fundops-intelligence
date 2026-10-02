"""Build a real-public-base, synthetic-incident benchmark without running models."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.baselines import observable_cases
from evaluation.profile_nport import profile_nport
from src.data.nport import NPortSnapshot, load_nport_tables
from src.data.nport_selection import select_disjoint_holdings
from src.data.semi_synthetic import CAUSES, SemiSyntheticSuite, build_real_nport_suite


def validate_suite(suite: SemiSyntheticSuite, snapshot: NPortSnapshot, *, per_family: int) -> dict:
    """Fail closed on split, provenance, label, numeric and evidence defects."""
    observable = suite.dataset.observable["exceptions"]
    truth = suite.dataset.ground_truth
    evidence = suite.dataset.observable["evidence_records"]
    memory = suite.memory_cases
    if len(observable) != len(truth) or observable.exception_id.duplicated().any() or truth.exception_id.duplicated().any():
        raise ValueError("Missing or duplicate evaluation cases")
    if len(memory) != per_family * len(CAUSES) or len({case.case_id for case in memory}) != len(memory):
        raise ValueError("Missing or duplicate memory cases")
    if Counter(truth.root_cause) != {cause: per_family for cause in CAUSES}:
        raise ValueError("Evaluation class distribution is not balanced")
    if Counter(case.root_cause for case in memory) != {cause: per_family for cause in CAUSES}:
        raise ValueError("Memory class distribution is not balanced")
    if observable.source_accession.str.cat(observable.source_holding_id, sep=":").duplicated().any():
        raise ValueError("Evaluation source holding repeated")
    if set(observable.source_accession + ":" + observable.source_holding_id) != set(suite.evaluation_holding_keys):
        raise ValueError("Evaluation source set differs from selected holdings")
    overlap = len(suite.memory_holding_keys & suite.evaluation_holding_keys)
    if overlap:
        raise ValueError("Source holding overlap between partitions")
    source = snapshot.holdings.set_index(snapshot.holdings.accession_number + ":" + snapshot.holdings.holding_id)
    if not source.index.is_unique:
        raise ValueError("Duplicate source holding key")
    selected = source.loc[list(suite.memory_holding_keys | suite.evaluation_holding_keys)]
    memory_source = source.loc[list(suite.memory_holding_keys)]
    evaluation_source = source.loc[list(suite.evaluation_holding_keys)]
    overlap_stats = {
        "holding": overlap,
        "issuer_cusip": len(set(memory_source.cusip.dropna()) & set(evaluation_source.cusip.dropna())),
        "issuer_name": len(set(memory_source.issuer_name.dropna().str.strip().str.casefold())
                           & set(evaluation_source.issuer_name.dropna().str.strip().str.casefold())),
        "fund_series": len(set(memory_source.series_id.dropna()) & set(evaluation_source.series_id.dropna())),
        "filing_period": len(set(memory_source.report_date.dropna()) & set(evaluation_source.report_date.dropna())),
    }
    if any(overlap_stats.values()):
        raise ValueError(f"Partition leakage: {overlap_stats}")
    if len(selected) != len(memory) + len(observable):
        raise ValueError("Source holdings reused within a partition")
    source_by_key = source
    for row in observable.itertuples(index=False):
        record = source_by_key.loc[f"{row.source_accession}:{row.source_holding_id}"]
        if (str(row.source_filed_balance_raw) != str(record.filed_balance_raw)
                or str(row.source_filed_value_raw) != str(record.filed_value_raw)
                or row.source_holding_sha256 != suite.source_sha256["FUND_REPORTED_HOLDING"]
                or not row.source_holding_file):
            raise ValueError("Missing or altered public source provenance")
    if truth[["source_accession", "source_holding_id", "source_holding_file",
              "source_holding_sha256", "perturbation_applied", "difficulty"]].isna().any().any():
        raise ValueError("Missing evaluator-only provenance")
    for field in ("source_reported_balance", "source_reported_value", "injected_comparison_balance",
                  "injected_comparison_unit_value", "price_variance_pct", "quantity_variance_pct",
                  "fx_deviation_pct", "financial_impact"):
        value = pd.to_numeric(observable[field], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(value).all() or (value < 0).any():
            raise ValueError(f"Invalid or non-finite {field}")
    if (observable[["source_reported_balance", "source_reported_value", "injected_comparison_balance",
                    "injected_comparison_unit_value"]] <= 0).any().any():
        raise ValueError("Impossible nonpositive source or comparison amount")
    if evidence.evidence_id.duplicated().any() or not set(evidence.source_type).issubset({
        "PUBLIC_FILING", "SYNTHETIC_RECONCILIATION",
    }):
        raise ValueError("Evidence IDs or source taxonomy invalid")
    public = evidence.loc[evidence.source_type.eq("PUBLIC_FILING")]
    synthetic = evidence.loc[evidence.source_type.eq("SYNTHETIC_RECONCILIATION")]
    if len(public) != len(observable) or public.source_sha256.isna().any() or synthetic.source_sha256.notna().any():
        raise ValueError("Public and synthetic evidence provenance mixed")
    if not set(synthetic.evidence_id).issuperset({item for refs in truth.relevant_evidence_ids for item in refs}):
        raise ValueError("Relevant evidence references missing or point to public filing")
    hidden = {"root_cause", "injection_mechanism", "secondary_causes", "difficulty",
              "perturbation_applied", "expected_escalation", "expected_recommendation_category"}
    for case in observable_cases(suite.dataset):
        context = case.to_context()
        if hidden & set(context) or hidden & set(context["observable_signals"]):
            raise ValueError("Answer-key fields leaked to investigator context")
        if hidden & {key for item in context["evidence_catalog"] for key in item}:
            raise ValueError("Answer-key fields leaked to evidence context")
    return overlap_stats


def build_manifest(directory: str | Path, *, quarter: str, seed: int = 42,
                   per_family: int = 100) -> tuple[dict, SemiSyntheticSuite]:
    """Profile first, then select/build cases; no evaluator is executed."""
    if quarter.lower() != "2025q4":
        raise ValueError("This two-period benchmark recipe is specific to SEC 2025Q4")
    profile = profile_nport(directory, quarter=quarter)
    selection = select_disjoint_holdings(directory, seed=seed, memory_period="30-SEP-2025",
                                         evaluation_period="31-OCT-2025", per_partition=per_family * len(CAUSES))
    keys = selection.memory_keys | selection.evaluation_keys
    snapshot = load_nport_tables(directory, accessions=list(selection.memory_accessions + selection.evaluation_accessions),
                                 holding_keys=keys)
    if {name: item["sha256"] for name, item in profile["files"].items()} != snapshot.source_sha256:
        raise ValueError("Source files changed after profiling")
    source_before = snapshot.holdings.copy(deep=True)
    suite = build_real_nport_suite(snapshot, memory_keys=selection.memory_keys,
                                   evaluation_keys=selection.evaluation_keys, seed=seed,
                                   per_family=per_family)
    pd.testing.assert_frame_equal(source_before, snapshot.holdings)
    overlap = validate_suite(suite, snapshot, per_family=per_family)
    return ({
        "dataset": "SEC N-PORT 2025Q4 real-public-base semi-synthetic benchmark",
        "source_profile": profile,
        "seed": seed,
        "filter": "ASSET_CAT=EC; UNIT=NS; nonmissing issuer name/CUSIP/currency/series; finite positive filed balance and value",
        "construction": "Unchanged SEC holding context plus controlled synthetic operational comparison; incident labels are synthetic, not SEC observations",
        "case_counts": {"historical_synthetic_label_cases": len(suite.memory_cases),
                        "held_out_evaluation_cases": len(suite.dataset.ground_truth),
                        "per_class_per_partition": per_family},
        "class_distribution": {cause: per_family for cause in CAUSES},
        "split": {"memory_report_date": "30-SEP-2025", "evaluation_report_date": "31-OCT-2025",
                  "available_eligible_series_by_period": selection.available_series_by_period,
                  "one_holding_per_series": True, "same_issuer_identifier_or_name_across_partitions": False,
                  "overlap_counts": overlap},
        "provenance": "Each generated case carries accession, holding ID, raw filed balance/value, currency, series, report date, source filename/hash, synthetic comparison fields, evidence IDs; labels and perturbation are evaluator-only",
        "limitations": [
            "N-PORT filings are as-filed public holdings, not verified prices or internal operations incidents.",
            "All root-cause labels, transaction/CA indicators, and operational comparisons are synthetic.",
            "Issuer CUSIP and series/period disjointness reduce but do not eliminate economic or manager-level dependence.",
            "The 100 cases per held-out class allow descriptive comparisons; uncertainty remains material.",
            "No model, memory experiment, or Sarvam evaluation was run when creating this benchmark.",
        ],
        "reproduce": "python -m evaluation.build_nport_benchmark --nport-dir <extracted 2025Q4 SEC N-PORT directory> --quarter 2025q4 --seed 42 --per-family 100 --output-dir <local output directory>",
    }, suite)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nport-dir", type=Path, required=True)
    parser.add_argument("--quarter", default="2025q4")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per-family", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest, suite = build_manifest(args.nport_dir, quarter=args.quarter,
                                     seed=args.seed, per_family=args.per_family)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    suite.dataset.observable["exceptions"].to_json(args.output_dir / "observable_exceptions.jsonl", orient="records", lines=True)
    suite.dataset.observable["evidence_records"].to_json(args.output_dir / "observable_evidence.jsonl", orient="records", lines=True)
    suite.dataset.ground_truth.to_json(args.output_dir / "hidden_ground_truth.jsonl", orient="records", lines=True)
    with (args.output_dir / "synthetic_historical_cases.jsonl").open("w", encoding="utf-8") as stream:
        for case in suite.memory_cases:
            stream.write(json.dumps(case.__dict__, default=str) + "\n")
    print(json.dumps({"case_counts": manifest["case_counts"], "overlap_counts": manifest["split"]["overlap_counts"],
                      "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
