"""Run the public-holdings / controlled-exception memory comparison.

Example: python -m evaluation.run_semi_synthetic --nport-dir path/to/quarter
"""

from __future__ import annotations

import argparse
import json

from evaluation.memory_experiment import evaluate_memory_experiment
from src.data.nport import load_nport_tables
from src.data.semi_synthetic import build_semi_synthetic_suite


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nport-dir", required=True, help="Extracted official SEC N-PORT quarterly TSV directory")
    parser.add_argument("--accession", action="append", help="Restrict to accession number (repeatable)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--evaluation-cases", type=int, default=120)
    args = parser.parse_args()
    snapshot = load_nport_tables(args.nport_dir, accessions=args.accession)
    suite = build_semi_synthetic_suite(snapshot, seed=args.seed, evaluation_count=args.evaluation_cases)
    print(json.dumps(evaluate_memory_experiment(suite), indent=2))


if __name__ == "__main__":
    main()

