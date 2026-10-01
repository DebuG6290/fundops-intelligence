from __future__ import annotations

import json

from evaluation.benchmark import evaluate_rules_and_ml
from src.data.benchmark import generate_benchmark_dataset


def main() -> None:
    dataset = generate_benchmark_dataset()
    print(json.dumps(evaluate_rules_and_ml(dataset), indent=2, default=str))


if __name__ == "__main__":
    main()
