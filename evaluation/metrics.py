from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class EvaluationResult:
    total_cases: int
    correct_root_causes: int
    root_cause_accuracy: float


def calculate_accuracy(predictions: Iterable[str], actuals: Iterable[str]) -> EvaluationResult:
    predictions = list(predictions)
    actuals = list(actuals)

    if len(predictions) != len(actuals):
        raise ValueError("predictions and actuals must have equal length")
    if not actuals:
        raise ValueError("evaluation set cannot be empty")

    correct = sum(pred == actual for pred, actual in zip(predictions, actuals))
    return EvaluationResult(
        total_cases=len(actuals),
        correct_root_causes=correct,
        root_cause_accuracy=correct / len(actuals),
    )
