"""Paired, held-out memory OFF/ON experiment for semi-synthetic cases.

This is a deterministic check-prioritization baseline, not a claim about
Sarvam or human outcomes. Historical labels are exposed only *after* safe
observable-context retrieval and are treated as analogies, never evidence.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from evaluation.baselines import BaselineResult, ObservableCase, RuleBaseline, observable_cases
from src.data.semi_synthetic import CHECKS, SemiSyntheticSuite, observable_symptom_query
from src.memory.cases import CaseMemory, HistoricalCase


@dataclass(frozen=True)
class MemoryPrediction:
    exception_id: str
    result: BaselineResult
    first_check: str | None
    checks: tuple[str, ...]
    retrieved_case_ids: tuple[str, ...]


def evaluate_memory_experiment(suite: SemiSyntheticSuite, *, top_k: int = 3) -> dict:
    """Score the same cases with and without historical analogies.

    The experiment corpus has synthetic ground-truth labels and is *not*
    allowed into live human-validated memory. CaseMemory filters exception
    type before ranking and scores only observable symptoms/useful evidence.
    """
    if top_k < 1:
        raise ValueError("top_k must be positive")
    memory = CaseMemory(suite.memory_cases)
    cases = observable_cases(suite.dataset)
    rows = suite.dataset.observable["exceptions"].to_dict("records")
    truth = suite.dataset.ground_truth.set_index("exception_id")
    off: list[MemoryPrediction] = []
    on: list[MemoryPrediction] = []
    for row, case in zip(rows, cases, strict=True):
        start = time.perf_counter()
        baseline = RuleBaseline().predict(case)
        off_checks = _rank_checks(baseline)
        off.append(MemoryPrediction(case.exception_id, baseline, off_checks[0] if off_checks else None,
                                    tuple(off_checks), ()))
        analogies = memory.search(
            observable_symptom_query(row), exception_type=row["exception_type"], top_k=top_k,
            validated_only=False,
        )
        # Relevance is fixed before historical root causes are inspected.
        on_checks = _memory_first_checks(analogies, off_checks)
        predicted = _cause_for_check(on_checks[0]) if on_checks else baseline.predicted_root_cause
        result = BaselineResult(
            approach="rules_memory_on", predicted_root_cause=predicted,
            ranked_hypotheses=[{"root_cause": _cause_for_check(check)} for check in on_checks[:3]],
            cited_evidence_ids=list(baseline.cited_evidence_ids),
            recommendation_category=baseline.recommendation_category,
            escalation_required=baseline.escalation_required,
            contradiction_detected=baseline.contradiction_detected,
            latency_seconds=time.perf_counter() - start,
            provider=None, model=None,
        )
        on.append(MemoryPrediction(case.exception_id, result, on_checks[0] if on_checks else None,
                                   tuple(on_checks), tuple(item.case_id for item in analogies)))
    return {
        "design": "paired held-out deterministic rules; historical analogy check-prioritization",
        "source": "SEC N-PORT filed holdings plus synthetic operational perturbations",
        "source_sha256": dict(suite.source_sha256),
        "memory_cases": len(suite.memory_cases), "evaluation_cases": len(cases),
        "source_holding_overlap": len(suite.memory_holding_keys & suite.evaluation_holding_keys),
        "memory_off": _score(off, truth), "memory_on": _score(on, truth),
        "llm_results": "Not evaluated", "token_usage": None, "estimated_cost": None,
        "caveat": "Memory labels are synthetic ground truth, not human validation. Reused holdings within each partition and injected operational records limit external validity.",
    }


def _rank_checks(result: BaselineResult) -> list[str]:
    causes = [item.get("root_cause") for item in result.ranked_hypotheses]
    if result.predicted_root_cause not in causes:
        causes.insert(0, result.predicted_root_cause)
    return list(dict.fromkeys([*(CHECKS[cause] for cause in causes if cause in CHECKS), *CHECKS.values()]))


def _memory_first_checks(analogies: list[HistoricalCase], base_checks: list[str]) -> list[str]:
    suggested = [check for case in analogies for check in case.investigation_path if check in CHECKS.values()]
    return list(dict.fromkeys([*suggested, *base_checks]))


def _cause_for_check(check: str) -> str | None:
    return next((cause for cause, candidate in CHECKS.items() if candidate == check), None)


def _score(rows: list[MemoryPrediction], truth) -> dict:
    if not rows:
        return {"cases": 0, "root_cause_accuracy": None, "first_check_accuracy": None,
                "mean_check_steps_to_target": None, "mean_latency_seconds": None}
    causes = [str(truth.loc[row.exception_id, "root_cause"]) for row in rows]
    target_checks = [CHECKS[cause] for cause in causes]
    target_steps = []
    for row, target in zip(rows, target_checks, strict=True):
        # The check list is derivable from ranked hypotheses, not a completed
        # tool trace. This is a prioritization proxy, not actual tool calls.
        checks = row.checks
        target_steps.append(checks.index(target) + 1 if target in checks else None)
    available_steps = [step for step in target_steps if step is not None]
    return {
        "cases": len(rows),
        "root_cause_accuracy": sum(row.result.predicted_root_cause == cause for row, cause in zip(rows, causes, strict=True)) / len(rows),
        "first_check_accuracy": sum(row.first_check == target for row, target in zip(rows, target_checks, strict=True)) / len(rows),
        "mean_check_steps_to_target": sum(available_steps) / len(available_steps) if available_steps else None,
        "target_check_coverage": len(available_steps) / len(rows),
        "mean_latency_seconds": sum(row.result.latency_seconds or 0 for row in rows) / len(rows),
        "retrieval_rate": sum(bool(row.retrieved_case_ids) for row in rows) / len(rows),
        "incorrect_analogy_first_rate": sum(bool(row.retrieved_case_ids) and row.first_check != target
                                          for row, target in zip(rows, target_checks, strict=True)) / len(rows),
        "token_usage": None, "estimated_cost": None,
    }

