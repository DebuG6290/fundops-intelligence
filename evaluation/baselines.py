"""Shared, leakage-conscious adapters for the four research approaches.

Ground truth and difficulty labels deliberately do not appear on
``ObservableCase``. They are joined by the evaluator only after prediction.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from src.agents.agent_loop import InvestigationAgentLoop
from src.agents.schemas import InvestigationReport
from src.agents.tool_registry import ToolDefinition, ToolRegistry


FEATURE_COLUMNS = (
    "price_variance_pct", "quantity_variance_pct", "nav_contribution_pct",
    "vendor_disagreement_pct", "transaction_age_days", "settlement_pending",
    "corporate_action_flag", "fx_deviation_pct", "source_conflict_count",
    "record_delay_hours",
)


@dataclass(frozen=True)
class ObservableCase:
    """The investigator-visible, allowlisted view of one synthetic case."""

    exception_id: str
    features: dict[str, Any]
    evidence: tuple[dict[str, Any], ...] = ()
    historical_analogies: tuple[dict[str, Any], ...] = ()
    operational_notes: tuple[dict[str, Any], ...] = ()

    def to_context(self) -> dict[str, Any]:
        # Construct anew from allowlisted fields; never serialize source rows.
        return {
            "exception_id": self.exception_id,
            "observable_signals": dict(self.features),
            "evidence_catalog": list(self.evidence),
            "historical_analogies": list(self.historical_analogies),
            "operational_notes": list(self.operational_notes),
        }


@dataclass
class BaselineResult:
    approach: str
    predicted_root_cause: str | None = None
    ranked_hypotheses: list[dict[str, Any]] = field(default_factory=list)
    cited_evidence_ids: list[str] = field(default_factory=list)
    recommendation_category: str | None = None
    escalation_required: bool | None = None
    contradiction_detected: bool | None = None
    latency_seconds: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    provider: str | None = None
    model: str | None = None
    error: str | None = None

    @property
    def failed(self) -> bool:
        return self.error is not None


class Baseline(Protocol):
    name: str

    def predict(self, case: ObservableCase) -> BaselineResult: ...


def observable_cases(dataset: Any) -> list[ObservableCase]:
    """Convert generated frames to a strict observable-only contract."""
    exceptions = dataset.observable["exceptions"]
    evidence = dataset.observable.get("evidence_records", pd.DataFrame())
    notes = dataset.observable.get("operational_notes", pd.DataFrame())
    history = dataset.observable.get("historical_cases", pd.DataFrame())
    rows: list[ObservableCase] = []
    for row in exceptions.to_dict(orient="records"):
        exception_id = str(row["exception_id"])
        # The feature list is explicit: neither labels, identifiers, nor
        # difficulty can accidentally ride along as model context.
        features = {name: row[name] for name in FEATURE_COLUMNS if name in row}
        records = evidence.loc[evidence.exception_id == exception_id].to_dict("records") if not evidence.empty else []
        operational = notes.loc[notes.exception_id == exception_id]
        analogies = history.loc[history.exception_type == row.get("exception_type")].head(3).to_dict("records") if not history.empty else []
        rows.append(ObservableCase(
            exception_id=exception_id,
            features=features,
            evidence=tuple(_json_safe(item) for item in records),
            historical_analogies=tuple(_json_safe(item) for item in analogies),
            operational_notes=tuple(_json_safe(item) for item in operational.to_dict(orient="records")),
        ))
    return rows


class RuleBaseline:
    name = "rules"

    def predict(self, case: ObservableCase) -> BaselineResult:
        start = time.perf_counter()
        row = case.features
        if not case.evidence:
            cause = "INSUFFICIENT_EVIDENCE"
        elif row.get("source_conflict_count", 0) >= 2:
            cause = "CONFLICTING_EVIDENCE"
        else:
            scores = {
                "STALE_PRICE": abs(float(row.get("price_variance_pct", 0))) / 10,
                "MISSING_TRANSACTION": abs(float(row.get("quantity_variance_pct", 0))) / 15,
                "WRONG_QUANTITY": abs(float(row.get("quantity_variance_pct", 0))) / 25,
                "CORPORATE_ACTION": 1.5 if row.get("corporate_action_flag") else 0,
                "FX_MISMATCH": abs(float(row.get("fx_deviation_pct", 0))) / 4,
                "PENDING_SETTLEMENT": 1.5 if row.get("settlement_pending") else 0,
                "VENDOR_DISCREPANCY": float(row.get("vendor_disagreement_pct", 0)) / 8,
            }
            cause = max(scores, key=scores.get)
        escalation = cause in {"INSUFFICIENT_EVIDENCE", "CONFLICTING_EVIDENCE"}
        return BaselineResult(
            approach=self.name, predicted_root_cause=cause,
            ranked_hypotheses=[{"root_cause": cause}],
            cited_evidence_ids=[], recommendation_category=("INVESTIGATE_FURTHER" if escalation else "REVIEW_RECOMMENDATION"),
            escalation_required=escalation, contradiction_detected=(cause == "CONFLICTING_EVIDENCE"),
            latency_seconds=time.perf_counter() - start,
            provider="python", model="transparent-rule-heuristic",
        )


class ClassicalMLBaseline:
    name = "classical_ml_random_forest"

    def __init__(self, model: RandomForestClassifier, encoder: LabelEncoder) -> None:
        self.model, self.encoder = model, encoder

    @classmethod
    def fit(cls, frame: pd.DataFrame, labels: np.ndarray, train_indices: np.ndarray, *, seed: int = 42) -> "ClassicalMLBaseline":
        X = _features(frame)
        encoder = LabelEncoder().fit(labels[train_indices])
        model = RandomForestClassifier(
            n_estimators=160, min_samples_leaf=2, class_weight="balanced",
            random_state=seed, n_jobs=-1,
        )
        model.fit(X.iloc[train_indices], encoder.transform(labels[train_indices]))
        return cls(model, encoder)

    def predict(self, case: ObservableCase) -> BaselineResult:
        start = time.perf_counter()
        vector = pd.DataFrame([{name: case.features.get(name, 0) for name in FEATURE_COLUMNS}])
        vector["settlement_pending"] = vector["settlement_pending"].astype(int)
        vector["corporate_action_flag"] = vector["corporate_action_flag"].astype(int)
        probabilities = self.model.predict_proba(vector)[0]
        order = np.argsort(probabilities)[::-1]
        ranked = [
            {"root_cause": str(self.encoder.inverse_transform([int(self.model.classes_[i])])[0]), "score": float(probabilities[i])}
            for i in order[:5]
        ]
        cause = ranked[0]["root_cause"]
        escalation = cause in {"INSUFFICIENT_EVIDENCE", "CONFLICTING_EVIDENCE"}
        return BaselineResult(
            approach=self.name, predicted_root_cause=cause, ranked_hypotheses=ranked,
            recommendation_category="INVESTIGATE_FURTHER" if escalation else "REVIEW_RECOMMENDATION",
            escalation_required=escalation, contradiction_detected=(cause == "CONFLICTING_EVIDENCE"),
            latency_seconds=time.perf_counter() - start,
            provider="scikit-learn", model="RandomForestClassifier",
        )


class SingleLLMBaseline:
    """One Sarvam completion, without tool calls or deterministic challenge."""

    name = "single_llm"

    def __init__(self, provider: Any) -> None:
        self.provider = provider

    def predict(self, case: ObservableCase) -> BaselineResult:
        prompt = _investigation_prompt()
        started = time.perf_counter()
        try:
            response = self.provider.create_response(
                system_instructions=prompt, context=case.to_context(), tools=[]
            )
            raw = getattr(response, "output_text", "") or ""
            try:
                payload = json.loads(raw)
                if not isinstance(payload, dict):
                    raise ValueError("Investigation report must be a JSON object")
                payload["human_review_required"] = True
                report = InvestigationReport.model_validate(payload)
            except Exception as validation_error:
                repair = getattr(self.provider, "repair_structured_response", None)
                if repair is None:
                    raise
                response = repair(
                    invalid_output=raw,
                    validation_error=str(validation_error),
                    system_instructions=prompt,
                )
                payload = json.loads(getattr(response, "output_text", "") or "")
                if not isinstance(payload, dict):
                    raise ValueError("Investigation report must be a JSON object")
                payload["human_review_required"] = True
                report = InvestigationReport.model_validate(payload)
            valid_ids = {str(item.get("evidence_id")) for item in case.evidence}
            citations = [item for item in report.cited_evidence_ids if item in valid_ids]
            hypotheses = _report_hypotheses(report)
            usage = getattr(self.provider, "last_call", {}) or {}
            return BaselineResult(
                approach=self.name, predicted_root_cause=report.probable_root_cause,
                ranked_hypotheses=hypotheses, cited_evidence_ids=citations,
                recommendation_category=report.recommendation_category,
                escalation_required=report.escalation_required, contradiction_detected=None,
                latency_seconds=_latency(usage, started), input_tokens=usage.get("input_tokens"),
                output_tokens=usage.get("output_tokens"), provider=usage.get("provider", "sarvam"),
                model=usage.get("model", getattr(self.provider, "model", None)),
            )
        except Exception as exc:
            return _failed_result(self.name, exc, self.provider, started)


class AgenticLLMBaseline:
    """Sarvam tool-using investigator plus a deterministic challenge check."""

    name = "agentic_llm"

    def __init__(self, provider: Any) -> None:
        self.provider = provider

    def predict(self, case: ObservableCase) -> BaselineResult:
        evidence = list(case.evidence)
        analogies = list(case.historical_analogies)
        registry = ToolRegistry([
            ToolDefinition("retrieve_current_evidence", "Retrieve current-case evidence by its observable evidence IDs.", {
                "type": "object", "properties": {"evidence_ids": {"type": "array", "items": {"type": "string"}}},
                "required": ["evidence_ids"], "additionalProperties": False,
            }, lambda evidence_ids: [item for item in evidence if item.get("evidence_id") in evidence_ids]),
            ToolDefinition("retrieve_historical_analogies", "Retrieve prior cases for analogy only; these are not evidence of current cause.", {
                "type": "object", "properties": {}, "required": [], "additionalProperties": False,
            }, lambda: analogies),
        ])
        started = time.perf_counter()
        try:
            run = InvestigationAgentLoop(self.provider, registry).run(
                _investigation_prompt() + " Use tools to retrieve evidence and historical analogies. Historical cases are analogies, not current-case proof. Cite only current evidence IDs.",
                case.to_context(),
            )
            if run.report is None:
                raise ValueError("No validated investigation report was returned")
            report = run.report
            hypotheses = _report_hypotheses(report)
            leading_id = hypotheses[0].get("hypothesis_id") if hypotheses else None
            assessments = getattr(report, "evidence_assessments", [])
            cited = {str(item) for item in report.cited_evidence_ids}
            valid_ids = {str(item.get("evidence_id")) for item in evidence}
            contradiction = any(
                getattr(item, "hypothesis_id", None) == leading_id
                and getattr(item, "relationship", None) == "CONTRADICTS"
                and getattr(item, "evidence_id", None) in cited & valid_ids
                for item in assessments
            )
            ambiguity = len(hypotheses) > 1 and abs(float(hypotheses[0].get("confidence", 0)) - float(hypotheses[1].get("confidence", 0))) < .15
            supporting = any(
                getattr(item, "hypothesis_id", None) == leading_id
                and getattr(item, "relationship", None) == "SUPPORTS"
                and getattr(item, "evidence_id", None) in cited & valid_ids
                for item in assessments
            )
            escalation = bool(contradiction or ambiguity or not supporting or not cited)
            usage = getattr(self.provider, "last_call", {}) or {}
            return BaselineResult(
                approach=self.name, predicted_root_cause=report.probable_root_cause,
                ranked_hypotheses=hypotheses, cited_evidence_ids=sorted(cited & valid_ids),
                recommendation_category="INVESTIGATE_FURTHER" if escalation else report.recommendation_category,
                escalation_required=escalation, contradiction_detected=contradiction,
                latency_seconds=run.telemetry.latency_seconds or _latency(usage, started),
                input_tokens=run.telemetry.input_tokens, output_tokens=run.telemetry.output_tokens,
                provider=run.telemetry.provider or "sarvam", model=run.telemetry.model or getattr(self.provider, "model", None),
                error=None,
            )
        except Exception as exc:
            return _failed_result(self.name, exc, self.provider, started)


def _investigation_prompt() -> str:
    return (
        "Investigate the exception using only observable signals and current-case evidence. "
        "Return valid InvestigationReport JSON with up to five ranked hypotheses, cited_evidence_ids, "
        "recommendation_category, and escalation_required. Never invent evidence or infer ground truth. "
        "Confidence is an investigator score, not a probability. Human review is mandatory."
    )


def _report_hypotheses(report: InvestigationReport) -> list[dict[str, Any]]:
    if report.hypotheses:
        return [item.model_dump() for item in report.hypotheses]
    return [{"hypothesis_id": "HYP-001", "root_cause": report.probable_root_cause, "confidence": report.confidence}]


def _failed_result(name: str, exc: Exception, provider: Any, started: float) -> BaselineResult:
    usage = getattr(provider, "last_call", {}) or {}
    import os
    key = os.getenv("SARVAM_API_KEY", "")
    message = str(exc).replace(key, "[redacted]") if key else str(exc)
    return BaselineResult(
        approach=name, latency_seconds=_latency(usage, started),
        input_tokens=usage.get("input_tokens"), output_tokens=usage.get("output_tokens"),
        provider=usage.get("provider", "sarvam"), model=usage.get("model", getattr(provider, "model", None)),
        error=f"{type(exc).__name__}: {message[:300]}",
    )


def _latency(usage: dict[str, Any], started: float) -> float:
    return float(usage.get("latency_seconds") or (time.perf_counter() - started))


def _features(frame: pd.DataFrame) -> pd.DataFrame:
    X = frame.loc[:, FEATURE_COLUMNS].copy()
    X["settlement_pending"] = X["settlement_pending"].astype(int)
    X["corporate_action_flag"] = X["corporate_action_flag"].astype(int)
    return X


def _json_safe(record: dict[str, Any]) -> dict[str, Any]:
    return {key: value.item() if isinstance(value, np.generic) else value for key, value in record.items()}
