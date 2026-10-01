from __future__ import annotations

from dataclasses import asdict
from typing import Any

from src.agents.corporate_action_agent import CorporateActionInvestigationAgent
from src.agents.evidence_challenge import EvidenceChallengeAgent, apply_challenge
from src.agents.resolution import ResolutionAgent
from src.agents.router import InvestigationRouter
from src.agents.rule_based_investigator import RuleBasedInvestigator
from src.agents.state import InvestigationState
from src.agents.transaction_agent import TransactionInvestigationAgent
from src.data.scenarios import InvestigationScenario
from src.data.scenarios_extra import (
    CorporateActionScenario,
    TransactionMismatchScenario,
)
from src.memory.cases import CaseMemory
from src.models.evidence import EvidenceItem
from src.review.models import HumanDecision, HumanReviewRecord, HumanReviewSubmission
from src.review.service import HumanReviewService


def _state_from_specialist_report(
    scenario: TransactionMismatchScenario | CorporateActionScenario,
    report: Any,
    evidence: list[EvidenceItem] | None = None,
) -> InvestigationState:
    """Adapt a specialist report to the shared deterministic control state."""
    exception_type = (
        "TRANSACTION_MISMATCH"
        if isinstance(scenario, TransactionMismatchScenario)
        else "CORPORATE_ACTION"
    )
    state = InvestigationState(
        exception={
            "exception_id": scenario.exception_id,
            "exception_type": exception_type,
        },
        recommended_action=report.recommended_next_step,
        confidence=report.confidence,
    )

    for observation in report.observations or []:
        state.add_observation(
            "specialist_observation",
            {"claim": observation, "source_type": "SPECIALIST_OBSERVATION"},
            "specialist_report",
        )
    state.evidence.extend(evidence or [])

    for claim in report.supporting_evidence or []:
        state.add_observation(
            "reported_supporting_claim",
            {"claim": claim, "source_type": "SPECIALIST_OBSERVATION"},
            "specialist_report",
        )

    # Preserve structured hypotheses when a specialist report implementation
    # supplies them; the current InvestigationReport schema has one root cause.
    hypotheses = getattr(report, "hypotheses", None) or []
    for hypothesis in hypotheses:
        values = (
            hypothesis.model_dump()
            if hasattr(hypothesis, "model_dump")
            else hypothesis
        )
        state.add_hypothesis(
            root_cause=values["root_cause"],
            rationale=values.get("rationale", ""),
            confidence=values["confidence"],
        )
    if not state.hypotheses and report.probable_root_cause:
        state.add_hypothesis(
            root_cause=report.probable_root_cause,
            rationale="",
            confidence=report.confidence,
        )

    for counter_evidence in report.counter_evidence or []:
        state.add_observation(
            "reported_counter_evidence",
            {"claim": counter_evidence, "source_type": "SPECIALIST_OBSERVATION"},
            "specialist_report",
        )
    return state


def _control_outputs(
    workflow: InvestigationWorkflow,
    state: InvestigationState,
    report_evidence_is_narrative: bool = False,
) -> dict[str, Any]:
    challenge = workflow.challenge_agent.review(state)
    apply_challenge(state, challenge)
    resolution = workflow.resolution_agent.resolve(state)
    serialized_evidence = [item.model_dump(mode="json") for item in state.evidence]
    return {
        "challenge": asdict(challenge),
        "resolution": asdict(resolution),
        "status": state.status,
        "probable_root_cause": (
            state.hypotheses[0]["root_cause"] if state.hypotheses else None
        ),
        "confidence": state.confidence,
        "evidence": serialized_evidence,
        "supporting_evidence_references": [
            item.evidence_id for item in state.evidence if item.supports
        ],
        "counter_evidence_references": [
            item.evidence_id for item in state.evidence if item.contradicts
        ],
        "human_review_required": True,
        "report_evidence_is_narrative": report_evidence_is_narrative,
    }


class InvestigationWorkflow:
    """
    Orchestration layer.

    Deterministic NAV workflow remains available as the baseline. Transaction
    and corporate-action routes use specialist agent loops when a provider is
    supplied, allowing direct comparison of routing and agentic investigation.
    """

    def __init__(
        self,
        memory: CaseMemory,
        review_service: HumanReviewService | None = None,
    ) -> None:
        self.router = InvestigationRouter()
        self.memory = memory
        self.investigator = RuleBasedInvestigator(memory)
        self.challenge_agent = EvidenceChallengeAgent()
        self.resolution_agent = ResolutionAgent()
        self.review_service = review_service or HumanReviewService()

    def run_nav(self, scenario: InvestigationScenario) -> dict[str, Any]:
        route = self.router.route("NAV_DISCREPANCY")
        state = self.investigator.investigate(scenario)

        return {
            "route": route,
            "exception": state.exception,
            "observations": state.observations,
            "hypotheses": state.hypotheses,
            **_control_outputs(self, state),
        }

    def run_transaction(
        self,
        scenario: TransactionMismatchScenario,
        provider: Any,
    ) -> dict[str, Any]:
        route = self.router.route("TRANSACTION_MISMATCH")
        run = TransactionInvestigationAgent(
            self.memory, provider=provider
        ).investigate(scenario)
        report = run.report.model_dump()
        report["human_review_required"] = True
        state = _state_from_specialist_report(scenario, run.report, run.evidence)
        return {
            "route": route,
            "exception_id": scenario.exception_id,
            "known_root_cause": scenario.known_root_cause,
            "report": report,
            "trace": [item.model_dump() for item in run.trace],
            "telemetry": run.telemetry.as_dict(),
            **_control_outputs(self, state, report_evidence_is_narrative=True),
        }

    def run_corporate_action(
        self,
        scenario: CorporateActionScenario,
        provider: Any,
    ) -> dict[str, Any]:
        route = self.router.route("CORPORATE_ACTION")
        run = CorporateActionInvestigationAgent(
            self.memory, provider=provider
        ).investigate(scenario)
        report = run.report.model_dump()
        report["human_review_required"] = True
        state = _state_from_specialist_report(scenario, run.report, run.evidence)
        return {
            "route": route,
            "exception_id": scenario.exception_id,
            "known_root_cause": scenario.known_root_cause,
            "report": report,
            "trace": [item.model_dump() for item in run.trace],
            "telemetry": run.telemetry.as_dict(),
            **_control_outputs(self, state, report_evidence_is_narrative=True),
        }

    def submit_human_review(
        self,
        investigation_result: dict[str, Any],
        human_decision: HumanDecision | str,
        reviewer_reason: str,
    ) -> HumanReviewRecord:
        """Record an explicit human decision; this method cannot edit finance data."""
        exception_id = investigation_result.get("exception_id") or (
            investigation_result.get("exception", {}).get("exception_id")
        )
        report = investigation_result.get("report", {})
        resolution = investigation_result.get("resolution", {})
        if not resolution.get("decision") or not resolution.get("rationale"):
            raise ValueError("Investigation result is missing its resolution recommendation")
        root_cause = investigation_result.get("probable_root_cause") or report.get(
            "probable_root_cause"
        )
        evidence = [
            EvidenceItem.model_validate(item)
            for item in investigation_result.get("evidence", [])
        ]
        submission = HumanReviewSubmission(
            exception_id=exception_id,
            agent_root_cause=root_cause,
            agent_confidence=investigation_result.get(
                "confidence", report.get("confidence", 0.0)
            ),
            agent_recommendation=(
                f"{resolution['decision']}: {resolution['rationale']}"
            ),
            supporting_evidence_references=tuple(
                item.evidence_id for item in evidence if item.supports
            ),
            counter_evidence_references=tuple(
                item.evidence_id for item in evidence if item.contradicts
            ),
            human_decision=human_decision,
            reviewer_reason=reviewer_reason,
            human_review_required=True,
        )
        return self.review_service.submit_review(submission, evidence)
