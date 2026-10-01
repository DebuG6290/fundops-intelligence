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


def _state_from_specialist_report(
    scenario: TransactionMismatchScenario | CorporateActionScenario,
    report: Any,
) -> InvestigationState:
    """Adapt a specialist report to the shared deterministic control state."""
    exception = dict(vars(scenario))
    exception["exception_type"] = (
        "TRANSACTION_MISMATCH"
        if isinstance(scenario, TransactionMismatchScenario)
        else "CORPORATE_ACTION"
    )
    state = InvestigationState(
        exception=exception,
        recommended_action=report.recommended_next_step,
        confidence=report.confidence,
    )

    for observation in report.observations or []:
        state.add_observation(
            "specialist_observation", observation, "specialist_report"
        )
    for evidence in report.supporting_evidence or []:
        state.evidence.append({"claim": evidence, "source": "specialist_report"})

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
            rationale="Leading hypothesis from specialist report.",
            confidence=report.confidence,
        )

    for counter_evidence in report.counter_evidence or []:
        state.add_observation(
            "counter_evidence",
            {"claim": counter_evidence, "contradicts": True},
            "specialist_report",
        )
    return state


def _control_outputs(
    workflow: InvestigationWorkflow,
    scenario: TransactionMismatchScenario | CorporateActionScenario,
    report: Any,
) -> dict[str, Any]:
    state = _state_from_specialist_report(scenario, report)
    challenge = workflow.challenge_agent.review(state)
    apply_challenge(state, challenge)
    resolution = workflow.resolution_agent.resolve(state)
    return {
        "challenge": asdict(challenge),
        "resolution": asdict(resolution),
        "status": state.status,
    }


class InvestigationWorkflow:
    """
    Orchestration layer.

    Deterministic NAV workflow remains available as the baseline. Transaction
    and corporate-action routes use specialist agent loops when a provider is
    supplied, allowing direct comparison of routing and agentic investigation.
    """

    def __init__(self, memory: CaseMemory) -> None:
        self.router = InvestigationRouter()
        self.memory = memory
        self.investigator = RuleBasedInvestigator(memory)
        self.challenge_agent = EvidenceChallengeAgent()
        self.resolution_agent = ResolutionAgent()

    def run_nav(self, scenario: InvestigationScenario) -> dict[str, Any]:
        route = self.router.route("NAV_DISCREPANCY")
        state = self.investigator.investigate(scenario)

        challenge = self.challenge_agent.review(state)
        apply_challenge(state, challenge)
        resolution = self.resolution_agent.resolve(state)

        return {
            "route": route,
            "exception": state.exception,
            "observations": state.observations,
            "hypotheses": state.hypotheses,
            "challenge": asdict(challenge),
            "resolution": asdict(resolution),
            "status": state.status,
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
        return {
            "route": route,
            "exception_id": scenario.exception_id,
            "known_root_cause": scenario.known_root_cause,
            "report": report,
            "trace": [item.model_dump() for item in run.trace],
            "telemetry": run.telemetry.as_dict(),
            **_control_outputs(self, scenario, run.report),
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
        return {
            "route": route,
            "exception_id": scenario.exception_id,
            "known_root_cause": scenario.known_root_cause,
            "report": report,
            "trace": [item.model_dump() for item in run.trace],
            "telemetry": run.telemetry.as_dict(),
            **_control_outputs(self, scenario, run.report),
        }
