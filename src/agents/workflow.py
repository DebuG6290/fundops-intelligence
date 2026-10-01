from __future__ import annotations

from dataclasses import asdict
from typing import Any

from src.agents.corporate_action_agent import CorporateActionInvestigationAgent
from src.agents.evidence_challenge import EvidenceChallengeAgent, apply_challenge
from src.agents.resolution import ResolutionAgent
from src.agents.router import InvestigationRouter
from src.agents.rule_based_investigator import RuleBasedInvestigator
from src.agents.transaction_agent import TransactionInvestigationAgent
from src.data.scenarios import InvestigationScenario
from src.data.scenarios_extra import (
    CorporateActionScenario,
    TransactionMismatchScenario,
)
from src.memory.cases import CaseMemory


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
        return {
            "route": route,
            "exception_id": scenario.exception_id,
            "known_root_cause": scenario.known_root_cause,
            "report": run.report.model_dump(),
            "trace": [item.model_dump() for item in run.trace],
            "telemetry": run.telemetry.as_dict(),
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
        return {
            "route": route,
            "exception_id": scenario.exception_id,
            "known_root_cause": scenario.known_root_cause,
            "report": run.report.model_dump(),
            "trace": [item.model_dump() for item in run.trace],
            "telemetry": run.telemetry.as_dict(),
        }
