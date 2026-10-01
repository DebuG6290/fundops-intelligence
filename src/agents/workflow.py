from __future__ import annotations

from dataclasses import asdict
from typing import Any

from src.agents.evidence_challenge import EvidenceChallengeAgent, apply_challenge
from src.agents.resolution import ResolutionAgent
from src.agents.router import InvestigationRouter
from src.agents.rule_based_investigator import RuleBasedInvestigator
from src.data.scenarios import InvestigationScenario
from src.memory.cases import CaseMemory


class InvestigationWorkflow:
    """
    First orchestration version.

    It demonstrates the intended state machine:
    route -> investigate -> challenge -> resolve.

    Specialist implementations can be swapped for LLM-driven agents later.
    """

    def __init__(self, memory: CaseMemory) -> None:
        self.router = InvestigationRouter()
        self.investigator = RuleBasedInvestigator(memory)
        self.challenge_agent = EvidenceChallengeAgent()
        self.resolution_agent = ResolutionAgent()

    def run_nav(
        self,
        scenario: InvestigationScenario,
    ) -> dict[str, Any]:
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
