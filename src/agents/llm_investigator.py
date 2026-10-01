"""Compatibility entry point for the Sarvam-backed NAV investigation agent."""

from __future__ import annotations

import os
from typing import Any

from src.agents.nav_agent import NavInvestigationAgent
from src.data.scenarios import InvestigationScenario
from src.llm.sarvam_provider import SarvamProvider
from src.memory.cases import CaseMemory


class LLMInvestigator:
    """Legacy name delegating to the provider-agnostic NAV agent loop."""

    def __init__(self, provider: Any, model: str, memory: CaseMemory) -> None:
        self.provider = provider
        self.model = model
        self.memory = memory

    def investigate(self, scenario: InvestigationScenario):
        return NavInvestigationAgent(self.memory, provider=self.provider).investigate(scenario)


def build_sarvam_investigator(memory: CaseMemory) -> LLMInvestigator:
    model = os.getenv("SARVAM_MODEL", "sarvam-105b")
    return LLMInvestigator(
        provider=SarvamProvider(model=model),
        model=model,
        memory=memory,
    )
