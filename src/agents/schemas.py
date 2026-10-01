from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Hypothesis(BaseModel):
    hypothesis_id: str = Field(min_length=1)
    root_cause: str
    rationale: str
    confidence: float = Field(ge=0.0, le=1.0)
    required_evidence: list[str] = Field(default_factory=list)
    uncertainty: str = ""


class InvestigationReport(BaseModel):
    """Compatibility report returned by the specialist LLM.

    Evidence text fields remain during the migration, but they are narrative
    claims, not factual EvidenceItems. Workflow code must build structured
    evidence only from deterministic or tool outputs.
    """

    probable_root_cause: str
    confidence: float = Field(ge=0.0, le=1.0)
    observations: list[str]
    supporting_evidence: list[str]
    counter_evidence: list[str]
    recommended_next_step: str
    human_review_required: bool = True
    # Transitional multi-hypothesis extension. The probable-root-cause fields
    # remain for existing consumers while newer agents can return alternatives.
    hypotheses: list[Hypothesis] = Field(default_factory=list, max_length=5)


class ToolTrace(BaseModel):
    tool_name: str
    arguments: dict[str, Any]
    result: Any
