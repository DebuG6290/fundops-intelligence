from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class Hypothesis(BaseModel):
    root_cause: str
    rationale: str
    confidence: float = Field(ge=0.0, le=1.0)


class InvestigationReport(BaseModel):
    probable_root_cause: str
    confidence: float = Field(ge=0.0, le=1.0)
    observations: list[str]
    supporting_evidence: list[str]
    counter_evidence: list[str]
    recommended_next_step: str
    human_review_required: bool = True


class ToolTrace(BaseModel):
    tool_name: str
    arguments: dict[str, Any]
    result: Any
