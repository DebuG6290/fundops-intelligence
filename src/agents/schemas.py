from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class Hypothesis(BaseModel):
    hypothesis_id: str = Field(min_length=1)
    root_cause: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    required_evidence: list[str] = Field(default_factory=list)
    uncertainty: str = ""


class EvidenceAssessment(BaseModel):
    evidence_id: str = Field(min_length=1)
    hypothesis_id: str = Field(min_length=1)
    relationship: Literal["SUPPORTS", "CONTRADICTS", "CONTEXT"]


class InvestigationReport(BaseModel):
    """Compatibility report returned by the specialist LLM.

    Evidence text fields remain during the migration, but they are narrative
    claims, not factual EvidenceItems. Workflow code must build structured
    evidence only from deterministic or tool outputs.
    """

    probable_root_cause: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    observations: list[str]
    supporting_evidence: list[str]
    counter_evidence: list[str]
    recommended_next_step: str = Field(min_length=1)
    human_review_required: bool = True
    # Transitional multi-hypothesis extension. The probable-root-cause fields
    # remain for existing consumers while newer agents can return alternatives.
    hypotheses: list[Hypothesis] = Field(default_factory=list, max_length=5)
    # Evaluation-facing fields are optional for old providers and reports.
    # IDs must refer to supplied evidence; narrative claims never become facts.
    cited_evidence_ids: list[str] = Field(default_factory=list)
    recommendation_category: Literal["REVIEW_RECOMMENDATION", "INVESTIGATE_FURTHER"] | None = None
    escalation_required: bool | None = None
    evidence_assessments: list[EvidenceAssessment] = Field(default_factory=list)

    @model_validator(mode="after")
    def investigation_is_review_only_and_well_linked(self) -> InvestigationReport:
        if not self.human_review_required:
            raise ValueError("Human review is mandatory")
        hypothesis_ids = [item.hypothesis_id for item in self.hypotheses]
        if len(hypothesis_ids) != len(set(hypothesis_ids)):
            raise ValueError("Hypothesis IDs must be unique within a report")
        if any(item.hypothesis_id not in hypothesis_ids for item in self.evidence_assessments):
            raise ValueError("Evidence assessments must reference a hypothesis in this report")
        evidence_links = [(item.evidence_id, item.hypothesis_id) for item in self.evidence_assessments]
        if len(evidence_links) != len(set(evidence_links)):
            raise ValueError("Evidence-to-hypothesis assessment links must be unique")
        if len(self.cited_evidence_ids) != len(set(self.cited_evidence_ids)):
            raise ValueError("Cited evidence IDs must be unique")
        return self


class ToolTrace(BaseModel):
    tool_name: str
    arguments: dict[str, Any]
    result: Any
