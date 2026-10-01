from __future__ import annotations

from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class EvidenceSourceType(str, Enum):
    DETERMINISTIC_ANALYTICS = "DETERMINISTIC_ANALYTICS"
    TOOL_RESULT = "TOOL_RESULT"
    PRICE_SOURCE = "PRICE_SOURCE"
    TRANSACTION_RECORD = "TRANSACTION_RECORD"
    CORPORATE_ACTION_RECORD = "CORPORATE_ACTION_RECORD"
    HISTORICAL_CASE = "HISTORICAL_CASE"
    SPECIALIST_OBSERVATION = "SPECIALIST_OBSERVATION"


class EvidenceItem(BaseModel):
    """A sourced claim with an explicit, optional relationship to a target.

    ``supports`` and ``contradicts`` refer to a hypothesis or exception key;
    neither means the evidence is context only. Reliability is left unknown
    unless a source provides a defensible value.
    """

    model_config = ConfigDict(frozen=True)

    evidence_id: str = Field(default_factory=lambda: str(uuid4()), min_length=1)
    source_type: EvidenceSourceType
    source_name: str = Field(min_length=1)
    claim: str = Field(min_length=1)
    exception_id: str | None = None
    supports_hypothesis: str | None = None
    contradicts_hypothesis: str | None = None
    # Transitional root-cause relationships retained for old serialized items.
    supports: str | None = None
    contradicts: str | None = None
    reliability: float | None = Field(default=None, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("evidence_id", "source_name", "claim", mode="before")
    @classmethod
    def required_text_is_not_blank(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            raise ValueError("Value must not be blank")
        return value

    @field_validator(
        "exception_id", "supports_hypothesis", "contradicts_hypothesis",
        "supports", "contradicts", mode="before"
    )
    @classmethod
    def relationship_target_is_not_blank(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            raise ValueError("Relationship target must not be blank")
        return value

    @model_validator(mode="after")
    def relationship_is_unambiguous(self) -> EvidenceItem:
        has_support = bool(self.supports_hypothesis or self.supports)
        has_contradiction = bool(self.contradicts_hypothesis or self.contradicts)
        if has_support and has_contradiction:
            raise ValueError("Evidence cannot support and contradict simultaneously")
        return self

    @property
    def direction(self) -> str:
        if self.supports_hypothesis or self.supports:
            return "SUPPORTS"
        if self.contradicts_hypothesis or self.contradicts:
            return "CONTRADICTS"
        return "CONTEXT"
