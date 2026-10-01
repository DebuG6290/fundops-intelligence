from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


class HumanDecision(str, Enum):
    ACCEPT = "ACCEPT"
    REJECT = "REJECT"
    INVESTIGATE_FURTHER = "INVESTIGATE_FURTHER"


class HumanReviewSubmission(BaseModel):
    exception_id: str = Field(min_length=1)
    agent_root_cause: str | None
    agent_confidence: float = Field(ge=0.0, le=1.0)
    agent_recommendation: str = Field(min_length=1)
    supporting_evidence_references: tuple[str, ...] = ()
    counter_evidence_references: tuple[str, ...] = ()
    human_decision: HumanDecision
    reviewer_reason: str = Field(min_length=1)
    human_review_required: Literal[True] = True

    @field_validator(
        "exception_id", "agent_recommendation", "reviewer_reason", mode="before"
    )
    @classmethod
    def required_text_is_not_blank(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            raise ValueError("Value must not be blank")
        return value

    @field_validator("agent_root_cause", mode="before")
    @classmethod
    def root_cause_is_not_blank(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            raise ValueError("Value must not be blank")
        return value

    @field_validator(
        "supporting_evidence_references", "counter_evidence_references"
    )
    @classmethod
    def evidence_references_are_unique_and_nonblank(
        cls, value: tuple[str, ...]
    ) -> tuple[str, ...]:
        if any(not item.strip() for item in value):
            raise ValueError("Evidence references must not be blank")
        if len(set(value)) != len(value):
            raise ValueError("Evidence references must be unique")
        return value


class HumanReviewRecord(HumanReviewSubmission):
    model_config = ConfigDict(frozen=True)

    review_id: str = Field(default_factory=lambda: str(uuid4()), min_length=1)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("timestamp")
    @classmethod
    def timestamp_must_be_timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("Review timestamp must include a timezone")
        return value
