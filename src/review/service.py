from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from src.models.evidence import EvidenceItem
from src.review.models import HumanReviewRecord, HumanReviewSubmission


class HumanReviewService:
    """Append-only in-memory audit store, replaceable by a persistence adapter.

    This service receives only review data and evidence references. It has no
    access to financial records and cannot apply accounting changes.
    """

    def __init__(self) -> None:
        self._records: dict[str, HumanReviewRecord] = {}
        self._history_by_exception: dict[str, list[str]] = defaultdict(list)

    def submit_review(
        self,
        submission: HumanReviewSubmission | dict,
        available_evidence: Iterable[EvidenceItem],
    ) -> HumanReviewRecord:
        validated = HumanReviewSubmission.model_validate(submission)
        items = list(available_evidence)
        evidence_by_id = {item.evidence_id: item for item in items}
        if len(evidence_by_id) != len(items):
            raise ValueError("Evidence catalog contains duplicate evidence IDs")

        supporting = set(validated.supporting_evidence_references)
        counter = set(validated.counter_evidence_references)
        if supporting & counter:
            raise ValueError("An evidence item cannot be both supporting and counter-evidence")

        for reference in supporting:
            item = evidence_by_id.get(reference)
            if item is None or not item.supports:
                raise ValueError(f"Unknown or non-supporting evidence reference: {reference}")
        for reference in counter:
            item = evidence_by_id.get(reference)
            if item is None or not item.contradicts:
                raise ValueError(f"Unknown or non-counter evidence reference: {reference}")

        record = HumanReviewRecord(**validated.model_dump())
        self._records[record.review_id] = record
        self._history_by_exception[record.exception_id].append(record.review_id)
        return record

    def get_review(self, review_id: str) -> HumanReviewRecord | None:
        return self._records.get(review_id)

    def get_review_history(self, exception_id: str) -> tuple[HumanReviewRecord, ...]:
        return tuple(
            self._records[review_id]
            for review_id in self._history_by_exception.get(exception_id, ())
        )
