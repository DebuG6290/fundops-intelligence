"""Canonical operational root-cause keys and legacy research-label aliases.

The optional ``:record_id`` scopes a cause to a security or transaction.
Historical and benchmark labels may be normalized for compatibility, but a
historical case never becomes current-case evidence by virtue of its label.
"""

from __future__ import annotations

from enum import Enum
from typing import Any


class RootCauseCode(str, Enum):
    PRICE_EXCEPTION = "PRICE_EXCEPTION"
    MISSING_TRANSACTION = "MISSING_TRANSACTION"
    EXTRA_TRANSACTION = "EXTRA_TRANSACTION"
    TRANSACTION_QUANTITY_MISMATCH = "TRANSACTION_QUANTITY_MISMATCH"
    TRANSACTION_TYPE_MISMATCH = "TRANSACTION_TYPE_MISMATCH"
    CORPORATE_ACTION = "CORPORATE_ACTION"
    WRONG_SECURITY_MAPPING = "WRONG_SECURITY_MAPPING"
    FX_MISMATCH = "FX_MISMATCH"
    PENDING_SETTLEMENT = "PENDING_SETTLEMENT"
    DUPLICATE_TRANSACTION = "DUPLICATE_TRANSACTION"
    VENDOR_DISCREPANCY = "VENDOR_DISCREPANCY"
    MULTI_CAUSE_EXCEPTION = "MULTI_CAUSE_EXCEPTION"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    CONFLICTING_EVIDENCE = "CONFLICTING_EVIDENCE"
    UNKNOWN = "UNKNOWN"


_ALIASES = {
    "STALE_PRICE": RootCauseCode.PRICE_EXCEPTION,
    "WRONG_QUANTITY": RootCauseCode.TRANSACTION_QUANTITY_MISMATCH,
}


def canonical_root_cause(value: str) -> str:
    """Reject invented labels; retain a valid optional record scope."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Root cause must be a nonblank taxonomy key")
    code, separator, scope = value.strip().partition(":")
    canonical = _ALIASES.get(code)
    if canonical is None:
        try:
            canonical = RootCauseCode(code)
        except ValueError as exc:
            raise ValueError(f"Unsupported root cause {code!r}; use the canonical taxonomy") from exc
    if separator and not scope.strip():
        raise ValueError("Root-cause record scope must not be blank")
    return f"{canonical.value}:{scope.strip()}" if separator else canonical.value


def root_cause_code(value: str) -> str:
    return canonical_root_cause(value).partition(":")[0]


def evidence_targets_hypothesis(target: str | None, root_cause: str, metadata: dict[str, Any]) -> bool:
    """Match a typed evidence target to a cause, honoring record scope."""
    if not target:
        return False
    try:
        target_key = canonical_root_cause(target)
        hypothesis_key = canonical_root_cause(root_cause)
    except ValueError:
        return target == root_cause  # legacy non-taxonomy reports stay compatible
    target_code, _, target_scope = target_key.partition(":")
    hypothesis_code, _, hypothesis_scope = hypothesis_key.partition(":")
    if target_code != hypothesis_code:
        return False
    if target_scope and hypothesis_scope:
        return target_scope == hypothesis_scope
    if hypothesis_scope and not target_scope:
        # Category-only evidence may support a scoped hypothesis only when the
        # deterministic record identifies the same security/transaction.
        return hypothesis_scope in {str(metadata.get(key)) for key in (
            "security_id", "transaction_id", "corporate_action_id",
        ) if metadata.get(key) is not None}
    return True
