from __future__ import annotations


class InvestigationRouter:
    """Route exceptions to the appropriate specialist workflow."""

    ROUTES = {
        "NAV_DISCREPANCY": "NAV_INVESTIGATOR",
        "TRANSACTION_MISMATCH": "TRANSACTION_INVESTIGATOR",
        "CORPORATE_ACTION": "CORPORATE_ACTION_INVESTIGATOR",
    }

    def route(self, exception_type: str) -> str:
        try:
            return self.ROUTES[exception_type]
        except KeyError as exc:
            raise ValueError(
                f"Unsupported exception type: {exception_type}"
            ) from exc
