from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.data.scenarios import InvestigationScenario, create_price_exception_scenario
from src.data.scenarios_extra import create_corporate_action_scenario, create_transaction_mismatch_scenario


@dataclass
class FinalDemoScenario:
    scenario_id: str
    title: str
    description: str
    scenario: InvestigationScenario
    expected_root_cause: str
    demo_difficulty: str


def create_final_demo_scenarios(seed: int = 42) -> dict[str, FinalDemoScenario]:
    price = create_price_exception_scenario(seed)
    transactions = create_transaction_mismatch_scenario(seed)
    transaction_scenario = create_price_exception_scenario(seed)
    transaction_scenario.calculated_prices = transaction_scenario.expected_prices.copy(deep=True)
    transaction_scenario.expected_holdings = transaction_scenario.dataset.holdings.copy(deep=True)
    transaction_scenario.expected_transactions = transactions.expected_transactions.copy(deep=True)
    transaction_scenario.actual_transactions = transactions.expected_transactions.iloc[1:].copy(deep=True)
    txn = transaction_scenario.expected_transactions.iloc[0]
    sec = txn.security_id
    delta = -float(txn.quantity)
    transaction_scenario.dataset.holdings.loc[
        transaction_scenario.dataset.holdings.security_id.eq(sec), "quantity"
    ] += delta
    transaction_scenario.known_root_cause = f"MISSING_TRANSACTION:{txn.transaction_id}"
    transaction_scenario.exception_id = "EXC_NAV_0002"

    ca = create_price_exception_scenario(seed)
    ca.calculated_prices = ca.expected_prices.copy(deep=True)
    ca.expected_holdings = ca.dataset.holdings.copy(deep=True)
    actions = create_corporate_action_scenario(seed)
    ca.corporate_actions = actions.corporate_actions.copy(deep=True)
    action_sec = str(ca.corporate_actions.iloc[0].security_id)
    ca.dataset.holdings.loc[ca.dataset.holdings.security_id.eq(action_sec), "quantity"] *= 2
    ca.known_root_cause = f"CORPORATE_ACTION:{action_sec}"
    ca.exception_id = "EXC_NAV_0003"

    insufficient = create_price_exception_scenario(seed)
    insufficient.calculated_prices = insufficient.expected_prices.copy(deep=True)
    insufficient.expected_holdings = insufficient.dataset.holdings.copy(deep=True)
    # A small current-case discrepancy remains, below attribution thresholds.
    sec = str(insufficient.dataset.securities.iloc[0].security_id)
    insufficient.calculated_prices["price"] *= 1.02
    insufficient.calculated_prices["source"] = "UNVERIFIED_FEED"
    insufficient.known_root_cause = "UNKNOWN"
    insufficient.exception_id = "EXC_NAV_0004"

    return {
        "NAV_PRICE": FinalDemoScenario("NAV_PRICE", "Pricing feed discrepancy", "A material disagreement between current price sources affects NAV.", price, price.known_root_cause, "controlled"),
        "NAV_TRANSACTION": FinalDemoScenario("NAV_TRANSACTION", "Transaction-related NAV break", "A position difference aligns with an expected-versus-actual transaction mismatch.", transaction_scenario, "MISSING_TRANSACTION", "controlled"),
        "NAV_CORPORATE_ACTION": FinalDemoScenario("NAV_CORPORATE_ACTION", "Corporate-action-related NAV break", "An effective security event coincides with a position adjustment; pricing sources agree.", ca, "CORPORATE_ACTION", "controlled"),
        "NAV_INSUFFICIENT": FinalDemoScenario("NAV_INSUFFICIENT", "Insufficient evidence", "The NAV variance exceeds the detection threshold, but current records do not identify a sufficiently reliable cause.", insufficient, "UNKNOWN", "insufficient"),
    }


def get_final_demo_scenario(scenario_id: str, seed: int = 42) -> FinalDemoScenario:
    try:
        return create_final_demo_scenarios(seed)[scenario_id]
    except KeyError as exc:
        raise ValueError(f"Unsupported final demo scenario: {scenario_id}") from exc

