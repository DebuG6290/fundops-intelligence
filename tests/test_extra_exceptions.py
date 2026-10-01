from src.analytics.corporate_actions import find_effective_corporate_actions
from src.analytics.transactions import find_transaction_mismatches
from src.data.scenarios_extra import (
    create_corporate_action_scenario,
    create_transaction_mismatch_scenario,
)


def test_transaction_mismatch_is_detected():
    scenario = create_transaction_mismatch_scenario()
    result = find_transaction_mismatches(
        scenario.expected_transactions,
        scenario.actual_transactions,
    )
    assert not result.empty
    assert scenario.known_root_cause.split(":")[1] in result.transaction_id.tolist()


def test_corporate_action_is_detected():
    scenario = create_corporate_action_scenario()
    result = find_effective_corporate_actions(
        scenario.corporate_actions,
        scenario.corporate_actions.iloc[0]["effective_date"],
    )
    assert len(result) == 1
    assert result.iloc[0]["action_type"] == "STOCK_SPLIT"
