from pprint import pprint

from src.data.scenarios import create_price_exception_scenario
from src.tools.nav_tools import (
    calculate_nav_variance_tool,
    compare_price_sources_tool,
    get_fund_snapshot_tool,
    identify_top_contributors_tool,
)


def main() -> None:
    scenario = create_price_exception_scenario()

    print("\n=== FUND SNAPSHOT ===")
    pprint(get_fund_snapshot_tool(scenario))

    print("\n=== NAV EXCEPTION ===")
    pprint(calculate_nav_variance_tool(scenario))

    print("\n=== TOP NAV CONTRIBUTORS ===")
    for record in identify_top_contributors_tool(scenario):
        print(
            record["security_id"],
            f"| contribution={record['contribution_pct']:.2f}%",
            f"| value_diff={record['value_difference']:.2f}",
        )

    print("\n=== PRICE SOURCE CHECK ===")
    pprint(
        compare_price_sources_tool(
            scenario,
            scenario.culprit_security_id,
        )
    )


if __name__ == "__main__":
    main()
