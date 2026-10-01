from evaluation.run_baseline import run_price_baseline


def test_baseline_evaluation_runs():
    result = run_price_baseline(case_count=5)

    assert result.total_cases == 5
    assert 0 <= result.root_cause_accuracy <= 1
