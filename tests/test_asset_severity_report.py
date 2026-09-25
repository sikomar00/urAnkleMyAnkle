import pandas as pd

from src.asset_severity_report import (
    build_failure_profile,
    render_experiment_summary,
)


def _profile_inputs():
    dates = pd.to_datetime(["2023-12-01", "2024-01-01", "2024-07-01"])
    raw_rows = pd.DataFrame(
        {
            "transaction_date": dates.repeat(2),
            "machine_type": ["Press"] * 6,
            "asset_tag": ["A-1"] * 6,
            "breakdown_flag": [0, 1, 0, 0, 1, 1],
        }
    )
    daily_rows = pd.DataFrame(
        {
            "transaction_date": dates,
            "machine_type": ["Press"] * 3,
            "asset_tag": ["A-1"] * 3,
            "failure_points": [4, 0, 12],
        }
    )
    return raw_rows, daily_rows


def test_failure_profile_keeps_rate_definitions():
    raw_rows, daily_rows = _profile_inputs()

    profile = build_failure_profile(
        raw_rows,
        daily_rows,
        high_risk_thresholds=(12, 13),
        validation_start="2024-01-01",
        test_start="2024-07-01",
    )

    assert {
        "part_breakdown_rate",
        "asset_issue_day_rate",
        "asset_high_risk_day_rate",
        "mean_failure_points",
        "median_failure_points",
    } <= set(profile)
    assert set(profile.scope_kind) == {"machine_type", "asset_tag"}
    assert set(profile.high_risk_threshold) == {12, 13}
    press_12 = profile.query(
        "split == 'test' and scope_kind == 'machine_type' "
        "and high_risk_threshold == 12"
    ).iloc[0]
    assert press_12["part_breakdown_rate"] == 1.0
    assert press_12["asset_high_risk_day_rate"] == 1.0


def test_summary_states_proxy_limit_and_threshold_difference():
    metrics = pd.DataFrame(
        [
            {
                "high_risk_threshold": threshold,
                "feature_set": feature_set,
                "scope_kind": "overall",
                "scope_name": "all",
                "model": "random_forest",
                "selected_model": True,
                "selected_feature_set": feature_set == "D",
                "split": "test",
                "status": "ok",
                "accuracy": 0.6,
                "macro_f1": macro_f1,
                "high_risk_precision": 0.4,
                "high_risk_recall": 0.3,
            }
            for threshold in (12, 13)
            for feature_set, macro_f1 in (("A", 0.48), ("D", 0.50))
        ]
    )
    predictions = pd.DataFrame(
        [
            {
                "high_risk_threshold": threshold,
                "feature_set": "D",
                "scope_kind": "overall",
                "scope_name": "all",
                "selected_feature_set": True,
                "actual_failure_points": 12,
                "actual_level": "high_risk" if threshold == 12 else "risk",
                "predicted_level": "risk",
            }
            for threshold in (12, 13)
        ]
    )
    raw_rows, daily_rows = _profile_inputs()
    profile = build_failure_profile(
        raw_rows,
        daily_rows,
        high_risk_thresholds=(12, 13),
        validation_start="2024-01-01",
        test_start="2024-07-01",
    )

    text = render_experiment_summary(metrics, predictions, profile)

    for phrase in (
        "실제 기계 정지율이 아닙니다",
        "12점 기준",
        "13점 기준",
        "12점 장비일",
        "Macro F1",
    ):
        assert phrase in text
