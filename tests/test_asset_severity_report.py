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
        + [
            {
                "high_risk_threshold": threshold,
                "feature_set": "D",
                "scope_kind": "machine_type",
                "scope_name": "Press",
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
    assert "| 12 | high_risk | risk | 1 |" in text
    assert "| 13 | risk | risk | 1 |" in text
    assert "| high_risk_threshold | actual_level | support | rate |" in text
    assert "| 12 | high_risk | 1 | 1.0000 |" in text


def test_summary_uses_machine_predictions_when_overall_is_absent():
    metrics = pd.DataFrame(
        [
            {
                "high_risk_threshold": 12,
                "feature_set": "A",
                "scope_kind": "machine_type",
                "scope_name": "Press",
                "model": "random_forest",
                "selected_model": True,
                "selected_feature_set": True,
                "split": "test",
                "status": "ok",
                "macro_f1": 0.5,
                "high_risk_precision": 0.4,
                "high_risk_recall": 0.3,
            }
        ]
    )
    predictions = pd.DataFrame(
        [
            {
                "high_risk_threshold": 12,
                "feature_set": "A",
                "scope_kind": "machine_type",
                "scope_name": "Press",
                "selected_feature_set": True,
                "actual_failure_points": 12,
                "actual_level": "high_risk",
                "predicted_level": "risk",
            }
        ]
    )
    raw_rows, daily_rows = _profile_inputs()
    profile = build_failure_profile(
        raw_rows,
        daily_rows,
        high_risk_thresholds=(12,),
        validation_start="2024-01-01",
        test_start="2024-07-01",
    )

    text = render_experiment_summary(metrics, predictions, profile)

    assert "| 12 | high_risk | risk | 1 |" in text
    assert "| 12 | high_risk | 1 | 1.0000 |" in text


def test_summary_includes_asset_performance():
    metrics = pd.DataFrame(
        [
            {
                "high_risk_threshold": 12,
                "feature_set": "D",
                "scope_kind": "asset_tag",
                "scope_name": "A-1",
                "source_scope_kind": "machine_type",
                "source_scope_name": "Press",
                "model": "logistic_regression",
                "selected_model": True,
                "selected_feature_set": True,
                "split": "test",
                "status": "ok",
                "macro_f1": 0.51,
                "high_risk_precision": 0.4,
                "high_risk_recall": 0.3,
            }
        ]
    )
    predictions = pd.DataFrame()
    raw_rows, daily_rows = _profile_inputs()
    profile = build_failure_profile(
        raw_rows,
        daily_rows,
        high_risk_thresholds=(12,),
        validation_start="2024-01-01",
        test_start="2024-07-01",
    )

    text = render_experiment_summary(metrics, predictions, profile)

    assert "개별 장비별 성능" in text
    assert "| 12 | A-1 | machine_type | Press | D | 0.5100 | 0.4000 | 0.3000 |" in text
