import numpy as np
import pandas as pd
import pytest

from src.asset_features import (
    add_asset_severity,
    build_asset_daily,
    prepare_asset_current,
    prepare_asset_forecast,
    prepare_asset_score_current,
    prepare_asset_severity_current,
)


def _rows(sensor_second: float = 1.0) -> pd.DataFrame:
    base = {
        "transaction_date": pd.Timestamp("2024-01-01"),
        "asset_tag": "A-1",
        "machine_type": "Press",
        "plant_code": "P1",
        "temp_bearing_degC": 50.0,
        "temp_motor_degC": 60.0,
        "vibration_h_mms": 1.0,
        "vibration_v_mms": 1.1,
        "oil_pressure_bar": 5.0,
        "load_pct": 70.0,
        "shaft_rpm": 1000.0,
        "power_consumption_kw": 20.0,
    }
    return pd.DataFrame(
        [
            {
                **base,
                "part_no": "P-A",
                "criticality": "A",
                "breakdown_flag": 1,
            },
            {
                **base,
                "part_no": "P-B",
                "criticality": "A",
                "breakdown_flag": 1,
                "vibration_h_mms": sensor_second,
            },
            {
                **base,
                "part_no": "P-C",
                "criticality": "A",
                "breakdown_flag": 1,
            },
        ]
    )


def test_asset_current_uses_greater_than_or_equal_boundary():
    prepared = prepare_asset_current(_rows(), score_threshold=12)

    assert prepared.target == "target_ge_12"
    assert prepared.frame["failure_points"].tolist() == [12]
    assert prepared.frame[prepared.target].tolist() == [1]


def test_asset_daily_rejects_conflicting_sensor_values():
    with pytest.raises(ValueError, match="vibration_h_mms"):
        build_asset_daily(_rows(sensor_second=2.0))


def test_asset_current_rejects_nonpositive_score_threshold():
    with pytest.raises(ValueError, match="1 이상"):
        prepare_asset_current(_rows(), score_threshold=0)


def test_asset_severity_uses_all_integer_boundaries():
    frame = pd.DataFrame({"failure_points": [0, 1, 5, 6, 11, 12, 30]})

    result = add_asset_severity(frame)

    assert result["severity_level"].astype("string").tolist() == [
        "normal",
        "caution",
        "caution",
        "risk",
        "risk",
        "high_risk",
        "high_risk",
    ]
    assert result["severity_code"].tolist() == [0, 1, 1, 2, 2, 3, 3]


@pytest.mark.parametrize("bad_value", [-1, np.nan, "bad"])
def test_asset_severity_rejects_invalid_failure_points(bad_value):
    with pytest.raises(ValueError, match="failure_points"):
        add_asset_severity(pd.DataFrame({"failure_points": [bad_value]}))


def test_asset_score_and_severity_tasks_keep_targets_out_of_features():
    score_task = prepare_asset_score_current(_rows())
    severity_task = prepare_asset_severity_current(_rows())

    assert score_task.target == "failure_points"
    assert severity_task.target == "severity_level"
    assert "failure_points" not in score_task.features
    assert "severity_level" not in severity_task.features
    assert "severity_code" not in severity_task.features
    assert score_task.frame["severity_level"].astype("string").tolist() == [
        "high_risk"
    ]


@pytest.fixture
def asset_history():
    rows = []
    for offset, date in enumerate(pd.date_range("2024-01-01", periods=12, freq="D")):
        for part in ("P-A", "P-B", "P-C"):
            rows.append({
                "transaction_date": date, "asset_tag": "A-1",
                "machine_type": "Press", "plant_code": "P1", "part_no": part,
                "criticality": "A",
                "breakdown_flag": int(date == pd.Timestamp("2024-01-04")),
                "temp_bearing_degC": 50.0 + offset,
                "temp_motor_degC": 60.0 + offset,
                "vibration_h_mms": 1.0 + offset / 10,
                "vibration_v_mms": 1.1 + offset / 10,
                "oil_pressure_bar": 5.0, "load_pct": 70.0 + offset,
                "shaft_rpm": 1000.0 + offset,
                "power_consumption_kw": 20.0 + offset,
            })
    return pd.DataFrame(rows)


def test_asset_forecast_looks_only_at_next_seven_calendar_days(asset_history):
    prepared = prepare_asset_forecast(asset_history, 12, 7, "any")
    row = prepared.frame.loc[
        prepared.frame["transaction_date"].eq(pd.Timestamp("2024-01-01"))
    ].iloc[0]
    assert row[prepared.target] == 1
    assert row["label_end_date"] == pd.Timestamp("2024-01-08")


def test_asset_forecast_excludes_incomplete_calendar_window(asset_history):
    broken = asset_history[
        asset_history["transaction_date"] != pd.Timestamp("2024-01-05")
    ]
    prepared = prepare_asset_forecast(broken, 12, 7, "any")
    assert pd.Timestamp("2024-01-01") not in set(prepared.frame["transaction_date"])


def test_new_risk_excludes_current_positive_but_any_keeps_it(asset_history):
    new = prepare_asset_forecast(asset_history, 12, 7, "new").frame
    any_risk = prepare_asset_forecast(asset_history, 12, 7, "any").frame
    current_positive_date = pd.Timestamp("2024-01-04")
    assert current_positive_date not in set(new["transaction_date"])
    assert current_positive_date in set(any_risk["transaction_date"])


def test_future_sensor_change_does_not_change_past_features(asset_history):
    before = prepare_asset_forecast(asset_history, 12, 7, "any").frame
    changed = asset_history.copy()
    changed.loc[
        changed["transaction_date"].eq(pd.Timestamp("2024-01-09")), "load_pct"
    ] = 999
    after = prepare_asset_forecast(changed, 12, 7, "any").frame
    columns = [column for column in before if column.startswith("load_pct_")]
    pd.testing.assert_series_equal(
        before.loc[before["transaction_date"].eq(pd.Timestamp("2024-01-01")), columns].iloc[0],
        after.loc[after["transaction_date"].eq(pd.Timestamp("2024-01-01")), columns].iloc[0],
    )


def test_days_since_last_risk_uses_actual_event_date(asset_history):
    prepared = prepare_asset_forecast(asset_history, 12, 7, "any").frame
    day_after_risk = prepared.loc[
        prepared["transaction_date"].eq(pd.Timestamp("2024-01-05"))
    ].iloc[0]
    assert day_after_risk["days_since_last_risk"] == 1


def test_asset_lag_means_previous_calendar_day(asset_history):
    missing_day = asset_history[
        ~asset_history["transaction_date"].eq(pd.Timestamp("2024-01-02"))
    ]
    prepared = prepare_asset_forecast(missing_day, 12, 7, "any").frame
    january_third = prepared.loc[
        prepared["transaction_date"].eq(pd.Timestamp("2024-01-03"))
    ].iloc[0]
    assert pd.isna(january_third["load_pct_lag1"])
