import pandas as pd
import pytest

from src.asset_anomaly_features import (
    RobustNormalBaseline,
    build_sensor_history_features,
    prepare_asset_experiment_features,
)
from src.industrial_data import SENSOR_COLUMNS


@pytest.fixture
def normal_daily():
    rows = []
    for asset_tag, machine_type, offset in (
        ("A-1", "Press", 0.0),
        ("A-2", "Press", 10.0),
    ):
        for day in range(4):
            row = {
                "transaction_date": pd.Timestamp("2023-01-01")
                + pd.Timedelta(days=day),
                "asset_tag": asset_tag,
                "machine_type": machine_type,
                "failure_points": 0,
            }
            for index, sensor in enumerate(SENSOR_COLUMNS):
                row[sensor] = offset + index + day
            rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture
def daily_sensor_rows():
    rows = []
    for day, date in enumerate(pd.date_range("2023-01-01", periods=10, freq="D")):
        row = {
            "transaction_date": date,
            "label_end_date": date,
            "asset_tag": "A-1",
            "machine_type": "Press",
            "failure_points": 0,
        }
        for index, sensor in enumerate(SENSOR_COLUMNS):
            row[sensor] = float(index + day)
        rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture
def raw_asset_history():
    rows = []
    dates = list(pd.date_range("2023-12-20", periods=12, freq="D"))
    dates += list(pd.date_range("2024-01-01", periods=5, freq="D"))
    for asset_offset, asset_tag in enumerate(("A-1", "A-2")):
        for day, date in enumerate(dates):
            row = {
                "transaction_date": date,
                "asset_tag": asset_tag,
                "machine_type": "Press",
                "plant_code": "P1",
                "part_no": "P-C",
                "criticality": "C",
                "breakdown_flag": 0,
            }
            for index, sensor in enumerate(SENSOR_COLUMNS):
                row[sensor] = float(index + day + asset_offset * 10)
            rows.append(row)
    return pd.DataFrame(rows)


def test_robust_baseline_uses_asset_normal_rows_only(normal_daily):
    abnormal = normal_daily.iloc[[0]].copy()
    abnormal["failure_points"] = 20
    abnormal["temp_bearing_degC"] = 999.0
    train = pd.concat([normal_daily, abnormal], ignore_index=True)
    fitted = RobustNormalBaseline(min_normal_rows=3).fit(train)

    transformed = fitted.transform(abnormal)
    table = fitted.baseline_table().query(
        "asset_tag == 'A-1' and sensor == 'temp_bearing_degC'"
    ).iloc[0]
    expected = (999.0 - table["median"]) / table["scale"]

    assert table["normal_rows"] == 4
    assert transformed.iloc[0]["temp_bearing_degC_robust_z"] == pytest.approx(
        expected
    )


def test_zero_asset_mad_falls_back_per_sensor_to_machine(normal_daily):
    normal_daily.loc[
        normal_daily.asset_tag.eq("A-1"), "oil_pressure_bar"
    ] = 5.0
    fitted = RobustNormalBaseline(min_normal_rows=3).fit(normal_daily)

    row = fitted.baseline_table().query(
        "asset_tag == 'A-1' and sensor == 'oil_pressure_bar'"
    ).iloc[0]

    assert row["selected_scope"] == "machine_type"
    assert row["fallback_reason"] == "asset_mad_zero"


def test_robust_baseline_requires_normal_training_rows(normal_daily):
    normal_daily["failure_points"] = 1

    with pytest.raises(ValueError, match="정상행이 없습니다"):
        RobustNormalBaseline(min_normal_rows=3).fit(normal_daily)


def test_unseen_asset_uses_known_machine_baseline(normal_daily):
    fitted = RobustNormalBaseline(min_normal_rows=3).fit(normal_daily)
    unseen = normal_daily.iloc[[0]].copy()
    unseen["asset_tag"] = "A-NEW"

    transformed = fitted.transform(unseen)

    assert transformed.filter(like="_robust_z").notna().all().all()


def test_history_lag_uses_previous_calendar_day(daily_sensor_rows):
    missing = daily_sensor_rows.loc[
        ~daily_sensor_rows.transaction_date.eq(pd.Timestamp("2023-01-02"))
    ]

    result = build_sensor_history_features(missing)
    january_third = result.loc[
        result.transaction_date.eq(pd.Timestamp("2023-01-03"))
    ].iloc[0]

    assert pd.isna(january_third["load_pct_lag1"])


def test_future_change_does_not_change_past_features(daily_sensor_rows):
    before = build_sensor_history_features(daily_sensor_rows)
    changed = daily_sensor_rows.copy()
    changed.loc[changed.transaction_date.eq("2023-01-10"), "load_pct"] = 999.0

    after = build_sensor_history_features(changed)
    columns = [column for column in before if column.startswith("load_pct_")]

    pd.testing.assert_series_equal(
        before.loc[before.transaction_date.eq("2023-01-05"), columns].iloc[0],
        after.loc[after.transaction_date.eq("2023-01-05"), columns].iloc[0],
    )


def test_feature_sets_are_nested_without_targets(raw_asset_history):
    prepared = prepare_asset_experiment_features(
        raw_asset_history,
        validation_start="2024-01-01",
        min_normal_rows=3,
    )

    assert tuple(prepared.feature_sets) == ("A", "B", "C", "D")
    assert set(prepared.feature_sets["A"]) < set(prepared.feature_sets["B"])
    assert set(prepared.feature_sets["A"]) < set(prepared.feature_sets["C"])
    assert set(prepared.feature_sets["B"]) | set(prepared.feature_sets["C"]) < set(
        prepared.feature_sets["D"]
    )
    forbidden = {
        "breakdown_flag",
        "failure_points",
        "severity_level",
        "severity_code",
    }
    assert not forbidden.intersection(prepared.feature_sets["D"])
