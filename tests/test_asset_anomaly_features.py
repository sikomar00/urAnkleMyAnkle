import pandas as pd
import pytest

from src.asset_anomaly_features import RobustNormalBaseline
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
