import pandas as pd
import pytest

from src.asset_features import build_asset_daily, prepare_asset_current


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
