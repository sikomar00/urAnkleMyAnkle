import pandas as pd
import pytest

from src.part_features import prepare_part_current, prepare_part_forecast


@pytest.fixture
def part_history():
    rows = []
    for offset, date in enumerate(pd.date_range("2024-01-01", periods=12, freq="D")):
        for part, criticality in (("P-1", "A"), ("P-2", "C")):
            rows.append({
                "transaction_date": date, "asset_tag": "A-1",
                "machine_type": "Press", "plant_code": "P1", "part_no": part,
                "criticality": criticality,
                "breakdown_flag": int(part == "P-1" and date == pd.Timestamp("2024-01-04")),
                "temp_bearing_degC": 50.0 + offset,
                "temp_motor_degC": 60.0 + offset,
                "vibration_h_mms": 1.0 + offset / 10,
                "vibration_v_mms": 1.1 + offset / 10,
                "oil_pressure_bar": 5.0, "load_pct": 70.0 + offset,
                "shaft_rpm": 1000.0 + offset,
                "power_consumption_kw": 20.0 + offset,
            })
    return pd.DataFrame(rows)


def test_part_current_includes_part_identity_and_criticality(part_history):
    prepared = prepare_part_current(part_history)
    assert prepared.target == "breakdown_flag"
    assert {"part_no", "criticality", "plant_code"} <= set(prepared.features)
    assert "breakdown_flag" not in prepared.features


def test_part_forecast_target_starts_tomorrow(part_history):
    prepared = prepare_part_forecast(part_history, horizon=7)
    row = prepared.frame.query("asset_tag == 'A-1' and part_no == 'P-1'").iloc[0]
    assert row[prepared.target] == 1
    assert row["label_end_date"] == row["transaction_date"] + pd.Timedelta(days=7)


def test_part_breakdown_history_does_not_cross_parts(part_history):
    prepared = prepare_part_forecast(part_history, horizon=7).frame
    p2 = prepared.query("asset_tag == 'A-1' and part_no == 'P-2'")
    assert (p2["breakdown_count_7d"].fillna(0) == 0).all()


def test_part_forecast_rejects_missing_calendar_day(part_history):
    missing = part_history[~(
        part_history["part_no"].eq("P-1")
        & part_history["transaction_date"].eq(pd.Timestamp("2024-01-05"))
    )]
    prepared = prepare_part_forecast(missing, horizon=7)
    key_rows = prepared.frame.query("asset_tag == 'A-1' and part_no == 'P-1'")
    assert pd.Timestamp("2024-01-01") not in set(key_rows["transaction_date"])
