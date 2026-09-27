import pandas as pd
import pytest

from src.family_features import (
    FAMILY_NAMES,
    build_family_daily,
    build_family_target_profile,
)


PARTS = (
    ("BR-1", "Bearing", "A"),
    ("BR-2", "Bearing", "A"),
    ("BR-3", "Bearing", "A"),
    ("SG-1", "Seal & Gasket", "A"),
    ("SG-2", "Seal & Gasket", "C"),
    ("SG-3", "Seal & Gasket", "B"),
    ("DB-1", "Drive Belt", "B"),
    ("DB-2", "Drive Belt", "B"),
    ("FI-1", "Filter", "B"),
    ("FI-2", "Filter", "B"),
    ("EL-1", "Electrical", "B"),
    ("EL-2", "Electrical", "A"),
    ("CO-1", "Coupling", "B"),
    ("CO-2", "Coupling", "B"),
    ("LU-1", "Lubrication", "C"),
    ("LU-2", "Lubrication", "C"),
    ("SE-1", "Sensor", "A"),
    ("SE-2", "Sensor", "B"),
    ("FA-1", "Fastener", "C"),
    ("FA-2", "Fastener", "C"),
)

SENSOR_VALUES = {
    "temp_bearing_degC": 62.0,
    "temp_motor_degC": 69.0,
    "vibration_h_mms": 2.1,
    "vibration_v_mms": 2.4,
    "oil_pressure_bar": 4.8,
    "load_pct": 71.0,
    "shaft_rpm": 1_480.0,
    "power_consumption_kw": 24.0,
}


@pytest.fixture
def raw_family_rows() -> pd.DataFrame:
    failures = {
        pd.Timestamp("2023-01-02"): {"BR-1", "EL-2"},
        pd.Timestamp("2023-01-03"): {"BR-1", "BR-2", "BR-3", "FI-1", "FI-2"},
    }
    rows = []
    for day_number, date in enumerate(pd.date_range("2023-01-01", periods=3)):
        sensors = {
            column: value + day_number for column, value in SENSOR_VALUES.items()
        }
        for part_no, family, criticality in PARTS:
            rows.append(
                {
                    "transaction_date": date,
                    "machine_type": "CNC Lathe",
                    "asset_tag": "CNC-001",
                    "plant_code": "P01",
                    "part_no": part_no,
                    "part_family": family,
                    "criticality": criticality,
                    "breakdown_flag": int(part_no in failures.get(date, set())),
                    **sensors,
                }
            )
    return pd.DataFrame(rows)


def select_day(result: pd.DataFrame, family: str, date: str) -> pd.Series:
    selected = result[
        result["part_family"].eq(family)
        & result["transaction_date"].eq(pd.Timestamp(date))
    ]
    assert len(selected) == 1
    return selected.iloc[0]


def test_build_family_daily_creates_affected_severe_and_all(raw_family_rows):
    result = build_family_daily(raw_family_rows)

    bearing = result.query("part_family == 'Bearing'").sort_values(
        "transaction_date"
    )
    assert bearing["affected"].tolist() == [0, 1, 1]
    assert bearing["severe"].tolist() == [0, 1, 1]
    assert bearing["all_failed"].tolist() == [0, 0, 1]
    assert (result["severe"] <= result["affected"]).all()


def test_one_a_or_two_non_a_parts_are_severe(raw_family_rows):
    result = build_family_daily(raw_family_rows)

    assert select_day(result, "Electrical", "2023-01-02")["severe"] == 1
    assert select_day(result, "Filter", "2023-01-03")["severe"] == 1


def test_multiple_families_remain_positive_on_same_asset_day(raw_family_rows):
    result = build_family_daily(raw_family_rows)

    day = result[result["transaction_date"].eq(pd.Timestamp("2023-01-03"))]
    assert set(day.loc[day["affected"].eq(1), "part_family"]) == {
        "Bearing",
        "Filter",
    }


@pytest.mark.parametrize(
    ("column", "bad_value", "message"),
    [
        ("breakdown_flag", 2, "breakdown_flag"),
        ("criticality", "D", "criticality"),
    ],
)
def test_build_family_daily_rejects_invalid_categorical_values(
    raw_family_rows, column, bad_value, message
):
    raw_family_rows.loc[0, column] = bad_value

    with pytest.raises(ValueError, match=message):
        build_family_daily(raw_family_rows)


def test_build_family_daily_rejects_conflicting_same_day_sensors(raw_family_rows):
    raw_family_rows.loc[0, "temp_bearing_degC"] += 10

    with pytest.raises(ValueError, match="센서"):
        build_family_daily(raw_family_rows)


def test_build_family_daily_rejects_family_composition_changes(raw_family_rows):
    changed = raw_family_rows.drop(
        raw_family_rows[
            raw_family_rows["transaction_date"].eq(pd.Timestamp("2023-01-03"))
            & raw_family_rows["part_no"].eq("BR-3")
        ].index
    )

    with pytest.raises(ValueError, match="구성"):
        build_family_daily(changed)


def test_build_family_daily_rejects_unexpected_family(raw_family_rows):
    raw_family_rows.loc[0, "part_family"] = "Unknown"

    with pytest.raises(ValueError, match="Family"):
        build_family_daily(raw_family_rows)


def test_build_family_daily_rejects_duplicate_part_rows(raw_family_rows):
    duplicated = pd.concat([raw_family_rows, raw_family_rows.iloc[[0]]])

    with pytest.raises(ValueError, match="중복"):
        build_family_daily(duplicated)


def test_family_target_profile_contains_all_periods_and_rates(raw_family_rows):
    daily = build_family_daily(raw_family_rows)

    profile = build_family_target_profile(daily)

    assert set(profile["period"]) == {"overall", "train", "valid", "test"}
    bearing_train = profile[
        profile["period"].eq("train")
        & profile["part_family"].eq("Bearing")
    ].iloc[0]
    assert bearing_train["row_count"] == 3
    assert bearing_train["affected_count"] == 2
    assert bearing_train["affected_rate"] == pytest.approx(2 / 3)
    assert bearing_train["total_parts"] == 3
    assert tuple(FAMILY_NAMES) == (
        "Bearing",
        "Seal & Gasket",
        "Drive Belt",
        "Filter",
        "Electrical",
        "Coupling",
        "Lubrication",
        "Sensor",
        "Fastener",
    )
