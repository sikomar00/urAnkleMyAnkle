import numpy as np
import pandas as pd
import pytest

from src.industrial_data import SENSOR_COLUMNS
from src.probabilistic_risk_features import (
    A2_HISTORY_FEATURES,
    prepare_probabilistic_risk_features,
)


@pytest.fixture
def raw_part_rows() -> pd.DataFrame:
    rows = []
    dates = pd.date_range("2023-01-01", periods=40, freq="D")
    criticalities = ["A"] * 6 + ["B"] * 9 + ["C"] * 5
    for asset_index, asset_tag in enumerate(("AST-1", "AST-2")):
        for day_index, date in enumerate(dates):
            sensors = {
                column: float(10 * (sensor_index + 1) + day_index + asset_index)
                for sensor_index, column in enumerate(SENSOR_COLUMNS)
            }
            for part_index in range(20):
                rows.append(
                    {
                        "transaction_date": date,
                        "machine_type": f"Machine-{asset_index + 1}",
                        "asset_tag": asset_tag,
                        "plant_code": f"Plant-{asset_index + 1}",
                        "part_no": f"P{part_index + 1:02d}",
                        "part_family": f"Family-{part_index // 4 + 1}",
                        "criticality": criticalities[part_index],
                        "breakdown_flag": int(
                            (day_index + part_index + asset_index) % 11 == 0
                        ),
                        **sensors,
                    }
                )
    return pd.DataFrame(rows)


def prepare(frame: pd.DataFrame):
    return prepare_probabilistic_risk_features(
        frame,
        selection_start="2023-01-11",
        calibration_start="2023-01-21",
        test_start="2023-01-31",
    )


def select_row(frame: pd.DataFrame, date: str, asset: str = "AST-1", part: str = "P01"):
    selected = frame.loc[
        frame["transaction_date"].eq(pd.Timestamp(date))
        & frame["asset_tag"].eq(asset)
        & frame["part_no"].eq(part)
    ]
    assert len(selected) == 1
    return selected.iloc[0]


def test_prepare_keeps_each_asset_day_in_one_period(raw_part_rows):
    prepared = prepare(raw_part_rows)

    counts = prepared.frame.groupby(["transaction_date", "asset_tag"])[
        "period"
    ].nunique()
    assert counts.eq(1).all()
    assert set(prepared.periods) == {"train", "selection", "calibration", "test"}
    assert {name: len(frame) for name, frame in prepared.periods.items()} == {
        "train": 400,
        "selection": 400,
        "calibration": 400,
        "test": 400,
    }


def test_a1_and_a2_never_include_current_targets(raw_part_rows):
    prepared = prepare(raw_part_rows)

    forbidden = {
        "breakdown_flag",
        "actual_failure_points",
        "failure_points",
        "severity_level",
    }
    assert forbidden.isdisjoint(prepared.feature_sets["A1"])
    assert forbidden.isdisjoint(prepared.feature_sets["A2"])
    assert set(A2_HISTORY_FEATURES).issubset(prepared.feature_sets["A2"])
    assert set(A2_HISTORY_FEATURES).isdisjoint(prepared.feature_sets["A1"])
    assert prepared.max_failure_points == 47


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda frame: frame.drop(columns=["part_family"]), "필요한 컬럼"),
        (lambda frame: pd.concat([frame, frame.iloc[[0]]]), "중복"),
        (
            lambda frame: frame.assign(
                breakdown_flag=np.where(frame.index == 0, 2, frame["breakdown_flag"])
            ),
            "0 또는 1",
        ),
    ],
)
def test_prepare_rejects_invalid_basic_contract(raw_part_rows, mutation, message):
    with pytest.raises(ValueError, match=message):
        prepare(mutation(raw_part_rows.copy()))


def test_prepare_rejects_sensor_conflict_within_asset_day(raw_part_rows):
    changed = raw_part_rows.copy()
    changed.loc[1, SENSOR_COLUMNS[0]] += 1

    with pytest.raises(ValueError, match="센서값"):
        prepare(changed)


def test_prepare_rejects_missing_or_changed_part_configuration(raw_part_rows):
    missing = raw_part_rows.drop(index=0)
    with pytest.raises(ValueError, match="20개"):
        prepare(missing)

    changed_family = raw_part_rows.copy()
    changed_family.loc[
        changed_family["transaction_date"].eq(pd.Timestamp("2023-01-02"))
        & changed_family["asset_tag"].eq("AST-1")
        & changed_family["part_no"].eq("P01"),
        "part_family",
    ] = "Other"
    with pytest.raises(ValueError, match="구성"):
        prepare(changed_family)

    changed_criticality = raw_part_rows.copy()
    changed_criticality.loc[
        changed_criticality["transaction_date"].eq(pd.Timestamp("2023-01-02"))
        & changed_criticality["asset_tag"].eq("AST-1")
        & changed_criticality["part_no"].eq("P01"),
        "criticality",
    ] = "B"
    with pytest.raises(ValueError, match="구성"):
        prepare(changed_criticality)


@pytest.mark.parametrize(
    ("selection_start", "calibration_start", "test_start"),
    [
        ("2023-01-21", "2023-01-11", "2023-01-31"),
        ("2023-01-11", "2023-02-20", "2023-03-01"),
    ],
)
def test_prepare_rejects_reversed_or_empty_periods(
    raw_part_rows, selection_start, calibration_start, test_start
):
    with pytest.raises(ValueError, match="시간 구간|비어"):
        prepare_probabilistic_risk_features(
            raw_part_rows,
            selection_start=selection_start,
            calibration_start=calibration_start,
            test_start=test_start,
        )


def test_current_failure_change_does_not_change_same_day_history(raw_part_rows):
    before = prepare(raw_part_rows).frame
    changed = raw_part_rows.copy()
    mask = changed["transaction_date"].eq(pd.Timestamp("2023-01-15"))
    changed.loc[mask, "breakdown_flag"] = 1 - changed.loc[mask, "breakdown_flag"]
    after = prepare(changed).frame

    for part in ("P01", "P11", "P20"):
        left = select_row(before, "2023-01-15", part=part)
        right = select_row(after, "2023-01-15", part=part)
        pd.testing.assert_series_equal(
            left[list(A2_HISTORY_FEATURES)],
            right[list(A2_HISTORY_FEATURES)],
            check_names=False,
        )


def test_lag1_uses_previous_calendar_day_not_previous_observation(raw_part_rows):
    missing_day = raw_part_rows.loc[
        ~(
            raw_part_rows["transaction_date"].eq(pd.Timestamp("2023-01-05"))
            & raw_part_rows["asset_tag"].eq("AST-1")
        )
    ]

    prepared = prepare(missing_day)
    row = select_row(prepared.frame, "2023-01-06")
    assert pd.isna(row["breakdown_lag1"])


def test_observed_test_history_only_affects_later_test_dates(raw_part_rows):
    before = prepare(raw_part_rows).frame
    changed = raw_part_rows.copy()
    mask = (
        changed["transaction_date"].eq(pd.Timestamp("2023-01-31"))
        & changed["asset_tag"].eq("AST-1")
        & changed["part_no"].eq("P01")
    )
    changed.loc[mask, "breakdown_flag"] = 1
    after = prepare(changed).frame

    same_day_before = select_row(before, "2023-01-31")
    same_day_after = select_row(after, "2023-01-31")
    pd.testing.assert_series_equal(
        same_day_before[list(A2_HISTORY_FEATURES)],
        same_day_after[list(A2_HISTORY_FEATURES)],
        check_names=False,
    )
    assert select_row(after, "2023-02-01")["breakdown_lag1"] == 1

