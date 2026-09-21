"""Feature and label preparation for industrial machine models.

The source file has one row per ``asset_tag``/``part_no``/date.  Sensor
values are repeated across the part rows for a machine-day, while the
breakdown label can differ by part.  This module keeps that grain explicit
and never fills an unavailable future date with a negative label.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DATE_COLUMN = "transaction_date"
ASSET_COLUMN = "asset_tag"
PART_COLUMN = "part_no"
MACHINE_COLUMN = "machine_type"
PLANT_COLUMN = "plant_code"
CURRENT_TARGET = "breakdown_flag"

SENSOR_COLUMNS = [
    "temp_bearing_degC",
    "temp_motor_degC",
    "vibration_h_mms",
    "vibration_v_mms",
    "oil_pressure_bar",
    "load_pct",
    "shaft_rpm",
    "power_consumption_kw",
]

REQUIRED_COLUMNS = [
    DATE_COLUMN,
    ASSET_COLUMN,
    MACHINE_COLUMN,
    PLANT_COLUMN,
    PART_COLUMN,
    CURRENT_TARGET,
    *SENSOR_COLUMNS,
]

# This is deliberately close to the feature_data the user started with.
# Metadata such as asset_tag remains in the prepared frame for grouping but is
# not silently used as a model feature.
CURRENT_FEATURES = [MACHINE_COLUMN, *SENSOR_COLUMNS]


def load_industrial_data(path: str | Path) -> pd.DataFrame:
    """Load and validate the CSV without dropping columns or rows silently."""

    frame = pd.read_csv(path, parse_dates=[DATE_COLUMN])
    missing = [column for column in REQUIRED_COLUMNS if column not in frame]
    if missing:
        raise ValueError(f"CSV에 필요한 컬럼이 없습니다: {missing}")
    frame[CURRENT_TARGET] = pd.to_numeric(frame[CURRENT_TARGET], errors="coerce")
    if frame[CURRENT_TARGET].isna().any():
        raise ValueError("breakdown_flag에 숫자가 아닌 값이 있습니다.")
    return frame


def _copy_and_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result[DATE_COLUMN] = pd.to_datetime(result[DATE_COLUMN])
    result = result.sort_values(
        [ASSET_COLUMN, PART_COLUMN, DATE_COLUMN]
    ).reset_index(drop=True)
    result["label_end_date"] = result[DATE_COLUMN]
    return result


def prepare_current(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str], str]:
    """Prepare current-state classification data.

    ``breakdown_flag`` is the same-day diagnostic target.  It is excluded from
    ``feature_data`` so callers cannot accidentally train on the answer.
    """

    missing = [column for column in REQUIRED_COLUMNS if column not in frame]
    if missing:
        raise ValueError(f"입력 데이터에 필요한 컬럼이 없습니다: {missing}")
    result = _copy_and_metadata(frame)
    result[CURRENT_TARGET] = pd.to_numeric(result[CURRENT_TARGET], errors="coerce")
    result = result.dropna(subset=[CURRENT_TARGET]).copy()
    result[CURRENT_TARGET] = result[CURRENT_TARGET].astype(int)
    return result, CURRENT_FEATURES.copy(), CURRENT_TARGET


def _future_label_for_group(group: pd.DataFrame, horizon: int) -> pd.Series:
    """Return a future max label only when all future calendar dates exist."""

    if horizon < 1:
        raise ValueError("horizon은 1 이상이어야 합니다.")
    indexed = group.set_index(DATE_COLUMN)[CURRENT_TARGET]
    labels: list[float] = []
    end_dates: list[pd.Timestamp | pd.NaT] = []
    for date in group[DATE_COLUMN]:
        future_dates = pd.date_range(
            date + pd.Timedelta(days=1), periods=horizon, freq="D"
        )
        if not future_dates.isin(indexed.index).all():
            labels.append(np.nan)
            end_dates.append(pd.NaT)
            continue
        labels.append(float(indexed.reindex(future_dates).max()))
        end_dates.append(future_dates[-1])
    return pd.Series(labels, index=group.index), pd.Series(end_dates, index=group.index)


def _future_labels(frame: pd.DataFrame, horizon: int) -> tuple[pd.Series, pd.Series]:
    # The source has a regular daily cadence for each asset/part.  Grouped
    # shifts preserve group boundaries and let us verify calendar continuity
    # without constructing a Python date range for every source row.
    grouped = frame.groupby([ASSET_COLUMN, PART_COLUMN], sort=False)
    future_values: list[pd.Series] = []
    valid = pd.Series(True, index=frame.index)
    for offset in range(1, horizon + 1):
        future_date = grouped[DATE_COLUMN].shift(-offset)
        valid &= (future_date - frame[DATE_COLUMN]).dt.days.eq(offset)
        future_values.append(grouped[CURRENT_TARGET].shift(-offset).astype(float))
    labels = pd.concat(future_values, axis=1).max(axis=1)
    labels = labels.where(valid, np.nan)
    end_dates = (frame[DATE_COLUMN] + pd.Timedelta(days=horizon)).where(
        valid, pd.NaT
    )
    return labels, end_dates


def _forecast_features(frame: pd.DataFrame) -> list[str]:
    features = [MACHINE_COLUMN, PART_COLUMN, PLANT_COLUMN]
    for column in SENSOR_COLUMNS:
        features.extend(
            [f"{column}_lag1", f"{column}_lag3", f"{column}_lag7", f"{column}_mean7"]
        )
    features.extend(
        [
            "breakdown_lag1",
            "breakdown_count_30d",
            "day_of_week",
            "month_sin",
            "month_cos",
        ]
    )
    return [column for column in features if column in frame.columns]


def prepare_forecast(
    frame: pd.DataFrame, horizon: int = 7
) -> tuple[pd.DataFrame, list[str], str]:
    """Prepare a part-level future-risk classification data set.

    A row is retained only when the same asset/part has every future calendar
    date required by ``horizon``.  Current breakdown rows are excluded from the
    prediction population, and only shifted historical breakdown features are
    generated.  The raw current ``breakdown_flag`` is never a feature.
    """

    missing = [column for column in REQUIRED_COLUMNS if column not in frame]
    if missing:
        raise ValueError(f"입력 데이터에 필요한 컬럼이 없습니다: {missing}")
    result = _copy_and_metadata(frame)
    result[CURRENT_TARGET] = pd.to_numeric(result[CURRENT_TARGET], errors="coerce")
    result = result.dropna(subset=[CURRENT_TARGET]).copy()
    result[CURRENT_TARGET] = result[CURRENT_TARGET].astype(int)

    target_name = f"target_{horizon}d"
    future_label, label_end = _future_labels(result, horizon)
    result[target_name] = future_label
    result["label_end_date"] = label_end

    group_keys = [ASSET_COLUMN, PART_COLUMN]
    result = result.sort_values(group_keys + [DATE_COLUMN]).copy()
    for column in SENSOR_COLUMNS:
        grouped = result.groupby(group_keys, sort=False)[column]
        result[f"{column}_lag1"] = grouped.shift(1)
        result[f"{column}_lag3"] = grouped.shift(3)
        result[f"{column}_lag7"] = grouped.shift(7)
        result[f"{column}_mean7"] = grouped.transform(
            lambda values: values.shift(1).rolling(7, min_periods=3).mean()
        )

    breakdown_group = result.groupby(group_keys, sort=False)[CURRENT_TARGET]
    result["breakdown_lag1"] = breakdown_group.shift(1)
    result["breakdown_count_30d"] = breakdown_group.transform(
        lambda values: values.shift(1).rolling(30, min_periods=1).sum()
    )
    result["day_of_week"] = result[DATE_COLUMN].dt.dayofweek
    result["month_sin"] = np.sin(2 * np.pi * result[DATE_COLUMN].dt.month / 12)
    result["month_cos"] = np.cos(2 * np.pi * result[DATE_COLUMN].dt.month / 12)

    # The model is a pre-failure model: rows already marked as broken are not
    # prediction opportunities. Future labels that cannot be observed remain
    # NaN and are removed here rather than converted to a negative class.
    features = _forecast_features(result)
    result = result[result[CURRENT_TARGET].eq(0)].copy()
    result = result.dropna(subset=[target_name]).copy()
    result[target_name] = result[target_name].astype(int)
    result = result.reset_index(drop=True)
    return result, features, target_name


def split_by_date(
    frame: pd.DataFrame,
    validation_start: str | pd.Timestamp = "2024-01-01",
    test_start: str | pd.Timestamp = "2024-07-01",
) -> dict[str, pd.DataFrame]:
    """Split on calendar dates and respect future-label end dates."""

    validation_start = pd.Timestamp(validation_start)
    test_start = pd.Timestamp(test_start)
    if validation_start >= test_start:
        raise ValueError("validation_start는 test_start보다 빨라야 합니다.")
    if DATE_COLUMN not in frame or "label_end_date" not in frame:
        raise ValueError("transaction_date와 label_end_date가 필요합니다.")
    result = frame.copy()
    result[DATE_COLUMN] = pd.to_datetime(result[DATE_COLUMN])
    result["label_end_date"] = pd.to_datetime(result["label_end_date"])
    train = result[result["label_end_date"] < validation_start].copy()
    valid = result[
        (result[DATE_COLUMN] >= validation_start)
        & (result["label_end_date"] < test_start)
    ].copy()
    test = result[result[DATE_COLUMN] >= test_start].copy()
    return {"train": train, "valid": valid, "test": test}


def _filter_frame(
    frame: pd.DataFrame,
    machine_type: str | None = None,
    asset_tag: str | None = None,
) -> pd.DataFrame:
    result = frame
    if machine_type is not None:
        result = result[result[MACHINE_COLUMN].eq(machine_type)]
    if asset_tag is not None:
        result = result[result[ASSET_COLUMN].eq(asset_tag)]
    if result.empty:
        raise ValueError("지정한 machine_type 또는 asset_tag에 해당하는 데이터가 없습니다.")
    return result


def iter_scopes(
    frame: pd.DataFrame,
    scope: str = "all",
    machine_type: str | None = None,
    asset_tag: str | None = None,
) -> Iterator[tuple[str, str, pd.DataFrame]]:
    """Yield overall, machine-type, and asset-level views in stable order."""

    valid_scopes = {"all", "overall", "machine_type", "asset_tag"}
    if scope not in valid_scopes:
        raise ValueError(f"scope는 다음 중 하나여야 합니다: {sorted(valid_scopes)}")
    filtered = _filter_frame(frame, machine_type, asset_tag)
    if scope in {"all", "overall"}:
        yield "overall", "all", filtered.copy()
    if scope in {"all", "machine_type"} and asset_tag is None:
        for value in sorted(filtered[MACHINE_COLUMN].dropna().unique()):
            subset = filtered[filtered[MACHINE_COLUMN].eq(value)].copy()
            yield "machine_type", str(value), subset
    if scope in {"all", "asset_tag"}:
        for value in sorted(filtered[ASSET_COLUMN].dropna().unique()):
            subset = filtered[filtered[ASSET_COLUMN].eq(value)].copy()
            yield "asset_tag", str(value), subset
