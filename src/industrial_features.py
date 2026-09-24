"""Feature and label preparation for industrial machine models.

The source file has one row per ``asset_tag``/``part_no``/date.  Sensor
values are repeated across the part rows for a machine-day, while the
breakdown label can differ by part.  This module keeps that grain explicit
and never fills an unavailable future date with a negative label.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .industrial_data import (
    ASSET_COLUMN,
    CURRENT_TARGET,
    DATE_COLUMN,
    MACHINE_COLUMN,
    PART_COLUMN,
    PLANT_COLUMN,
    REQUIRED_COLUMNS,
    SENSOR_COLUMNS,
    iter_scopes,
    load_industrial_data,
    split_by_date,
)

# This is deliberately close to the feature_data the user started with.
# Metadata such as asset_tag remains in the prepared frame for grouping but is
# not silently used as a model feature.
CURRENT_FEATURES = [MACHINE_COLUMN, *SENSOR_COLUMNS]


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


def _future_label_for_group(group: pd.DataFrame, horizon: int) -> tuple[pd.Series, pd.Series]:
    """Return a future max label only when all future calendar dates exist."""

    if horizon < 1:
        raise ValueError("horizon은 1 이상이어야 합니다.")
    indexed = group.set_index(DATE_COLUMN)[CURRENT_TARGET]
    labels: list[float] = []
    end_dates: list[Any] = []  # Timestamp values and the NaT singleton.
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
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
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
