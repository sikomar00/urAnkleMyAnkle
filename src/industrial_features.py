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

from .part_features import prepare_part_current, prepare_part_forecast

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

    prepared = prepare_part_current(frame)
    return prepared.frame, list(prepared.features), prepared.target


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

    prepared = prepare_part_forecast(frame, horizon)
    return prepared.frame, list(prepared.features), prepared.target
