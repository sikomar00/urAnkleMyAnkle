"""부품 단위 당일 고장 탐지와 미래 7일 예측 Feature를 만든다."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .industrial_data import (
    ASSET_COLUMN, CURRENT_TARGET, DATE_COLUMN, MACHINE_COLUMN, PART_COLUMN,
    PLANT_COLUMN, REQUIRED_COLUMNS, SENSOR_COLUMNS, PreparedTask,
    add_calendar_features,
)

PART_CURRENT_FEATURES = (
    MACHINE_COLUMN, ASSET_COLUMN, PART_COLUMN, "criticality", PLANT_COLUMN,
    *SENSOR_COLUMNS, "day_of_week", "month_sin", "month_cos",
)


def _validated_part_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """부품 원본의 필수 컬럼·중복·정답 값을 검사하고 시간순으로 정렬한다."""
    missing = [column for column in REQUIRED_COLUMNS if column not in frame]
    if missing:
        raise ValueError(f"입력 데이터에 필요한 컬럼이 없습니다: {missing}")
    result = frame.copy()
    result[DATE_COLUMN] = pd.to_datetime(result[DATE_COLUMN]).dt.normalize()
    result[CURRENT_TARGET] = pd.to_numeric(result[CURRENT_TARGET], errors="coerce")
    if result[CURRENT_TARGET].isna().any() or not result[CURRENT_TARGET].isin([0, 1]).all():
        raise ValueError("breakdown_flag는 0 또는 1이어야 합니다.")
    keys = [DATE_COLUMN, ASSET_COLUMN, PART_COLUMN]
    if result.duplicated(keys).any():
        raise ValueError("같은 날짜·장비·부품의 중복 행이 있습니다.")
    return result.sort_values([ASSET_COLUMN, PART_COLUMN, DATE_COLUMN]).reset_index(drop=True)


def prepare_part_current(frame: pd.DataFrame) -> PreparedTask:
    """부품별 당일 ``breakdown_flag`` 탐지 과제를 준비한다."""
    result = add_calendar_features(_validated_part_frame(frame))
    result["label_end_date"] = result[DATE_COLUMN]
    return PreparedTask(result, PART_CURRENT_FEATURES, CURRENT_TARGET, "part", "current")


def _sensor_history(result: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """장비·날짜별 유일 센서 표에서 현재값과 과거 통계를 계산한다."""
    keys = [ASSET_COLUMN, DATE_COLUMN]
    counts = result.groupby(keys)[SENSOR_COLUMNS].nunique(dropna=False)
    if counts.gt(1).any().any():
        columns = counts.columns[counts.gt(1).any()].tolist()
        raise ValueError(f"같은 장비·날짜의 센서값이 다릅니다: {columns}")
    sensors = result[keys + SENSOR_COLUMNS].drop_duplicates(keys).sort_values(keys)
    grouped = sensors.groupby(ASSET_COLUMN, sort=False)
    features: list[str] = []
    for column in SENSOR_COLUMNS:
        values = grouped[column]
        names = [f"{column}_current", f"{column}_lag1", f"{column}_lag3",
                 f"{column}_lag7", f"{column}_mean7", f"{column}_std7",
                 f"{column}_diff1"]
        sensors[names[0]] = sensors[column]
        sensors[names[1]] = values.shift(1)
        sensors[names[2]] = values.shift(3)
        sensors[names[3]] = values.shift(7)
        sensors[names[4]] = values.transform(lambda s: s.rolling(7, min_periods=3).mean())
        sensors[names[5]] = values.transform(lambda s: s.rolling(7, min_periods=3).std())
        sensors[names[6]] = values.diff(1)
        features.extend(names)
    return sensors[keys + features], features


def prepare_part_forecast(frame: pd.DataFrame, horizon: int = 7) -> PreparedTask:
    """오늘 정상인 부품의 내일부터 ``horizon``일까지 고장 여부를 준비한다."""
    if horizon < 1:
        raise ValueError("horizon은 1 이상이어야 합니다.")
    result = _validated_part_frame(frame)
    groups = result.groupby([ASSET_COLUMN, PART_COLUMN], sort=False)
    valid = pd.Series(True, index=result.index)
    future = []
    for offset in range(1, horizon + 1):
        future_date = groups[DATE_COLUMN].shift(-offset)
        valid &= (future_date - result[DATE_COLUMN]).dt.days.eq(offset)
        future.append(groups[CURRENT_TARGET].shift(-offset).astype(float))
    target = f"target_{horizon}d"
    result[target] = pd.concat(future, axis=1).max(axis=1).where(valid)
    result["label_end_date"] = (
        result[DATE_COLUMN] + pd.Timedelta(days=horizon)
    ).where(valid, pd.NaT)

    sensor_history, sensor_features = _sensor_history(result)
    result = result.merge(sensor_history, on=[ASSET_COLUMN, DATE_COLUMN], how="left")
    groups = result.groupby([ASSET_COLUMN, PART_COLUMN], sort=False)
    shifted = groups[CURRENT_TARGET].shift(1)
    shifted_groups = shifted.groupby([result[ASSET_COLUMN], result[PART_COLUMN]], sort=False)
    result["breakdown_lag1"] = shifted
    result["breakdown_count_7d"] = shifted_groups.transform(
        lambda s: s.rolling(7, min_periods=1).sum()
    )
    result["breakdown_count_30d"] = shifted_groups.transform(
        lambda s: s.rolling(30, min_periods=1).sum()
    )
    previous_date = groups[DATE_COLUMN].shift(1).where(shifted.eq(1))
    last_date = previous_date.groupby(
        [result[ASSET_COLUMN], result[PART_COLUMN]], sort=False
    ).ffill()
    result["days_since_last_breakdown"] = (result[DATE_COLUMN] - last_date).dt.days
    result = add_calendar_features(result)

    static = [MACHINE_COLUMN, ASSET_COLUMN, PART_COLUMN, "criticality", PLANT_COLUMN]
    history = ["breakdown_lag1", "breakdown_count_7d", "breakdown_count_30d",
               "days_since_last_breakdown", "day_of_week", "month_sin", "month_cos"]
    result = result[result[CURRENT_TARGET].eq(0)].dropna(subset=[target]).copy()
    result[target] = result[target].astype(int)
    return PreparedTask(
        result.reset_index(drop=True), tuple(static + sensor_features + history), target,
        "part", "forecast", risk_definition="new", horizon=horizon,
    )
