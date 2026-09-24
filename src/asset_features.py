"""부품 원본 데이터를 장비·날짜 단위 위험 데이터로 변환한다.

이 모듈은 장비별 당일 고장점수를 만들고, 점수 기준 이상 여부를 당일 탐지
Target으로 준비한다. 같은 장비·날짜의 센서가 서로 다르면 임의 집계하지 않고
오류를 발생시켜 데이터 의미가 바뀌는 것을 막는다.
"""

from __future__ import annotations

import pandas as pd

from .industrial_data import (
    ASSET_COLUMN,
    CURRENT_TARGET,
    DATE_COLUMN,
    MACHINE_COLUMN,
    PART_COLUMN,
    SENSOR_COLUMNS,
    PreparedTask,
    add_calendar_features,
)

CRITICALITY_WEIGHTS = {"A": 4, "B": 2, "C": 1}

ASSET_CURRENT_FEATURES = (
    MACHINE_COLUMN,
    ASSET_COLUMN,
    *SENSOR_COLUMNS,
    "day_of_week",
    "month_sin",
    "month_cos",
)


def build_asset_daily(frame: pd.DataFrame) -> pd.DataFrame:
    """부품 단위 원본을 장비·날짜별 고장점수와 센서 한 행으로 집계한다.

    Args:
        frame: 날짜, 장비, 부품, 중요도, 고장 여부와 센서값이 포함된 원본 데이터.

    Returns:
        장비·날짜별 ``failure_points``와 센서·달력 Feature가 포함된 데이터.

    Raises:
        ValueError: 중요도가 A/B/C가 아니거나 키가 중복되거나 같은 장비·날짜의
            센서값이 서로 다를 때.
    """
    result = frame.copy()
    result[DATE_COLUMN] = pd.to_datetime(result[DATE_COLUMN]).dt.normalize()
    result["criticality_weight"] = (
        result["criticality"]
        .astype("string")
        .str.strip()
        .str.upper()
        .map(CRITICALITY_WEIGHTS)
    )
    if result["criticality_weight"].isna().any():
        raise ValueError("criticality에는 A, B, C만 사용할 수 있습니다.")

    duplicate_keys = [DATE_COLUMN, ASSET_COLUMN, PART_COLUMN]
    if result.duplicated(duplicate_keys).any():
        raise ValueError("같은 날짜·장비·부품의 중복 행이 있습니다.")

    group_keys = [DATE_COLUMN, MACHINE_COLUMN, ASSET_COLUMN]
    sensor_counts = result.groupby(group_keys)[SENSOR_COLUMNS].nunique(dropna=False)
    conflicts = sensor_counts.gt(1)
    if conflicts.any().any():
        conflict_columns = conflicts.columns[conflicts.any()].tolist()
        raise ValueError(
            f"같은 장비·날짜의 센서값이 다릅니다: {conflict_columns}"
        )

    result["failure_points"] = (
        result[CURRENT_TARGET] * result["criticality_weight"]
    )
    daily = result.groupby(group_keys, as_index=False).agg(
        {
            "failure_points": "sum",
            **{column: "first" for column in SENSOR_COLUMNS},
        }
    )
    daily["label_end_date"] = daily[DATE_COLUMN]
    return add_calendar_features(daily)


def prepare_asset_current(
    frame: pd.DataFrame,
    score_threshold: int,
) -> PreparedTask:
    """장비 단위 당일 위험 탐지 과제를 준비한다.

    Args:
        frame: 부품 단위 원본 산업 데이터.
        score_threshold: 위험으로 판정할 최소 고장점수.

    Returns:
        장비 당일 Feature, Target과 메타데이터를 담은 ``PreparedTask``.

    Raises:
        ValueError: 점수 기준이 1보다 작거나 장비 일별 집계가 유효하지 않을 때.
    """
    if score_threshold < 1:
        raise ValueError("score_threshold는 1 이상이어야 합니다.")

    daily = build_asset_daily(frame)
    target = f"target_ge_{score_threshold}"
    daily[target] = (daily["failure_points"] >= score_threshold).astype(int)
    return PreparedTask(
        frame=daily,
        features=ASSET_CURRENT_FEATURES,
        target=target,
        grain="asset",
        mode="current",
        score_threshold=score_threshold,
    )
