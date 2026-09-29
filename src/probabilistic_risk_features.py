"""부품확률 기반 장비 위험 모델의 입력 계약과 A1·A2 Feature를 만든다."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .asset_features import CRITICALITY_WEIGHTS
from .industrial_data import (
    ASSET_COLUMN,
    CURRENT_TARGET,
    DATE_COLUMN,
    MACHINE_COLUMN,
    PART_COLUMN,
    PLANT_COLUMN,
    REQUIRED_COLUMNS,
    SENSOR_COLUMNS,
    add_calendar_features,
)

PART_FAMILY_COLUMN = "part_family"
PERIOD_NAMES = ("train", "selection", "calibration", "test")

A1_FEATURES = (
    MACHINE_COLUMN,
    ASSET_COLUMN,
    PLANT_COLUMN,
    PART_COLUMN,
    PART_FAMILY_COLUMN,
    "criticality",
    *SENSOR_COLUMNS,
    "day_of_week",
    "month_sin",
    "month_cos",
)

A2_HISTORY_FEATURES = (
    "breakdown_lag1",
    "breakdown_count_7d",
    "breakdown_count_30d",
    "historical_breakdown_rate",
    "days_since_last_breakdown",
)


@dataclass(frozen=True)
class ProbabilisticRiskFeatures:
    """검증된 부품행, Feature 집합과 네 시간 구간을 함께 보관한다."""

    frame: pd.DataFrame
    feature_sets: dict[str, tuple[str, ...]]
    periods: dict[str, pd.DataFrame]
    max_failure_points: int


def _validate_boundaries(
    selection_start: str | pd.Timestamp,
    calibration_start: str | pd.Timestamp,
    test_start: str | pd.Timestamp,
) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    boundaries = tuple(
        pd.Timestamp(value).normalize()
        for value in (selection_start, calibration_start, test_start)
    )
    if not boundaries[0] < boundaries[1] < boundaries[2]:
        raise ValueError(
            "시간 구간은 selection_start < calibration_start < test_start여야 합니다."
        )
    return boundaries


def _validate_raw(frame: pd.DataFrame, expected_parts_per_asset: int) -> pd.DataFrame:
    if expected_parts_per_asset < 1:
        raise ValueError("expected_parts_per_asset는 1 이상이어야 합니다.")
    required = [*REQUIRED_COLUMNS, PART_FAMILY_COLUMN]
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"입력 데이터에 필요한 컬럼이 없습니다: {missing}")

    result = frame.copy()
    result[DATE_COLUMN] = pd.to_datetime(result[DATE_COLUMN], errors="coerce").dt.normalize()
    if result[DATE_COLUMN].isna().any():
        raise ValueError("transaction_date에 유효하지 않은 날짜가 있습니다.")

    result[CURRENT_TARGET] = pd.to_numeric(result[CURRENT_TARGET], errors="coerce")
    if result[CURRENT_TARGET].isna().any() or not result[CURRENT_TARGET].isin([0, 1]).all():
        raise ValueError("breakdown_flag는 0 또는 1이어야 합니다.")
    result[CURRENT_TARGET] = result[CURRENT_TARGET].astype(int)

    keys = [DATE_COLUMN, ASSET_COLUMN, PART_COLUMN]
    if result.duplicated(keys).any():
        raise ValueError("같은 날짜·장비·부품의 중복 행이 있습니다.")

    asset_day = [DATE_COLUMN, ASSET_COLUMN]
    counts = result.groupby(asset_day, dropna=False)[PART_COLUMN].size()
    if not counts.eq(expected_parts_per_asset).all():
        raise ValueError(
            f"각 장비·날짜에는 정확히 {expected_parts_per_asset}개 부품이 있어야 합니다."
        )

    shared_columns = [MACHINE_COLUMN, PLANT_COLUMN, *SENSOR_COLUMNS]
    shared_counts = result.groupby(asset_day, dropna=False)[shared_columns].nunique(
        dropna=False
    )
    conflicts = shared_counts.gt(1)
    if conflicts.any().any():
        sensor_conflicts = [
            column for column in SENSOR_COLUMNS if conflicts[column].any()
        ]
        if sensor_conflicts:
            raise ValueError(
                f"같은 장비·날짜의 센서값이 다릅니다: {sensor_conflicts}"
            )
        other = conflicts.columns[conflicts.any()].tolist()
        raise ValueError(f"같은 장비·날짜의 장비 정보가 다릅니다: {other}")

    if result[[ASSET_COLUMN, PART_COLUMN, PART_FAMILY_COLUMN, "criticality"]].isna().any().any():
        raise ValueError("부품 구성 컬럼에는 결측값을 사용할 수 없습니다.")
    result["criticality"] = (
        result["criticality"].astype("string").str.strip().str.upper()
    )
    result["criticality_weight"] = result["criticality"].map(CRITICALITY_WEIGHTS)
    if result["criticality_weight"].isna().any():
        raise ValueError("criticality에는 A, B, C만 사용할 수 있습니다.")
    result["criticality_weight"] = result["criticality_weight"].astype(int)

    static_counts = result.groupby([ASSET_COLUMN, PART_COLUMN], dropna=False)[
        [PART_FAMILY_COLUMN, "criticality"]
    ].nunique(dropna=False)
    if static_counts.gt(1).any().any():
        raise ValueError("장비별 부품 Family 또는 중요도 구성이 날짜에 따라 달라집니다.")

    for _, group in result.groupby(ASSET_COLUMN, sort=False):
        expected_parts = frozenset(group[PART_COLUMN].unique())
        daily_parts = group.groupby(DATE_COLUMN)[PART_COLUMN].agg(
            lambda values: frozenset(values)
        )
        if not daily_parts.map(lambda values: values == expected_parts).all():
            raise ValueError("장비별 부품 구성이 날짜에 따라 달라집니다.")

    maximums = (
        result.drop_duplicates([ASSET_COLUMN, PART_COLUMN])
        .groupby(ASSET_COLUMN)["criticality_weight"]
        .sum()
    )
    if set(maximums.astype(int)) != {47}:
        raise ValueError("장비별 중요도 구성의 최대 고장점수는 47점이어야 합니다.")

    result["actual_failure_points"] = (
        result[CURRENT_TARGET] * result["criticality_weight"]
    ).groupby([result[DATE_COLUMN], result[ASSET_COLUMN]]).transform("sum")
    return result.sort_values(keys).reset_index(drop=True)


def _build_part_history(frame: pd.DataFrame) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for (asset_tag, part_no), group in frame.groupby(
        [ASSET_COLUMN, PART_COLUMN], sort=False
    ):
        indexed = (
            group[[DATE_COLUMN, CURRENT_TARGET]]
            .sort_values(DATE_COLUMN)
            .set_index(DATE_COLUMN)
        )
        calendar = indexed.reindex(
            pd.date_range(indexed.index.min(), indexed.index.max(), freq="D")
        )
        calendar.index.name = DATE_COLUMN
        shifted = calendar[CURRENT_TARGET].shift(1)
        calendar["breakdown_lag1"] = shifted
        calendar["breakdown_count_7d"] = shifted.rolling(7, min_periods=1).sum()
        calendar["breakdown_count_30d"] = shifted.rolling(30, min_periods=1).sum()
        calendar["historical_breakdown_rate"] = shifted.expanding(min_periods=1).mean()

        previous_dates = calendar.index.to_series().shift(1)
        last_breakdown_date = previous_dates.where(shifted.eq(1)).ffill()
        calendar["days_since_last_breakdown"] = (
            calendar.index.to_series() - last_breakdown_date
        ).dt.days

        piece = calendar.loc[indexed.index, list(A2_HISTORY_FEATURES)].reset_index()
        piece[ASSET_COLUMN] = asset_tag
        piece[PART_COLUMN] = part_no
        pieces.append(piece)
    return pd.concat(pieces, ignore_index=True)[
        [DATE_COLUMN, ASSET_COLUMN, PART_COLUMN, *A2_HISTORY_FEATURES]
    ]


def prepare_probabilistic_risk_features(
    raw: pd.DataFrame,
    *,
    selection_start: str | pd.Timestamp = "2024-01-01",
    calibration_start: str | pd.Timestamp = "2024-04-01",
    test_start: str | pd.Timestamp = "2024-07-01",
    expected_parts_per_asset: int = 20,
) -> ProbabilisticRiskFeatures:
    """부품 확률모델의 A1·A2 Feature와 네 시간 구간을 준비한다."""

    selection_date, calibration_date, test_date = _validate_boundaries(
        selection_start, calibration_start, test_start
    )
    frame = _validate_raw(raw, expected_parts_per_asset)
    history = _build_part_history(frame)
    frame = frame.merge(
        history,
        on=[DATE_COLUMN, ASSET_COLUMN, PART_COLUMN],
        how="left",
        validate="one_to_one",
    )
    frame = add_calendar_features(frame)
    frame["period"] = np.select(
        [
            frame[DATE_COLUMN].lt(selection_date),
            frame[DATE_COLUMN].lt(calibration_date),
            frame[DATE_COLUMN].lt(test_date),
        ],
        ["train", "selection", "calibration"],
        default="test",
    )

    periods = {
        name: frame.loc[frame["period"].eq(name)].copy()
        for name in PERIOD_NAMES
    }
    empty = [name for name, period in periods.items() if period.empty]
    if empty:
        raise ValueError(f"다음 시간 구간이 비어 있습니다: {empty}")

    feature_sets = {
        "A1": A1_FEATURES,
        "A2": (*A1_FEATURES, *A2_HISTORY_FEATURES),
    }
    return ProbabilisticRiskFeatures(
        frame=frame,
        feature_sets=feature_sets,
        periods=periods,
        max_failure_points=47,
    )
