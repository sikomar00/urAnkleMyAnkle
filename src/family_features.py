"""부품 행을 장비·날짜·Family 단위의 당일 진단 데이터로 집계한다."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from .asset_anomaly_features import HISTORY_FEATURES, build_sensor_history_features
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

FAMILY_NAMES: tuple[str, ...] = (
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
FAMILY_TARGETS: tuple[str, ...] = ("affected", "severe")
FAMILY_HISTORY_FEATURES: tuple[str, ...] = (
    "affected_lag1",
    "affected_count_7d",
    "affected_count_30d",
    "days_since_last_affected",
    "severe_lag1",
    "severe_count_30d",
)
FAMILY_STATIC_FEATURES: tuple[str, ...] = (
    MACHINE_COLUMN,
    ASSET_COLUMN,
    *SENSOR_COLUMNS,
    "day_of_week",
    "month_sin",
    "month_cos",
)

FAMILY_COLUMN = "part_family"
_ASSET_DAY = [DATE_COLUMN, MACHINE_COLUMN, ASSET_COLUMN]
_FAMILY_DAY = [*_ASSET_DAY, FAMILY_COLUMN]


def _validate_family_input(raw: pd.DataFrame) -> pd.DataFrame:
    """Family 집계 전에 원본 스키마와 장비·날짜 계약을 검사한다."""
    required = [*REQUIRED_COLUMNS, FAMILY_COLUMN]
    missing = [column for column in required if column not in raw.columns]
    if missing:
        raise ValueError(f"Family 집계에 필요한 컬럼이 없습니다: {missing}")

    frame = raw.copy()
    frame[DATE_COLUMN] = pd.to_datetime(frame[DATE_COLUMN])

    breakdown_values = set(frame[CURRENT_TARGET].dropna().unique())
    if frame[CURRENT_TARGET].isna().any() or not breakdown_values <= {0, 1}:
        raise ValueError("breakdown_flag는 0 또는 1이어야 합니다.")
    if frame["criticality"].isna().any() or not set(
        frame["criticality"].unique()
    ) <= {"A", "B", "C"}:
        raise ValueError("criticality는 A, B, C 중 하나여야 합니다.")

    actual_families = set(frame[FAMILY_COLUMN].dropna().unique())
    expected_families = set(FAMILY_NAMES)
    if actual_families != expected_families:
        missing_families = sorted(expected_families - actual_families)
        unexpected = sorted(actual_families - expected_families)
        raise ValueError(
            "Family 목록이 고정 스키마와 다릅니다: "
            f"누락={missing_families}, 예상 밖={unexpected}"
        )

    duplicate_key = [DATE_COLUMN, ASSET_COLUMN, PART_COLUMN]
    if frame.duplicated(duplicate_key).any():
        raise ValueError("같은 장비·날짜·부품 행이 중복되었습니다.")

    day_family_sets = frame.groupby(_ASSET_DAY, observed=True)[FAMILY_COLUMN].agg(set)
    if not day_family_sets.map(lambda values: values == expected_families).all():
        raise ValueError("장비·날짜별 Family 구성이 고정 목록과 다릅니다.")

    daily_sensor_counts = frame.groupby(_ASSET_DAY, observed=True)[
        [PLANT_COLUMN, *SENSOR_COLUMNS]
    ].nunique(dropna=False)
    if daily_sensor_counts.gt(1).any(axis=None):
        raise ValueError("같은 장비·날짜 안에서 센서 또는 공장 값이 다릅니다.")

    signature_columns = [PART_COLUMN, FAMILY_COLUMN, "criticality"]
    signatures = (
        frame.groupby([MACHINE_COLUMN, ASSET_COLUMN], observed=True)[signature_columns]
        .apply(
            lambda group: frozenset(
                group.drop_duplicates().itertuples(index=False, name=None)
            ),
            include_groups=False,
        )
        .to_dict()
    )
    expected_signatures = frame.groupby(
        [DATE_COLUMN, MACHINE_COLUMN, ASSET_COLUMN], observed=True
    ).apply(
        lambda group: frozenset(
            group[signature_columns].itertuples(index=False, name=None)
        ),
        include_groups=False,
    )
    for (_, machine_type, asset_tag), signature in expected_signatures.items():
        if signature != signatures[(machine_type, asset_tag)]:
            raise ValueError("날짜에 따라 장비의 Family 부품 구성이 달라졌습니다.")

    return frame


def build_family_daily(raw: pd.DataFrame) -> pd.DataFrame:
    """부품별 원본을 장비·날짜·Family별 당일 Target으로 집계한다.

    ``affected``는 한 부품 이상 고장, ``severe``는 두 부품 이상 또는 A등급
    부품 고장, ``all_failed``는 Family의 모든 부품 고장을 뜻한다.
    """
    frame = _validate_family_input(raw)
    frame["failed_a_part"] = (
        frame[CURRENT_TARGET].eq(1) & frame["criticality"].eq("A")
    ).astype(int)

    aggregations: dict[str, tuple[str, str]] = {
        PLANT_COLUMN: (PLANT_COLUMN, "first"),
        "total_parts": (PART_COLUMN, "nunique"),
        "failed_parts": (CURRENT_TARGET, "sum"),
        "failed_a_parts": ("failed_a_part", "sum"),
    }
    aggregations.update(
        {column: (column, "first") for column in SENSOR_COLUMNS}
    )
    result = (
        frame.groupby(_FAMILY_DAY, observed=True, sort=False)
        .agg(**aggregations)
        .reset_index()
    )
    result["affected"] = result["failed_parts"].ge(1).astype(int)
    result["severe"] = (
        result["failed_parts"].ge(2) | result["failed_a_parts"].ge(1)
    ).astype(int)
    result["all_failed"] = result["failed_parts"].eq(result["total_parts"]).astype(int)
    if result["severe"].gt(result["affected"]).any():
        raise ValueError("severe Target은 affected Target보다 클 수 없습니다.")

    result["label_end_date"] = result[DATE_COLUMN]
    result[FAMILY_COLUMN] = pd.Categorical(
        result[FAMILY_COLUMN], categories=FAMILY_NAMES, ordered=True
    )
    result = add_calendar_features(result)
    return result.sort_values([DATE_COLUMN, ASSET_COLUMN, FAMILY_COLUMN]).reset_index(
        drop=True
    )


def build_family_target_profile(family_daily: pd.DataFrame) -> pd.DataFrame:
    """Family별 전체·학습·검증·테스트 Target 발생률을 반환한다."""
    required = {
        DATE_COLUMN,
        FAMILY_COLUMN,
        "total_parts",
        "affected",
        "severe",
        "all_failed",
    }
    missing = sorted(required - set(family_daily.columns))
    if missing:
        raise ValueError(f"Family Target 프로필에 필요한 컬럼이 없습니다: {missing}")

    frame = family_daily.copy()
    frame[DATE_COLUMN] = pd.to_datetime(frame[DATE_COLUMN])
    validation_start = pd.Timestamp("2024-01-01")
    test_start = pd.Timestamp("2024-07-01")
    periods = {
        "overall": pd.Series(True, index=frame.index),
        "train": frame[DATE_COLUMN].lt(validation_start),
        "valid": frame[DATE_COLUMN].ge(validation_start)
        & frame[DATE_COLUMN].lt(test_start),
        "test": frame[DATE_COLUMN].ge(test_start),
    }

    rows: list[dict[str, object]] = []
    for period, mask in periods.items():
        period_frame = frame.loc[mask]
        for family in FAMILY_NAMES:
            selected = period_frame[period_frame[FAMILY_COLUMN].eq(family)]
            row_count = len(selected)
            row: dict[str, object] = {
                "period": period,
                FAMILY_COLUMN: family,
                "row_count": row_count,
                "total_parts": (
                    int(selected["total_parts"].iloc[0]) if row_count else pd.NA
                ),
            }
            for target in (*FAMILY_TARGETS, "all_failed"):
                count = int(selected[target].sum()) if row_count else 0
                row[f"{target}_count"] = count
                row[f"{target}_rate"] = count / row_count if row_count else float("nan")
            rows.append(row)
    return pd.DataFrame(rows)


def _family_target_history(group: pd.DataFrame) -> pd.DataFrame:
    """한 장비·Family의 과거 Target을 빈 달력일까지 보존해 계산한다."""
    indexed = group.sort_values(DATE_COLUMN).set_index(DATE_COLUMN)
    original_dates = indexed.index
    calendar = indexed.reindex(
        pd.date_range(original_dates.min(), original_dates.max(), freq="D")
    )
    calendar.index.name = DATE_COLUMN

    shifted_affected = calendar["affected"].shift(1)
    shifted_severe = calendar["severe"].shift(1)
    calendar["affected_lag1"] = shifted_affected
    calendar["affected_count_7d"] = shifted_affected.rolling(
        7, min_periods=1
    ).sum()
    calendar["affected_count_30d"] = shifted_affected.rolling(
        30, min_periods=1
    ).sum()
    previous_dates = calendar.index.to_series() - pd.Timedelta(days=1)
    last_affected_date = previous_dates.where(shifted_affected.eq(1)).ffill()
    calendar["days_since_last_affected"] = (
        calendar.index.to_series() - last_affected_date
    ).dt.days
    calendar["severe_lag1"] = shifted_severe
    calendar["severe_count_30d"] = shifted_severe.rolling(
        30, min_periods=1
    ).sum()
    return calendar.loc[original_dates].reset_index()


def build_family_history_features(family_daily: pd.DataFrame) -> pd.DataFrame:
    """Family 과거 상태와 장비 센서 과거 Feature를 누수 없이 추가한다."""
    required = {
        DATE_COLUMN,
        MACHINE_COLUMN,
        ASSET_COLUMN,
        FAMILY_COLUMN,
        "affected",
        "severe",
        *SENSOR_COLUMNS,
    }
    missing = sorted(required - set(family_daily.columns))
    if missing:
        raise ValueError(f"Family 이력에 필요한 컬럼이 없습니다: {missing}")
    if family_daily.duplicated(_FAMILY_DAY).any():
        raise ValueError("같은 장비·날짜·Family 행이 중복되었습니다.")

    frame = family_daily.copy()
    frame[DATE_COLUMN] = pd.to_datetime(frame[DATE_COLUMN])
    family_pieces = [
        _family_target_history(group)
        for _, group in frame.groupby(
            [ASSET_COLUMN, FAMILY_COLUMN], observed=True, sort=False
        )
    ]
    with_family_history = pd.concat(family_pieces, ignore_index=True)

    sensor_keys = [DATE_COLUMN, MACHINE_COLUMN, ASSET_COLUMN]
    asset_daily = frame[[*sensor_keys, *SENSOR_COLUMNS]].drop_duplicates()
    if asset_daily.duplicated([DATE_COLUMN, ASSET_COLUMN]).any():
        raise ValueError("같은 장비·날짜에 서로 다른 센서 행이 있습니다.")
    sensor_history = build_sensor_history_features(asset_daily)
    sensor_history = sensor_history[[*sensor_keys, *HISTORY_FEATURES]]
    result = with_family_history.merge(
        sensor_history,
        on=sensor_keys,
        how="left",
        validate="many_to_one",
    )
    return result.sort_values(
        [DATE_COLUMN, ASSET_COLUMN, FAMILY_COLUMN]
    ).reset_index(drop=True)


@dataclass(frozen=True)
class FamilyDiagnosisFeatures:
    """Family 진단 Frame과 실험별 Feature 목록을 함께 보관한다."""

    frame: pd.DataFrame
    feature_sets: dict[str, tuple[str, ...]]
    residual_transformer: object | None = None


def prepare_family_features(
    raw: pd.DataFrame,
    validation_start: str | pd.Timestamp = "2024-01-01",
    feature_sets: Sequence[str] = ("A", "B"),
) -> FamilyDiagnosisFeatures:
    """원본 부품 행에서 정적 A와 과거 이력 B 실험 입력을 준비한다."""
    requested = tuple(dict.fromkeys(feature_sets))
    unknown = sorted(set(requested) - {"A", "B"})
    if unknown:
        raise ValueError(f"지원하지 않는 Feature 집합입니다: {unknown}")
    if not requested:
        raise ValueError("Feature 집합을 하나 이상 요청해야 합니다.")
    pd.Timestamp(validation_start)

    frame = build_family_history_features(build_family_daily(raw))
    available = {
        "A": FAMILY_STATIC_FEATURES,
        "B": (*FAMILY_STATIC_FEATURES, *HISTORY_FEATURES, *FAMILY_HISTORY_FEATURES),
    }
    selected = {name: tuple(available[name]) for name in requested}
    forbidden = {
        CURRENT_TARGET,
        "affected",
        "severe",
        "all_failed",
        "failed_parts",
        "failed_a_parts",
    }
    for name, columns in selected.items():
        overlap = sorted(forbidden.intersection(columns))
        if overlap:
            raise ValueError(f"{name} Feature 집합에 Target 열이 포함됐습니다: {overlap}")
    return FamilyDiagnosisFeatures(frame=frame, feature_sets=selected)
