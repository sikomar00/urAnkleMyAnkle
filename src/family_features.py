"""부품 행을 장비·날짜·Family 단위의 당일 진단 데이터로 집계한다."""

from __future__ import annotations

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
