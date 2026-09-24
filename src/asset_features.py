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
    PreparedRegressionTask,
    PreparedTask,
    add_calendar_features,
)

CRITICALITY_WEIGHTS = {"A": 4, "B": 2, "C": 1}
SEVERITY_LEVELS = ("normal", "caution", "risk", "high_risk")
SEVERITY_CODE = {name: index for index, name in enumerate(SEVERITY_LEVELS)}

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


def add_asset_severity(frame: pd.DataFrame) -> pd.DataFrame:
    """정수 고장점수를 정상·주의·위험·고위험으로 변환한다."""
    if "failure_points" not in frame:
        raise ValueError("failure_points 컬럼이 필요합니다.")

    result = frame.copy()
    scores = pd.to_numeric(result["failure_points"], errors="coerce")
    if scores.isna().any() or scores.lt(0).any():
        raise ValueError("failure_points는 결측이 없는 0 이상의 숫자여야 합니다.")

    labels = pd.Series("high_risk", index=result.index, dtype="string")
    labels.loc[scores.eq(0)] = "normal"
    labels.loc[scores.between(1, 5, inclusive="both")] = "caution"
    labels.loc[scores.between(6, 11, inclusive="both")] = "risk"
    result["failure_points"] = scores
    result["severity_level"] = pd.Categorical(
        labels,
        categories=SEVERITY_LEVELS,
        ordered=True,
    )
    result["severity_code"] = labels.map(SEVERITY_CODE).astype(int)
    return result


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


def prepare_asset_score_current(frame: pd.DataFrame) -> PreparedRegressionTask:
    """장비 당일 고장점수 회귀 과제를 준비한다."""
    daily = add_asset_severity(build_asset_daily(frame))
    return PreparedRegressionTask(
        frame=daily,
        features=ASSET_CURRENT_FEATURES,
        target="failure_points",
        grain="asset",
        mode="current",
    )


def prepare_asset_severity_current(frame: pd.DataFrame) -> PreparedTask:
    """장비 당일 4단계 위험도 분류 과제를 준비한다."""
    daily = add_asset_severity(build_asset_daily(frame))
    return PreparedTask(
        frame=daily,
        features=ASSET_CURRENT_FEATURES,
        target="severity_level",
        grain="asset",
        mode="current",
        risk_definition="severity_4class",
    )


def _asset_history_features(
    daily: pd.DataFrame,
    current_target: str,
) -> tuple[pd.DataFrame, list[str]]:
    """달력 날짜를 보존하여 장비별 센서·과거 위험 Feature를 계산한다."""
    pieces: list[pd.DataFrame] = []
    feature_names: list[str] = []
    for column in SENSOR_COLUMNS:
        feature_names.extend([
            f"{column}_current", f"{column}_lag1", f"{column}_lag3",
            f"{column}_lag7", f"{column}_mean7", f"{column}_std7",
            f"{column}_diff1",
        ])
    risk_features = [
        "failure_points_lag1", "failure_points_mean7", "failure_points_max7",
        "risk_event_count_30d", "days_since_last_risk",
    ]
    for asset_tag, group in daily.groupby(ASSET_COLUMN, sort=False):
        indexed = group.sort_values(DATE_COLUMN).set_index(DATE_COLUMN)
        calendar = indexed.reindex(
            pd.date_range(indexed.index.min(), indexed.index.max(), freq="D")
        )
        calendar.index.name = DATE_COLUMN
        for column in SENSOR_COLUMNS:
            values = calendar[column]
            calendar[f"{column}_current"] = values
            calendar[f"{column}_lag1"] = values.shift(1)
            calendar[f"{column}_lag3"] = values.shift(3)
            calendar[f"{column}_lag7"] = values.shift(7)
            calendar[f"{column}_mean7"] = values.rolling(7, min_periods=3).mean()
            calendar[f"{column}_std7"] = values.rolling(7, min_periods=3).std()
            calendar[f"{column}_diff1"] = values.diff(1)

        shifted_points = calendar["failure_points"].shift(1)
        calendar["failure_points_lag1"] = shifted_points
        calendar["failure_points_mean7"] = shifted_points.rolling(
            7, min_periods=1
        ).mean()
        calendar["failure_points_max7"] = shifted_points.rolling(
            7, min_periods=1
        ).max()
        shifted_risk = calendar[current_target].shift(1)
        calendar["risk_event_count_30d"] = shifted_risk.rolling(
            30, min_periods=1
        ).sum()
        previous_dates = calendar.index.to_series().shift(1)
        event_dates = previous_dates.where(shifted_risk.eq(1)).ffill()
        calendar["days_since_last_risk"] = (
            calendar.index.to_series() - event_dates
        ).dt.days

        original_dates = indexed.index
        piece = calendar.loc[original_dates, feature_names + risk_features].reset_index()
        piece[ASSET_COLUMN] = asset_tag
        pieces.append(piece)
    return pd.concat(pieces, ignore_index=True), feature_names + risk_features


def prepare_asset_forecast(
    frame: pd.DataFrame,
    score_threshold: int,
    horizon: int = 7,
    risk_definition: str = "new",
) -> PreparedTask:
    """오늘까지의 장비 센서·위험 이력으로 미래 위험 과제를 준비한다.

    ``new``는 오늘 정상인 장비만 남겨 신규 위험을 예측하고, ``any``는 오늘
    상태와 관계없이 미래 위험을 예측한다. 미래 달력 날짜가 연속으로 모두 존재하는
    행만 Target을 만든다.
    """
    if score_threshold < 1:
        raise ValueError("score_threshold는 1 이상이어야 합니다.")
    if horizon < 1:
        raise ValueError("horizon은 1 이상이어야 합니다.")
    if risk_definition not in {"new", "any"}:
        raise ValueError("risk_definition은 new 또는 any여야 합니다.")

    daily = build_asset_daily(frame).sort_values(
        [ASSET_COLUMN, DATE_COLUMN]
    ).reset_index(drop=True)
    current_target = f"target_ge_{score_threshold}"
    daily[current_target] = (daily["failure_points"] >= score_threshold).astype(int)
    grouped = daily.groupby(ASSET_COLUMN, sort=False)

    valid = pd.Series(True, index=daily.index)
    future_values = []
    for offset in range(1, horizon + 1):
        future_date = grouped[DATE_COLUMN].shift(-offset)
        valid &= (future_date - daily[DATE_COLUMN]).dt.days.eq(offset)
        future_values.append(grouped[current_target].shift(-offset).astype(float))

    target = f"target_{risk_definition}_risk_{horizon}d_ge_{score_threshold}"
    daily[target] = pd.concat(future_values, axis=1).max(axis=1).where(valid)
    daily["label_end_date"] = (
        daily[DATE_COLUMN] + pd.Timedelta(days=horizon)
    ).where(valid, pd.NaT)

    history, history_features = _asset_history_features(daily, current_target)
    daily = daily.merge(history, on=[DATE_COLUMN, ASSET_COLUMN], how="left")
    feature_names = [
        MACHINE_COLUMN, ASSET_COLUMN, *history_features,
        "day_of_week", "month_sin", "month_cos",
    ]

    if risk_definition == "new":
        daily = daily[daily[current_target].eq(0)].copy()
    daily = daily.dropna(subset=[target]).copy()
    daily[target] = daily[target].astype(int)
    return PreparedTask(
        frame=daily.reset_index(drop=True),
        features=tuple(feature_names),
        target=target,
        grain="asset",
        mode="forecast",
        risk_definition=risk_definition,
        score_threshold=score_threshold,
        horizon=horizon,
    )
