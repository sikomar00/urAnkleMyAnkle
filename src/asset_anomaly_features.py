"""장비별 정상 센서 기준과 이상 정도 Feature를 제공한다."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .asset_features import ASSET_CURRENT_FEATURES, build_asset_daily
from .industrial_data import (
    ASSET_COLUMN,
    DATE_COLUMN,
    MACHINE_COLUMN,
    SENSOR_COLUMNS,
)

ROBUST_Z_FEATURES = tuple(f"{column}_robust_z" for column in SENSOR_COLUMNS)
HISTORY_SUFFIXES = (
    "lag1",
    "lag3",
    "lag7",
    "diff1",
    "diff7",
    "median3",
    "median7",
    "mad7",
    "slope7",
)
HISTORY_FEATURES = tuple(
    f"{sensor}_{suffix}"
    for sensor in SENSOR_COLUMNS
    for suffix in HISTORY_SUFFIXES
)
ANOMALY_SUMMARY_FEATURES = (
    "max_abs_robust_z",
    "mean_abs_robust_z",
    "sensor_count_abs_z_ge_2",
    "sensor_count_abs_z_ge_3",
    "temperature_and_vibration_anomaly",
)
FEATURE_SET_NAMES = ("A", "B", "C", "D")
BASELINE_COLUMNS = (
    ASSET_COLUMN,
    MACHINE_COLUMN,
    "sensor",
    "normal_rows",
    "median",
    "mad",
    "scale",
    "selected_scope",
    "fallback_reason",
)


def _sensor_stats(values: pd.Series) -> dict[str, float | int]:
    numeric = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if numeric.empty:
        return {"normal_rows": 0, "median": np.nan, "mad": np.nan, "std": np.nan}
    median = float(numeric.median())
    mad = float((numeric - median).abs().median())
    return {
        "normal_rows": int(len(numeric)),
        "median": median,
        "mad": mad,
        "std": float(numeric.std(ddof=0)),
    }


def _usable_robust(
    stats: dict[str, float | int],
    min_normal_rows: int,
    epsilon: float,
) -> bool:
    return (
        int(stats["normal_rows"]) >= min_normal_rows
        and np.isfinite(float(stats["mad"]))
        and float(stats["mad"]) > epsilon
    )


def _selected_row(
    *,
    asset_tag: Any,
    machine_type: Any,
    sensor: str,
    stats: dict[str, float | int],
    selected_scope: str,
    fallback_reason: str,
    use_std: bool = False,
) -> dict[str, Any]:
    scale = float(stats["std"]) if use_std else 1.4826 * float(stats["mad"])
    if selected_scope == "constant":
        scale = 1.0
    return {
        ASSET_COLUMN: asset_tag,
        MACHINE_COLUMN: machine_type,
        "sensor": sensor,
        "normal_rows": int(stats["normal_rows"]),
        "median": float(stats["median"]),
        "mad": float(stats["mad"]),
        "scale": scale,
        "selected_scope": selected_scope,
        "fallback_reason": fallback_reason,
    }


def _global_choice(
    normal: pd.DataFrame,
    sensor: str,
    epsilon: float,
) -> tuple[dict[str, float | int], str, str, bool]:
    stats = _sensor_stats(normal[sensor])
    if float(stats["mad"]) > epsilon:
        return stats, "global", "machine_baseline_unavailable", False
    if float(stats["std"]) > epsilon:
        return stats, "global_std", "global_mad_zero", True
    return stats, "constant", "global_scale_zero", False


def _build_baseline_table(
    normal: pd.DataFrame,
    min_normal_rows: int,
    epsilon: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for sensor in SENSOR_COLUMNS:
        global_stats, global_scope, global_reason, global_use_std = _global_choice(
            normal, sensor, epsilon
        )
        rows.append(
            _selected_row(
                asset_tag=pd.NA,
                machine_type=pd.NA,
                sensor=sensor,
                stats=global_stats,
                selected_scope=global_scope,
                fallback_reason=global_reason,
                use_std=global_use_std,
            )
        )

        machine_choices: dict[Any, tuple[dict[str, float | int], str, str, bool]] = {}
        for machine_type, machine_rows in normal.groupby(MACHINE_COLUMN, dropna=False):
            machine_stats = _sensor_stats(machine_rows[sensor])
            if _usable_robust(machine_stats, min_normal_rows, epsilon):
                choice = (machine_stats, "machine_type", "", False)
            else:
                machine_reason = (
                    "machine_insufficient_rows"
                    if int(machine_stats["normal_rows"]) < min_normal_rows
                    else "machine_mad_zero"
                )
                choice = (
                    global_stats,
                    global_scope,
                    machine_reason,
                    global_use_std,
                )
            machine_choices[machine_type] = choice
            rows.append(
                _selected_row(
                    asset_tag=pd.NA,
                    machine_type=machine_type,
                    sensor=sensor,
                    stats=choice[0],
                    selected_scope=choice[1],
                    fallback_reason=choice[2],
                    use_std=choice[3],
                )
            )

        for (machine_type, asset_tag), asset_rows in normal.groupby(
            [MACHINE_COLUMN, ASSET_COLUMN], dropna=False
        ):
            asset_stats = _sensor_stats(asset_rows[sensor])
            if _usable_robust(asset_stats, min_normal_rows, epsilon):
                choice = (asset_stats, "asset_tag", "", False)
            else:
                reason = (
                    "asset_insufficient_rows"
                    if int(asset_stats["normal_rows"]) < min_normal_rows
                    else "asset_mad_zero"
                )
                machine_choice = machine_choices[machine_type]
                choice = (
                    machine_choice[0],
                    machine_choice[1],
                    reason,
                    machine_choice[3],
                )
            rows.append(
                _selected_row(
                    asset_tag=asset_tag,
                    machine_type=machine_type,
                    sensor=sensor,
                    stats=choice[0],
                    selected_scope=choice[1],
                    fallback_reason=choice[2],
                    use_std=choice[3],
                )
            )
    return pd.DataFrame(rows).reindex(columns=BASELINE_COLUMNS)


def _lookup_baseline(
    table: pd.DataFrame,
    machine_type: Any,
    asset_tag: Any,
    sensor: str,
) -> pd.Series:
    exact = table[
        table["sensor"].eq(sensor)
        & table[ASSET_COLUMN].eq(asset_tag)
        & table[MACHINE_COLUMN].eq(machine_type)
    ]
    if not exact.empty:
        return exact.iloc[0]
    machine = table[
        table["sensor"].eq(sensor)
        & table[ASSET_COLUMN].isna()
        & table[MACHINE_COLUMN].eq(machine_type)
    ]
    if not machine.empty:
        return machine.iloc[0]
    return table[
        table["sensor"].eq(sensor)
        & table[ASSET_COLUMN].isna()
        & table[MACHINE_COLUMN].isna()
    ].iloc[0]


def _apply_baseline_table(frame: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    required = {ASSET_COLUMN, MACHINE_COLUMN, *SENSOR_COLUMNS}
    missing = sorted(required - set(result.columns))
    if missing:
        raise ValueError(f"Robust Z-score에 필요한 컬럼이 없습니다: {missing}")
    for sensor in SENSOR_COLUMNS:
        transformed: list[float] = []
        for _, row in result.iterrows():
            baseline = _lookup_baseline(
                table,
                row[MACHINE_COLUMN],
                row[ASSET_COLUMN],
                sensor,
            )
            value = pd.to_numeric(pd.Series([row[sensor]]), errors="coerce").iloc[0]
            if pd.isna(value):
                transformed.append(np.nan)
            elif baseline["selected_scope"] == "constant":
                transformed.append(0.0)
            else:
                transformed.append(
                    (float(value) - float(baseline["median"]))
                    / float(baseline["scale"])
                )
        result[f"{sensor}_robust_z"] = transformed
    return result


@dataclass
class RobustNormalBaseline:
    """학습 정상행으로 장비·기계·전체 센서 기준을 적합한다."""

    min_normal_rows: int = 30
    epsilon: float = 1e-12
    _table: pd.DataFrame = field(default_factory=pd.DataFrame, init=False)

    def fit(self, train: pd.DataFrame) -> "RobustNormalBaseline":
        """학습 구간의 0점 행만 사용해 정상 기준을 고정한다."""
        if self.min_normal_rows < 1:
            raise ValueError("min_normal_rows는 1 이상이어야 합니다.")
        if "failure_points" not in train:
            raise ValueError("failure_points 컬럼이 필요합니다.")
        normal = train.loc[train["failure_points"].eq(0)].copy()
        if normal.empty:
            raise ValueError("학습 구간에 failure_points == 0인 정상행이 없습니다.")
        self._table = _build_baseline_table(
            normal,
            self.min_normal_rows,
            self.epsilon,
        )
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """적합된 기준을 바꾸지 않고 Robust Z-score 열을 추가한다."""
        if self._table.empty:
            raise ValueError("RobustNormalBaseline.fit()을 먼저 호출해야 합니다.")
        return _apply_baseline_table(frame, self._table)

    def baseline_table(self) -> pd.DataFrame:
        """선택된 센서별 기준과 fallback 근거의 복사본을 반환한다."""
        if self._table.empty:
            raise ValueError("RobustNormalBaseline.fit()을 먼저 호출해야 합니다.")
        return self._table.copy()


def _rolling_slope(values: np.ndarray) -> float:
    if np.isnan(values).any():
        return np.nan
    x = np.arange(len(values), dtype=float)
    return float(np.polyfit(x, values.astype(float), 1)[0])


def build_sensor_history_features(daily: pd.DataFrame) -> pd.DataFrame:
    """장비별 이전 달력일 센서값으로 변화·추세 Feature를 만든다."""
    required = {DATE_COLUMN, ASSET_COLUMN, *SENSOR_COLUMNS}
    missing = sorted(required - set(daily.columns))
    if missing:
        raise ValueError(f"센서 이력에 필요한 컬럼이 없습니다: {missing}")
    if daily.duplicated([DATE_COLUMN, ASSET_COLUMN]).any():
        raise ValueError("같은 날짜·장비의 중복 행이 있습니다.")

    pieces: list[pd.DataFrame] = []
    for _, group in daily.groupby(ASSET_COLUMN, sort=False):
        indexed = group.sort_values(DATE_COLUMN).set_index(DATE_COLUMN)
        original_dates = indexed.index
        calendar = indexed.reindex(
            pd.date_range(original_dates.min(), original_dates.max(), freq="D")
        )
        calendar.index.name = DATE_COLUMN
        history: dict[str, pd.Series] = {}
        for sensor in SENSOR_COLUMNS:
            shifted = calendar[sensor].shift(1)
            history[f"{sensor}_lag1"] = calendar[sensor].shift(1)
            history[f"{sensor}_lag3"] = calendar[sensor].shift(3)
            history[f"{sensor}_lag7"] = calendar[sensor].shift(7)
            history[f"{sensor}_diff1"] = (
                calendar[sensor] - history[f"{sensor}_lag1"]
            )
            history[f"{sensor}_diff7"] = (
                calendar[sensor] - history[f"{sensor}_lag7"]
            )
            history[f"{sensor}_median3"] = shifted.rolling(
                3, min_periods=2
            ).median()
            history[f"{sensor}_median7"] = shifted.rolling(
                7, min_periods=3
            ).median()
            history[f"{sensor}_mad7"] = shifted.rolling(
                7, min_periods=3
            ).apply(
                lambda values: np.nanmedian(
                    np.abs(values - np.nanmedian(values))
                ),
                raw=True,
            )
            history[f"{sensor}_slope7"] = shifted.rolling(
                7, min_periods=7
            ).apply(_rolling_slope, raw=True)
        history_frame = pd.DataFrame(history, index=calendar.index)
        calendar = pd.concat([calendar, history_frame], axis=1)
        pieces.append(calendar.loc[original_dates].reset_index())
    if not pieces:
        return daily.copy()
    return pd.concat(pieces, ignore_index=True)


@dataclass(frozen=True)
class AssetExperimentFeatures:
    """실험용 장비 일별 Frame, Feature 집합과 정상 기준을 보관한다."""

    frame: pd.DataFrame
    feature_sets: dict[str, tuple[str, ...]]
    baselines: pd.DataFrame
    baseline_transformer: RobustNormalBaseline | None
    unavailable_feature_sets: dict[str, str] = field(default_factory=dict)


def _add_anomaly_summaries(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    absolute = result.loc[:, ROBUST_Z_FEATURES].abs()
    result["max_abs_robust_z"] = absolute.max(axis=1)
    result["mean_abs_robust_z"] = absolute.mean(axis=1)
    result["sensor_count_abs_z_ge_2"] = absolute.ge(2).sum(axis=1)
    result["sensor_count_abs_z_ge_3"] = absolute.ge(3).sum(axis=1)
    temperature = absolute[
        ["temp_bearing_degC_robust_z", "temp_motor_degC_robust_z"]
    ].ge(2).any(axis=1)
    vibration = absolute[
        ["vibration_h_mms_robust_z", "vibration_v_mms_robust_z"]
    ].ge(2).any(axis=1)
    result["temperature_and_vibration_anomaly"] = (
        temperature & vibration
    ).astype(int)
    return result


def prepare_asset_experiment_features(
    raw: pd.DataFrame,
    *,
    validation_start: str | pd.Timestamp = "2024-01-01",
    min_normal_rows: int = 30,
    requested_feature_sets: Sequence[str] = FEATURE_SET_NAMES,
    allow_missing_baseline: bool = False,
) -> AssetExperimentFeatures:
    """학습 정상 기준과 과거값으로 A~D 실험 Feature를 준비한다."""
    requested = tuple(dict.fromkeys(requested_feature_sets))
    daily = build_asset_daily(raw).sort_values(
        [ASSET_COLUMN, DATE_COLUMN]
    ).reset_index(drop=True)
    validation_start = pd.Timestamp(validation_start)
    train = daily.loc[daily["label_end_date"].lt(validation_start)]
    with_history = build_sensor_history_features(daily)
    base = tuple(ASSET_CURRENT_FEATURES)
    available = {
        "A": base,
        "C": (*base, *HISTORY_FEATURES),
    }
    combined = with_history
    baselines = pd.DataFrame(columns=BASELINE_COLUMNS)
    baseline: RobustNormalBaseline | None = None
    unavailable: dict[str, str] = {}

    needs_baseline = bool({"B", "D"}.intersection(requested))
    if needs_baseline:
        baseline_error = ""
        if train.empty:
            baseline_error = "정상 기준을 적합할 학습 구간이 비어 있습니다."
        else:
            try:
                baseline = RobustNormalBaseline(
                    min_normal_rows=min_normal_rows
                ).fit(train)
            except ValueError as error:
                baseline_error = str(error)
        if baseline_error:
            if not allow_missing_baseline:
                raise ValueError(baseline_error)
            unavailable.update(
                {name: baseline_error for name in ("B", "D") if name in requested}
            )
        else:
            with_z = baseline.transform(daily)
            history_columns = [
                DATE_COLUMN,
                MACHINE_COLUMN,
                ASSET_COLUMN,
                *HISTORY_FEATURES,
            ]
            combined = with_z.merge(
                with_history.loc[:, history_columns],
                on=[DATE_COLUMN, MACHINE_COLUMN, ASSET_COLUMN],
                how="left",
                validate="one_to_one",
            )
            combined = _add_anomaly_summaries(combined)
            baselines = baseline.baseline_table()
            available["B"] = (*base, *ROBUST_Z_FEATURES)
            available["D"] = (
                *base,
                *ROBUST_Z_FEATURES,
                *HISTORY_FEATURES,
                *ANOMALY_SUMMARY_FEATURES,
            )

    feature_sets = {
        name: available[name]
        for name in FEATURE_SET_NAMES
        if name in requested and name in available
    }
    return AssetExperimentFeatures(
        frame=combined,
        feature_sets=feature_sets,
        baselines=baselines,
        baseline_transformer=baseline,
        unavailable_feature_sets=unavailable,
    )
