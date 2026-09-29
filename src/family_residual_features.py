"""운전조건으로 설명되는 정상 변화를 제거한 Family 진단 잔차 Feature."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from .industrial_data import ASSET_COLUMN, DATE_COLUMN, MACHINE_COLUMN
from .industrial_regression import build_regression_model

HEALTH_SENSOR_COLUMNS = [
    "temp_bearing_degC",
    "temp_motor_degC",
    "vibration_h_mms",
    "vibration_v_mms",
    "oil_pressure_bar",
]
OPERATING_CONTEXT_COLUMNS = [
    "load_pct",
    "shaft_rpm",
    "power_consumption_kw",
]
CALENDAR_CONTEXT_COLUMNS = ["day_of_week", "month_sin", "month_cos"]
RESIDUAL_HISTORY_SUFFIXES = ("lag1", "lag3", "lag7", "median7", "mad7", "slope7")

_RESIDUAL_CURRENT_FEATURES = tuple(
    feature
    for sensor in HEALTH_SENSOR_COLUMNS
    for feature in (
        f"{sensor}_expected",
        f"{sensor}_residual",
        f"{sensor}_residual_robust_z",
    )
)
_RESIDUAL_HISTORY_FEATURES = tuple(
    f"{source}_{suffix}"
    for sensor in HEALTH_SENSOR_COLUMNS
    for source in (f"{sensor}_residual", f"{sensor}_residual_robust_z")
    for suffix in RESIDUAL_HISTORY_SUFFIXES
)
RESIDUAL_FEATURES = (*_RESIDUAL_CURRENT_FEATURES, *_RESIDUAL_HISTORY_FEATURES)

_DAY_KEYS = [DATE_COLUMN, MACHINE_COLUMN, ASSET_COLUMN]
_MODEL_NAME = "hist_gradient_boosting_regressor"


def _rolling_slope(values: np.ndarray) -> float:
    if np.isnan(values).any():
        return np.nan
    x = np.arange(len(values), dtype=float)
    return float(np.polyfit(x, values.astype(float), 1)[0])


def _scale_stats(values: pd.Series, epsilon: float) -> dict[str, float | str]:
    numeric = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if numeric.empty:
        raise ValueError("잔차 기준을 계산할 정상 잔차가 없습니다.")
    median = float(numeric.median())
    mad = float((numeric - median).abs().median())
    std = float(numeric.std(ddof=0))
    if mad > epsilon:
        scale = 1.4826 * mad
        method = "mad"
    elif std > epsilon:
        scale = std
        method = "std"
    else:
        scale = 1.0
        method = "constant"
    return {
        "residual_median": median,
        "residual_mad": mad,
        "residual_std": std,
        "residual_scale": scale,
        "scale_method": method,
    }


def _asset_daily(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        *_DAY_KEYS,
        "affected",
        *HEALTH_SENSOR_COLUMNS,
        *OPERATING_CONTEXT_COLUMNS,
        *CALENDAR_CONTEXT_COLUMNS,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"운전조건 잔차에 필요한 컬럼이 없습니다: {missing}")

    value_columns = [
        *HEALTH_SENSOR_COLUMNS,
        *OPERATING_CONTEXT_COLUMNS,
        *CALENDAR_CONTEXT_COLUMNS,
    ]
    counts = frame.groupby(_DAY_KEYS, observed=True)[value_columns].nunique(
        dropna=False
    )
    if counts.gt(1).any(axis=None):
        raise ValueError("같은 장비·날짜에서 잔차 입력값이 서로 다릅니다.")

    aggregations: dict[str, tuple[str, str]] = {
        column: (column, "first") for column in value_columns
    }
    aggregations["any_affected"] = ("affected", "max")
    return (
        frame.groupby(_DAY_KEYS, observed=True, as_index=False)
        .agg(**aggregations)
        .sort_values([ASSET_COLUMN, DATE_COLUMN])
        .reset_index(drop=True)
    )


def _history_for_asset(group: pd.DataFrame) -> pd.DataFrame:
    indexed = group.sort_values(DATE_COLUMN).set_index(DATE_COLUMN)
    original_dates = indexed.index
    calendar = indexed.reindex(
        pd.date_range(original_dates.min(), original_dates.max(), freq="D")
    )
    calendar.index.name = DATE_COLUMN
    history: dict[str, pd.Series] = {}
    for sensor in HEALTH_SENSOR_COLUMNS:
        for source in (f"{sensor}_residual", f"{sensor}_residual_robust_z"):
            shifted = calendar[source].shift(1)
            history[f"{source}_lag1"] = calendar[source].shift(1)
            history[f"{source}_lag3"] = calendar[source].shift(3)
            history[f"{source}_lag7"] = calendar[source].shift(7)
            history[f"{source}_median7"] = shifted.rolling(
                7, min_periods=3
            ).median()
            history[f"{source}_mad7"] = shifted.rolling(
                7, min_periods=3
            ).apply(
                lambda values: np.nanmedian(
                    np.abs(values - np.nanmedian(values))
                ),
                raw=True,
            )
            history[f"{source}_slope7"] = shifted.rolling(
                7, min_periods=7
            ).apply(_rolling_slope, raw=True)
    calendar = pd.concat([calendar, pd.DataFrame(history, index=calendar.index)], axis=1)
    return calendar.loc[original_dates].reset_index()


@dataclass
class OperatingResidualTransformer:
    """학습 정상일에서 운전조건 기대 센서와 잔차 기준을 적합한다."""

    min_normal_rows: int = 30
    max_iter: int = 100
    random_state: int = 42
    epsilon: float = 1e-12
    _global_models: dict[str, Pipeline] = field(default_factory=dict, init=False)
    _asset_models: dict[tuple[str, str], Pipeline] = field(
        default_factory=dict, init=False
    )
    _artifacts: pd.DataFrame = field(default_factory=pd.DataFrame, init=False)

    def fit(self, train: pd.DataFrame) -> "OperatingResidualTransformer":
        """학습 구간의 완전 정상 장비일만 사용해 모델과 잔차 기준을 고정한다."""
        if self.min_normal_rows < 1:
            raise ValueError("min_normal_rows는 1 이상이어야 합니다.")
        daily = _asset_daily(train)
        normal = daily.loc[daily["any_affected"].eq(0)].copy()
        if normal.empty:
            raise ValueError("학습 구간에 모든 Family가 정상인 장비일이 없습니다.")
        if normal[HEALTH_SENSOR_COLUMNS].isna().any(axis=None):
            raise ValueError("건강상태 센서 Target에는 결측을 사용할 수 없습니다.")

        global_features = [
            MACHINE_COLUMN,
            ASSET_COLUMN,
            *OPERATING_CONTEXT_COLUMNS,
            *CALENDAR_CONTEXT_COLUMNS,
        ]
        asset_features = [*OPERATING_CONTEXT_COLUMNS, *CALENDAR_CONTEXT_COLUMNS]
        eligible_assets = {
            str(asset_tag)
            for asset_tag, rows in normal.groupby(ASSET_COLUMN, observed=True)
            if len(rows) >= self.min_normal_rows
        }
        self._global_models = {}
        self._asset_models = {}
        global_residuals: dict[str, pd.Series] = {}
        asset_residuals: dict[tuple[str, str], pd.Series] = {}

        for sensor in HEALTH_SENSOR_COLUMNS:
            global_model = build_regression_model(
                _MODEL_NAME,
                normal[global_features],
                max_iter=self.max_iter,
                random_state=self.random_state,
            )
            global_model.fit(normal[global_features], normal[sensor])
            self._global_models[sensor] = global_model
            global_residuals[sensor] = normal[sensor] - global_model.predict(
                normal[global_features]
            )

            for asset_tag in sorted(eligible_assets):
                asset_rows = normal.loc[normal[ASSET_COLUMN].astype(str).eq(asset_tag)]
                asset_model = build_regression_model(
                    _MODEL_NAME,
                    asset_rows[asset_features],
                    max_iter=self.max_iter,
                    random_state=self.random_state,
                )
                asset_model.fit(asset_rows[asset_features], asset_rows[sensor])
                self._asset_models[(asset_tag, sensor)] = asset_model
                asset_residuals[(asset_tag, sensor)] = pd.Series(
                    asset_rows[sensor].to_numpy()
                    - asset_model.predict(asset_rows[asset_features]),
                    index=asset_rows.index,
                )

        artifact_rows: list[dict[str, Any]] = []
        global_stats: dict[str, dict[str, float | str]] = {}
        for sensor in HEALTH_SENSOR_COLUMNS:
            stats = _scale_stats(global_residuals[sensor], self.epsilon)
            global_stats[sensor] = stats
            artifact_rows.append(
                {
                    ASSET_COLUMN: pd.NA,
                    "sensor": sensor,
                    "normal_rows": len(normal),
                    "selected_scope": "global",
                    "fallback_reason": "",
                    **stats,
                }
            )
            for asset_tag, asset_rows in normal.groupby(ASSET_COLUMN, observed=True):
                asset_key = str(asset_tag)
                if asset_key in eligible_assets:
                    selected_scope = "asset_tag"
                    fallback_reason = ""
                    stats = _scale_stats(
                        asset_residuals[(asset_key, sensor)], self.epsilon
                    )
                else:
                    selected_scope = "global"
                    fallback_reason = "asset_insufficient_normal_rows"
                    stats = global_stats[sensor]
                artifact_rows.append(
                    {
                        ASSET_COLUMN: asset_tag,
                        "sensor": sensor,
                        "normal_rows": len(asset_rows),
                        "selected_scope": selected_scope,
                        "fallback_reason": fallback_reason,
                        **stats,
                    }
                )
        self._artifacts = pd.DataFrame(artifact_rows).reset_index(drop=True)
        return self

    def _baseline(self, asset_tag: Any, sensor: str) -> pd.Series:
        selected = self._artifacts[
            self._artifacts[ASSET_COLUMN].eq(asset_tag)
            & self._artifacts["sensor"].eq(sensor)
        ]
        if selected.empty:
            selected = self._artifacts[
                self._artifacts[ASSET_COLUMN].isna()
                & self._artifacts["sensor"].eq(sensor)
            ]
        return selected.iloc[0]

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """고정된 기대값 모델과 잔차 기준으로 전체 기간 Feature를 만든다."""
        if self._artifacts.empty:
            raise ValueError("OperatingResidualTransformer.fit()을 먼저 호출해야 합니다.")
        daily = _asset_daily(frame)
        global_features = [
            MACHINE_COLUMN,
            ASSET_COLUMN,
            *OPERATING_CONTEXT_COLUMNS,
            *CALENDAR_CONTEXT_COLUMNS,
        ]
        asset_features = [*OPERATING_CONTEXT_COLUMNS, *CALENDAR_CONTEXT_COLUMNS]

        for sensor in HEALTH_SENSOR_COLUMNS:
            expected = pd.Series(index=daily.index, dtype=float)
            for asset_tag, index in daily.groupby(ASSET_COLUMN, observed=True).groups.items():
                asset_key = str(asset_tag)
                rows = daily.loc[index]
                model = self._asset_models.get((asset_key, sensor))
                if model is None:
                    predictions = self._global_models[sensor].predict(
                        rows[global_features]
                    )
                else:
                    predictions = model.predict(rows[asset_features])
                expected.loc[index] = predictions
            residual_column = f"{sensor}_residual"
            daily[f"{sensor}_expected"] = expected
            daily[residual_column] = daily[sensor] - expected
            z_values = pd.Series(index=daily.index, dtype=float)
            for index, row in daily.iterrows():
                baseline = self._baseline(row[ASSET_COLUMN], sensor)
                if baseline["scale_method"] == "constant":
                    z_values.loc[index] = 0.0
                else:
                    z_values.loc[index] = (
                        row[residual_column] - baseline["residual_median"]
                    ) / baseline["residual_scale"]
            daily[f"{sensor}_residual_robust_z"] = z_values

        with_history = pd.concat(
            [
                _history_for_asset(group)
                for _, group in daily.groupby(ASSET_COLUMN, observed=True, sort=False)
            ],
            ignore_index=True,
        )
        residual_daily = with_history[[*_DAY_KEYS, *RESIDUAL_FEATURES]]
        return frame.merge(
            residual_daily,
            on=_DAY_KEYS,
            how="left",
            validate="many_to_one",
        )

    def artifact_table(self) -> pd.DataFrame:
        """선택 모델 범위와 학습 정상 잔차 기준을 복사해 반환한다."""
        if self._artifacts.empty:
            raise ValueError("OperatingResidualTransformer.fit()을 먼저 호출해야 합니다.")
        return self._artifacts.copy()
