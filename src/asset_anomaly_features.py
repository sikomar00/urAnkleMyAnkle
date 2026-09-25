"""장비별 정상 센서 기준과 이상 정도 Feature를 제공한다."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .industrial_data import ASSET_COLUMN, MACHINE_COLUMN, SENSOR_COLUMNS

ROBUST_Z_FEATURES = tuple(f"{column}_robust_z" for column in SENSOR_COLUMNS)
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
