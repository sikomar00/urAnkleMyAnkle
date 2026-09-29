"""부품 고장확률을 순위와 장비 고장점수 확률분포로 집계한다."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from .industrial_data import ASSET_COLUMN, CURRENT_TARGET, DATE_COLUMN, PART_COLUMN

MODEL_VARIANT_COLUMN = "model_variant"
RANK_GROUP_COLUMNS = [DATE_COLUMN, ASSET_COLUMN, MODEL_VARIANT_COLUMN]
SEVERITY_LEVELS = ("normal", "caution", "risk", "high_risk")


def weighted_score_distribution(
    probabilities: Sequence[float],
    weights: Sequence[int],
    *,
    max_score: int | None = None,
) -> np.ndarray:
    """독립 Bernoulli 부품들의 가중합 확률분포를 동적계획법으로 계산한다."""

    probability_array = np.asarray(probabilities, dtype=float)
    raw_weights = np.asarray(weights)
    if probability_array.ndim != 1 or raw_weights.ndim != 1:
        raise ValueError("확률과 가중치는 1차원이어야 합니다.")
    if len(probability_array) != len(raw_weights) or not len(probability_array):
        raise ValueError("확률과 가중치 개수는 같고 1개 이상이어야 합니다.")
    if not np.isfinite(probability_array).all():
        raise ValueError("확률에는 결측값이나 무한값을 사용할 수 없습니다.")
    if ((probability_array < 0) | (probability_array > 1)).any():
        raise ValueError("확률은 0과 1 사이여야 합니다.")
    numeric_weights = pd.to_numeric(pd.Series(raw_weights), errors="coerce").to_numpy(
        dtype=float
    )
    if (
        not np.isfinite(numeric_weights).all()
        or (numeric_weights <= 0).any()
        or not np.equal(numeric_weights, np.floor(numeric_weights)).all()
    ):
        raise ValueError("가중치는 양의 정수여야 합니다.")
    weight_array = numeric_weights.astype(int)
    required_score = int(weight_array.sum())
    if max_score is None:
        max_score = required_score
    if isinstance(max_score, bool) or not isinstance(max_score, int) or max_score < required_score:
        raise ValueError("max_score는 가중치 합 이상의 정수여야 합니다.")

    distribution = np.zeros(max_score + 1, dtype=float)
    distribution[0] = 1.0
    reached = 0
    for probability, weight in zip(probability_array, weight_array, strict=True):
        updated = np.zeros_like(distribution)
        updated[: reached + 1] += distribution[: reached + 1] * (1.0 - probability)
        updated[weight : reached + weight + 1] += (
            distribution[: reached + 1] * probability
        )
        distribution = updated
        reached += int(weight)

    if not np.isclose(distribution.sum(), 1.0, atol=1e-10):
        raise ValueError("점수분포 확률의 합이 1이 아닙니다.")
    direct_expectation = float(np.dot(probability_array, weight_array))
    distribution_expectation = float(
        np.dot(np.arange(len(distribution)), distribution)
    )
    if not np.isclose(direct_expectation, distribution_expectation, atol=1e-10):
        raise ValueError("점수분포 기대값이 부품확률 가중합과 일치하지 않습니다.")
    return distribution


def _required_columns(frame: pd.DataFrame, required: set[str], context: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{context}에 필요한 컬럼이 없습니다: {missing}")


def rank_part_probabilities(
    frame: pd.DataFrame,
    *,
    probability_column: str = "failure_probability",
) -> pd.DataFrame:
    """장비일별 부품확률·예상점수 기여도 순위를 결정적으로 계산한다."""

    required = {
        *RANK_GROUP_COLUMNS,
        PART_COLUMN,
        "criticality_weight",
        probability_column,
    }
    _required_columns(frame, required, "부품확률 순위")
    result = frame.copy()
    probabilities = pd.to_numeric(result[probability_column], errors="coerce")
    if (
        probabilities.isna().any()
        or not np.isfinite(probabilities).all()
        or not probabilities.between(0, 1).all()
    ):
        raise ValueError("부품 고장확률은 결측 없이 0과 1 사이여야 합니다.")
    result[probability_column] = probabilities.astype(float)
    weights = pd.to_numeric(result["criticality_weight"], errors="coerce")
    if weights.isna().any() or weights.le(0).any():
        raise ValueError("criticality_weight는 양수여야 합니다.")
    result["criticality_weight"] = weights.astype(int)
    duplicate_keys = [*RANK_GROUP_COLUMNS, PART_COLUMN]
    if result.duplicated(duplicate_keys).any():
        raise ValueError("같은 장비·날짜·모델에 중복 부품이 있습니다.")

    result["expected_score_contribution"] = (
        result[probability_column] * result["criticality_weight"]
    )
    pieces: list[pd.DataFrame] = []
    for _, group in result.groupby(RANK_GROUP_COLUMNS, sort=True, dropna=False):
        ranked = group.copy()
        probability_order = ranked.sort_values(
            [probability_column, "criticality_weight", PART_COLUMN],
            ascending=[False, False, True],
            kind="stable",
        ).index
        contribution_order = ranked.sort_values(
            ["expected_score_contribution", "criticality_weight", PART_COLUMN],
            ascending=[False, False, True],
            kind="stable",
        ).index
        ranked.loc[probability_order, "probability_rank"] = np.arange(
            1, len(ranked) + 1
        )
        ranked.loc[contribution_order, "contribution_rank"] = np.arange(
            1, len(ranked) + 1
        )
        pieces.append(ranked)
    ranked_result = pd.concat(pieces, ignore_index=True)
    ranked_result[["probability_rank", "contribution_rank"]] = ranked_result[
        ["probability_rank", "contribution_rank"]
    ].astype(int)
    return ranked_result.sort_values(
        [*RANK_GROUP_COLUMNS, PART_COLUMN]
    ).reset_index(drop=True)


def severity_probabilities(
    score_distribution: Sequence[float], high_risk_threshold: int
) -> dict[str, float | str]:
    """점수분포를 정상·주의·위험·고위험 확률과 대표 등급으로 변환한다."""

    distribution = np.asarray(score_distribution, dtype=float)
    if distribution.ndim != 1 or not len(distribution):
        raise ValueError("점수분포는 비어 있지 않은 1차원이어야 합니다.")
    if (
        not np.isfinite(distribution).all()
        or (distribution < 0).any()
        or not np.isclose(distribution.sum(), 1.0, atol=1e-10)
    ):
        raise ValueError("점수분포는 유효한 확률이며 합이 1이어야 합니다.")
    if (
        isinstance(high_risk_threshold, bool)
        or not isinstance(high_risk_threshold, int)
        or high_risk_threshold < 7
        or high_risk_threshold >= len(distribution)
    ):
        raise ValueError("고위험 기준은 7 이상이며 점수분포 범위 안의 정수여야 합니다.")

    values = {
        "prob_normal": float(distribution[0]),
        "prob_caution": float(distribution[1:6].sum()),
        "prob_risk": float(distribution[6:high_risk_threshold].sum()),
        "prob_high_risk": float(distribution[high_risk_threshold:].sum()),
    }
    ordered_values = np.array([values[f"prob_{level}"] for level in SEVERITY_LEVELS])
    # np.flatnonzero의 마지막 값을 사용해 동률이면 더 위험한 등급을 택한다.
    predicted_index = int(np.flatnonzero(np.isclose(ordered_values, ordered_values.max()))[-1])
    return {**values, "predicted_severity": SEVERITY_LEVELS[predicted_index]}


def _actual_severity(score: int, high_risk_threshold: int) -> str:
    if score == 0:
        return "normal"
    if score <= 5:
        return "caution"
    if score < high_risk_threshold:
        return "risk"
    return "high_risk"


def aggregate_asset_risk(
    ranked_parts: pd.DataFrame,
    *,
    high_risk_thresholds: tuple[int, ...] = (12, 13),
) -> pd.DataFrame:
    """부품확률로 장비일별 0~47점 분포와 위험 상태확률을 계산한다."""

    required = {
        *RANK_GROUP_COLUMNS,
        PART_COLUMN,
        "criticality_weight",
        "failure_probability",
        "probability_rank",
        "contribution_rank",
        "expected_score_contribution",
        CURRENT_TARGET,
    }
    _required_columns(ranked_parts, required, "장비 위험 집계")
    thresholds = tuple(dict.fromkeys(high_risk_thresholds))
    if not thresholds or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 7
        for value in thresholds
    ):
        raise ValueError("고위험 기준은 7 이상의 정수여야 합니다.")

    rows: list[dict[str, Any]] = []
    for keys, group in ranked_parts.groupby(
        RANK_GROUP_COLUMNS, sort=True, dropna=False
    ):
        if len(group) != 20 or group[PART_COLUMN].nunique() != 20:
            raise ValueError("각 장비·날짜·모델에는 정확히 20개 부품이 필요합니다.")
        distribution = weighted_score_distribution(
            group["failure_probability"],
            group["criticality_weight"],
            max_score=47,
        )
        expected_score = float(group["expected_score_contribution"].sum())
        actual_score = int(
            (group[CURRENT_TARGET].astype(int) * group["criticality_weight"]).sum()
        )
        base: dict[str, Any] = {
            DATE_COLUMN: keys[0],
            ASSET_COLUMN: keys[1],
            MODEL_VARIANT_COLUMN: keys[2],
            "actual_failure_points": actual_score,
            "expected_failure_points": expected_score,
            "score_error": actual_score - expected_score,
            "prob_ge_12": float(distribution[12:].sum()),
            "prob_ge_13": float(distribution[13:].sum()),
            "top3_probability_parts": ";".join(
                group.nsmallest(3, "probability_rank")[PART_COLUMN].astype(str)
            ),
            "top3_contribution_parts": ";".join(
                group.nsmallest(3, "contribution_rank")[PART_COLUMN].astype(str)
            ),
            **{
                f"prob_score_{score}": float(probability)
                for score, probability in enumerate(distribution)
            },
        }
        for metadata in ("machine_type", "plant_code", "split"):
            if metadata in group:
                if group[metadata].nunique(dropna=False) != 1:
                    raise ValueError(f"같은 장비일의 {metadata} 값이 다릅니다.")
                base[metadata] = group[metadata].iloc[0]

        for threshold in thresholds:
            severity = severity_probabilities(distribution, threshold)
            rows.append(
                {
                    **base,
                    "high_risk_threshold": threshold,
                    "actual_severity": _actual_severity(actual_score, threshold),
                    **severity,
                }
            )
    result = pd.DataFrame(rows)
    if (result["prob_ge_13"] > result["prob_ge_12"] + 1e-12).any():
        raise ValueError("13점 이상 확률은 12점 이상 확률보다 클 수 없습니다.")
    return result.sort_values(
        [DATE_COLUMN, ASSET_COLUMN, MODEL_VARIANT_COLUMN, "high_risk_threshold"]
    ).reset_index(drop=True)
