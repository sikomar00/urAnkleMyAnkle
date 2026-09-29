"""부품 확률모델의 후보 선택, 재적합과 독립 확률 보정을 수행한다."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import average_precision_score, brier_score_loss

from .industrial_data import CURRENT_TARGET
from .industrial_training import MODEL_NAMES, build_model
from .probabilistic_risk_aggregation import (
    aggregate_asset_risk,
    rank_part_probabilities,
)
from .probabilistic_risk_evaluation import build_ranking_metrics
from .probabilistic_risk_features import ProbabilisticRiskFeatures

MODEL_SELECTION_ORDER = (
    "logistic_regression",
    "hist_gradient_boosting",
    "random_forest",
)


@dataclass(frozen=True)
class PartModelSelection:
    """Feature 구성별 선택 모델과 모든 후보의 선택구간 지표다."""

    feature_set: str
    selected_model: str
    selection_metrics: pd.DataFrame


@dataclass(frozen=True)
class CalibratedPartModel:
    """학습+선택 구간 기반 모델과 보정 구간 sigmoid calibrator다."""

    feature_set: str
    model_name: str
    estimator: Any
    calibrator: CalibratedClassifierCV
    features: tuple[str, ...]


def choose_model_from_selection_metrics(
    metrics: pd.DataFrame,
    *,
    model_order: tuple[str, ...] = MODEL_SELECTION_ORDER,
) -> str:
    """점수 MAE→고위험 Brier→부품 AP→고정순서로 후보를 선택한다."""

    required = {
        "model_name",
        "score_mae",
        "mean_high_risk_brier",
        "part_average_precision",
    }
    missing = sorted(required - set(metrics.columns))
    if missing or metrics.empty:
        raise ValueError(f"모델 선택 지표가 없거나 필요한 컬럼이 없습니다: {missing}")
    work = metrics.copy()
    numeric = ["score_mae", "mean_high_risk_brier", "part_average_precision"]
    for column in numeric:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    if work[numeric].isna().any().any():
        raise ValueError("모델 선택 지표에는 결측값을 사용할 수 없습니다.")
    unknown = sorted(set(work["model_name"]) - set(model_order))
    if unknown:
        raise ValueError(f"고정 선택 순서에 없는 모델입니다: {unknown}")
    order = {name: index for index, name in enumerate(model_order)}
    work["_model_order"] = work["model_name"].map(order)
    selected = work.sort_values(
        [
            "score_mae",
            "mean_high_risk_brier",
            "part_average_precision",
            "_model_order",
        ],
        ascending=[True, True, False, True],
        kind="stable",
    ).iloc[0]
    return str(selected["model_name"])


def _require_two_classes(frame: pd.DataFrame, period_label: str) -> None:
    if frame.empty:
        raise ValueError(f"{period_label}이 비어 있습니다.")
    if frame[CURRENT_TARGET].nunique(dropna=False) != 2:
        raise ValueError(f"{period_label} 정답이 단일 클래스라 모델을 적합할 수 없습니다.")


def _prediction_input(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    *,
    model_variant: str,
    split: str,
) -> pd.DataFrame:
    result = frame.copy()
    result["failure_probability"] = np.asarray(probabilities, dtype=float)
    result["model_variant"] = model_variant
    result["split"] = split
    return rank_part_probabilities(result)


def _candidate_metrics(ranked: pd.DataFrame, model_name: str) -> dict[str, Any]:
    asset = aggregate_asset_risk(ranked, high_risk_thresholds=(12, 13))
    score_rows = asset.loc[asset["high_risk_threshold"].eq(12)]
    score_mae = float(
        np.abs(
            score_rows["actual_failure_points"]
            - score_rows["expected_failure_points"]
        ).mean()
    )
    briers = []
    for threshold in (12, 13):
        rows = asset.loc[asset["high_risk_threshold"].eq(threshold)]
        target = rows["actual_failure_points"].ge(threshold).astype(int)
        briers.append(
            brier_score_loss(target, rows[f"prob_ge_{threshold}"])
        )
    part_ap = float(
        average_precision_score(
            ranked[CURRENT_TARGET].astype(int), ranked["failure_probability"]
        )
    )
    ranking = build_ranking_metrics(ranked)
    overall = ranking.loc[ranking["scope_kind"].eq("overall")].iloc[0]
    return {
        "model_name": model_name,
        "split": "selection",
        "score_mae": score_mae,
        "mean_high_risk_brier": float(np.mean(briers)),
        "brier_ge_12": float(briers[0]),
        "brier_ge_13": float(briers[1]),
        "part_average_precision": part_ap,
        "hit_rate_at_3": overall.get("hit_rate_at_3", np.nan),
        "recall_at_3": overall.get("recall_at_3", np.nan),
    }


def select_part_model(
    prepared: ProbabilisticRiskFeatures,
    feature_set: str,
    *,
    model_names: Sequence[str] = MODEL_NAMES,
    max_iter: int = 100,
    random_state: int = 42,
) -> PartModelSelection:
    """학습 구간에 후보를 적합하고 선택 구간 장비 위험지표로 하나를 고른다."""

    if feature_set not in prepared.feature_sets:
        raise ValueError(f"지원하지 않는 Feature 구성입니다: {feature_set}")
    model_names = tuple(dict.fromkeys(model_names))
    if not model_names:
        raise ValueError("후보 모델이 하나 이상 필요합니다.")
    train = prepared.periods["train"]
    selection = prepared.periods["selection"]
    _require_two_classes(train, "학습 구간")
    _require_two_classes(selection, "선택 구간")
    features = list(prepared.feature_sets[feature_set])

    rows = []
    for model_name in model_names:
        estimator = build_model(
            model_name,
            train[features],
            max_iter=max_iter,
            random_state=random_state,
        )
        estimator.fit(train[features], train[CURRENT_TARGET].astype(int))
        probabilities = estimator.predict_proba(selection[features])[:, 1]
        ranked = _prediction_input(
            selection,
            probabilities,
            model_variant=model_name,
            split="selection",
        )
        rows.append(_candidate_metrics(ranked, model_name))
    metrics = pd.DataFrame(rows)
    selected = choose_model_from_selection_metrics(
        metrics,
        model_order=tuple(
            name for name in MODEL_SELECTION_ORDER if name in model_names
        ),
    )
    metrics["selected_model"] = metrics["model_name"].eq(selected)
    metrics.insert(0, "feature_set", feature_set)
    return PartModelSelection(feature_set, selected, metrics)


def fit_calibrated_part_model(
    prepared: ProbabilisticRiskFeatures,
    selection: PartModelSelection,
    *,
    method: str = "sigmoid",
    max_iter: int = 100,
    random_state: int = 42,
) -> CalibratedPartModel:
    """선택 모델을 학습+선택에 재적합하고 보정 구간에서 sigmoid 보정한다."""

    if method != "sigmoid":
        raise ValueError("현재 확률 보정 방식은 sigmoid만 지원합니다.")
    if selection.feature_set not in prepared.feature_sets:
        raise ValueError("선택 결과의 Feature 구성이 준비 데이터에 없습니다.")
    calibration = prepared.periods["calibration"]
    _require_two_classes(calibration, "보정 구간")
    refit = pd.concat(
        [prepared.periods["train"], prepared.periods["selection"]],
        ignore_index=True,
    )
    _require_two_classes(refit, "학습+선택 구간")
    features = prepared.feature_sets[selection.feature_set]
    estimator = build_model(
        selection.selected_model,
        refit[list(features)],
        max_iter=max_iter,
        random_state=random_state,
    )
    estimator.fit(refit[list(features)], refit[CURRENT_TARGET].astype(int))
    calibrator = CalibratedClassifierCV(
        FrozenEstimator(estimator), method=method
    )
    calibrator.fit(
        calibration[list(features)], calibration[CURRENT_TARGET].astype(int)
    )
    return CalibratedPartModel(
        feature_set=selection.feature_set,
        model_name=selection.selected_model,
        estimator=estimator,
        calibrator=calibrator,
        features=tuple(features),
    )


def predict_part_probabilities(
    model: CalibratedPartModel,
    frame: pd.DataFrame,
    *,
    split: str,
) -> pd.DataFrame:
    """저장 가능한 보정 부품확률과 결정적 순위를 반환한다."""

    if frame.empty:
        raise ValueError("부품확률을 예측할 입력이 비어 있습니다.")
    missing = [feature for feature in model.features if feature not in frame]
    if missing:
        raise ValueError(f"부품확률 예측 Feature가 없습니다: {missing}")
    probabilities = model.calibrator.predict_proba(frame[list(model.features)])[:, 1]
    if not np.isfinite(probabilities).all() or not np.logical_and(
        probabilities >= 0, probabilities <= 1
    ).all():
        raise ValueError("보정 모델이 유효하지 않은 확률을 반환했습니다.")
    return _prediction_input(
        frame,
        probabilities,
        model_variant=model.feature_set,
        split=split,
    )
