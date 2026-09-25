"""장비 4단계 위험도 비교 실험을 실행한다."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight

from .asset_anomaly_features import FEATURE_SET_NAMES
from .industrial_training import build_model

EXPERIMENT_MODELS = (
    "dummy_classifier",
    "logistic_regression",
    "random_forest",
    "hist_gradient_boosting",
    "hist_gradient_boosting_balanced",
)
LEARNED_EXPERIMENT_MODELS = EXPERIMENT_MODELS[1:]


def fit_experiment_model(
    model_name: str,
    frame: pd.DataFrame,
    target: pd.Series,
    *,
    max_iter: int,
    random_state: int,
) -> Pipeline:
    """실험 후보 모델을 학습하고 적합된 Pipeline을 반환한다."""
    if model_name == "dummy_classifier":
        model = Pipeline(
            [
                (
                    "classifier",
                    DummyClassifier(
                        strategy="prior",
                        random_state=random_state,
                    ),
                )
            ]
        )
        return model.fit(frame, target.astype(str))
    if model_name not in LEARNED_EXPERIMENT_MODELS:
        raise ValueError(f"지원하지 않는 실험 모델입니다: {model_name}")

    build_name = (
        "hist_gradient_boosting"
        if model_name == "hist_gradient_boosting_balanced"
        else model_name
    )
    model = build_model(
        build_name,
        frame,
        max_iter=max_iter,
        random_state=random_state,
    )
    fit_params: dict[str, Any] = {}
    if model_name == "hist_gradient_boosting_balanced":
        fit_params["classifier__sample_weight"] = compute_sample_weight(
            class_weight="balanced",
            y=target.astype(str),
        )
    return model.fit(frame, target.astype(str), **fit_params)


def _select_name(
    metrics: pd.DataFrame,
    name_column: str,
    candidates: Sequence[str],
    tolerance: float,
) -> str:
    if tolerance < 0:
        raise ValueError("tolerance는 0 이상이어야 합니다.")
    required = {
        name_column,
        "macro_f1",
        "high_risk_recall",
        "high_risk_precision",
    }
    missing = sorted(required - set(metrics.columns))
    if missing:
        raise ValueError(f"모델 선택 지표 컬럼이 없습니다: {missing}")

    eligible = metrics.loc[metrics[name_column].isin(candidates)].copy()
    eligible = eligible.dropna(
        subset=["macro_f1", "high_risk_recall", "high_risk_precision"]
    )
    if eligible.empty:
        raise ValueError("선택할 검증 지표가 없습니다.")
    best = float(eligible["macro_f1"].max())
    eligible = eligible.loc[
        eligible["macro_f1"].ge(best - tolerance - 1e-12)
    ]
    order = {name: index for index, name in enumerate(candidates)}
    eligible["candidate_order"] = eligible[name_column].map(order)
    ranked = eligible.sort_values(
        [
            "high_risk_recall",
            "high_risk_precision",
            "macro_f1",
            "candidate_order",
        ],
        ascending=[False, False, False, True],
    )
    return str(ranked.iloc[0][name_column])


def select_candidate(metrics: pd.DataFrame, tolerance: float = 0.01) -> str:
    """검증 지표로 학습 모델을 선택한다."""
    return _select_name(
        metrics,
        "model",
        LEARNED_EXPERIMENT_MODELS,
        tolerance,
    )


def select_feature_set(metrics: pd.DataFrame, tolerance: float = 0.01) -> str:
    """검증 지표로 A~D 대표 Feature 집합을 선택한다."""
    return _select_name(
        metrics,
        "feature_set",
        FEATURE_SET_NAMES,
        tolerance,
    )
