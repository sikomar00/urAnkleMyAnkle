"""장비 고장점수 회귀와 4단계 위험도 평가 도구를 제공한다."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.pipeline import Pipeline

from .asset_features import SEVERITY_LEVELS
from .industrial_training import build_preprocessor

REGRESSION_MODELS = (
    "dummy_regressor",
    "random_forest_regressor",
    "hist_gradient_boosting_regressor",
)
LEARNED_REGRESSION_MODELS = REGRESSION_MODELS[1:]


def build_regression_model(
    model_name: str,
    frame: pd.DataFrame,
    max_iter: int = 100,
    random_state: int = 42,
) -> Pipeline:
    """학습 데이터에서만 적합되는 회귀 파이프라인을 만든다."""
    preprocess = build_preprocessor(frame)
    if model_name == "dummy_regressor":
        regressor = DummyRegressor(strategy="mean")
    elif model_name == "random_forest_regressor":
        regressor = RandomForestRegressor(
            n_estimators=max(50, max_iter),
            random_state=random_state,
            n_jobs=-1,
        )
    elif model_name == "hist_gradient_boosting_regressor":
        regressor = HistGradientBoostingRegressor(
            max_iter=max_iter,
            early_stopping=False,
            random_state=random_state,
        )
    else:
        raise ValueError(f"지원하지 않는 회귀 모델입니다: {model_name}")
    return Pipeline([("preprocess", preprocess), ("regressor", regressor)])


def observed_severity(scores: Any) -> np.ndarray:
    """관측된 0 이상의 정수 점수를 4단계로 변환한다."""
    values = np.asarray(scores, dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("관측 점수는 결측이 없는 0 이상의 숫자여야 합니다.")
    return np.select(
        [values == 0, values <= 5, values <= 11],
        ["normal", "caution", "risk"],
        default="high_risk",
    )


def predicted_severity(scores: Any) -> np.ndarray:
    """연속 회귀 예측점수를 반점 경계로 4단계로 변환한다."""
    values = np.maximum(np.asarray(scores, dtype=float), 0.0)
    return np.select(
        [values < 0.5, values < 5.5, values < 11.5],
        ["normal", "caution", "risk"],
        default="high_risk",
    )


def severity_classification_metrics(y_true: Any, y_pred: Any) -> dict[str, Any]:
    """고정된 등급 순서로 다중분류 성능을 계산한다."""
    actual = np.asarray(y_true, dtype=str)
    predicted = np.asarray(y_pred, dtype=str)
    labels = list(SEVERITY_LEVELS)
    precision, recall, f1, support = precision_recall_fscore_support(
        actual,
        predicted,
        labels=labels,
        zero_division=0,
    )
    result: dict[str, Any] = {
        "accuracy": float(accuracy_score(actual, predicted)),
        "macro_precision": float(
            precision_score(
                actual,
                predicted,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_recall": float(
            recall_score(
                actual,
                predicted,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_f1": float(
            f1_score(
                actual,
                predicted,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                actual,
                predicted,
                labels=labels,
                average="weighted",
                zero_division=0,
            )
        ),
        "confusion_matrix": confusion_matrix(
            actual,
            predicted,
            labels=labels,
        ).tolist(),
    }
    for index, level in enumerate(SEVERITY_LEVELS):
        result[f"{level}_precision"] = float(precision[index])
        result[f"{level}_recall"] = float(recall[index])
        result[f"{level}_f1"] = float(f1[index])
        result[f"{level}_support"] = int(support[index])

    actual_high = actual == "high_risk"
    predicted_high = predicted == "high_risk"
    result["high_risk_precision"] = float(
        precision_score(actual_high, predicted_high, zero_division=0)
    )
    result["high_risk_recall"] = float(
        recall_score(actual_high, predicted_high, zero_division=0)
    )
    result["high_risk_binary_f1"] = float(
        f1_score(actual_high, predicted_high, zero_division=0)
    )
    return result


def score_regression_metrics(y_true: Any, y_pred: Any) -> dict[str, Any]:
    """회귀 오차와 점수를 4단계로 변환한 성능을 함께 계산한다."""
    actual = np.asarray(y_true, dtype=float)
    predicted = np.maximum(np.asarray(y_pred, dtype=float), 0.0)
    severity = severity_classification_metrics(
        observed_severity(actual),
        predicted_severity(predicted),
    )
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(mean_squared_error(actual, predicted) ** 0.5),
        "r2": float(r2_score(actual, predicted)) if len(actual) >= 2 else None,
        "spearman": (
            float(pd.Series(actual).corr(pd.Series(predicted), method="spearman"))
            if len(actual) >= 2
            else None
        ),
        **severity,
    }
