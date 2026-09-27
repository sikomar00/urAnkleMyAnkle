"""장비 고장점수 회귀와 4단계 위험도 평가 도구를 제공한다."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
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
from .industrial_data import (
    PreparedRegressionTask,
    PreparedTask,
    iter_scopes,
    split_by_date,
)
from .industrial_training import MODEL_NAMES, build_model, build_preprocessor

REGRESSION_MODELS = (
    "dummy_regressor",
    "random_forest_regressor",
    "hist_gradient_boosting_regressor",
)
LEARNED_REGRESSION_MODELS = REGRESSION_MODELS[1:]

REGRESSION_METRIC_COLUMNS = (
    "task_type",
    "scope_kind",
    "scope_name",
    "model",
    "selected_model",
    "split",
    "status",
    "reason",
    "train_rows",
    "valid_rows",
    "test_rows",
    "mae",
    "rmse",
    "r2",
    "spearman",
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_f1",
    "confusion_matrix_json",
    "normal_precision",
    "normal_recall",
    "normal_f1",
    "normal_support",
    "caution_precision",
    "caution_recall",
    "caution_f1",
    "caution_support",
    "risk_precision",
    "risk_recall",
    "risk_f1",
    "risk_support",
    "high_risk_precision",
    "high_risk_recall",
    "high_risk_f1",
    "high_risk_support",
    "high_risk_binary_f1",
)

SEVERITY_METRIC_COLUMNS = tuple(
    column
    for column in REGRESSION_METRIC_COLUMNS
    if column not in {"mae", "rmse", "r2", "spearman"}
)

PREDICTION_COLUMNS = (
    "transaction_date",
    "label_end_date",
    "machine_type",
    "asset_tag",
    "scope_kind",
    "scope_name",
    "actual_failure_points",
    "predicted_failure_points",
    "actual_severity_level",
    "regression_severity_level",
    "predicted_severity_level",
    "prob_normal",
    "prob_caution",
    "prob_risk",
    "prob_high_risk",
    "regression_model",
    "severity_model",
)

IMPORTANCE_COLUMNS = (
    "task_type",
    "scope_kind",
    "scope_name",
    "model",
    "feature",
    "importance_mean",
    "importance_std",
)


@dataclass(frozen=True)
class NonnegativeRegressor:
    """저장 모델의 회귀 예측을 점수 도메인인 0 이상으로 제한한다."""

    estimator: Any

    def predict(self, frame: Any) -> np.ndarray:
        """음수 원시 예측을 0으로 바꿔 반환한다."""
        return np.maximum(np.asarray(self.estimator.predict(frame), dtype=float), 0.0)


@dataclass(frozen=True)
class OrderedSeverityClassifier:
    """저장 모델의 확률 컬럼을 고정된 4단계 순서로 제공한다."""

    estimator: Pipeline
    class_order: tuple[str, ...] = SEVERITY_LEVELS

    @property
    def classes_(self) -> np.ndarray:
        """`predict_proba()` 컬럼과 같은 순서의 등급을 반환한다."""
        return np.asarray(self.class_order)

    def predict(self, frame: Any) -> np.ndarray:
        """원본 분류기의 예측 등급을 반환한다."""
        return np.asarray(self.estimator.predict(frame), dtype=str)

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        """확률을 `normal`, `caution`, `risk`, `high_risk` 순서로 반환한다."""
        return ordered_probabilities(self.estimator, frame)


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
    spearman: float | None = None
    if (
        len(actual) >= 2
        and np.unique(actual).size > 1
        and np.unique(predicted).size > 1
    ):
        correlation = pd.Series(actual).corr(
            pd.Series(predicted),
            method="spearman",
        )
        if pd.notna(correlation):
            spearman = float(correlation)
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(mean_squared_error(actual, predicted) ** 0.5),
        "r2": float(r2_score(actual, predicted)) if len(actual) >= 2 else None,
        "spearman": spearman,
        **severity,
    }


def ordered_probabilities(model: Pipeline, frame: pd.DataFrame) -> np.ndarray:
    """모델의 클래스 순서와 무관하게 고정된 4단계 확률을 반환한다."""
    probabilities = model.predict_proba(frame)
    classes = list(model.named_steps["classifier"].classes_)
    ordered = np.zeros((len(frame), len(SEVERITY_LEVELS)), dtype=float)
    for target_index, level in enumerate(SEVERITY_LEVELS):
        if level in classes:
            ordered[:, target_index] = probabilities[:, classes.index(level)]
    return ordered


def _metric_row(metrics: dict[str, Any]) -> dict[str, Any]:
    result = dict(metrics)
    matrix = result.pop("confusion_matrix", None)
    result["confusion_matrix_json"] = (
        json.dumps(matrix, ensure_ascii=False) if matrix is not None else None
    )
    return result


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "all"


def _split_issue(parts: dict[str, pd.DataFrame]) -> str | None:
    empty = [name for name, frame in parts.items() if frame.empty]
    if empty:
        return "; ".join(f"empty {name} split" for name in empty)
    return None


def _base_metric_row(
    *,
    task_type: str,
    scope_kind: str,
    scope_name: str,
    model: str,
    selected_model: bool,
    split: str,
    status: str,
    reason: str,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "task_type": task_type,
        "scope_kind": scope_kind,
        "scope_name": scope_name,
        "model": model,
        "selected_model": selected_model,
        "split": split,
        "status": status,
        "reason": reason,
        "train_rows": len(train),
        "valid_rows": len(valid),
        "test_rows": len(test),
    }


def _importance_rows(
    *,
    task_type: str,
    scope_kind: str,
    scope_name: str,
    model_name: str,
    model: Pipeline,
    valid: pd.DataFrame,
    features: list[str],
    target: str,
    scoring: str,
    random_state: int,
) -> list[dict[str, Any]]:
    sample = valid.sample(min(1000, len(valid)), random_state=random_state)
    importance = permutation_importance(
        model,
        sample[features],
        sample[target],
        scoring=scoring,
        n_repeats=3,
        random_state=random_state,
    )
    return [
        {
            "task_type": task_type,
            "scope_kind": scope_kind,
            "scope_name": scope_name,
            "model": model_name,
            "feature": feature,
            "importance_mean": float(mean),
            "importance_std": float(std),
        }
        for feature, mean, std in zip(
            features,
            importance.importances_mean,
            importance.importances_std,
        )
    ]


def run_asset_score_suite(
    score_task: PreparedRegressionTask,
    severity_task: PreparedTask,
    *,
    output_dir: str | Path,
    scope: str = "all",
    machine_type: str | None = None,
    asset_tag: str | None = None,
    max_iter: int = 100,
    validation_start: str = "2024-01-01",
    test_start: str = "2024-07-01",
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """점수 회귀와 4단계 분류를 Scope별로 학습·선택·저장한다."""
    if tuple(score_task.features) != tuple(severity_task.features):
        raise ValueError("회귀와 분류 과제의 Feature가 같아야 합니다.")
    required_targets = {
        score_task.target,
        severity_task.target,
        "failure_points",
        "severity_level",
    }
    missing = sorted(required_targets - set(score_task.frame.columns))
    if missing:
        raise ValueError(f"점수 학습 데이터에 컬럼이 없습니다: {missing}")

    output = Path(output_dir)
    model_dir = output / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    print(
        "[Info] DummyRegressor Spearman: N/A "
        "(상수 예측을 사용하는 기준선)",
        flush=True,
    )
    features = list(score_task.features)
    regression_rows: list[dict[str, Any]] = []
    severity_rows: list[dict[str, Any]] = []
    prediction_rows: list[pd.DataFrame] = []
    importance_rows: list[dict[str, Any]] = []

    if score_task.frame.empty:
        empty = score_task.frame
        base = _base_metric_row(
            task_type="regression",
            scope_kind="overall",
            scope_name="all",
            model="none",
            selected_model=False,
            split="test",
            status="skipped",
            reason="empty prepared task",
            train=empty,
            valid=empty,
            test=empty,
        )
        regression_rows.append(base)
        severity_rows.append({**base, "task_type": "severity"})
    else:
        for scope_kind, scope_name, subset in iter_scopes(
            score_task.frame,
            scope,
            machine_type,
            asset_tag,
        ):
            parts = split_by_date(subset, validation_start, test_start)
            train, valid, test = parts["train"], parts["valid"], parts["test"]
            issue = _split_issue(parts)
            if issue:
                for task_type, rows in (
                    ("regression", regression_rows),
                    ("severity", severity_rows),
                ):
                    rows.append(
                        _base_metric_row(
                            task_type=task_type,
                            scope_kind=scope_kind,
                            scope_name=scope_name,
                            model="none",
                            selected_model=False,
                            split="test",
                            status="skipped",
                            reason=issue,
                            train=train,
                            valid=valid,
                            test=test,
                        )
                    )
                continue

            print(
                f"[Asset score] {scope_kind}: {scope_name} "
                f"(train={len(train):,}, valid={len(valid):,}, test={len(test):,})",
                flush=True,
            )

            regression_models: dict[str, Pipeline] = {}
            regression_predictions: dict[str, dict[str, np.ndarray]] = {}
            regression_scores: dict[str, float] = {}
            for model_name in REGRESSION_MODELS:
                model = build_regression_model(
                    model_name,
                    train[features],
                    max_iter=max_iter,
                    random_state=random_state,
                )
                model.fit(train[features], train[score_task.target])
                valid_prediction = np.maximum(model.predict(valid[features]), 0.0)
                test_prediction = np.maximum(model.predict(test[features]), 0.0)
                regression_models[model_name] = model
                regression_predictions[model_name] = {
                    "validation": valid_prediction,
                    "test": test_prediction,
                }
                regression_scores[model_name] = mean_absolute_error(
                    valid[score_task.target],
                    valid_prediction,
                )

            selected_regression = min(
                LEARNED_REGRESSION_MODELS,
                key=lambda name: (
                    regression_scores[name],
                    LEARNED_REGRESSION_MODELS.index(name),
                ),
            )
            for model_name in REGRESSION_MODELS:
                for split_name, frame in (("validation", valid), ("test", test)):
                    metrics = score_regression_metrics(
                        frame[score_task.target],
                        regression_predictions[model_name][split_name],
                    )
                    regression_rows.append(
                        {
                            **_base_metric_row(
                                task_type="regression",
                                scope_kind=scope_kind,
                                scope_name=scope_name,
                                model=model_name,
                                selected_model=model_name == selected_regression,
                                split=split_name,
                                status="ok",
                                reason="",
                                train=train,
                                valid=valid,
                                test=test,
                            ),
                            **_metric_row(metrics),
                        }
                    )

            selected_regression_model = regression_models[selected_regression]
            selected_regression_predictor = NonnegativeRegressor(
                selected_regression_model
            )
            verification_rows = subset.loc[:, features].head(3)
            regression_path = model_dir / (
                f"regression__{_safe_name(scope_kind)}__{_safe_name(scope_name)}__"
                f"{selected_regression}__selected.joblib"
            )
            joblib.dump(
                {
                    "pipeline": selected_regression_predictor,
                    "raw_pipeline": selected_regression_model,
                    "feature_data": features,
                    "target_data": score_task.target,
                    "selected_model": selected_regression,
                    "validation_start": str(validation_start),
                    "test_start": str(test_start),
                    "verification_predictions": selected_regression_predictor.predict(
                        verification_rows
                    ),
                },
                regression_path,
            )
            importance_rows.extend(
                _importance_rows(
                    task_type="regression",
                    scope_kind=scope_kind,
                    scope_name=scope_name,
                    model_name=selected_regression,
                    model=selected_regression_model,
                    valid=valid,
                    features=features,
                    target=score_task.target,
                    scoring="neg_mean_absolute_error",
                    random_state=random_state,
                )
            )

            regression_prediction = selected_regression_predictor.predict(
                test[features]
            )
            prediction = test[
                [
                    "transaction_date",
                    "label_end_date",
                    "machine_type",
                    "asset_tag",
                    "failure_points",
                    "severity_level",
                ]
            ].rename(
                columns={
                    "failure_points": "actual_failure_points",
                    "severity_level": "actual_severity_level",
                }
            )
            prediction["scope_kind"] = scope_kind
            prediction["scope_name"] = scope_name
            prediction["predicted_failure_points"] = regression_prediction
            prediction["regression_severity_level"] = predicted_severity(
                regression_prediction
            )
            prediction["regression_model"] = selected_regression

            if train[severity_task.target].nunique() < 2:
                severity_rows.append(
                    _base_metric_row(
                        task_type="severity",
                        scope_kind=scope_kind,
                        scope_name=scope_name,
                        model="none",
                        selected_model=False,
                        split="test",
                        status="skipped",
                        reason="single train class",
                        train=train,
                        valid=valid,
                        test=test,
                    )
                )
                prediction["predicted_severity_level"] = pd.NA
                for level in SEVERITY_LEVELS:
                    prediction[f"prob_{level}"] = pd.NA
                prediction["severity_model"] = "skipped"
                prediction_rows.append(prediction)
                continue

            severity_models: dict[str, Pipeline] = {}
            severity_scores: dict[str, float] = {}
            dummy = Pipeline([("classifier", DummyClassifier(strategy="prior"))])
            dummy.fit(train[features], train[severity_task.target].astype(str))
            severity_models["dummy_classifier"] = dummy
            for model_name in MODEL_NAMES:
                model = build_model(
                    model_name,
                    train[features],
                    max_iter=max_iter,
                    random_state=random_state,
                )
                model.fit(train[features], train[severity_task.target].astype(str))
                severity_models[model_name] = model
                severity_scores[model_name] = severity_classification_metrics(
                    valid[severity_task.target],
                    model.predict(valid[features]),
                )["macro_f1"]

            selected_severity = max(
                MODEL_NAMES,
                key=lambda name: (
                    severity_scores[name],
                    -MODEL_NAMES.index(name),
                ),
            )
            for model_name, model in severity_models.items():
                for split_name, frame in (("validation", valid), ("test", test)):
                    metrics = severity_classification_metrics(
                        frame[severity_task.target],
                        model.predict(frame[features]),
                    )
                    severity_rows.append(
                        {
                            **_base_metric_row(
                                task_type="severity",
                                scope_kind=scope_kind,
                                scope_name=scope_name,
                                model=model_name,
                                selected_model=model_name == selected_severity,
                                split=split_name,
                                status="ok",
                                reason="",
                                train=train,
                                valid=valid,
                                test=test,
                            ),
                            **_metric_row(metrics),
                        }
                    )

            selected_severity_model = severity_models[selected_severity]
            selected_severity_predictor = OrderedSeverityClassifier(
                selected_severity_model
            )
            severity_path = model_dir / (
                f"severity__{_safe_name(scope_kind)}__{_safe_name(scope_name)}__"
                f"{selected_severity}__selected.joblib"
            )
            joblib.dump(
                {
                    "pipeline": selected_severity_predictor,
                    "raw_pipeline": selected_severity_model,
                    "feature_data": features,
                    "target_data": severity_task.target,
                    "selected_model": selected_severity,
                    "class_order": list(SEVERITY_LEVELS),
                    "validation_start": str(validation_start),
                    "test_start": str(test_start),
                    "verification_probabilities": (
                        selected_severity_predictor.predict_proba(verification_rows)
                    ),
                },
                severity_path,
            )
            importance_rows.extend(
                _importance_rows(
                    task_type="severity",
                    scope_kind=scope_kind,
                    scope_name=scope_name,
                    model_name=selected_severity,
                    model=selected_severity_model,
                    valid=valid,
                    features=features,
                    target=severity_task.target,
                    scoring="f1_macro",
                    random_state=random_state,
                )
            )

            severity_prediction = selected_severity_predictor.predict(test[features])
            severity_probability = selected_severity_predictor.predict_proba(
                test[features]
            )
            prediction["predicted_severity_level"] = severity_prediction
            for index, level in enumerate(SEVERITY_LEVELS):
                prediction[f"prob_{level}"] = severity_probability[:, index]
            prediction["severity_model"] = selected_severity
            prediction_rows.append(prediction)

    regression_frame = pd.DataFrame(regression_rows).reindex(
        columns=REGRESSION_METRIC_COLUMNS
    )
    severity_frame = pd.DataFrame(severity_rows).reindex(
        columns=SEVERITY_METRIC_COLUMNS
    )
    predictions = (
        pd.concat(prediction_rows, ignore_index=True, sort=False)
        if prediction_rows
        else pd.DataFrame()
    ).reindex(columns=PREDICTION_COLUMNS)
    importance_frame = pd.DataFrame(importance_rows).reindex(
        columns=IMPORTANCE_COLUMNS
    )
    regression_frame.to_csv(output / "regression_metrics.csv", index=False)
    severity_frame.to_csv(output / "severity_metrics.csv", index=False)
    predictions.to_csv(output / "test_predictions.csv", index=False)
    importance_frame.to_csv(output / "feature_importance.csv", index=False)

    config = {
        "regression_models": list(REGRESSION_MODELS),
        "severity_models": ["dummy_classifier", *MODEL_NAMES],
        "severity_levels": list(SEVERITY_LEVELS),
        "observed_score_boundaries": [0, 5, 11, 12],
        "predicted_score_boundaries": [0.5, 5.5, 11.5],
        "scope": scope,
        "machine_type": machine_type,
        "asset_tag": asset_tag,
        "validation_start": str(validation_start),
        "test_start": str(test_start),
        "max_iter": max_iter,
        "random_state": random_state,
    }
    (output / "run_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return regression_frame, severity_frame
