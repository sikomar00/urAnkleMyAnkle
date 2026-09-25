"""장비 4단계 위험도 비교 실험을 실행한다."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight

from .asset_anomaly_features import (
    FEATURE_SET_NAMES,
    AssetExperimentFeatures,
    prepare_asset_experiment_features,
)
from .asset_features import SEVERITY_LEVELS, severity_labels
from .asset_severity_report import build_failure_profile, render_experiment_summary
from .industrial_data import (
    ASSET_COLUMN,
    DATE_COLUMN,
    MACHINE_COLUMN,
    split_by_date,
)
from .industrial_regression import (
    OrderedSeverityClassifier,
    severity_classification_metrics,
)
from .industrial_training import build_model

EXPERIMENT_MODELS = (
    "dummy_classifier",
    "logistic_regression",
    "random_forest",
    "hist_gradient_boosting",
    "hist_gradient_boosting_balanced",
)
LEARNED_EXPERIMENT_MODELS = EXPERIMENT_MODELS[1:]

METRIC_ID_COLUMNS = (
    "high_risk_threshold",
    "feature_set",
    "scope_kind",
    "scope_name",
    "model",
    "selected_model",
    "selected_feature_set",
    "split",
    "status",
    "reason",
    "train_rows",
    "valid_rows",
    "test_rows",
)


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


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "all"


def _validate_options(
    high_risk_thresholds: Sequence[int],
    feature_sets: Sequence[str],
    scope: str,
) -> tuple[tuple[int, ...], tuple[str, ...]]:
    thresholds = tuple(dict.fromkeys(high_risk_thresholds))
    selected_features = tuple(dict.fromkeys(feature_sets))
    if not thresholds:
        raise ValueError("high_risk_thresholds를 하나 이상 지정해야 합니다.")
    for threshold in thresholds:
        severity_labels([0], high_risk_threshold=threshold)
    invalid = [name for name in selected_features if name not in FEATURE_SET_NAMES]
    if not selected_features or invalid:
        raise ValueError("feature_sets는 A, B, C, D 중 하나 이상이어야 합니다.")
    if scope not in {"all", "overall", "machine_type"}:
        raise ValueError("scope는 all, overall, machine_type 중 하나여야 합니다.")
    return thresholds, selected_features


def _filter_raw(
    raw: pd.DataFrame,
    machine_type: str | None,
    asset_tag: str | None,
) -> pd.DataFrame:
    result = raw
    if machine_type is not None:
        result = result.loc[result[MACHINE_COLUMN].eq(machine_type)]
    if asset_tag is not None:
        result = result.loc[result[ASSET_COLUMN].eq(asset_tag)]
    if result.empty:
        raise ValueError("지정한 machine_type 또는 asset_tag 데이터가 없습니다.")
    return result.copy()


def _iter_experiment_scopes(
    frame: pd.DataFrame,
    scope: str,
):
    if scope in {"all", "overall"}:
        yield "overall", "all", frame.copy()
    if scope in {"all", "machine_type"}:
        for value in sorted(frame[MACHINE_COLUMN].dropna().unique()):
            yield (
                "machine_type",
                str(value),
                frame.loc[frame[MACHINE_COLUMN].eq(value)].copy(),
            )


def _split_issue(parts: dict[str, pd.DataFrame]) -> str | None:
    empty = [name for name, frame in parts.items() if frame.empty]
    if empty:
        return "; ".join(f"empty {name} split" for name in empty)
    return None


def _flatten_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in metrics.items()
        if key != "confusion_matrix"
    }


def _base_row(
    *,
    threshold: int,
    feature_set: str,
    scope_kind: str,
    scope_name: str,
    model: str,
    split: str,
    status: str,
    reason: str,
    parts: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    return {
        "high_risk_threshold": threshold,
        "feature_set": feature_set,
        "scope_kind": scope_kind,
        "scope_name": scope_name,
        "model": model,
        "selected_model": False,
        "selected_feature_set": False,
        "split": split,
        "status": status,
        "reason": reason,
        "train_rows": len(parts["train"]),
        "valid_rows": len(parts["valid"]),
        "test_rows": len(parts["test"]),
    }


def _metric_detail_rows(
    *,
    identifiers: dict[str, Any],
    metrics: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    class_rows = []
    for level in SEVERITY_LEVELS:
        class_rows.append(
            {
                **identifiers,
                "level": level,
                "precision": metrics[f"{level}_precision"],
                "recall": metrics[f"{level}_recall"],
                "f1": metrics[f"{level}_f1"],
                "support": metrics[f"{level}_support"],
            }
        )
    confusion_rows = []
    matrix = metrics["confusion_matrix"]
    for actual_index, actual_level in enumerate(SEVERITY_LEVELS):
        for predicted_index, predicted_level in enumerate(SEVERITY_LEVELS):
            confusion_rows.append(
                {
                    **identifiers,
                    "actual_level": actual_level,
                    "predicted_level": predicted_level,
                    "count": int(matrix[actual_index][predicted_index]),
                }
            )
    return class_rows, confusion_rows


def _importance_rows(
    model: Pipeline,
    valid: pd.DataFrame,
    features: list[str],
    target: str,
    identifiers: dict[str, Any],
    random_state: int,
) -> list[dict[str, Any]]:
    sample = valid.sample(min(1000, len(valid)), random_state=random_state)
    result = permutation_importance(
        model,
        sample[features],
        sample[target].astype(str),
        scoring="f1_macro",
        n_repeats=3,
        random_state=random_state,
    )
    return [
        {
            **identifiers,
            "feature": feature,
            "importance_mean": float(result.importances_mean[index]),
            "importance_std": float(result.importances_std[index]),
        }
        for index, feature in enumerate(features)
    ]


def _mark_selection(
    rows: list[dict[str, Any]],
    *,
    threshold: int,
    scope_kind: str,
    scope_name: str,
    feature_set: str,
    model: str | None = None,
    selected_feature: bool = False,
) -> None:
    for row in rows:
        if (
            row.get("high_risk_threshold") == threshold
            and row.get("scope_kind") == scope_kind
            and row.get("scope_name") == scope_name
            and row.get("feature_set") == feature_set
        ):
            if model is not None and row.get("model") == model:
                row["selected_model"] = True
            if selected_feature:
                row["selected_feature_set"] = True


def run_asset_severity_experiments(
    raw: pd.DataFrame,
    *,
    output_dir: str | Path,
    high_risk_thresholds: Sequence[int] = (12, 13),
    feature_sets: Sequence[str] = FEATURE_SET_NAMES,
    scope: str = "all",
    machine_type: str | None = None,
    asset_tag: str | None = None,
    validation_start: str = "2024-01-01",
    test_start: str = "2024-07-01",
    min_normal_rows: int = 30,
    max_iter: int = 100,
    random_state: int = 42,
) -> pd.DataFrame:
    """두 고위험 기준과 A~D Feature 조합을 학습·평가·저장한다."""
    thresholds, selected_features = _validate_options(
        high_risk_thresholds,
        feature_sets,
        scope,
    )
    filtered_raw = _filter_raw(raw, machine_type, asset_tag)
    prepared: AssetExperimentFeatures = prepare_asset_experiment_features(
        filtered_raw,
        validation_start=validation_start,
        min_normal_rows=min_normal_rows,
    )
    frame = prepared.frame.copy()
    output = Path(output_dir)
    model_dir = output / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    metric_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    prediction_rows: list[pd.DataFrame] = []
    importance_rows: list[dict[str, Any]] = []

    for threshold in thresholds:
        target = f"severity_level_ge_{threshold}"
        frame[target] = severity_labels(
            frame["failure_points"],
            high_risk_threshold=threshold,
        )
        for scope_kind, scope_name, subset in _iter_experiment_scopes(frame, scope):
            parts = split_by_date(subset, validation_start, test_start)
            issue = _split_issue(parts)
            selected_models: dict[str, str] = {}
            for feature_set in selected_features:
                features = list(prepared.feature_sets[feature_set])
                if issue or parts["train"][target].astype(str).nunique() < 2:
                    reason = issue or "single train class"
                    metric_rows.append(
                        _base_row(
                            threshold=threshold,
                            feature_set=feature_set,
                            scope_kind=scope_kind,
                            scope_name=scope_name,
                            model="none",
                            split="test",
                            status="skipped",
                            reason=reason,
                            parts=parts,
                        )
                    )
                    continue

                models: dict[str, Pipeline] = {}
                valid_metrics: list[dict[str, Any]] = []
                for model_name in EXPERIMENT_MODELS:
                    model = fit_experiment_model(
                        model_name,
                        parts["train"][features],
                        parts["train"][target],
                        max_iter=max_iter,
                        random_state=random_state,
                    )
                    models[model_name] = model
                    for split_name in ("validation", "test"):
                        split_key = "valid" if split_name == "validation" else "test"
                        split_frame = parts[split_key]
                        metrics = severity_classification_metrics(
                            split_frame[target],
                            model.predict(split_frame[features]),
                        )
                        identifiers = _base_row(
                            threshold=threshold,
                            feature_set=feature_set,
                            scope_kind=scope_kind,
                            scope_name=scope_name,
                            model=model_name,
                            split=split_name,
                            status="ok",
                            reason="",
                            parts=parts,
                        )
                        metric_rows.append(
                            {**identifiers, **_flatten_metrics(metrics)}
                        )
                        added_classes, added_confusion = _metric_detail_rows(
                            identifiers=identifiers,
                            metrics=metrics,
                        )
                        class_rows.extend(added_classes)
                        confusion_rows.extend(added_confusion)
                        if split_name == "validation":
                            valid_metrics.append(
                                {"model": model_name, **_flatten_metrics(metrics)}
                            )

                selected_model_name = select_candidate(pd.DataFrame(valid_metrics))
                selected_models[feature_set] = selected_model_name
                for rows in (metric_rows, class_rows, confusion_rows):
                    _mark_selection(
                        rows,
                        threshold=threshold,
                        scope_kind=scope_kind,
                        scope_name=scope_name,
                        feature_set=feature_set,
                        model=selected_model_name,
                    )

                selected_model = models[selected_model_name]
                predictor = OrderedSeverityClassifier(selected_model)
                verification_rows = parts["valid"].loc[:, features].head(3).copy()
                model_path = model_dir / (
                    f"severity__{threshold}__{feature_set}__{scope_kind}__"
                    f"{_safe_name(scope_name)}__{selected_model_name}.joblib"
                )
                joblib.dump(
                    {
                        "pipeline": predictor,
                        "raw_pipeline": selected_model,
                        "feature_data": features,
                        "feature_set": feature_set,
                        "high_risk_threshold": threshold,
                        "class_order": list(SEVERITY_LEVELS),
                        "baseline_transformer": prepared.baseline_transformer,
                        "verification_rows": verification_rows,
                        "verification_probabilities": predictor.predict_proba(
                            verification_rows
                        ),
                    },
                    model_path,
                )

                test = parts["test"]
                probabilities = predictor.predict_proba(test[features])
                predictions = test[
                    [DATE_COLUMN, "label_end_date", MACHINE_COLUMN, ASSET_COLUMN, "failure_points"]
                ].rename(columns={"failure_points": "actual_failure_points"})
                predictions["high_risk_threshold"] = threshold
                predictions["feature_set"] = feature_set
                predictions["scope_kind"] = scope_kind
                predictions["scope_name"] = scope_name
                predictions["model"] = selected_model_name
                predictions["selected_feature_set"] = False
                predictions["actual_level"] = test[target].astype(str).to_numpy()
                predictions["predicted_level"] = predictor.predict(test[features])
                for index, level in enumerate(SEVERITY_LEVELS):
                    predictions[f"prob_{level}"] = probabilities[:, index]
                prediction_rows.append(predictions)

                importance_rows.extend(
                    _importance_rows(
                        selected_model,
                        parts["valid"],
                        features,
                        target,
                        {
                            "high_risk_threshold": threshold,
                            "feature_set": feature_set,
                            "scope_kind": scope_kind,
                            "scope_name": scope_name,
                            "model": selected_model_name,
                        },
                        random_state,
                    )
                )

            if selected_models:
                validation_selected = pd.DataFrame(
                    [
                        row
                        for row in metric_rows
                        if row.get("high_risk_threshold") == threshold
                        and row.get("scope_kind") == scope_kind
                        and row.get("scope_name") == scope_name
                        and row.get("split") == "validation"
                        and row.get("selected_model") is True
                    ]
                )
                selected_feature = select_feature_set(validation_selected)
                for rows in (metric_rows, class_rows, confusion_rows):
                    _mark_selection(
                        rows,
                        threshold=threshold,
                        scope_kind=scope_kind,
                        scope_name=scope_name,
                        feature_set=selected_feature,
                        selected_feature=True,
                    )
                for predictions in prediction_rows:
                    mask = (
                        predictions["high_risk_threshold"].eq(threshold)
                        & predictions["scope_kind"].eq(scope_kind)
                        & predictions["scope_name"].eq(scope_name)
                        & predictions["feature_set"].eq(selected_feature)
                    )
                    predictions.loc[mask, "selected_feature_set"] = True

    metrics_frame = pd.DataFrame(metric_rows)
    classes_frame = pd.DataFrame(class_rows)
    confusion_frame = pd.DataFrame(confusion_rows)
    predictions_frame = (
        pd.concat(prediction_rows, ignore_index=True)
        if prediction_rows
        else pd.DataFrame()
    )
    importance_frame = pd.DataFrame(importance_rows)

    metrics_frame.to_csv(output / "experiment_metrics.csv", index=False)
    classes_frame.to_csv(output / "class_metrics.csv", index=False)
    confusion_frame.to_csv(output / "confusion_matrices.csv", index=False)
    predictions_frame.to_csv(output / "test_predictions.csv", index=False)
    importance_frame.to_csv(output / "feature_importance.csv", index=False)
    prepared.baselines.to_csv(output / "zscore_baselines.csv", index=False)
    profile_frame = build_failure_profile(
        filtered_raw,
        frame,
        high_risk_thresholds=thresholds,
        validation_start=validation_start,
        test_start=test_start,
    )
    profile_frame.to_csv(output / "machine_failure_profile.csv", index=False)
    (output / "experiment_summary.md").write_text(
        render_experiment_summary(metrics_frame, predictions_frame, profile_frame),
        encoding="utf-8",
    )
    config = {
        "high_risk_thresholds": list(thresholds),
        "feature_sets": list(selected_features),
        "scope": scope,
        "machine_type": machine_type,
        "asset_tag": asset_tag,
        "validation_start": str(validation_start),
        "test_start": str(test_start),
        "min_normal_rows": min_normal_rows,
        "max_iter": max_iter,
        "random_state": random_state,
        "models": list(EXPERIMENT_MODELS),
        "sklearn_version": sklearn.__version__,
    }
    (output / "run_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metrics_frame
