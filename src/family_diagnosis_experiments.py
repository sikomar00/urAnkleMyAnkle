"""Family별 당일 이상·심각 진단 모델을 비교하고 검증 기준으로 선택한다."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score

from .family_features import FAMILY_TARGETS, FamilyDiagnosisFeatures
from .industrial_data import ASSET_COLUMN, DATE_COLUMN, MACHINE_COLUMN, split_by_date
from .industrial_training import (
    MODEL_NAMES,
    ThresholdSelection,
    build_model,
    classification_metrics,
    ranking_metrics,
)

MODEL_ORDER = tuple(MODEL_NAMES)
FEATURE_SET_ORDER = ("A", "B", "C")
THRESHOLD_POLICIES = (
    "f1",
    "min_precision_70",
    "min_precision_80",
    "top_10pct",
)
MIN_POSITIVE_ROWS = 10


@dataclass(frozen=True)
class FamilyExperimentResult:
    """Family 실험의 지표·예측·중요도와 저장 모델 경로를 보관한다."""

    metrics: pd.DataFrame
    predictions: pd.DataFrame
    importances: pd.DataFrame
    models: dict[str, Path]
    multilabel_input: pd.DataFrame


def _validation_candidates(metrics: pd.DataFrame) -> pd.DataFrame:
    if "split" not in metrics:
        raise ValueError("선택 지표에 split 컬럼이 필요합니다.")
    validation = metrics.loc[metrics["split"].eq("validation")].copy()
    if validation.empty:
        raise ValueError("모델 선택에는 검증 지표가 필요합니다.")
    return validation


def _select_name(
    metrics: pd.DataFrame,
    column: str,
    declaration_order: tuple[str, ...],
    tolerance: float,
) -> str:
    if tolerance < 0:
        raise ValueError("tolerance는 0 이상이어야 합니다.")
    validation = _validation_candidates(metrics)
    if column not in validation:
        raise ValueError(f"선택 지표에 {column} 컬럼이 필요합니다.")
    validation = validation.dropna(subset=["average_precision", column])
    if validation.empty:
        raise ValueError("선택 가능한 검증 후보가 없습니다.")

    best_ap = float(validation["average_precision"].max())
    candidates = validation.loc[
        validation["average_precision"].ge(best_ap - tolerance)
    ].copy()
    observed_order = list(dict.fromkeys(candidates[column].astype(str)))
    order = [name for name in declaration_order if name in observed_order]
    order.extend(name for name in observed_order if name not in order)
    rank = {name: index for index, name in enumerate(order)}
    for metric in ("f1", "recall", "precision"):
        candidates[metric] = pd.to_numeric(
            candidates.get(metric), errors="coerce"
        ).fillna(float("-inf"))
    candidates["_declaration_order"] = candidates[column].astype(str).map(rank)
    selected = candidates.sort_values(
        ["f1", "recall", "precision", "_declaration_order"],
        ascending=[False, False, False, True],
        kind="stable",
    ).iloc[0]
    return str(selected[column])


def select_family_model(
    validation_metrics: pd.DataFrame, tolerance: float = 0.01
) -> str:
    """검증 AP와 동률 보조 지표로 대표 분류 모델을 선택한다."""
    candidates = validation_metrics.loc[
        ~validation_metrics["model"].eq("dummy_classifier")
    ]
    return _select_name(candidates, "model", MODEL_ORDER, tolerance)


def select_family_feature_set(
    validation_metrics: pd.DataFrame, tolerance: float = 0.01
) -> str:
    """검증 AP와 동률 보조 지표로 대표 Feature 집합을 선택한다."""
    return _select_name(
        validation_metrics, "feature_set", FEATURE_SET_ORDER, tolerance
    )


def _threshold_at_min_precision(
    y: np.ndarray, scores: np.ndarray, minimum: float
) -> ThresholdSelection:
    policy = f"min_precision_{int(minimum * 100)}"
    candidates = np.unique(np.clip(scores, 0.0, 1.0))
    feasible: list[tuple[float, float]] = []
    positives = int(y.sum())
    for cutoff in candidates:
        predicted = scores >= cutoff
        predicted_count = int(predicted.sum())
        true_positive = int(y[predicted].sum())
        precision = true_positive / predicted_count if predicted_count else 0.0
        recall = true_positive / positives if positives else 0.0
        if precision >= minimum:
            feasible.append((recall, float(cutoff)))
    if not feasible:
        return ThresholdSelection(policy, None, "unavailable")
    best_recall = max(item[0] for item in feasible)
    cutoff = max(item[1] for item in feasible if item[0] == best_recall)
    return ThresholdSelection(policy, cutoff, "ok")


def choose_family_thresholds(y_true: Any, scores: Any) -> list[ThresholdSelection]:
    """검증 점수에서 F1·Precision 0.70/0.80·상위 10% 임계값을 고른다."""
    y = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(scores, dtype=float)
    if len(y) != len(probabilities) or not len(y):
        raise ValueError("정답과 점수는 길이가 같은 비어 있지 않은 배열이어야 합니다.")
    if not set(np.unique(y)) <= {0, 1}:
        raise ValueError("정답은 0 또는 1이어야 합니다.")

    candidates = np.unique(np.clip(probabilities, 0.0, 1.0))
    f1_rows: list[tuple[float, float]] = []
    for cutoff in candidates:
        metrics = classification_metrics(y, probabilities, float(cutoff))
        f1_rows.append((float(metrics["f1"] or 0.0), float(cutoff)))
    best_f1 = max(item[0] for item in f1_rows)
    f1_cutoff = max(item[1] for item in f1_rows if item[0] == best_f1)

    count = max(1, math.ceil(len(probabilities) * 0.10))
    top_cutoff = float(np.sort(probabilities)[::-1][count - 1])
    return [
        ThresholdSelection("f1", f1_cutoff, "ok"),
        _threshold_at_min_precision(y, probabilities, 0.70),
        _threshold_at_min_precision(y, probabilities, 0.80),
        ThresholdSelection("top_10pct", top_cutoff, "ok"),
    ]


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def _metric_row(
    *,
    family: str,
    target: str,
    feature_set: str,
    model_name: str,
    split_name: str,
    y_true: pd.Series,
    scores: np.ndarray,
    threshold: ThresholdSelection,
    prior_ap: float,
    selected_model: bool = False,
    selected_feature_set: bool = False,
    selected_policy: bool = False,
) -> dict[str, Any]:
    return {
        "part_family": family,
        "target": target,
        "feature_set": feature_set,
        "model": model_name,
        "split": split_name,
        "status": threshold.status,
        "reason": "",
        "threshold_policy": threshold.policy,
        "probability_cutoff": threshold.cutoff,
        "prior_average_precision": prior_ap,
        "selected_model": selected_model,
        "selected_feature_set": selected_feature_set,
        "selected_policy": selected_policy,
        "support": int(len(y_true)),
        "positive_rate": float(y_true.mean()) if len(y_true) else np.nan,
        **classification_metrics(y_true, scores, threshold.cutoff),
        **ranking_metrics(y_true, scores),
    }


def _skip_row(
    family: str, target: str, status: str, reason: str
) -> dict[str, Any]:
    return {
        "part_family": family,
        "target": target,
        "feature_set": "none",
        "model": "none",
        "split": "validation",
        "status": status,
        "reason": reason,
        "selected_model": False,
        "selected_feature_set": False,
        "selected_policy": False,
    }


def run_family_experiments(
    prepared: FamilyDiagnosisFeatures,
    output_dir: str | Path,
    max_iter: int = 100,
    random_state: int = 42,
) -> FamilyExperimentResult:
    """Family별 이진 모델을 검증에서 선택하고 테스트 결과와 모델을 저장한다."""
    frame = prepared.frame.copy()
    output = Path(output_dir)
    model_dir = output / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    metric_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    importance_rows: list[dict[str, Any]] = []
    model_paths: dict[str, Path] = {}
    completed: dict[tuple[str, str], dict[str, Any]] = {}

    families = list(dict.fromkeys(frame["part_family"].astype(str)))
    for family in families:
        family_frame = frame.loc[frame["part_family"].astype(str).eq(family)].copy()
        for target in FAMILY_TARGETS:
            if target not in family_frame:
                metric_rows.append(
                    _skip_row(family, target, "missing_target", f"{target} 컬럼 없음")
                )
                continue
            if (
                target == "severe"
                and "affected" in family_frame
                and family_frame[target].equals(family_frame["affected"])
                and (family, "affected") in completed
            ):
                cached = completed[(family, "affected")]
                copied_metrics = cached["metrics"].copy()
                copied_metrics["target"] = target
                copied_metrics["status"] = "identical_target"
                copied_metrics["reason"] = "affected Target과 동일하여 모델 재사용"
                metric_rows.extend(copied_metrics.to_dict("records"))
                copied_predictions = cached["predictions"].copy()
                copied_predictions["target"] = target
                prediction_frames.append(copied_predictions)
                copied_importances = cached["importances"].copy()
                copied_importances["target"] = target
                importance_rows.extend(copied_importances.to_dict("records"))
                model_paths[f"{family}__{target}"] = cached["model_path"]
                completed[(family, target)] = {
                    **cached,
                    "metrics": copied_metrics,
                    "predictions": copied_predictions,
                    "importances": copied_importances,
                }
                continue

            parts = split_by_date(family_frame)
            train, valid, test = parts["train"], parts["valid"], parts["test"]
            issue = ""
            for split_name, subset in (("train", train), ("validation", valid)):
                positive = int(subset[target].sum()) if not subset.empty else 0
                if subset.empty or subset[target].nunique() < 2 or positive < MIN_POSITIVE_ROWS:
                    issue = (
                        f"{split_name} 양성={positive}, 행={len(subset)}, "
                        f"클래스={subset[target].nunique() if not subset.empty else 0}"
                    )
                    break
            if issue:
                metric_rows.append(
                    _skip_row(
                        family, target, "insufficient_positive_rows", issue
                    )
                )
                continue
            if test.empty:
                metric_rows.append(_skip_row(family, target, "empty_test", "테스트 없음"))
                continue

            candidate_rows: list[dict[str, Any]] = []
            fitted: dict[tuple[str, str], Any] = {}
            valid_scores: dict[tuple[str, str], np.ndarray] = {}
            prior_ap = float(valid[target].mean())
            for feature_set, features in prepared.feature_sets.items():
                feature_names = list(features)
                for model_name in MODEL_ORDER:
                    model = build_model(
                        model_name,
                        train[feature_names],
                        max_iter=max_iter,
                        random_state=random_state,
                    )
                    model.fit(train[feature_names], train[target].astype(int))
                    scores = model.predict_proba(valid[feature_names])[:, 1]
                    fitted[(feature_set, model_name)] = model
                    valid_scores[(feature_set, model_name)] = scores
                    f1_threshold = choose_family_thresholds(valid[target], scores)[0]
                    candidate_rows.append(
                        _metric_row(
                            family=family,
                            target=target,
                            feature_set=feature_set,
                            model_name=model_name,
                            split_name="validation",
                            y_true=valid[target],
                            scores=scores,
                            threshold=f1_threshold,
                            prior_ap=prior_ap,
                        )
                    )

            candidate_metrics = pd.DataFrame(candidate_rows)
            selected_models: dict[str, str] = {}
            for feature_set in prepared.feature_sets:
                selected_models[feature_set] = select_family_model(
                    candidate_metrics.loc[
                        candidate_metrics["feature_set"].eq(feature_set)
                    ]
                )
            feature_candidates = pd.concat(
                [
                    candidate_metrics.loc[
                        candidate_metrics["feature_set"].eq(feature_set)
                        & candidate_metrics["model"].eq(model_name)
                    ]
                    for feature_set, model_name in selected_models.items()
                ],
                ignore_index=True,
            )
            selected_feature = select_family_feature_set(feature_candidates)
            selected_model = selected_models[selected_feature]
            for row in candidate_rows:
                row["selected_model"] = row["model"] == selected_models[row["feature_set"]]
                row["selected_feature_set"] = row["feature_set"] == selected_feature
            metric_rows.extend(candidate_rows)

            features = list(prepared.feature_sets[selected_feature])
            model = fitted[(selected_feature, selected_model)]
            validation_scores = valid_scores[(selected_feature, selected_model)]
            test_scores = model.predict_proba(test[features])[:, 1]
            thresholds = choose_family_thresholds(valid[target], validation_scores)
            selected_rows: list[dict[str, Any]] = []
            target_predictions: list[pd.DataFrame] = []
            for selection in thresholds:
                for split_name, subset, scores in (
                    ("validation", valid, validation_scores),
                    ("test", test, test_scores),
                ):
                    selected_rows.append(
                        _metric_row(
                            family=family,
                            target=target,
                            feature_set=selected_feature,
                            model_name=selected_model,
                            split_name=split_name,
                            y_true=subset[target],
                            scores=scores,
                            threshold=selection,
                            prior_ap=prior_ap,
                            selected_model=True,
                            selected_feature_set=True,
                            selected_policy=True,
                        )
                    )
                predictions = test[
                    [DATE_COLUMN, "label_end_date", MACHINE_COLUMN, ASSET_COLUMN]
                ].copy()
                predictions["part_family"] = family
                predictions["target"] = target
                predictions["actual"] = test[target].astype(int).to_numpy()
                predictions["feature_set"] = selected_feature
                predictions["model"] = selected_model
                predictions["threshold_policy"] = selection.policy
                predictions["probability_cutoff"] = selection.cutoff
                predictions["risk_score"] = test_scores
                predictions["prediction"] = (
                    (test_scores >= selection.cutoff).astype(int)
                    if selection.cutoff is not None
                    else pd.NA
                )
                target_predictions.append(predictions)
            metric_rows.extend(selected_rows)
            target_prediction_frame = pd.concat(target_predictions, ignore_index=True)
            prediction_frames.append(target_prediction_frame)

            try:
                importance = permutation_importance(
                    model,
                    test[features],
                    test[target],
                    scoring="average_precision",
                    n_repeats=3,
                    random_state=random_state,
                )
                target_importances = pd.DataFrame(
                    {
                        "part_family": family,
                        "target": target,
                        "feature_set": selected_feature,
                        "model": selected_model,
                        "feature": features,
                        "importance_mean": importance.importances_mean,
                        "importance_std": importance.importances_std,
                    }
                )
            except ValueError:
                target_importances = pd.DataFrame(
                    {
                        "part_family": family,
                        "target": target,
                        "feature_set": selected_feature,
                        "model": selected_model,
                        "feature": features,
                        "importance_mean": np.nan,
                        "importance_std": np.nan,
                    }
                )
            importance_rows.extend(target_importances.to_dict("records"))

            model_path = model_dir / (
                f"{_safe_name(family)}__{target}__{selected_feature}__"
                f"{selected_model}.joblib"
            )
            joblib.dump(
                {
                    "pipeline": model,
                    "part_family": family,
                    "target": target,
                    "feature_set": selected_feature,
                    "features": features,
                    "thresholds": {
                        item.policy: {"cutoff": item.cutoff, "status": item.status}
                        for item in thresholds
                    },
                    "residual_transformer": prepared.residual_transformer,
                },
                model_path,
            )
            key = f"{family}__{target}"
            model_paths[key] = model_path
            completed[(family, target)] = {
                "metrics": pd.DataFrame([*candidate_rows, *selected_rows]),
                "predictions": target_prediction_frame,
                "importances": target_importances,
                "model_path": model_path,
            }

    metrics = pd.DataFrame(metric_rows)
    predictions = (
        pd.concat(prediction_frames, ignore_index=True)
        if prediction_frames
        else pd.DataFrame()
    )
    importances = pd.DataFrame(importance_rows)
    multilabel_input = (
        predictions.loc[predictions["threshold_policy"].eq("f1")].copy()
        if not predictions.empty
        else pd.DataFrame()
    )
    return FamilyExperimentResult(
        metrics=metrics,
        predictions=predictions,
        importances=importances,
        models=model_paths,
        multilabel_input=multilabel_input,
    )
