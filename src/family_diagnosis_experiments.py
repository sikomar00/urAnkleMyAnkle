"""Family별 당일 이상·심각 진단 모델을 비교하고 검증 기준으로 선택한다."""

from __future__ import annotations

import json
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

from .asset_anomaly_features import HISTORY_FEATURES
from .family_features import (
    FAMILY_TARGETS,
    FamilyDiagnosisFeatures,
    build_family_target_profile,
)
from .family_residual_features import RESIDUAL_FEATURES
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


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _atomic_text(text: str, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


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


def _scope_metric_rows(predictions: pd.DataFrame) -> list[dict[str, Any]]:
    """선택 테스트 예측을 기계 종류·장비 범위로 나눠 재평가한다."""
    rows: list[dict[str, Any]] = []
    if predictions.empty:
        return rows
    group_keys = ["part_family", "target", "threshold_policy"]
    for scope_kind, scope_column in (
        ("machine_type", MACHINE_COLUMN),
        ("asset_tag", ASSET_COLUMN),
    ):
        for keys, group in predictions.groupby(
            [*group_keys, scope_column], sort=False, dropna=False
        ):
            cutoff = float(group["probability_cutoff"].iloc[0])
            threshold = ThresholdSelection(str(keys[2]), cutoff, "ok")
            row = _metric_row(
                family=str(keys[0]),
                target=str(keys[1]),
                feature_set=str(group["feature_set"].iloc[0]),
                model_name=str(group["model"].iloc[0]),
                split_name="test",
                y_true=group["actual"].astype(int),
                scores=group["risk_score"].to_numpy(),
                threshold=threshold,
                prior_ap=float(group["actual"].mean()),
                selected_model=True,
                selected_feature_set=True,
                selected_policy=True,
            )
            row["scope_kind"] = scope_kind
            row["scope_name"] = str(keys[3])
            rows.append(row)
    return rows


def run_family_experiments(
    prepared: FamilyDiagnosisFeatures,
    output_dir: str | Path,
    max_iter: int = 100,
    random_state: int = 42,
    targets: tuple[str, ...] = FAMILY_TARGETS,
    validation_start: str | pd.Timestamp = "2024-01-01",
    test_start: str | pd.Timestamp = "2024-07-01",
) -> FamilyExperimentResult:
    """Family별 이진 모델을 검증에서 선택하고 테스트 결과와 모델을 저장한다."""
    frame = prepared.frame.copy()
    targets = tuple(dict.fromkeys(targets))
    unknown_targets = sorted(set(targets) - set(FAMILY_TARGETS))
    if not targets or unknown_targets:
        raise ValueError(
            "targets는 affected, severe 중 하나 이상이어야 합니다: "
            f"{unknown_targets}"
        )
    validation_start = pd.Timestamp(validation_start)
    test_start = pd.Timestamp(test_start)
    if validation_start >= test_start:
        raise ValueError("validation_start는 test_start보다 빨라야 합니다.")
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
        for target in targets:
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

            parts = split_by_date(family_frame, validation_start, test_start)
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
            candidate_test_rows: list[dict[str, Any]] = []
            for feature_set, feature_model in selected_models.items():
                feature_names = list(prepared.feature_sets[feature_set])
                feature_valid_scores = valid_scores[(feature_set, feature_model)]
                feature_test_scores = fitted[(feature_set, feature_model)].predict_proba(
                    test[feature_names]
                )[:, 1]
                feature_threshold = choose_family_thresholds(
                    valid[target], feature_valid_scores
                )[0]
                candidate_test_rows.append(
                    _metric_row(
                        family=family,
                        target=target,
                        feature_set=feature_set,
                        model_name=feature_model,
                        split_name="test",
                        y_true=test[target],
                        scores=feature_test_scores,
                        threshold=feature_threshold,
                        prior_ap=prior_ap,
                        selected_model=True,
                        selected_feature_set=feature_set == selected_feature,
                    )
                )
            metric_rows.extend(candidate_rows)
            metric_rows.extend(candidate_test_rows)

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
                if selection.cutoff is None:
                    continue
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
                predictions["prediction"] = pd.Series(
                    (test_scores >= selection.cutoff).astype(int),
                    index=predictions.index,
                    dtype="Int64",
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
                "metrics": pd.DataFrame(
                    [*candidate_rows, *candidate_test_rows, *selected_rows]
                ),
                "predictions": target_prediction_frame,
                "importances": target_importances,
                "model_path": model_path,
            }

    metrics = pd.DataFrame(metric_rows)
    metrics["scope_kind"] = "overall"
    metrics["scope_name"] = "all"
    predictions = (
        pd.concat(prediction_frames, ignore_index=True)
        if prediction_frames
        else pd.DataFrame()
    )
    importances = pd.DataFrame(importance_rows)
    scope_rows = _scope_metric_rows(predictions)
    if scope_rows:
        metrics = pd.concat([metrics, pd.DataFrame(scope_rows)], ignore_index=True)
    metrics["temporal_importance"] = 0.0
    temporal_features = set(HISTORY_FEATURES) | set(RESIDUAL_FEATURES)
    if not importances.empty:
        temporal = importances.loc[importances["feature"].isin(temporal_features)].copy()
        temporal["positive_importance"] = pd.to_numeric(
            temporal["importance_mean"], errors="coerce"
        ).clip(lower=0).fillna(0.0)
        for keys, group in temporal.groupby(
            ["part_family", "target", "feature_set", "model"], sort=False
        ):
            mask = (
                metrics["part_family"].eq(keys[0])
                & metrics["target"].eq(keys[1])
                & metrics["feature_set"].eq(keys[2])
                & metrics["model"].eq(keys[3])
            )
            metrics.loc[mask, "temporal_importance"] = float(
                group["positive_importance"].sum()
            )
    multilabel_input = (
        predictions.loc[predictions["threshold_policy"].eq("f1")].copy()
        if not predictions.empty
        else pd.DataFrame()
    )
    from .family_diagnosis_report import (
        build_multilabel_metrics,
        evaluate_sequence_gate,
        render_family_summary,
    )

    multilabel_metrics = (
        build_multilabel_metrics(multilabel_input)
        if not multilabel_input.empty
        else pd.DataFrame()
    )
    profile = build_family_target_profile(
        prepared.frame, validation_start, test_start
    )
    sequence_gate = evaluate_sequence_gate(metrics)
    residual_baselines = (
        prepared.residual_transformer.artifact_table()
        if prepared.residual_transformer is not None
        else pd.DataFrame(
            columns=[
                ASSET_COLUMN,
                "sensor",
                "normal_rows",
                "selected_scope",
                "fallback_reason",
                "residual_median",
                "residual_mad",
                "residual_std",
                "residual_scale",
                "scale_method",
            ]
        )
    )
    run_config = {
        "random_state": random_state,
        "max_iter": max_iter,
        "validation_start": validation_start.strftime("%Y-%m-%d"),
        "test_start": test_start.strftime("%Y-%m-%d"),
        "feature_sets": {
            name: list(features) for name, features in prepared.feature_sets.items()
        },
        "targets": {
            "affected": "failed_parts >= 1",
            "severe": "failed_parts >= 2 or failed_a_parts >= 1",
            "all_failed": "통계 전용",
        },
        "threshold_policies": list(THRESHOLD_POLICIES),
        "sequence_gate": sequence_gate,
    }
    summary = render_family_summary(
        metrics, multilabel_metrics, profile, sequence_gate
    )
    output.mkdir(parents=True, exist_ok=True)
    _atomic_csv(metrics, output / "metrics.csv")
    _atomic_csv(multilabel_metrics, output / "multilabel_metrics.csv")
    _atomic_csv(profile, output / "target_profile.csv")
    _atomic_csv(predictions, output / "test_predictions.csv")
    _atomic_csv(importances, output / "feature_importance.csv")
    _atomic_csv(residual_baselines, output / "residual_baselines.csv")
    _atomic_text(
        json.dumps(run_config, ensure_ascii=False, indent=2),
        output / "run_config.json",
    )
    _atomic_text(summary, output / "experiment_summary.md")
    return FamilyExperimentResult(
        metrics=metrics,
        predictions=predictions,
        importances=importances,
        models=model_paths,
        multilabel_input=multilabel_input,
    )
