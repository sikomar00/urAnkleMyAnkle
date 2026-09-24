"""데이터 누수를 막는 분류 모델 비교·선택·평가·저장을 공통으로 수행한다."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Sequence
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.inspection import permutation_importance

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "src"

from .industrial_data import PreparedTask
from .industrial_features import iter_scopes, split_by_date

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = PROJECT_ROOT / "data" / "raw" / "synthetic_industrial_machine_data.csv"
if sys.platform == "win32":
    # Keep the desktop responsive and avoid the physical-core probe.
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(max(1, min(4, (os.cpu_count() or 2) // 2))))


POST_EVENT_FEATURES = {
    "wo_type",
    "qty_issued",
    "issue_value_inr",
    "target_7d",
    "target_1d",
    "target_2d",
    "target_3d",
    "target_7d",
}


def validate_features(features: list[str], target: str) -> None:
    """Reject direct targets and fields likely recorded after an event."""

    overlap = set(features) & {target, "breakdown_flag"}
    if overlap:
        raise ValueError(f"target 컬럼을 feature에 넣을 수 없습니다: {sorted(overlap)}")
    forbidden = set(features) & POST_EVENT_FEATURES
    if forbidden:
        raise ValueError(f"사후 정보 컬럼을 feature에 넣을 수 없습니다: {sorted(forbidden)}")


def choose_threshold(y_true: Any, scores: Any) -> float:
    """Choose the validation threshold with the highest F1."""

    y = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(scores, dtype=float)
    if len(y) == 0 or len(np.unique(y)) < 2:
        return 0.5
    candidates = np.unique(np.clip(probabilities, 0.0, 1.0))
    candidates = np.unique(np.r_[0.5, candidates])
    order = np.argsort(probabilities)
    sorted_scores = probabilities[order]
    positive_prefix = np.r_[0, np.cumsum(y[order])]
    starts = np.searchsorted(sorted_scores, candidates, side="left")
    true_positives = positive_prefix[-1] - positive_prefix[starts]
    predicted_positives = len(y) - starts
    denominators = predicted_positives + positive_prefix[-1]
    f1_values = np.divide(
        2.0 * true_positives, denominators,
        out=np.zeros(len(candidates), dtype=float), where=denominators != 0,
    )
    # Candidates are sorted; retain the original highest-threshold tie break.
    return float(candidates[np.flatnonzero(f1_values == f1_values.max())[-1]])


@dataclass(frozen=True)
class ThresholdSelection:
    """검증 데이터에서 선택한 한 가지 임계값 정책의 결과다."""

    policy: str
    cutoff: float | None
    status: str


def choose_thresholds(
    y_true: Any,
    scores: Any,
    *,
    min_precision: float = 0.30,
    top_fraction: float = 0.10,
) -> list[ThresholdSelection]:
    """F1·최소 Precision·상위 비율 기준 임계값을 검증 점수에서 선택한다."""
    if not 0 < min_precision <= 1:
        raise ValueError("min_precision은 0보다 크고 1 이하여야 합니다.")
    if not 0 < top_fraction <= 1:
        raise ValueError("top_fraction은 0보다 크고 1 이하여야 합니다.")
    y = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(scores, dtype=float)
    if len(y) == 0 or len(np.unique(y)) < 2:
        return [
            ThresholdSelection("f1", 0.5, "ok"),
            ThresholdSelection("min_precision", None, "unavailable"),
            ThresholdSelection("top_fraction", 0.5, "ok"),
        ]

    candidates = np.unique(np.clip(probabilities, 0.0, 1.0))
    feasible: list[tuple[float, float]] = []
    for cutoff in candidates:
        predictions = probabilities >= cutoff
        precision = precision_score(y, predictions, zero_division=0)
        recall = recall_score(y, predictions, zero_division=0)
        if precision >= min_precision:
            feasible.append((float(recall), float(cutoff)))
    min_precision_cutoff = max(feasible)[1] if feasible else None

    count = max(1, math.ceil(len(probabilities) * top_fraction))
    top_cutoff = float(np.sort(probabilities)[::-1][count - 1])
    return [
        ThresholdSelection("f1", choose_threshold(y, probabilities), "ok"),
        ThresholdSelection(
            "min_precision", min_precision_cutoff,
            "ok" if min_precision_cutoff is not None else "unavailable",
        ),
        ThresholdSelection("top_fraction", top_cutoff, "ok"),
    ]


def ranking_metrics(y_true: Any, scores: Any) -> dict[str, float | None]:
    """위험점수 상위 5·10·20%의 Precision, Recall과 Lift를 계산한다."""
    y = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(scores, dtype=float)
    result: dict[str, float | None] = {}
    prevalence = float(y.mean()) if len(y) else 0.0
    order = np.argsort(-probabilities, kind="stable")
    for percent in (5, 10, 20):
        key = f"{percent}pct"
        if not len(y):
            result[f"precision_at_{key}"] = None
            result[f"recall_at_{key}"] = None
            result[f"lift_at_{key}"] = None
            continue
        count = max(1, math.ceil(len(y) * percent / 100))
        selected = y[order[:count]]
        precision = float(selected.mean())
        recall = float(selected.sum() / y.sum()) if y.sum() else 0.0
        result[f"precision_at_{key}"] = precision
        result[f"recall_at_{key}"] = recall
        result[f"lift_at_{key}"] = precision / prevalence if prevalence else None
    return result


def classification_metrics(
    y_true: Any, scores: Any, threshold: float
) -> dict[str, float | int | None]:
    """확률점수와 판단 임계값으로 분류 성능과 혼동행렬 값을 계산한다."""
    y = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(scores, dtype=float)
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predictions, labels=[0, 1]).ravel()
    roc_auc = None
    average_precision = None
    if len(np.unique(y)) == 2:
        roc_auc = float(roc_auc_score(y, probabilities))
        average_precision = float(average_precision_score(y, probabilities))
    return {
        "accuracy": float(accuracy_score(y, predictions)) if len(y) else None,
        "precision": float(precision_score(y, predictions, zero_division=0)) if len(y) else None,
        "recall": float(recall_score(y, predictions, zero_division=0)) if len(y) else None,
        "f1": float(f1_score(y, predictions, zero_division=0)) if len(y) else None,
        "roc_auc": roc_auc,
        "average_precision": average_precision,
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def _column_types(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    # pandas 4 can infer plain CSV strings as ``StringDtype`` rather than the
    # older object dtype.  Treat every non-numeric feature as categorical so
    # both pandas representations follow the same preprocessing path.
    numeric = [column for column in frame.columns if pd.api.types.is_numeric_dtype(frame[column])]
    categorical = [column for column in frame.columns if column not in numeric]
    return categorical, numeric


def build_model(
    model_name: str,
    frame: pd.DataFrame,
    max_iter: int = 100,
    random_state: int = 42,
) -> Pipeline:
    """Build a pipeline whose preprocessing is fitted on training data only."""

    categorical, numeric = _column_types(frame)
    transformers: list[tuple[str, Any, list[str]]] = []
    if categorical:
        transformers.append(
            (
                "category",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical,
            )
        )
    if numeric:
        transformers.append(
            (
                "numeric",
                SimpleImputer(strategy="median"),
                numeric,
            )
        )
    preprocess = ColumnTransformer(transformers=transformers, remainder="drop")

    if model_name == "hist_gradient_boosting":
        classifier = HistGradientBoostingClassifier(
            max_iter=max_iter,
            early_stopping=False,
            random_state=random_state,
        )
    elif model_name == "logistic_regression":
        classifier = LogisticRegression(
            max_iter=max(1000, max_iter), class_weight="balanced",
            random_state=random_state,
        )
    elif model_name == "random_forest":
        classifier = RandomForestClassifier(
            n_estimators=max(50, max_iter),
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )
    else:
        raise ValueError(
            "지원하지 않는 모델입니다. hist_gradient_boosting, "
            "logistic_regression, random_forest 중 하나를 사용하세요."
        )
    return Pipeline([("preprocess", preprocess), ("classifier", classifier)])


MODEL_NAMES = ("logistic_regression", "random_forest", "hist_gradient_boosting")


def run_experiment_suite(
    tasks: Sequence[PreparedTask], *, output_dir: str | Path, scope: str = "all",
    machine_type: str | None = None, asset_tag: str | None = None,
    max_iter: int = 100, validation_start: str = "2024-01-01",
    test_start: str = "2024-07-01",
    threshold_policies: tuple[str, ...] = ("f1", "min_precision", "top_fraction"),
    min_precision: float = 0.30, top_fraction: float = 0.10,
    random_state: int = 42,
) -> pd.DataFrame:
    """여러 준비 과제를 같은 모델·평가 계약으로 학습하고 산출물을 저장한다."""
    allowed = {"f1", "min_precision", "top_fraction"}
    if not threshold_policies or not set(threshold_policies) <= allowed:
        raise ValueError(f"threshold_policies는 {sorted(allowed)} 중 하나 이상이어야 합니다.")
    output = Path(output_dir)
    (output / "models").mkdir(parents=True, exist_ok=True)
    metric_rows, prediction_rows, importance_rows = [], [], []
    for task in tasks:
        validate_features(list(task.features), task.target)
        for scope_kind, scope_name, subset in iter_scopes(
            task.frame, scope, machine_type, asset_tag
        ):
            parts = split_by_date(subset, validation_start, test_start)
            train, valid, test = parts["train"], parts["valid"], parts["test"]
            meta = {
                "grain": task.grain, "mode": task.mode, "target": task.target,
                "risk_definition": task.risk_definition,
                "score_threshold": task.score_threshold, "horizon": task.horizon,
                "scope_kind": scope_kind, "scope_name": scope_name,
            }
            if any(part.empty for part in parts.values()) or train[task.target].nunique() < 2 or valid[task.target].nunique() < 2:
                metric_rows.append({**meta, "model": "none", "selected_model": False,
                                    "split": "test", "status": "skipped",
                                    "reason": "empty split or single training/validation class"})
                continue
            fitted = {}
            validation_scores = {}
            for model_name in MODEL_NAMES:
                model = build_model(model_name, train[list(task.features)], max_iter, random_state)
                model.fit(train[list(task.features)], train[task.target].astype(int))
                fitted[model_name] = model
                validation_scores[model_name] = model.predict_proba(valid[list(task.features)])[:, 1]
            ap = {name: average_precision_score(valid[task.target], scores)
                  for name, scores in validation_scores.items()}
            selected_name = max(MODEL_NAMES, key=lambda name: (ap[name], -MODEL_NAMES.index(name)))
            for model_name, model in fitted.items():
                scores_by_split = {
                    "validation": validation_scores[model_name],
                    "test": model.predict_proba(test[list(task.features)])[:, 1],
                }
                selections = choose_thresholds(
                    valid[task.target], scores_by_split["validation"],
                    min_precision=min_precision, top_fraction=top_fraction,
                )
                for split_name, part in (("validation", valid), ("test", test)):
                    scores = scores_by_split[split_name]
                    ranks = ranking_metrics(part[task.target], scores)
                    for selection in selections:
                        if selection.policy not in threshold_policies:
                            continue
                        metrics = ({key: None for key in classification_metrics(part[task.target], scores, 0.5)}
                                   if selection.cutoff is None else
                                   classification_metrics(part[task.target], scores, selection.cutoff))
                        metric_rows.append({
                            **meta, "model": model_name,
                            "selected_model": model_name == selected_name,
                            "split": split_name, "status": selection.status, "reason": "",
                            "threshold_policy": selection.policy,
                            "probability_cutoff": selection.cutoff,
                            "train_rows": len(train), "valid_rows": len(valid), "test_rows": len(test),
                            "train_positive_rate": train[task.target].mean(),
                            "valid_positive_rate": valid[task.target].mean(),
                            "test_positive_rate": test[task.target].mean(), **metrics, **ranks,
                        })
                        if split_name == "test" and model_name == selected_name:
                            pred = test[[c for c in ["transaction_date", "label_end_date", "asset_tag", "machine_type", "part_no", task.target] if c in test]].copy()
                            pred.update({})
                            pred = pred.assign(**meta, model=model_name,
                                               threshold_policy=selection.policy,
                                               probability_cutoff=selection.cutoff,
                                               risk_score=scores,
                                               prediction=(scores >= selection.cutoff).astype(int) if selection.cutoff is not None else pd.NA)
                            prediction_rows.append(pred)
            selected = fitted[selected_name]
            model_path = output / "models" / f"{_safe_name(task.target)}__{_safe_name(scope_kind)}__{_safe_name(scope_name)}__{selected_name}__selected.joblib"
            verification_rows = task.frame.loc[:, list(task.features)].head(3)
            joblib.dump({"pipeline": selected, "feature_data": list(task.features),
                         "target_data": task.target, "metadata": meta,
                         "verification_scores": selected.predict_proba(verification_rows)[:, 1]}, model_path)
            sample = valid.sample(min(1000, len(valid)), random_state=random_state)
            perm = permutation_importance(selected, sample[list(task.features)], sample[task.target],
                                          scoring="average_precision", n_repeats=3, random_state=random_state)
            importance_rows.extend({**meta, "model": selected_name, "feature": feature,
                                    "importance_mean": mean, "importance_std": std}
                                   for feature, mean, std in zip(task.features, perm.importances_mean, perm.importances_std))
    metrics_frame = pd.DataFrame(metric_rows)
    metrics_frame.to_csv(output / "metrics.csv", index=False)
    pd.concat(prediction_rows, ignore_index=True).to_csv(output / "test_predictions.csv", index=False) if prediction_rows else pd.DataFrame().to_csv(output / "test_predictions.csv", index=False)
    pd.DataFrame(importance_rows).to_csv(output / "feature_importance.csv", index=False)
    (output / "run_config.json").write_text(json.dumps({"models": MODEL_NAMES, "validation_start": validation_start, "test_start": test_start, "threshold_policies": threshold_policies, "random_state": random_state}, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics_frame


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "all"


def _positive_rate(frame: pd.DataFrame, target: str) -> float | None:
    if frame.empty:
        return None
    return float(pd.to_numeric(frame[target]).mean())


def _metrics_row(
    *,
    scope_kind: str,
    scope_name: str,
    model: str,
    status: str,
    metrics: dict[str, Any] | None = None,
    threshold: float | None = None,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
) -> dict[str, Any]:
    metrics = metrics or {}
    return {
        "scope_kind": scope_kind,
        "scope_name": scope_name,
        "model": model,
        "status": status,
        "threshold": threshold,
        "train_rows": len(train),
        "valid_rows": len(valid),
        "test_rows": len(test),
        "train_positive_rate": _positive_rate(train, target),
        "valid_positive_rate": _positive_rate(valid, target),
        "test_positive_rate": _positive_rate(test, target),
        **metrics,
    }


def _skipped_rows(
    scope_kind: str,
    scope_name: str,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
) -> list[dict[str, Any]]:
    return [
        _metrics_row(
            scope_kind=scope_kind,
            scope_name=scope_name,
            model=model,
            status="skipped",
            train=train,
            valid=valid,
            test=test,
            target=target,
        )
        for model in ["prior", "hist_gradient_boosting"]
    ]


def run_experiments(
    frame: pd.DataFrame,
    features: list[str],
    target: str,
    *,
    output_dir: str | Path,
    mode: str = "current",
    scope: str = "all",
    machine_type: str | None = None,
    asset_tag: str | None = None,
    max_iter: int = 100,
    validation_start: str = "2024-01-01",
    test_start: str = "2024-07-01",
) -> pd.DataFrame:
    """Train baseline and boosting models for all requested scopes."""

    validate_features(features, target)
    missing = [column for column in [*features, target] if column not in frame]
    if missing:
        raise ValueError(f"학습 데이터에 컬럼이 없습니다: {missing}")
    output_path = Path(output_dir)
    (output_path / "models").mkdir(parents=True, exist_ok=True)
    scopes = iter_scopes(frame, scope, machine_type, asset_tag)

    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[pd.DataFrame] = []
    for scope_kind, scope_name, subset in scopes:
        print(f"[Training] {scope_kind}: {scope_name} ({len(subset):,} rows)", flush=True)
        splits = split_by_date(subset, validation_start, test_start)
        train, valid, test = splits["train"], splits["valid"], splits["test"]
        if train.empty or valid.empty or test.empty or train[target].nunique() < 2:
            print("[Skipped] Empty date split or single training class.", flush=True)
            metric_rows.extend(
                _skipped_rows(scope_kind, scope_name, train, valid, test, target)
            )
            continue

        train_y = train[target].astype(int)
        valid_y = valid[target].astype(int)
        test_y = test[target].astype(int)
        prior_score = float(train_y.mean())
        prior_valid_scores = np.full(len(valid), prior_score)
        prior_threshold = choose_threshold(valid_y, prior_valid_scores)
        prior_test_scores = np.full(len(test), prior_score)
        prior_metrics = classification_metrics(test_y, prior_test_scores, prior_threshold)
        metric_rows.append(
            _metrics_row(
                scope_kind=scope_kind,
                scope_name=scope_name,
                model="prior",
                status="ok",
                metrics=prior_metrics,
                threshold=prior_threshold,
                train=train,
                valid=valid,
                test=test,
                target=target,
            )
        )

        model = build_model("hist_gradient_boosting", train[features], max_iter)
        model.fit(train[features], train_y)
        valid_scores = model.predict_proba(valid[features])[:, 1]
        threshold = choose_threshold(valid_y, valid_scores)
        test_scores = model.predict_proba(test[features])[:, 1]
        model_metrics = classification_metrics(test_y, test_scores, threshold)
        metric_rows.append(
            _metrics_row(
                scope_kind=scope_kind,
                scope_name=scope_name,
                model="hist_gradient_boosting",
                status="ok",
                metrics=model_metrics,
                threshold=threshold,
                train=train,
                valid=valid,
                test=test,
                target=target,
            )
        )

        model_file = (
            output_path
            / "models"
            / f"{_safe_name(scope_kind)}__{_safe_name(scope_name)}__hist_gradient_boosting.joblib"
        )
        joblib.dump(
            {
                "pipeline": model,
                "feature_data": features,
                "target_data": target,
                "mode": mode,
                "scope_kind": scope_kind,
                "scope_name": scope_name,
                "threshold": threshold,
            },
            model_file,
        )

        predictions = test[
            [
                column
                for column in [
                    "transaction_date",
                    "label_end_date",
                    "asset_tag",
                    "machine_type",
                    "part_no",
                    target,
                ]
                if column in test.columns
            ]
        ].copy()
        prior_predictions = predictions.copy()
        prior_predictions["scope_kind"] = scope_kind
        prior_predictions["scope_name"] = scope_name
        prior_predictions["model"] = "prior"
        prior_predictions["risk_score"] = prior_test_scores
        prior_predictions["threshold"] = prior_threshold
        prior_predictions["prediction"] = (
            prior_test_scores >= prior_threshold
        ).astype(int)
        predictions["scope_kind"] = scope_kind
        predictions["scope_name"] = scope_name
        predictions["model"] = "hist_gradient_boosting"
        predictions["risk_score"] = test_scores
        predictions["threshold"] = threshold
        predictions["prediction"] = (test_scores >= threshold).astype(int)
        prediction_rows.extend([prior_predictions, predictions])

    metrics_frame = pd.DataFrame(metric_rows)
    metrics_frame.to_csv(output_path / "metrics.csv", index=False)
    if prediction_rows:
        pd.concat(prediction_rows, ignore_index=True).to_csv(
            output_path / "test_predictions.csv", index=False
        )
    else:
        pd.DataFrame().to_csv(output_path / "test_predictions.csv", index=False)
    print(f"[Done] Results: {output_path.resolve()}", flush=True)
    print(metrics_frame.groupby('status').size().to_string(), flush=True)
    return metrics_frame


def _cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="산업 기계 고장 위험 모델 학습")
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--output", default=None)
    parser.add_argument("--mode", choices=["current", "forecast"], default="current")
    parser.add_argument("--scope", choices=["all", "overall", "machine_type", "asset_tag"], default="all")
    parser.add_argument("--machine-type")
    parser.add_argument("--asset-tag")
    parser.add_argument("--max-iter", type=int, default=100)
    return parser


if __name__ == "__main__":
    args = _cli().parse_args()
    from .industrial_features import load_industrial_data, prepare_current, prepare_forecast

    raw = load_industrial_data(args.data)
    if args.mode == "current":
        prepared, feature_data, target_data = prepare_current(raw)
    else:
        prepared, feature_data, target_data = prepare_forecast(raw)
    run_experiments(
        prepared,
        feature_data,
        target_data,
        output_dir=args.output or PROJECT_ROOT / "outputs" / ("current_state" if args.mode == "current" else "timeseries"),
        mode=args.mode,
        scope=args.scope,
        machine_type=args.machine_type,
        asset_tag=args.asset_tag,
        max_iter=args.max_iter,
    )
