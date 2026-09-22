"""Leakage-safe classifiers and scope-wide experiment runner."""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
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

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "src"

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


def classification_metrics(
    y_true: Any, scores: Any, threshold: float
) -> dict[str, float | int | None]:
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
        classifier = LogisticRegression(max_iter=max(1000, max_iter), random_state=random_state)
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
