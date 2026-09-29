"""부품 확률모델의 후보 선택, 재적합과 독립 확률 보정을 수행한다."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.inspection import permutation_importance

from .industrial_data import CURRENT_TARGET
from .industrial_training import MODEL_NAMES, build_model
from .probabilistic_risk_aggregation import (
    aggregate_asset_risk,
    rank_part_probabilities,
)
from .probabilistic_risk_evaluation import build_ranking_metrics
from .probabilistic_risk_evaluation import (
    build_high_risk_metrics,
    build_score_metrics,
    build_severity_metrics,
    choose_fpr_cutoff,
    decide_localization_status,
    paired_hit_rate_bootstrap,
    random_ranking_expectations,
)
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


@dataclass
class ProbabilisticRiskRun:
    """전체 실행의 데이터·모델·지표와 출력 위치를 보관한다."""

    output_dir: Path
    prepared: ProbabilisticRiskFeatures
    selections: dict[str, PartModelSelection]
    models: dict[str, CalibratedPartModel]
    part_probabilities: pd.DataFrame
    asset_predictions: pd.DataFrame
    ranking_metrics: pd.DataFrame
    score_metrics: pd.DataFrame
    high_risk_metrics: pd.DataFrame
    severity_metrics: pd.DataFrame
    confusion_matrices: pd.DataFrame
    calibration_table: pd.DataFrame
    feature_importance: pd.DataFrame
    localization: dict[str, dict[str, Any]]
    config: dict[str, Any]


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


def _asset_score_baselines(
    prepared: ProbabilisticRiskFeatures,
) -> pd.DataFrame:
    reference = pd.concat(
        [prepared.periods["train"], prepared.periods["selection"]],
        ignore_index=True,
    )
    global_mean = float(reference["actual_failure_points"].mean())
    asset_means = reference.groupby("asset_tag")["actual_failure_points"].mean()
    rows: list[pd.DataFrame] = []
    for period_name, period in prepared.periods.items():
        daily = period.drop_duplicates(["transaction_date", "asset_tag"])[
            ["transaction_date", "asset_tag", "machine_type", "actual_failure_points"]
        ].copy()
        for variant, values in (
            ("baseline_global_mean", pd.Series(global_mean, index=daily.index)),
            (
                "baseline_asset_mean",
                daily["asset_tag"].map(asset_means).fillna(global_mean),
            ),
        ):
            result = daily.copy()
            result["model_variant"] = variant
            result["split"] = period_name
            result["expected_failure_points"] = values.to_numpy(dtype=float)
            result["score_error"] = (
                result["actual_failure_points"] - result["expected_failure_points"]
            )
            rows.append(result)
    return pd.concat(rows, ignore_index=True)


def _build_localization_results(
    part_predictions: pd.DataFrame,
    *,
    bootstrap_samples: int,
    random_state: int,
) -> dict[str, dict[str, Any]]:
    ranking = build_ranking_metrics(part_predictions)
    random_expectation = random_ranking_expectations(part_predictions)
    calibration = ranking.loc[
        ranking["split"].eq("calibration") & ranking["scope_kind"].eq("overall")
    ]
    random_calibration = random_expectation.loc[
        random_expectation["split"].eq("calibration")
    ]
    results: dict[str, dict[str, Any]] = {
        "B0": {
            "localization_status": "insufficient_evidence",
            "display_label": "참고용 위험 순위",
            "reasons": ["B0는 비교 기준모델입니다."],
        }
    }
    baseline = part_predictions.loc[
        part_predictions["model_variant"].eq("B0")
        & part_predictions["split"].eq("calibration")
    ]
    base_row = calibration.loc[calibration["model_variant"].eq("B0")].iloc[0]
    random_row = random_calibration.loc[random_calibration["model_variant"].eq("B0")].iloc[0]
    for variant in sorted(
        set(part_predictions["model_variant"]) - {"B0"}
    ):
        model = part_predictions.loc[
            part_predictions["model_variant"].eq(variant)
            & part_predictions["split"].eq("calibration")
        ]
        row = calibration.loc[calibration["model_variant"].eq(variant)].iloc[0]
        bootstrap = paired_hit_rate_bootstrap(
            model,
            baseline,
            k=3,
            samples=bootstrap_samples,
            random_state=random_state,
        )
        results[variant] = decide_localization_status(
            {
                "split": "calibration",
                "model_hit_rate_at_3": float(row["hit_rate_at_3"]),
                "baseline_hit_rate_at_3": float(base_row["hit_rate_at_3"]),
                "random_hit_rate_at_3": float(random_row["expected_hit_rate_at_3"]),
                "model_recall_at_3": float(row["recall_at_3"]),
                "baseline_recall_at_3": float(base_row["recall_at_3"]),
                "bootstrap_ci_lower": float(bootstrap["ci_lower"]),
                "unique_top3_sets": int(row["unique_top3_sets"]),
                **bootstrap,
            }
        )
    return results


def _cutoffs_from_calibration(
    asset_predictions: pd.DataFrame,
    *,
    fpr_policies: tuple[float, ...],
) -> dict[tuple[Any, Any, int, str], dict[str, Any]]:
    cutoffs: dict[tuple[Any, Any, int, str], dict[str, Any]] = {}
    calibration = asset_predictions.loc[asset_predictions["split"].eq("calibration")]
    for (variant, threshold), group in calibration.groupby(
        ["model_variant", "high_risk_threshold"], sort=False
    ):
        y = group["actual_failure_points"].ge(int(threshold)).astype(int)
        probabilities = group[f"prob_ge_{int(threshold)}"]
        for fpr in fpr_policies:
            policy = f"fpr_{int(round(fpr * 100)):02d}"
            setting = choose_fpr_cutoff(y, probabilities, max_fpr=fpr)
            for split in asset_predictions["split"].unique():
                cutoffs[(variant, split, int(threshold), policy)] = setting
    return cutoffs


def _apply_operational_columns(
    asset_predictions: pd.DataFrame,
    cutoffs: dict[tuple[Any, Any, int, str], dict[str, Any]],
    localization: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    result = asset_predictions.copy()
    result["localization_status"] = result["model_variant"].map(
        lambda value: localization.get(value, localization["B0"])["localization_status"]
    )
    result["top3_display_label"] = result["model_variant"].map(
        lambda value: localization.get(value, localization["B0"])["display_label"]
    )
    for key, setting in cutoffs.items():
        variant, split, threshold, policy = key
        mask = (
            result["model_variant"].eq(variant)
            & result["split"].eq(split)
            & result["high_risk_threshold"].eq(threshold)
        )
        probability_column = f"prob_ge_{threshold}"
        if setting.get("cutoff") is None:
            result.loc[mask, f"inspection_recommended_{policy}"] = False
        else:
            result.loc[mask, f"inspection_recommended_{policy}"] = (
                result.loc[mask, probability_column] >= float(setting["cutoff"])
            )
    for column in [column for column in result if column.startswith("inspection_recommended_")]:
        result[column] = result[column].astype("boolean").fillna(False).astype(bool)
    return result


def _source_hash(raw: pd.DataFrame) -> str:
    return hashlib.sha256(
        pd.util.hash_pandas_object(raw, index=True).to_numpy().tobytes()
    ).hexdigest()


def _git_metadata() -> dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": None, "git_dirty": None}
    return {"git_commit": commit, "git_dirty": dirty}


def run_probabilistic_asset_risk(
    raw: pd.DataFrame,
    *,
    output_dir: str | Path,
    selection_start: str = "2024-01-01",
    calibration_start: str = "2024-04-01",
    test_start: str = "2024-07-01",
    high_risk_thresholds: tuple[int, ...] = (12, 13),
    fpr_policies: tuple[float, ...] = (0.05, 0.10),
    bootstrap_samples: int = 2_000,
    max_iter: int = 100,
    random_state: int = 42,
    model_names: Sequence[str] = MODEL_NAMES,
    source_path: str | Path | None = None,
) -> ProbabilisticRiskRun:
    """Feature 준비부터 모델·평가지표·CSV 저장까지 한 번 실행한다."""

    prepared = __import__(
        "src.probabilistic_risk_features", fromlist=["prepare_probabilistic_risk_features"]
    ).prepare_probabilistic_risk_features(
        raw,
        selection_start=selection_start,
        calibration_start=calibration_start,
        test_start=test_start,
    )
    selections: dict[str, PartModelSelection] = {}
    models: dict[str, CalibratedPartModel] = {}
    part_frames: list[pd.DataFrame] = []
    for feature_set in ("A1", "A2"):
        selection = select_part_model(
            prepared,
            feature_set,
            model_names=model_names,
            max_iter=max_iter,
            random_state=random_state,
        )
        selections[feature_set] = selection
        models[feature_set] = fit_calibrated_part_model(
            prepared,
            selection,
            max_iter=max_iter,
            random_state=random_state,
        )
        for split, period in prepared.periods.items():
            part_frames.append(
                predict_part_probabilities(models[feature_set], period, split=split)
            )

    b0 = __import__(
        "src.probabilistic_risk_evaluation", fromlist=["SmoothedBaseRateModel"]
    ).SmoothedBaseRateModel()
    b0.fit(pd.concat([prepared.periods["train"], prepared.periods["selection"]]))
    for split, period in prepared.periods.items():
        b0_frame = period.copy()
        b0_frame["failure_probability"] = b0.predict_proba(period)
        b0_frame["model_variant"] = "B0"
        b0_frame["split"] = split
        part_frames.append(rank_part_probabilities(b0_frame))
    part_probabilities = pd.concat(part_frames, ignore_index=True)
    asset_predictions = aggregate_asset_risk(
        part_probabilities,
        high_risk_thresholds=high_risk_thresholds,
    )
    localization = _build_localization_results(
        part_probabilities,
        bootstrap_samples=bootstrap_samples,
        random_state=random_state,
    )
    cutoffs = _cutoffs_from_calibration(
        asset_predictions,
        fpr_policies=tuple(fpr_policies),
    )
    asset_predictions = _apply_operational_columns(
        asset_predictions, cutoffs, localization
    )

    baseline_scores = _asset_score_baselines(prepared)
    score_input = pd.concat(
        [asset_predictions, baseline_scores], ignore_index=True, sort=False
    )
    ranking_metrics = build_ranking_metrics(part_probabilities)
    score_metrics = build_score_metrics(score_input)
    high_risk_metrics, confusion_high_risk, calibration_table = build_high_risk_metrics(
        asset_predictions, cutoffs
    )
    severity_metrics, confusion_severity = build_severity_metrics(asset_predictions)
    confusion_matrices = pd.concat(
        [confusion_high_risk.assign(confusion_type="high_risk"), confusion_severity.assign(confusion_type="severity")],
        ignore_index=True,
        sort=False,
    )
    feature_importance_rows: list[dict[str, Any]] = []
    for feature_set, model in models.items():
        selection_frame = prepared.periods["selection"]
        importance = permutation_importance(
            model.estimator,
            selection_frame[list(model.features)],
            selection_frame[CURRENT_TARGET].astype(int),
            scoring="average_precision",
            n_repeats=3,
            random_state=random_state,
            n_jobs=-1,
        )
        for feature, mean, std in zip(
            model.features, importance.importances_mean, importance.importances_std, strict=True
        ):
            feature_importance_rows.append(
                {
                    "feature_set": feature_set,
                    "model_name": model.model_name,
                    "feature": feature,
                    "importance_mean": float(mean),
                    "importance_std": float(std),
                    "score": "average_precision",
                    "split": "selection",
                }
            )
    feature_importance = pd.DataFrame(feature_importance_rows)

    output = Path(output_dir)
    (output / "models").mkdir(parents=True, exist_ok=True)
    part_probabilities.to_csv(output / "part_probabilities.csv", index=False)
    asset_predictions.to_csv(output / "asset_risk_predictions.csv", index=False)
    ranking_metrics.to_csv(output / "ranking_metrics.csv", index=False)
    score_metrics.to_csv(output / "score_metrics.csv", index=False)
    high_risk_metrics.to_csv(output / "high_risk_metrics.csv", index=False)
    severity_metrics.to_csv(output / "severity_metrics.csv", index=False)
    confusion_matrices.to_csv(output / "confusion_matrices.csv", index=False)
    calibration_table.to_csv(output / "calibration_table.csv", index=False)
    feature_importance.to_csv(output / "feature_importance.csv", index=False)
    for feature_set, model in models.items():
        joblib.dump(model, output / "models" / f"{feature_set}__{model.model_name}.joblib")

    config = {
        "source_path": str(source_path) if source_path is not None else None,
        "source_sha256": _source_hash(raw),
        "selection_start": selection_start,
        "calibration_start": calibration_start,
        "test_start": test_start,
        "high_risk_thresholds": list(high_risk_thresholds),
        "fpr_policies": list(fpr_policies),
        "bootstrap_samples": bootstrap_samples,
        "max_iter": max_iter,
        "random_state": random_state,
        "feature_sets": {name: list(values) for name, values in prepared.feature_sets.items()},
        "selected_models": {name: value.selected_model for name, value in selections.items()},
        "candidate_models": list(model_names),
        "calibration_method": "sigmoid",
        "b0_prior_strength": 20.0,
        "cutoffs": {"|".join(map(str, key)): value for key, value in cutoffs.items()},
        "localization": localization,
        "python": sys.version,
        "pandas": pd.__version__,
        "sklearn": __import__("sklearn").__version__,
        **_git_metadata(),
    }
    with (output / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2, default=str)

    return ProbabilisticRiskRun(
        output_dir=output,
        prepared=prepared,
        selections=selections,
        models=models,
        part_probabilities=part_probabilities,
        asset_predictions=asset_predictions,
        ranking_metrics=ranking_metrics,
        score_metrics=score_metrics,
        high_risk_metrics=high_risk_metrics,
        severity_metrics=severity_metrics,
        confusion_matrices=confusion_matrices,
        calibration_table=calibration_table,
        feature_importance=feature_importance,
        localization=localization,
        config=config,
    )
