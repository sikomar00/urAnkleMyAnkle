"""확률 기반 장비 위험 모델의 기준선, 순위와 통계적 승인 기준을 제공한다."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from math import comb
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)
from .probabilistic_risk_aggregation import SEVERITY_LEVELS

from .industrial_data import ASSET_COLUMN, CURRENT_TARGET, DATE_COLUMN, PART_COLUMN

DAY_KEYS = [DATE_COLUMN, ASSET_COLUMN]


class SmoothedBaseRateModel:
    """장비×부품 과거 고장률을 Beta-binomial 사전분포로 평활한다."""

    def __init__(self, prior_strength: float = 20.0) -> None:
        if not np.isfinite(prior_strength) or prior_strength <= 0:
            raise ValueError("prior_strength는 0보다 큰 유한수여야 합니다.")
        self.prior_strength = float(prior_strength)

    def fit(self, frame: pd.DataFrame) -> "SmoothedBaseRateModel":
        required = {ASSET_COLUMN, PART_COLUMN, CURRENT_TARGET}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"B0 적합에 필요한 컬럼이 없습니다: {missing}")
        target = pd.to_numeric(frame[CURRENT_TARGET], errors="coerce")
        if target.isna().any() or not target.isin([0, 1]).all() or frame.empty:
            raise ValueError("B0 적합 breakdown_flag는 비어 있지 않은 0·1 값이어야 합니다.")
        self.global_rate_ = float(target.mean())
        self.alpha_ = self.prior_strength * self.global_rate_
        self.beta_ = self.prior_strength * (1.0 - self.global_rate_)
        work = frame[[ASSET_COLUMN, PART_COLUMN]].copy()
        work[CURRENT_TARGET] = target.to_numpy(dtype=float)
        grouped = work.groupby([ASSET_COLUMN, PART_COLUMN], dropna=False)[CURRENT_TARGET].agg(
            failures="sum", observations="size"
        )
        grouped["probability"] = (
            grouped["failures"] + self.alpha_
        ) / (grouped["observations"] + self.alpha_ + self.beta_)
        self.rates_ = grouped["probability"].to_dict()
        return self

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "rates_"):
            raise ValueError("B0 모델을 먼저 fit해야 합니다.")
        required = {ASSET_COLUMN, PART_COLUMN}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"B0 예측에 필요한 컬럼이 없습니다: {missing}")
        return np.asarray(
            [
                self.rates_.get((asset, part), self.global_rate_)
                for asset, part in frame[[ASSET_COLUMN, PART_COLUMN]].itertuples(
                    index=False, name=None
                )
            ],
            dtype=float,
        )


def _evaluation_group_columns(frame: pd.DataFrame) -> list[str]:
    columns = [column for column in ("split", "model_variant") if column in frame]
    if "model_variant" not in columns:
        raise ValueError("순위 평가에는 model_variant 컬럼이 필요합니다.")
    return columns


def _validate_ranking_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        *DAY_KEYS,
        "model_variant",
        PART_COLUMN,
        CURRENT_TARGET,
        "probability_rank",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"순위 평가에 필요한 컬럼이 없습니다: {missing}")
    result = frame.copy()
    result[CURRENT_TARGET] = pd.to_numeric(result[CURRENT_TARGET], errors="coerce")
    result["probability_rank"] = pd.to_numeric(
        result["probability_rank"], errors="coerce"
    )
    if (
        result[[CURRENT_TARGET, "probability_rank"]].isna().any().any()
        or not result[CURRENT_TARGET].isin([0, 1]).all()
        or result["probability_rank"].le(0).any()
    ):
        raise ValueError("순위 평가의 정답은 0·1, 순위는 양수여야 합니다.")
    group_columns = [*_evaluation_group_columns(result), *DAY_KEYS]
    for _, day in result.groupby(group_columns, sort=False, dropna=False):
        if day[PART_COLUMN].duplicated().any() or day["probability_rank"].duplicated().any():
            raise ValueError("같은 장비일에 중복 부품 또는 중복 순위가 있습니다.")
    return result


def _group_metadata(group_columns: list[str], keys: Any) -> dict[str, Any]:
    if not isinstance(keys, tuple):
        keys = (keys,)
    return dict(zip(group_columns, keys, strict=True))


def build_ranking_metrics(
    ranked_parts: pd.DataFrame,
    *,
    ks: tuple[int, ...] = (1, 3, 5),
) -> pd.DataFrame:
    """부품순위의 Hit Rate, Recall, Precision, MRR과 편중도를 계산한다."""

    if not ks or any(isinstance(k, bool) or not isinstance(k, int) or k < 1 for k in ks):
        raise ValueError("ks는 1 이상의 정수여야 합니다.")
    ks = tuple(dict.fromkeys(ks))
    frame = _validate_ranking_frame(ranked_parts)
    group_columns = _evaluation_group_columns(frame)
    rows: list[dict[str, Any]] = []
    grouper: str | list[str] = group_columns[0] if len(group_columns) == 1 else group_columns
    for keys, group in frame.groupby(grouper, sort=False, dropna=False):
        metadata = _group_metadata(group_columns, keys)
        daily = list(group.groupby(DAY_KEYS, sort=False, dropna=False))
        positive_days = [day for _, day in daily if day[CURRENT_TARGET].sum() > 0]
        total_positive_parts = int(group[CURRENT_TARGET].sum())
        overall: dict[str, Any] = {
            **metadata,
            "scope_kind": "overall",
            "scope_value": "all",
            "status": "ok" if positive_days else "unavailable_no_positive_days",
            "total_asset_days": len(daily),
            "positive_asset_days": len(positive_days),
            "positive_parts": total_positive_parts,
        }
        for k in ks:
            hit_count = sum(
                int(((day[CURRENT_TARGET] == 1) & (day["probability_rank"] <= k)).any())
                for day in positive_days
            )
            found = sum(
                int(((day[CURRENT_TARGET] == 1) & (day["probability_rank"] <= k)).sum())
                for day in positive_days
            )
            overall[f"hit_rate_at_{k}"] = (
                hit_count / len(positive_days) if positive_days else np.nan
            )
            overall[f"recall_at_{k}"] = (
                found / total_positive_parts if total_positive_parts else np.nan
            )
        recommended = sum(min(3, len(day)) for _, day in daily)
        found_at_3 = sum(
            int(((day[CURRENT_TARGET] == 1) & (day["probability_rank"] <= 3)).sum())
            for _, day in daily
        )
        overall["precision_at_3"] = found_at_3 / recommended if recommended else np.nan
        reciprocal_ranks = []
        for day in positive_days:
            first_rank = float(day.loc[day[CURRENT_TARGET].eq(1), "probability_rank"].min())
            reciprocal_ranks.append(1.0 / first_rank)
        overall["mrr"] = float(np.mean(reciprocal_ranks)) if reciprocal_ranks else np.nan
        top3_sets = [
            tuple(sorted(day.nsmallest(3, "probability_rank")[PART_COLUMN].astype(str)))
            for _, day in daily
        ]
        counts = Counter(top3_sets)
        overall["unique_top3_sets"] = len(counts)
        overall["most_common_top3_set_share"] = (
            max(counts.values()) / len(top3_sets) if top3_sets else np.nan
        )
        rows.append(overall)

        for scope_column, scope_kind in (
            ("criticality", "criticality"),
            ("part_family", "part_family"),
        ):
            if scope_column not in group:
                continue
            failed = group.loc[group[CURRENT_TARGET].eq(1)]
            for scope_value, scoped in failed.groupby(scope_column, sort=True, dropna=False):
                positives = len(scoped)
                rows.append(
                    {
                        **metadata,
                        "scope_kind": scope_kind,
                        "scope_value": scope_value,
                        "status": "ok",
                        "total_asset_days": len(daily),
                        "positive_asset_days": len(positive_days),
                        "positive_parts": positives,
                        "recall_at_3": float(scoped["probability_rank"].le(3).mean()),
                    }
                )
    return pd.DataFrame(rows)


def random_ranking_expectations(
    ranked_parts: pd.DataFrame,
    *,
    ks: tuple[int, ...] = (1, 3, 5),
) -> pd.DataFrame:
    """각 장비일의 후보·고장 부품 수로 무작위 Top K 기대성능을 계산한다."""

    frame = _validate_ranking_frame(ranked_parts)
    group_columns = _evaluation_group_columns(frame)
    rows: list[dict[str, Any]] = []
    grouper: str | list[str] = group_columns[0] if len(group_columns) == 1 else group_columns
    for keys, group in frame.groupby(grouper, sort=False, dropna=False):
        metadata = _group_metadata(group_columns, keys)
        positive_days = [
            day
            for _, day in group.groupby(DAY_KEYS, sort=False, dropna=False)
            if day[CURRENT_TARGET].sum() > 0
        ]
        row: dict[str, Any] = {
            **metadata,
            "status": "ok" if positive_days else "unavailable_no_positive_days",
            "positive_asset_days": len(positive_days),
        }
        for k in ks:
            hit_expectations = []
            recall_numerators = 0.0
            positive_parts = 0
            for day in positive_days:
                n = len(day)
                m = int(day[CURRENT_TARGET].sum())
                selected = min(k, n)
                miss = comb(n - m, selected) / comb(n, selected) if n - m >= selected else 0.0
                hit_expectations.append(1.0 - miss)
                recall_numerators += m * selected / n
                positive_parts += m
            row[f"expected_hit_rate_at_{k}"] = (
                float(np.mean(hit_expectations)) if hit_expectations else np.nan
            )
            row[f"expected_recall_at_{k}"] = (
                recall_numerators / positive_parts if positive_parts else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _daily_hit_map(frame: pd.DataFrame, k: int) -> tuple[dict[tuple[Any, Any], int], dict]:
    validated = _validate_ranking_frame(frame)
    variants = validated["model_variant"].drop_duplicates()
    if len(variants) != 1:
        raise ValueError("paired bootstrap 입력에는 모델변형이 하나만 있어야 합니다.")
    hits: dict[tuple[Any, Any], int] = {}
    failures: dict[tuple[Any, Any], frozenset[str]] = {}
    for keys, day in validated.groupby(DAY_KEYS, sort=False, dropna=False):
        failed = day.loc[day[CURRENT_TARGET].eq(1), PART_COLUMN].astype(str)
        if failed.empty:
            continue
        key = tuple(keys) if isinstance(keys, tuple) else (keys,)
        failures[key] = frozenset(failed)
        hits[key] = int(
            ((day[CURRENT_TARGET] == 1) & (day["probability_rank"] <= k)).any()
        )
    return hits, failures


def paired_hit_rate_bootstrap(
    model_parts: pd.DataFrame,
    baseline_parts: pd.DataFrame,
    *,
    k: int = 3,
    samples: int = 2_000,
    random_state: int = 42,
) -> dict[str, float | int]:
    """같은 장비일의 모델-B0 Hit 차이를 장비일 단위로 재표집한다."""

    if k < 1 or samples < 1:
        raise ValueError("k와 samples는 1 이상이어야 합니다.")
    model_hits, model_failures = _daily_hit_map(model_parts, k)
    baseline_hits, baseline_failures = _daily_hit_map(baseline_parts, k)
    if model_failures != baseline_failures or not model_failures:
        raise ValueError("모델과 B0는 같은 양성 장비일과 실제 고장 부품을 가져야 합니다.")
    keys = sorted(model_failures, key=str)
    differences = np.asarray(
        [model_hits[key] - baseline_hits[key] for key in keys], dtype=float
    )
    rng = np.random.default_rng(random_state)
    indices = rng.integers(0, len(differences), size=(samples, len(differences)))
    bootstrap_means = differences[indices].mean(axis=1)
    return {
        "asset_days": len(differences),
        "samples": samples,
        "mean_difference": float(differences.mean()),
        "ci_lower": float(np.quantile(bootstrap_means, 0.025)),
        "ci_upper": float(np.quantile(bootstrap_means, 0.975)),
    }


def decide_localization_status(calibration_results: dict[str, Any]) -> dict[str, Any]:
    """보정 구간 지표 네 조건으로 부품 위치 특정 표시 권한을 고정한다."""

    if calibration_results.get("split") != "calibration":
        raise ValueError("부품 위치 특정 승인은 보정 구간 결과로만 결정해야 합니다.")
    required = (
        "model_hit_rate_at_3",
        "baseline_hit_rate_at_3",
        "random_hit_rate_at_3",
        "model_recall_at_3",
        "baseline_recall_at_3",
        "bootstrap_ci_lower",
        "unique_top3_sets",
    )
    missing = [key for key in required if key not in calibration_results]
    if missing:
        raise ValueError(f"위치 특정 승인에 필요한 값이 없습니다: {missing}")

    reasons: list[str] = []
    if not (
        calibration_results["model_hit_rate_at_3"]
        > calibration_results["baseline_hit_rate_at_3"]
        and calibration_results["model_hit_rate_at_3"]
        > calibration_results["random_hit_rate_at_3"]
    ):
        reasons.append("Hit Rate@3가 B0와 무작위 기준을 모두 넘지 못했습니다.")
    if calibration_results["bootstrap_ci_lower"] <= 0:
        reasons.append("B0 대비 Hit Rate@3 차이의 95% 신뢰구간 하한이 0보다 크지 않습니다.")
    if calibration_results["model_recall_at_3"] < calibration_results["baseline_recall_at_3"]:
        reasons.append("Recall@3가 B0보다 낮습니다.")
    if calibration_results["unique_top3_sets"] <= 1:
        reasons.append("모든 장비일에 같은 Top 3 조합만 반복했습니다.")

    approved = not reasons
    return {
        "localization_status": (
            "approved_on_validation" if approved else "insufficient_evidence"
        ),
        "display_label": "우선 점검 후보" if approved else "참고용 위험 순위",
        "reasons": reasons,
    }


def choose_fpr_cutoff(
    y_true: Sequence[int],
    probabilities: Sequence[float],
    *,
    max_fpr: float,
) -> dict[str, float | str | None]:
    """음성 장비일 FPR 제한을 만족하는 가장 낮은 확률 cutoff를 고른다."""

    if not 0 < max_fpr <= 1:
        raise ValueError("max_fpr는 0보다 크고 1 이하여야 합니다.")
    y = np.asarray(y_true, dtype=int)
    scores = np.asarray(probabilities, dtype=float)
    if len(y) != len(scores) or not len(y):
        raise ValueError("정답과 확률은 같은 길이의 비어 있지 않은 배열이어야 합니다.")
    if not np.isfinite(scores).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("정답은 0·1, 확률은 유한수여야 합니다.")
    negatives = int((y == 0).sum())
    positives = int((y == 1).sum())
    if negatives == 0:
        return {"cutoff": None, "status": "unavailable", "fpr": None}
    candidates = np.unique(scores)[::-1]
    feasible: list[tuple[float, float, float]] = []
    for cutoff in candidates:
        prediction = scores >= cutoff
        false_positive = int(((y == 0) & prediction).sum())
        true_positive = int(((y == 1) & prediction).sum())
        fpr = false_positive / negatives
        recall = true_positive / positives if positives else 0.0
        if fpr <= max_fpr + 1e-12:
            feasible.append((float(cutoff), fpr, recall))
    if not feasible:
        return {"cutoff": None, "status": "unavailable", "fpr": None}
    # 같은 Recall이면 낮은 cutoff를 택해 제한 안에서 더 많은 후보를 본다.
    cutoff, fpr, _ = sorted(feasible, key=lambda row: (-row[2], row[0]))[0]
    return {"cutoff": cutoff, "status": "ok", "fpr": fpr}


def _safe_correlations(actual: pd.Series, expected: pd.Series) -> dict[str, Any]:
    if actual.nunique(dropna=True) < 2 or expected.nunique(dropna=True) < 2:
        return {
            "pearson": np.nan,
            "spearman": np.nan,
            "pearson_status": "not_defined_constant_input",
            "spearman_status": "not_defined_constant_input",
        }
    actual_values = actual.to_numpy(dtype=float)
    expected_values = expected.to_numpy(dtype=float)
    pearson = float(np.corrcoef(actual_values, expected_values)[0, 1])
    actual_rank = pd.Series(actual_values).rank(method="average").to_numpy()
    expected_rank = pd.Series(expected_values).rank(method="average").to_numpy()
    spearman = float(np.corrcoef(actual_rank, expected_rank)[0, 1])
    return {
        "pearson": pearson,
        "spearman": spearman,
        "pearson_status": "ok",
        "spearman_status": "ok",
    }


def _score_row(frame: pd.DataFrame, metadata: dict[str, Any]) -> dict[str, Any]:
    actual = pd.to_numeric(frame["actual_failure_points"], errors="coerce")
    expected = pd.to_numeric(frame["expected_failure_points"], errors="coerce")
    usable = pd.DataFrame({"actual": actual, "expected": expected}).dropna()
    if usable.empty:
        return {**metadata, "status": "unavailable", "row_count": 0}
    errors = usable["actual"] - usable["expected"]
    result: dict[str, Any] = {
        **metadata,
        "status": "ok",
        "row_count": len(usable),
        "actual_mean": float(usable["actual"].mean()),
        "expected_mean": float(usable["expected"].mean()),
        "score_mae": float(errors.abs().mean()),
        "score_rmse": float(np.sqrt(np.mean(np.square(errors)))),
    }
    result.update(_safe_correlations(usable["actual"], usable["expected"]))
    return result


def build_score_metrics(asset_predictions: pd.DataFrame) -> pd.DataFrame:
    """장비 예상점수의 전체·장비·기계·점수구간별 오차를 계산한다."""

    required = {
        DATE_COLUMN,
        ASSET_COLUMN,
        "model_variant",
        "actual_failure_points",
        "expected_failure_points",
    }
    missing = sorted(required - set(asset_predictions.columns))
    if missing:
        raise ValueError(f"점수 지표에 필요한 컬럼이 없습니다: {missing}")
    frame = asset_predictions.copy()
    if "high_risk_threshold" in frame:
        frame = frame.drop_duplicates(
            [column for column in ["split", "model_variant", DATE_COLUMN, ASSET_COLUMN] if column in frame]
        )
    group_columns = [column for column in ("split", "model_variant") if column in frame]
    if "model_variant" not in group_columns:
        raise ValueError("점수 지표에는 model_variant 컬럼이 필요합니다.")
    rows: list[dict[str, Any]] = []
    grouper: str | list[str] = group_columns[0] if len(group_columns) == 1 else group_columns
    for keys, group in frame.groupby(grouper, sort=False, dropna=False):
        metadata = _group_metadata(group_columns, keys)
        rows.append(_score_row(group, {**metadata, "scope_kind": "overall", "scope_value": "all"}))
        for column, kind in (("machine_type", "machine_type"), (ASSET_COLUMN, "asset_tag")):
            if column not in group:
                continue
            for value, scoped in group.groupby(column, sort=True, dropna=False):
                rows.append(
                    _score_row(
                        scoped,
                        {**metadata, "scope_kind": kind, "scope_value": value},
                    )
                )
        decile_frame = group.copy()
        ranks = decile_frame["expected_failure_points"].rank(method="first", pct=True)
        decile_frame["expected_score_decile"] = np.ceil(ranks * 10).clip(1, 10).astype(int)
        for decile, scoped in decile_frame.groupby("expected_score_decile", sort=True):
            row = _score_row(
                scoped,
                {**metadata, "scope_kind": "expected_score_decile", "scope_value": int(decile)},
            )
            row["actual_mean_in_decile"] = float(scoped["actual_failure_points"].mean())
            rows.append(row)
    return pd.DataFrame(rows)


def _calibration_rows(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    order = np.argsort(probabilities, kind="stable")
    rank_bins = np.ceil((np.arange(len(order)) + 1) * 10 / len(order)).astype(int)
    assigned = np.empty(len(order), dtype=int)
    assigned[order] = rank_bins
    rows = []
    for bin_number in range(1, 11):
        mask = assigned == bin_number
        rows.append(
            {
                **metadata,
                "bin": bin_number,
                "count": int(mask.sum()),
                "predicted_probability": float(probabilities[mask].mean()) if mask.any() else np.nan,
                "actual_rate": float(y_true[mask].mean()) if mask.any() else np.nan,
            }
        )
    return rows


def build_high_risk_metrics(
    asset_predictions: pd.DataFrame,
    cutoffs: dict[tuple[Any, Any, int, str], dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """12·13점 고위험 확률, 정책별 분류지표와 보정표를 만든다."""

    required = {
        DATE_COLUMN,
        ASSET_COLUMN,
        "model_variant",
        "high_risk_threshold",
        "actual_failure_points",
    }
    missing = sorted(required - set(asset_predictions.columns))
    if missing:
        raise ValueError(f"고위험 지표에 필요한 컬럼이 없습니다: {missing}")
    frame = asset_predictions.copy()
    group_columns = [column for column in ("split", "model_variant") if column in frame]
    policies = sorted({key[3] for key in cutoffs})
    metric_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(
        [*group_columns, "high_risk_threshold"], sort=False, dropna=False
    ):
        if not isinstance(keys, tuple):
            keys = (keys,)
        metadata = dict(zip([*group_columns, "high_risk_threshold"], keys, strict=True))
        threshold = int(metadata["high_risk_threshold"])
        probability_column = f"prob_ge_{threshold}"
        if probability_column not in group:
            raise ValueError(f"고위험 확률 컬럼이 없습니다: {probability_column}")
        y_true = group["actual_failure_points"].ge(threshold).astype(int).to_numpy()
        probabilities = group[probability_column].astype(float).to_numpy()
        for policy in policies:
            setting = cutoffs.get(
                (metadata.get("model_variant"), metadata.get("split"), threshold, policy),
                {"cutoff": None, "status": "unavailable"},
            )
            cutoff = setting.get("cutoff")
            status = setting.get("status", "unavailable")
            metrics: dict[str, Any] = {
                **metadata,
                "policy": policy,
                "cutoff": cutoff,
                "status": status,
                "positive_rate": float(y_true.mean()) if len(y_true) else np.nan,
            }
            if len(np.unique(y_true)) == 2:
                metrics.update(
                    {
                        "roc_auc": float(roc_auc_score(y_true, probabilities)),
                        "average_precision": float(average_precision_score(y_true, probabilities)),
                        "brier_score": float(brier_score_loss(y_true, probabilities)),
                        "log_loss": float(log_loss(y_true, probabilities, labels=[0, 1])),
                    }
                )
            else:
                metrics.update(
                    {
                        "roc_auc": np.nan,
                        "average_precision": np.nan,
                        "brier_score": float(brier_score_loss(y_true, probabilities)),
                        "log_loss": np.nan,
                    }
                )
            if cutoff is None:
                metrics.update({
                    "accuracy": np.nan,
                    "precision": np.nan,
                    "recall": np.nan,
                    "f1": np.nan,
                    "fpr": np.nan,
                })
                predictions = np.zeros(len(y_true), dtype=int)
            else:
                predictions = (probabilities >= float(cutoff)).astype(int)
                tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
                metrics.update(
                    {
                        "accuracy": float(accuracy_score(y_true, predictions)),
                        "precision": float(precision_score(y_true, predictions, zero_division=0)),
                        "recall": float(recall_score(y_true, predictions, zero_division=0)),
                        "f1": float(f1_score(y_true, predictions, zero_division=0)),
                        "fpr": float(fp / (fp + tn)) if (fp + tn) else np.nan,
                    }
                )
            tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
            confusion_rows.append(
                {
                    **metadata,
                    "policy": policy,
                    "true_negative": int(tn),
                    "false_positive": int(fp),
                    "false_negative": int(fn),
                    "true_positive": int(tp),
                }
            )
            metric_rows.append(metrics)
        calibration_rows.extend(
            _calibration_rows(
                y_true,
                probabilities,
                {**metadata, "probability_column": probability_column},
            )
        )
    return (
        pd.DataFrame(metric_rows),
        pd.DataFrame(confusion_rows),
        pd.DataFrame(calibration_rows),
    )


def build_severity_metrics(
    asset_predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """4단계 등급의 전체 지표와 4×4 혼동행렬을 계산한다."""

    required = {
        DATE_COLUMN,
        ASSET_COLUMN,
        "model_variant",
        "high_risk_threshold",
        "actual_severity",
        "predicted_severity",
    }
    missing = sorted(required - set(asset_predictions.columns))
    if missing:
        raise ValueError(f"4단계 지표에 필요한 컬럼이 없습니다: {missing}")
    group_columns = [column for column in ("split", "model_variant", "high_risk_threshold") if column in asset_predictions]
    metric_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    for keys, group in asset_predictions.groupby(group_columns, sort=False, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        metadata = dict(zip(group_columns, keys, strict=True))
        actual = pd.Categorical(group["actual_severity"], categories=SEVERITY_LEVELS, ordered=True)
        predicted = pd.Categorical(group["predicted_severity"], categories=SEVERITY_LEVELS, ordered=True)
        y_true = pd.Series(actual).astype(str).to_numpy()
        y_pred = pd.Series(predicted).astype(str).to_numpy()
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=list(SEVERITY_LEVELS),
            zero_division=0,
        )
        metric_rows.append(
            {
                **metadata,
                "status": "ok",
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "macro_precision": float(precision.mean()),
                "macro_recall": float(recall.mean()),
                "macro_f1": float(f1.mean()),
                "weighted_f1": float(f1_score(y_true, y_pred, labels=list(SEVERITY_LEVELS), average="weighted", zero_division=0)),
                **{
                    f"{level}_precision": float(precision[index])
                    for index, level in enumerate(SEVERITY_LEVELS)
                },
                **{
                    f"{level}_recall": float(recall[index])
                    for index, level in enumerate(SEVERITY_LEVELS)
                },
                **{
                    f"{level}_f1": float(f1[index])
                    for index, level in enumerate(SEVERITY_LEVELS)
                },
                **{
                    f"{level}_support": int(support[index])
                    for index, level in enumerate(SEVERITY_LEVELS)
                },
            }
        )
        matrix = confusion_matrix(y_true, y_pred, labels=list(SEVERITY_LEVELS))
        for actual_index, actual_level in enumerate(SEVERITY_LEVELS):
            for predicted_index, predicted_level in enumerate(SEVERITY_LEVELS):
                confusion_rows.append(
                    {
                        **metadata,
                        "actual_severity": actual_level,
                        "predicted_severity": predicted_level,
                        "count": int(matrix[actual_index, predicted_index]),
                    }
                )
    return pd.DataFrame(metric_rows), pd.DataFrame(confusion_rows)
