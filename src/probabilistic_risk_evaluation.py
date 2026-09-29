"""확률 기반 장비 위험 모델의 기준선, 순위와 통계적 승인 기준을 제공한다."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from math import comb
from typing import Any

import numpy as np
import pandas as pd

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
