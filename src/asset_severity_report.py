"""장비 위험도 실험의 발생률 프로파일과 한글 요약을 만든다."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd

from .industrial_data import ASSET_COLUMN, CURRENT_TARGET, DATE_COLUMN, MACHINE_COLUMN

PROFILE_COLUMNS = (
    "split",
    "scope_kind",
    "scope_name",
    "high_risk_threshold",
    "part_rows",
    "asset_days",
    "part_breakdown_rate",
    "asset_issue_day_rate",
    "asset_high_risk_day_rate",
    "mean_failure_points",
    "median_failure_points",
)


def _split_name(dates: pd.Series, validation_start: str, test_start: str) -> pd.Series:
    values = pd.to_datetime(dates)
    validation = pd.Timestamp(validation_start)
    test = pd.Timestamp(test_start)
    if validation >= test:
        raise ValueError("validation_start는 test_start보다 빨라야 합니다.")
    result = pd.Series("train", index=dates.index, dtype="string")
    result.loc[values.ge(validation) & values.lt(test)] = "validation"
    result.loc[values.ge(test)] = "test"
    return result


def _scope_values(frame: pd.DataFrame, column: str) -> list[Any]:
    return sorted(frame[column].dropna().unique(), key=str)


def build_failure_profile(
    raw_rows: pd.DataFrame,
    daily_rows: pd.DataFrame,
    *,
    high_risk_thresholds: Sequence[int] = (12, 13),
    validation_start: str = "2024-01-01",
    test_start: str = "2024-07-01",
) -> pd.DataFrame:
    """분할·기계·장비별로 서로 다른 세 가지 발생률을 계산한다.

    ``part_breakdown_rate``는 원본 부품 행 중 고장 플래그 비율이고,
    ``asset_issue_day_rate``는 고장점수가 1 이상인 장비일 비율이며,
    ``asset_high_risk_day_rate``는 해당 임계값 이상인 장비일 비율이다.
    """
    required_raw = {DATE_COLUMN, MACHINE_COLUMN, ASSET_COLUMN, CURRENT_TARGET}
    required_daily = {
        DATE_COLUMN,
        MACHINE_COLUMN,
        ASSET_COLUMN,
        "failure_points",
    }
    missing_raw = sorted(required_raw - set(raw_rows.columns))
    missing_daily = sorted(required_daily - set(daily_rows.columns))
    if missing_raw:
        raise ValueError(f"원본 발생률 계산 컬럼이 없습니다: {missing_raw}")
    if missing_daily:
        raise ValueError(f"장비일 발생률 계산 컬럼이 없습니다: {missing_daily}")

    raw = raw_rows.copy()
    daily = daily_rows.copy()
    raw["split"] = _split_name(raw[DATE_COLUMN], validation_start, test_start)
    daily["split"] = _split_name(daily[DATE_COLUMN], validation_start, test_start)
    raw[CURRENT_TARGET] = pd.to_numeric(raw[CURRENT_TARGET], errors="coerce")
    daily["failure_points"] = pd.to_numeric(
        daily["failure_points"], errors="coerce"
    )

    rows: list[dict[str, Any]] = []
    for threshold in dict.fromkeys(high_risk_thresholds):
        for split in ("train", "validation", "test"):
            raw_split = raw.loc[raw["split"].eq(split)]
            daily_split = daily.loc[daily["split"].eq(split)]
            for scope_kind, column in (
                ("machine_type", MACHINE_COLUMN),
                ("asset_tag", ASSET_COLUMN),
            ):
                scope_names = sorted(
                    set(_scope_values(raw_split, column))
                    | set(_scope_values(daily_split, column)),
                    key=str,
                )
                for scope_name in scope_names:
                    raw_scope = raw_split.loc[raw_split[column].eq(scope_name)]
                    daily_scope = daily_split.loc[daily_split[column].eq(scope_name)]
                    points = daily_scope["failure_points"]
                    rows.append(
                        {
                            "split": split,
                            "scope_kind": scope_kind,
                            "scope_name": str(scope_name),
                            "high_risk_threshold": int(threshold),
                            "part_rows": int(len(raw_scope)),
                            "asset_days": int(len(daily_scope)),
                            "part_breakdown_rate": float(
                                raw_scope[CURRENT_TARGET].mean()
                            ),
                            "asset_issue_day_rate": float(points.ge(1).mean()),
                            "asset_high_risk_day_rate": float(
                                points.ge(threshold).mean()
                            ),
                            "mean_failure_points": float(points.mean()),
                            "median_failure_points": float(points.median()),
                        }
                    )
    return pd.DataFrame(rows).reindex(columns=PROFILE_COLUMNS)


def _format_value(value: Any) -> str:
    if pd.isna(value):
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _markdown_table(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    if frame.empty:
        return "데이터가 없습니다."
    view = frame.loc[:, [column for column in columns if column in frame]].copy()
    header = "| " + " | ".join(view.columns) + " |"
    rule = "| " + " | ".join("---" for _ in view.columns) + " |"
    lines = [header, rule]
    for row in view.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(_format_value(value) for value in row) + " |")
    return "\n".join(lines)


def _selected_test_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return metrics.copy()
    mask = metrics.get("split", pd.Series(index=metrics.index)).eq("test")
    mask &= metrics.get("status", pd.Series(index=metrics.index)).eq("ok")
    if "selected_model" in metrics:
        mask &= metrics["selected_model"].eq(True)  # noqa: E712
    return metrics.loc[mask].copy()


def render_experiment_summary(
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    profile: pd.DataFrame,
) -> str:
    """CSV 결과를 사람이 검토하기 쉬운 한글 Markdown 보고서로 변환한다."""
    selected = _selected_test_metrics(metrics)
    overall = selected.loc[
        selected.get("scope_kind", pd.Series(index=selected.index)).eq("overall")
    ].copy()
    representative = overall.loc[
        overall.get(
            "selected_feature_set", pd.Series(False, index=overall.index)
        ).eq(True)  # noqa: E712
    ].copy()

    profile_test = profile.loc[profile["split"].eq("test")].copy()
    machine_profile = profile_test.loc[
        profile_test["scope_kind"].eq("machine_type")
    ]
    asset_profile = profile_test.loc[profile_test["scope_kind"].eq("asset_tag")]

    feature_view = overall.sort_values(["high_risk_threshold", "feature_set"])
    machine_view = selected.loc[
        selected.get("scope_kind", pd.Series(index=selected.index)).eq(
            "machine_type"
        )
        & selected.get(
            "selected_feature_set", pd.Series(False, index=selected.index)
        ).eq(True)  # noqa: E712
    ].sort_values(["high_risk_threshold", "scope_name"])

    selected_predictions = predictions.copy()
    if not selected_predictions.empty and "selected_feature_set" in selected_predictions:
        selected_predictions = selected_predictions.loc[
            selected_predictions["selected_feature_set"].eq(True)  # noqa: E712
        ]
    score_12 = selected_predictions.loc[
        selected_predictions.get(
            "actual_failure_points",
            pd.Series(index=selected_predictions.index, dtype=float),
        ).eq(12)
    ]
    score_12_counts = (
        score_12.groupby(
            ["high_risk_threshold", "actual_level", "predicted_level"],
            dropna=False,
        )
        .size()
        .rename("장비일 수")
        .reset_index()
        if not score_12.empty
        else pd.DataFrame()
    )

    threshold_distribution = representative[
        [
            column
            for column in (
                "high_risk_threshold",
                "feature_set",
                "model",
                "accuracy",
                "macro_f1",
                "high_risk_precision",
                "high_risk_recall",
            )
            if column in representative
        ]
    ].sort_values("high_risk_threshold") if not representative.empty else representative

    delta_text = "A와 D를 비교할 수 있는 전체 테스트 결과가 없습니다."
    if not feature_view.empty and {"A", "D"} <= set(feature_view["feature_set"]):
        changes = []
        for threshold, group in feature_view.groupby("high_risk_threshold"):
            indexed = group.drop_duplicates("feature_set").set_index("feature_set")
            if {"A", "D"} <= set(indexed.index):
                delta = float(indexed.loc["D", "macro_f1"] - indexed.loc["A", "macro_f1"])
                direction = "증가" if delta > 0 else "감소" if delta < 0 else "동일"
                changes.append(f"{int(threshold)}점 기준 D-A Macro F1: {delta:+.4f} ({direction})")
        if changes:
            delta_text = "; ".join(changes)

    return "\n\n".join(
        [
            "# 산업 장비 4단계 위험도 실험 결과",
            "## 1. 해석 전 주의사항\n\n"
            "이 실험의 정답은 부품별 `breakdown_flag`에 중요도 가중치를 적용한 "
            "대리 점수입니다. 따라서 아래 고위험률은 실제 기계 정지율이 아닙니다. "
            "모델 성능은 Accuracy 하나가 아니라 Macro F1과 고위험 Precision·Recall을 함께 봅니다.",
            "## 2. 12점·13점 등급 분포\n\n"
            "12점 기준에서는 12점 이상을 고위험, 13점 기준에서는 13점 이상을 "
            "고위험으로 둡니다. 같은 12점 장비일은 전자에서 고위험, 후자에서 위험입니다.\n\n"
            + _markdown_table(
                threshold_distribution,
                (
                    "high_risk_threshold",
                    "feature_set",
                    "model",
                    "accuracy",
                    "macro_f1",
                    "high_risk_precision",
                    "high_risk_recall",
                ),
            ),
            "## 3. Feature A~D 성능 비교\n\n"
            + _markdown_table(
                feature_view,
                (
                    "high_risk_threshold",
                    "feature_set",
                    "model",
                    "accuracy",
                    "macro_f1",
                    "high_risk_precision",
                    "high_risk_recall",
                ),
            ),
            "## 4. 기계 종류별 결과\n\n"
            + _markdown_table(
                machine_view,
                (
                    "high_risk_threshold",
                    "scope_name",
                    "feature_set",
                    "model",
                    "macro_f1",
                    "high_risk_precision",
                    "high_risk_recall",
                ),
            )
            + "\n\n발생률 정의를 분리한 테스트 프로파일:\n\n"
            + _markdown_table(
                machine_profile,
                (
                    "high_risk_threshold",
                    "scope_name",
                    "part_breakdown_rate",
                    "asset_issue_day_rate",
                    "asset_high_risk_day_rate",
                ),
            ),
            "## 5. 개별 장비별 발생률\n\n"
            + _markdown_table(
                asset_profile,
                (
                    "high_risk_threshold",
                    "scope_name",
                    "part_breakdown_rate",
                    "asset_issue_day_rate",
                    "asset_high_risk_day_rate",
                    "mean_failure_points",
                ),
            ),
            "## 6. 12점 장비일 분석\n\n"
            "임계값 변경의 영향을 직접 확인하기 위해 실제 점수가 정확히 12점인 "
            "장비일의 정답·예측 조합을 집계했습니다.\n\n"
            + _markdown_table(
                score_12_counts,
                (
                    "high_risk_threshold",
                    "actual_level",
                    "predicted_level",
                    "장비일 수",
                ),
            ),
            "## 7. 다음 실험 권고\n\n"
            f"{delta_text}. D가 A보다 개선되지 않으면 이상치·이력 Feature 자체보다 "
            "대리 정답과 센서의 연결이 약한지 먼저 확인해야 합니다. 실제 정지·정비 "
            "이력 확보 전에는 높은 점수를 실제 고장 확률로 해석하지 않습니다.",
        ]
    ) + "\n"
