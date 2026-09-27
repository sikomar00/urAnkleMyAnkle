"""Family 당일 진단의 다중라벨 지표와 한글 결과 보고서를 만든다."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
)

from .family_features import FAMILY_NAMES


def build_multilabel_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """선택 예측을 장비·날짜별 다중라벨 행렬로 묶어 전체 지표를 계산한다."""
    required = {
        "transaction_date",
        "asset_tag",
        "target",
        "threshold_policy",
        "part_family",
        "actual",
        "prediction",
        "risk_score",
    }
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"다중라벨 지표에 필요한 컬럼이 없습니다: {missing}")

    rows: list[dict[str, Any]] = []
    group_columns = ["target", "threshold_policy"]
    index_columns = ["transaction_date", "asset_tag"]
    for keys, group in predictions.groupby(group_columns, sort=False, dropna=False):
        numeric_prediction = pd.to_numeric(group["prediction"], errors="coerce")
        usable = group.loc[numeric_prediction.notna()].copy()
        if usable.empty:
            continue
        actual = usable.pivot(
            index=index_columns, columns="part_family", values="actual"
        )
        predicted = usable.pivot(
            index=index_columns, columns="part_family", values="prediction"
        )
        scores = usable.pivot(
            index=index_columns, columns="part_family", values="risk_score"
        )
        common_columns = [
            family
            for family in FAMILY_NAMES
            if family in actual.columns
            and family in predicted.columns
            and family in scores.columns
        ]
        evaluated = [
            family
            for family in common_columns
            if actual[family].dropna().nunique() == 2
            and not (
                actual[family].isna().any()
                or predicted[family].isna().any()
                or scores[family].isna().any()
            )
        ]
        excluded = [family for family in FAMILY_NAMES if family not in evaluated]
        if not evaluated:
            rows.append(
                {
                    "target": keys[0],
                    "threshold_policy": keys[1],
                    "status": "unavailable",
                    "evaluated_family_count": 0,
                    "excluded_families": ";".join(excluded),
                }
            )
            continue

        y_true = actual[evaluated].astype(int).to_numpy()
        y_pred = predicted[evaluated].astype(int).to_numpy()
        y_score = scores[evaluated].astype(float).to_numpy()
        family_precision = []
        family_recall = []
        family_f1 = []
        family_ap = []
        for index in range(len(evaluated)):
            family_precision.append(
                precision_score(y_true[:, index], y_pred[:, index], zero_division=0)
            )
            family_recall.append(
                recall_score(y_true[:, index], y_pred[:, index], zero_division=0)
            )
            family_f1.append(
                f1_score(y_true[:, index], y_pred[:, index], zero_division=0)
            )
            family_ap.append(average_precision_score(y_true[:, index], y_score[:, index]))
        rows.append(
            {
                "target": keys[0],
                "threshold_policy": keys[1],
                "status": "ok",
                "evaluated_family_count": len(evaluated),
                "excluded_families": ";".join(excluded),
                "macro_precision": float(np.mean(family_precision)),
                "macro_recall": float(np.mean(family_recall)),
                "macro_f1": float(np.mean(family_f1)),
                "macro_average_precision": float(np.mean(family_ap)),
                "micro_precision": float(
                    precision_score(y_true.ravel(), y_pred.ravel(), zero_division=0)
                ),
                "micro_recall": float(
                    recall_score(y_true.ravel(), y_pred.ravel(), zero_division=0)
                ),
                "micro_f1": float(
                    f1_score(y_true.ravel(), y_pred.ravel(), zero_division=0)
                ),
                "micro_average_precision": float(
                    average_precision_score(y_true.ravel(), y_score.ravel())
                ),
                "hamming_loss": float(hamming_loss(y_true, y_pred)),
                "exact_match_ratio": float(np.all(y_true == y_pred, axis=1).mean()),
            }
        )
    return pd.DataFrame(rows)


def evaluate_sequence_gate(metrics: pd.DataFrame) -> dict[str, object]:
    """A 대비 B/C 개선이 순차 모델 검토 조건 네 가지를 충족하는지 판정한다."""
    required = {
        "part_family",
        "target",
        "feature_set",
        "split",
        "average_precision",
    }
    missing = sorted(required - set(metrics.columns))
    if missing:
        raise ValueError(f"순차 모델 판정에 필요한 컬럼이 없습니다: {missing}")
    selected = metrics.loc[metrics["target"].eq("affected")].copy()
    if "scope_kind" in selected:
        selected = selected.loc[selected["scope_kind"].eq("overall")]
    if "threshold_policy" in selected:
        selected = selected.loc[selected["threshold_policy"].eq("f1")]
    if "selected_model" in selected:
        selected = selected.loc[selected["selected_model"].fillna(False)]
    if "status" in selected:
        selected = selected.loc[selected["status"].isin(["ok", "identical_target"])]
    selected = selected.dropna(subset=["average_precision"])
    selected = selected.sort_values("average_precision", ascending=False).drop_duplicates(
        ["part_family", "feature_set", "split"]
    )

    validation = selected.loc[selected["split"].eq("validation")]
    test = selected.loc[selected["split"].eq("test")]
    validation_macro = validation.groupby("feature_set")["average_precision"].mean()
    test_macro = test.groupby("feature_set")["average_precision"].mean()
    reasons: list[str] = []
    temporal_candidates = [
        name for name in ("B", "C") if name in validation_macro.index
    ]
    if "A" not in validation_macro.index or not temporal_candidates:
        return {
            "eligible": False,
            "selected_temporal_feature_set": None,
            "validation_macro_ap_delta": None,
            "test_macro_ap_delta": None,
            "improved_family_count": 0,
            "temporal_importance": 0.0,
            "reasons": ["A와 B/C 검증 지표가 모두 필요합니다."],
        }

    temporal = max(
        temporal_candidates,
        key=lambda name: (validation_macro[name], -temporal_candidates.index(name)),
    )
    validation_delta = float(validation_macro[temporal] - validation_macro["A"])
    if validation_delta < 0.02 - 1e-12:
        reasons.append("검증 Macro AP 개선이 0.02 미만입니다.")

    paired = validation.loc[
        validation["feature_set"].isin(["A", temporal])
    ].pivot(index="part_family", columns="feature_set", values="average_precision")
    paired = paired.dropna(subset=["A", temporal])
    improved_count = int(paired[temporal].gt(paired["A"]).sum())
    if improved_count < 5:
        reasons.append("affected 개선 Family가 5개 미만입니다.")

    test_delta: float | None = None
    if "A" not in test_macro.index or temporal not in test_macro.index:
        reasons.append("A와 선택 시간 Feature의 테스트 Macro AP가 필요합니다.")
    else:
        test_delta = float(test_macro[temporal] - test_macro["A"])
        if test_delta < -1e-12:
            reasons.append("테스트 Macro AP가 정적 A보다 낮습니다.")

    temporal_importance = 0.0
    if "temporal_importance" in selected:
        temporal_importance = float(
            pd.to_numeric(
                selected.loc[
                    selected["feature_set"].eq(temporal), "temporal_importance"
                ],
                errors="coerce",
            ).fillna(0.0).max()
        )
    if temporal_importance <= 0:
        reasons.append("센서 시계열·잔차 Feature 중요도가 0보다 크지 않습니다.")

    return {
        "eligible": not reasons,
        "selected_temporal_feature_set": temporal,
        "validation_macro_ap_delta": validation_delta,
        "test_macro_ap_delta": test_delta,
        "improved_family_count": improved_count,
        "temporal_importance": temporal_importance,
        "reasons": reasons,
    }


def _format_value(value: Any) -> str:
    if pd.isna(value):
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _markdown_table(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    available = [column for column in columns if column in frame.columns]
    if frame.empty or not available:
        return "표시할 데이터가 없습니다."
    view = frame.loc[:, available]
    lines = [
        "| " + " | ".join(available) + " |",
        "| " + " | ".join("---" for _ in available) + " |",
    ]
    lines.extend(
        "| " + " | ".join(_format_value(value) for value in row) + " |"
        for row in view.itertuples(index=False, name=None)
    )
    return "\n".join(lines)


def render_family_summary(
    metrics: pd.DataFrame,
    multilabel_metrics: pd.DataFrame,
    profile: pd.DataFrame,
    sequence_gate: dict[str, object],
) -> str:
    """실험 결과와 제한을 숨김없이 설명하는 한글 Markdown을 만든다."""
    validation_mask = metrics.get(
        "split", pd.Series(index=metrics.index)
    ).eq("validation")
    validation_mask &= metrics.get(
        "scope_kind", pd.Series("overall", index=metrics.index)
    ).eq("overall")
    if "selected_policy" in metrics:
        validation_mask &= ~metrics["selected_policy"].fillna(False)
    validation = metrics.loc[validation_mask]
    precision_mask = metrics.get(
        "threshold_policy", pd.Series(index=metrics.index)
    ).isin(["min_precision_70", "min_precision_80"])
    precision_mask &= metrics.get(
        "scope_kind", pd.Series("overall", index=metrics.index)
    ).eq("overall")
    if "selected_policy" in metrics:
        precision_mask &= metrics["selected_policy"].fillna(False)
    precision_rows = metrics.loc[precision_mask].drop_duplicates(
        ["part_family", "target", "threshold_policy"]
    )
    unavailable = int(precision_rows.get("status", pd.Series(index=precision_rows.index)).eq("unavailable").sum())
    gate_text = "진행 가능" if sequence_gate.get("eligible") else "진행 보류"
    reasons = sequence_gate.get("reasons") or ["조건을 모두 충족했습니다."]

    sections = [
        "# 산업 장비 당일 Family 이상·심각 진단 결과",
        "## 세 줄 요약\n\n"
        "1. 장비·날짜 센서로 9개 Family의 `affected`와 `severe`를 독립 진단했습니다.\n"
        "2. 성능은 Accuracy만이 아니라 Family별 Average Precision과 다중라벨 Macro·Micro 지표로 판단합니다.\n"
        f"3. 순차 모델 판정은 **{gate_text}**이며, 높은 Precision 정책 불가 항목은 {unavailable}건입니다.",
        "## 한 페이지 요약\n\n"
        "이 결과는 당일 센서와 과거 이력으로 어느 Family를 점검할지 범위를 줄이는 실험입니다. "
        "`affected`는 구성품 하나 이상 표시, `severe`는 두 개 이상 또는 A등급 부품 표시입니다. "
        "출력은 Family 점검 우선순위이며 실제 장비 고장 확정이 아닙니다. 모델과 임계값은 검증 구간에서만 "
        "선택했고 테스트 구간은 최종 평가에만 사용했습니다. 낮은 성능과 사용할 수 없는 Precision 정책도 "
        "그대로 기록합니다.",
        "## 1. Family별 발생률\n\n"
        + _markdown_table(
            profile,
            (
                "period",
                "part_family",
                "row_count",
                "affected_rate",
                "severe_rate",
                "all_failed_rate",
            ),
        ),
        "## 2. A·B·C 비교\n\n"
        + _markdown_table(
            validation,
            (
                "part_family",
                "target",
                "feature_set",
                "model",
                "average_precision",
                "precision",
                "recall",
                "f1",
            ),
        ),
        "## 3. Precision 0.70·0.80 가용성\n\n"
        f"검증에서 목표 Precision을 달성하지 못해 `unavailable`로 기록된 행은 {unavailable}건입니다.\n\n"
        + _markdown_table(
            precision_rows,
            (
                "part_family",
                "target",
                "threshold_policy",
                "status",
                "probability_cutoff",
            ),
        ),
        "## 4. 다중라벨 전체 결과\n\n"
        + _markdown_table(
            multilabel_metrics,
            (
                "target",
                "threshold_policy",
                "macro_average_precision",
                "macro_f1",
                "micro_average_precision",
                "micro_f1",
                "hamming_loss",
                "exact_match_ratio",
                "excluded_families",
            ),
        ),
        "## 5. 기계 종류·장비별 결과\n\n"
        "현재 저장 지표에 기계 종류 또는 장비 범위 열이 있으면 아래 표로 표시합니다. "
        "전체 범위 실행만 했다면 별도 범위 결과가 없다는 뜻입니다.\n\n"
        + _markdown_table(
            metrics.loc[
                metrics.get("scope_kind", pd.Series(index=metrics.index)).isin(
                    ["machine_type", "asset_tag"]
                )
                & metrics.get(
                    "threshold_policy", pd.Series(index=metrics.index)
                ).eq("f1")
            ],
            (
                "scope_kind",
                "scope_name",
                "part_family",
                "target",
                "average_precision",
                "precision",
                "recall",
                "f1",
            ),
        ),
        "## 6. 순차 모델 판정\n\n"
        f"판정: **{gate_text}**  \n"
        f"선택 시간 Feature: {sequence_gate.get('selected_temporal_feature_set')}  \n"
        f"검증 Macro AP 차이: {_format_value(sequence_gate.get('validation_macro_ap_delta'))}  \n"
        f"개선 Family 수: {sequence_gate.get('improved_family_count')}  \n"
        "판정 사유:\n"
        + "\n".join(f"- {reason}" for reason in reasons),
        "## 7. 합성 데이터 제한과 다음 단계\n\n"
        "현재 `breakdown_flag`는 합성 데이터의 대리 정답이며 실제 정지 시각·정비 확정·생산 손실이 아닙니다. "
        "따라서 이 결과만으로 부품 교체를 지시하지 않습니다. 당일 Family 식별력이 충분하고 순차 모델 조건을 "
        "통과할 때만 미래 1·3·7일 예측을 별도 설계합니다.",
    ]
    return "\n\n".join(sections) + "\n"
