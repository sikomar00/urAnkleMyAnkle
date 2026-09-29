"""확률 기반 장비 위험 실험을 한글 Markdown으로 요약한다."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .probabilistic_asset_risk import ProbabilisticRiskRun


def _value(value: Any) -> str:
    if pd.isna(value):
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _table(frame: pd.DataFrame, columns: list[str]) -> str:
    available = [column for column in columns if column in frame]
    if frame.empty or not available:
        return "표시할 결과가 없습니다."
    lines = [
        "| " + " | ".join(available) + " |",
        "| " + " | ".join("---" for _ in available) + " |",
    ]
    lines.extend(
        "| " + " | ".join(_value(value) for value in row) + " |"
        for row in frame[available].itertuples(index=False, name=None)
    )
    return "\n".join(lines)


def render_probabilistic_risk_summary(run: ProbabilisticRiskRun) -> str:
    """장비 위험과 부품 위치 특정 결과를 섞지 않는 한글 보고서를 만든다."""

    score = run.score_metrics.copy()
    score_test = score.loc[
        score.get("split", pd.Series(index=score.index)).eq("test")
        & score["scope_kind"].eq("overall")
    ]
    score_view = score_test.loc[
        score_test["model_variant"].isin(["baseline_asset_mean", "A1", "A2"])
    ]
    best_score = score_view.sort_values("score_mae").iloc[0] if not score_view.empty else None
    high = run.high_risk_metrics.copy()
    high_test = high.loc[high.get("split", pd.Series(index=high.index)).eq("test")]
    loc_lines = []
    for variant, result in run.localization.items():
        loc_lines.append(
            f"- `{variant}`: `{result['localization_status']}` — {result['display_label']}"
        )
        if result.get("reasons"):
            loc_lines.extend(f"  - {reason}" for reason in result["reasons"])
        if result.get("localization_status") == "approved_on_validation":
            if result.get("test_reproduced") is False:
                loc_lines.append("  - 검증 구간 승인 결과가 테스트 구간에서 재현되지 않았습니다.")
            elif result.get("test_reproduced") is True:
                loc_lines.append("  - 검증 구간 승인 결과가 테스트 구간에서도 재현되었습니다.")
    selected_models = ", ".join(
        f"{name}={selection.selected_model}"
        for name, selection in run.selections.items()
    )
    best_sentence = (
        "테스트에서 가장 낮은 예상점수 MAE는 "
        f"`{best_score['model_variant']}`의 {_value(best_score['score_mae'])}점입니다."
        if best_score is not None
        else "테스트 점수 MAE를 계산할 수 있는 결과가 없습니다."
    )
    localization_has_failure = any(
        result["localization_status"] == "insufficient_evidence"
        for result in run.localization.values()
    )

    return f"""# 확률 기반 장비 당일 위험 실험 결과

## 세 줄 요약

1. 부품별 보정 고장확률을 중요도 가중합해 장비 예상 고장점수와 12점·13점 이상 확률을 계산했습니다.
2. {best_sentence}
3. Top 3는 별도 승인 조건으로 검증했으며, 장비 위험 추정과 부품 위치 특정은 서로 다른 결론으로 분리했습니다.

## 한 페이지 요약

원본의 같은 장비·날짜 센서 8개가 20개 부품 행에 반복되는 구조를 그대로 유지했습니다. A1은 당일 센서와 정적 부품정보를 사용하고, A2는 현재일 이전의 부품 고장 이력을 추가합니다. B0는 센서 없이 장비×부품 과거 고장률만 사용하는 기준모델입니다.

모델 선택은 `{selected_models}`이며, 선택 구간에서 예상점수 MAE, 12·13점 평균 Brier Score, 부품 PR-AUC 순으로 정했습니다. 선택 후 학습·선택 자료로 재적합하고 보정 구간에서 sigmoid 확률 보정을 적용했습니다. 테스트 자료는 모델·보정기·오경보 기준을 바꾸는 데 사용하지 않았습니다.

부품확률은 가중 Poisson-binomial 점수분포로 변환했습니다. 따라서 `expected_failure_points`는 평균적인 예상 점수이고, `prob_ge_12`와 `prob_ge_13`은 각각 12점·13점 이상이 될 확률입니다. 이 확률은 특정 부품의 확정 고장을 의미하지 않습니다.

## 장비 위험 추정

{_table(score_view, ['model_variant', 'score_mae', 'score_rmse', 'actual_mean', 'expected_mean'])}

테스트 고위험 확률 지표:

{_table(high_test, ['model_variant', 'high_risk_threshold', 'policy', 'average_precision', 'brier_score', 'precision', 'recall', 'f1'])}

이 표의 `fpr_05`, `fpr_10`은 보정 구간의 실제 음성 장비일에서 오경보율을 각각 5%, 10% 이내로 제한해 고정한 정책입니다. 테스트에서 성능이 낮아도 기준을 다시 조정하지 않았습니다.

## 부품 위치 특정

Top 3는 다음 조건을 모두 만족할 때만 `우선 점검 후보`로 표시합니다.

1. B0와 무작위 Top 3보다 Hit Rate@3가 높음
2. B0 대비 paired bootstrap 95% 신뢰구간 하한이 0보다 큼
3. Recall@3가 B0보다 낮지 않음
4. 같은 세 부품만 반복 추천하지 않음

{chr(10).join(loc_lines)}

위 조건을 통과하지 못한 결과는 `참고용 위험 순위`로만 표시합니다. 즉, 확률 순위가 높다는 이유만으로 특정 부품의 고장 원인을 확정하지 않습니다. 정확한 고장 부품을 확정하지 않습니다. 승인된 경우에도 고장 원인 확정은 하지 않습니다.

## 한계와 다음 판단

- 장비 센서가 부품별로 다르지 않으므로 부품 위치 특정은 부품 속성·기저율·과거 이력에 의존할 수 있습니다.
- 가중 점수분포는 부품 고장의 조건부 독립을 근사하므로 고위험 확률이 실제 결합 확률과 다를 수 있습니다.
- 합성 데이터의 결과를 실제 기계 정지율이나 교체 필요성으로 해석하면 안 됩니다.
- 현재 보고서에서 장비 위험 추정이 개선되어도 `부품 특정 근거 부족`이면 점검 순위는 설명용으로만 사용합니다.
- {"일부 또는 전체 모델에서 부품 특정 근거 부족 상태입니다." if localization_has_failure else "부품 위치 특정 승인 상태를 테스트 구간에서도 재현하는지 추가 확인해야 합니다."}

이 결과는 당일 장비 위험을 확률로 표시하기 위한 검증 단계입니다. 미래 1일·3일·7일 고장 예측은 당일 위험 확률이 충분히 검증된 이후의 별도 과제로 남깁니다.
"""
