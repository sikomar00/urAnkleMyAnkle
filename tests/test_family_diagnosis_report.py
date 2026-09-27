import pandas as pd
import pytest

from src.family_diagnosis_report import (
    build_multilabel_metrics,
    evaluate_sequence_gate,
    render_family_summary,
)


def _multilabel_predictions() -> pd.DataFrame:
    rows = []
    values = {
        "2024-07-01": {
            "Bearing": (1, 1, 0.9),
            "Filter": (0, 1, 0.8),
        },
        "2024-07-02": {
            "Bearing": (0, 0, 0.1),
            "Filter": (1, 0, 0.4),
        },
    }
    for date, families in values.items():
        for family, (actual, prediction, score) in families.items():
            rows.append(
                {
                    "transaction_date": pd.Timestamp(date),
                    "asset_tag": "A-1",
                    "machine_type": "Press",
                    "target": "affected",
                    "threshold_policy": "f1",
                    "part_family": family,
                    "actual": actual,
                    "prediction": prediction,
                    "risk_score": score,
                }
            )
    return pd.DataFrame(rows)


def test_multilabel_metrics_match_hand_calculated_values():
    result = build_multilabel_metrics(_multilabel_predictions()).iloc[0]

    assert result["macro_precision"] == pytest.approx(0.5)
    assert result["macro_recall"] == pytest.approx(0.5)
    assert result["macro_f1"] == pytest.approx(0.5)
    assert result["macro_average_precision"] == pytest.approx(0.75)
    assert result["micro_precision"] == pytest.approx(0.5)
    assert result["micro_recall"] == pytest.approx(0.5)
    assert result["micro_f1"] == pytest.approx(0.5)
    assert result["micro_average_precision"] == pytest.approx(5 / 6)
    assert result["hamming_loss"] == pytest.approx(0.5)
    assert result["exact_match_ratio"] == pytest.approx(0.0)
    assert "Electrical" in result["excluded_families"]


def _gate_metrics(*, test_drop: bool = False, importance: float = 0.1):
    rows = []
    for family_index in range(9):
        family = f"Family-{family_index}"
        for feature_set, valid_ap, test_ap in (
            ("A", 0.40, 0.42),
            ("B", 0.43 if family_index < 6 else 0.40, 0.41 if test_drop else 0.43),
            ("C", 0.39, 0.40),
        ):
            for split, average_precision in (
                ("validation", valid_ap),
                ("test", test_ap),
            ):
                rows.append(
                    {
                        "part_family": family,
                        "target": "affected",
                        "feature_set": feature_set,
                        "model": "logistic_regression",
                        "split": split,
                        "status": "ok",
                        "threshold_policy": "f1",
                        "selected_model": True,
                        "average_precision": average_precision,
                        "temporal_importance": importance if feature_set == "B" else 0.0,
                    }
                )
    return pd.DataFrame(rows)


def test_sequence_gate_requires_all_four_conditions():
    eligible = evaluate_sequence_gate(_gate_metrics())
    test_regression = evaluate_sequence_gate(_gate_metrics(test_drop=True))
    no_temporal_signal = evaluate_sequence_gate(_gate_metrics(importance=0.0))

    assert eligible["eligible"] is True
    assert eligible["selected_temporal_feature_set"] == "B"
    assert eligible["improved_family_count"] == 6
    assert test_regression["eligible"] is False
    assert any("테스트" in reason for reason in test_regression["reasons"])
    assert no_temporal_signal["eligible"] is False
    assert any("중요도" in reason for reason in no_temporal_signal["reasons"])


def test_korean_summary_contains_required_sections_and_limitations():
    metrics = _gate_metrics()
    metrics["precision"] = 0.5
    metrics["recall"] = 0.6
    metrics["f1"] = 0.55
    metrics["threshold_policy"] = "min_precision_70"
    unavailable = metrics.iloc[[0]].copy()
    unavailable["threshold_policy"] = "min_precision_80"
    unavailable["status"] = "unavailable"
    metrics = pd.concat([metrics, unavailable], ignore_index=True)
    multilabel = build_multilabel_metrics(_multilabel_predictions())
    profile = pd.DataFrame(
        {
            "period": ["test"],
            "part_family": ["Bearing"],
            "row_count": [20],
            "affected_rate": [0.5],
            "severe_rate": [0.5],
            "all_failed_rate": [0.1],
        }
    )
    gate = evaluate_sequence_gate(_gate_metrics())

    text = render_family_summary(metrics, multilabel, profile, gate)

    for phrase in (
        "세 줄 요약",
        "한 페이지 요약",
        "Family별 발생률",
        "A·B·C 비교",
        "Precision 0.70·0.80",
        "기계 종류·장비별 결과",
        "순차 모델 판정",
        "합성 데이터",
        "고장 확정이 아닙니다",
    ):
        assert phrase in text
