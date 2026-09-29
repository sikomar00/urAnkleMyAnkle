from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from src.current_probabilistic_asset_risk import (
    DEFAULT_OUTPUT,
    build_parser,
)
from src.probabilistic_risk_report import render_probabilistic_risk_summary


def fake_run(localization_status: str = "insufficient_evidence"):
    score_metrics = pd.DataFrame(
        [
            {
                "split": "test",
                "model_variant": "baseline_asset_mean",
                "scope_kind": "overall",
                "score_mae": 4.0,
                "score_rmse": 5.0,
            },
            {
                "split": "test",
                "model_variant": "A1",
                "scope_kind": "overall",
                "score_mae": 2.0,
                "score_rmse": 3.0,
            },
            {
                "split": "test",
                "model_variant": "A2",
                "scope_kind": "overall",
                "score_mae": 2.5,
                "score_rmse": 3.5,
            },
        ]
    )
    high_risk_metrics = pd.DataFrame(
        [
            {
                "split": "test",
                "model_variant": "A1",
                "high_risk_threshold": 12,
                "policy": "fpr_05",
                "average_precision": 0.42,
                "brier_score": 0.18,
                "precision": 0.30,
                "recall": 0.50,
                "f1": 0.375,
            }
        ]
    )
    severity_metrics = pd.DataFrame(
        [
            {
                "split": "test",
                "model_variant": "A1",
                "high_risk_threshold": 12,
                "accuracy": 0.55,
                "macro_f1": 0.32,
                "weighted_f1": 0.48,
            }
        ]
    )
    return SimpleNamespace(
        score_metrics=score_metrics,
        high_risk_metrics=high_risk_metrics,
        severity_metrics=severity_metrics,
        ranking_metrics=pd.DataFrame(),
        localization={
            "B0": {
                "localization_status": "insufficient_evidence",
                "display_label": "참고용 위험 순위",
                "reasons": ["B0는 비교 기준모델입니다."],
            },
            "A1": {
                "localization_status": localization_status,
                "display_label": (
                    "우선 점검 후보"
                    if localization_status == "approved_on_validation"
                    else "참고용 위험 순위"
                ),
                "reasons": [] if localization_status == "approved_on_validation" else [
                    "부품 특정 근거가 충분하지 않습니다."
                ],
            },
            "A2": {
                "localization_status": "insufficient_evidence",
                "display_label": "참고용 위험 순위",
                "reasons": ["부품 특정 근거가 충분하지 않습니다."],
            },
        },
        selections={"A1": SimpleNamespace(selected_model="logistic_regression"), "A2": SimpleNamespace(selected_model="random_forest")},
        config={"selection_start": "2024-01-01", "calibration_start": "2024-04-01", "test_start": "2024-07-01"},
    )


def test_report_separates_equipment_risk_from_part_localization():
    report = render_probabilistic_risk_summary(fake_run())

    for heading in (
        "세 줄 요약",
        "한 페이지 요약",
        "장비 위험 추정",
        "부품 위치 특정",
        "한계와 다음 판단",
    ):
        assert heading in report
    assert "예상 점수" in report
    assert "참고용 위험 순위" in report
    assert "부품 특정 근거 부족" in report
    assert "정확한 고장 부품을 확정하지 않습니다" in report


def test_report_can_describe_approved_localization_without_overclaiming():
    report = render_probabilistic_risk_summary(
        fake_run(localization_status="approved_on_validation")
    )
    assert "우선 점검 후보" in report
    assert "고장 원인 확정" in report


def test_probabilistic_cli_defaults_and_rejects_invalid_values():
    args = build_parser().parse_args([])
    assert args.data.name == "synthetic_industrial_machine_data.csv"
    assert args.output == DEFAULT_OUTPUT
    assert args.high_risk_thresholds == [12, 13]
    assert args.fpr_policies == [0.05, 0.10]
    assert args.bootstrap_samples == 2_000
    assert args.max_iter == 100
    assert args.random_state == 42

    with pytest.raises(SystemExit):
        build_parser().parse_args(["--fpr-policies", "1.2"])
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--high-risk-thresholds", "6"])

