import pytest

from src.industrial_regression import (
    predicted_severity,
    score_regression_metrics,
    severity_classification_metrics,
)


def test_predicted_severity_uses_half_point_boundaries():
    result = predicted_severity([-2.0, 0.49, 0.5, 5.49, 5.5, 11.49, 11.5])

    assert result.tolist() == [
        "normal",
        "normal",
        "caution",
        "caution",
        "risk",
        "risk",
        "high_risk",
    ]


def test_regression_metrics_include_score_and_high_risk_results():
    metrics = score_regression_metrics(
        [0, 4, 8, 12],
        [0.1, 4.4, 8.2, 12.1],
    )

    assert metrics["mae"] == pytest.approx(0.2)
    assert metrics["macro_f1"] == 1.0
    assert metrics["high_risk_recall"] == 1.0


def test_severity_metrics_keep_fixed_class_order_when_class_is_absent():
    metrics = severity_classification_metrics(
        ["normal", "risk"],
        ["normal", "normal"],
    )

    assert metrics["confusion_matrix"] == [
        [1, 0, 0, 0],
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    assert metrics["high_risk_support"] == 0
