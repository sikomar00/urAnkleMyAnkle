import pandas as pd

from src.asset_severity_experiments import (
    fit_experiment_model,
    select_candidate,
    select_feature_set,
)


def test_balanced_hist_gradient_boosting_fits_all_classes():
    frame = pd.DataFrame({"signal": range(12)})
    target = pd.Series(
        ["normal"] * 6
        + ["caution"] * 3
        + ["risk"] * 2
        + ["high_risk"]
    )

    model = fit_experiment_model(
        "hist_gradient_boosting_balanced",
        frame,
        target,
        max_iter=10,
        random_state=42,
    )

    assert set(model.named_steps["classifier"].classes_) == set(target)


def test_select_candidate_uses_recall_at_exact_tolerance():
    metrics = pd.DataFrame(
        [
            {
                "model": "random_forest",
                "macro_f1": 0.500,
                "high_risk_recall": 0.40,
                "high_risk_precision": 0.60,
            },
            {
                "model": "hist_gradient_boosting",
                "macro_f1": 0.490,
                "high_risk_recall": 0.70,
                "high_risk_precision": 0.30,
            },
        ]
    )

    assert select_candidate(metrics, tolerance=0.01) == "hist_gradient_boosting"


def test_select_candidate_uses_precision_after_recall_tie():
    metrics = pd.DataFrame(
        [
            {
                "model": "logistic_regression",
                "macro_f1": 0.50,
                "high_risk_recall": 0.70,
                "high_risk_precision": 0.20,
            },
            {
                "model": "random_forest",
                "macro_f1": 0.50,
                "high_risk_recall": 0.70,
                "high_risk_precision": 0.40,
            },
        ]
    )

    assert select_candidate(metrics) == "random_forest"


def test_select_feature_set_uses_same_validation_rule():
    metrics = pd.DataFrame(
        [
            {
                "feature_set": "A",
                "macro_f1": 0.510,
                "high_risk_recall": 0.30,
                "high_risk_precision": 0.60,
            },
            {
                "feature_set": "D",
                "macro_f1": 0.505,
                "high_risk_recall": 0.50,
                "high_risk_precision": 0.40,
            },
        ]
    )

    assert select_feature_set(metrics) == "D"
