import joblib
import numpy as np
import pandas as pd
import pytest

from src.asset_features import SEVERITY_LEVELS
from src.asset_severity_experiments import (
    fit_experiment_model,
    run_asset_severity_experiments,
    select_candidate,
    select_feature_set,
)
from src.industrial_data import SENSOR_COLUMNS


@pytest.fixture
def experiment_raw():
    rows = []
    dates = [
        *pd.date_range("2023-12-01", periods=12, freq="D"),
        *pd.date_range("2024-01-01", periods=12, freq="D"),
        *pd.date_range("2024-07-01", periods=12, freq="D"),
    ]
    for day, date in enumerate(dates):
        failure_count = day % 4
        for part_index in range(3):
            row = {
                "transaction_date": date,
                "asset_tag": "A-1",
                "machine_type": "Press",
                "plant_code": "P1",
                "part_no": f"P-A-{part_index}",
                "criticality": "A",
                "breakdown_flag": int(part_index < failure_count),
            }
            for sensor_index, sensor in enumerate(SENSOR_COLUMNS):
                row[sensor] = float(sensor_index + day + failure_count)
            rows.append(row)
    return pd.DataFrame(rows)


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


def test_runner_writes_threshold_outputs(tmp_path, experiment_raw):
    metrics = run_asset_severity_experiments(
        experiment_raw,
        output_dir=tmp_path,
        high_risk_thresholds=(12, 13),
        feature_sets=("A",),
        scope="overall",
        min_normal_rows=2,
        max_iter=10,
    )

    assert set(metrics.high_risk_threshold) == {12, 13}
    required = {
        "experiment_metrics.csv",
        "class_metrics.csv",
        "confusion_matrices.csv",
        "test_predictions.csv",
        "feature_importance.csv",
        "zscore_baselines.csv",
        "machine_failure_profile.csv",
        "experiment_summary.md",
        "run_config.json",
        "models",
    }
    assert required <= {path.name for path in tmp_path.iterdir()}
    predictions = pd.read_csv(tmp_path / "test_predictions.csv")
    score_12 = predictions.loc[predictions.actual_failure_points.eq(12)]
    assert set(
        score_12.loc[
            score_12.high_risk_threshold.eq(12), "actual_level"
        ]
    ) == {"high_risk"}
    assert set(
        score_12.loc[
            score_12.high_risk_threshold.eq(13), "actual_level"
        ]
    ) == {"risk"}


def test_saved_model_reproduces_ordered_probabilities(tmp_path, experiment_raw):
    run_asset_severity_experiments(
        experiment_raw,
        output_dir=tmp_path,
        high_risk_thresholds=(13,),
        feature_sets=("A",),
        scope="overall",
        min_normal_rows=2,
        max_iter=10,
    )
    model_file = next(
        (tmp_path / "models").glob("severity__13__A__overall__*.joblib")
    )
    payload = joblib.load(model_file)

    assert payload["class_order"] == list(SEVERITY_LEVELS)
    assert np.allclose(
        payload["pipeline"].predict_proba(payload["verification_rows"]),
        payload["verification_probabilities"],
    )


@pytest.mark.parametrize(
    ("thresholds", "feature_sets", "message"),
    [
        ((6,), ("A",), "7 이상의 정수"),
        ((12,), ("E",), "A, B, C, D"),
    ],
)
def test_runner_rejects_invalid_experiment_options(
    tmp_path,
    experiment_raw,
    thresholds,
    feature_sets,
    message,
):
    with pytest.raises(ValueError, match=message):
        run_asset_severity_experiments(
            experiment_raw,
            output_dir=tmp_path,
            high_risk_thresholds=thresholds,
            feature_sets=feature_sets,
            scope="overall",
            min_normal_rows=2,
            max_iter=10,
        )


def test_runner_records_single_training_class_as_skipped(
    tmp_path,
    experiment_raw,
):
    train = pd.to_datetime(experiment_raw.transaction_date).lt("2024-01-01")
    experiment_raw.loc[train, "breakdown_flag"] = 0

    metrics = run_asset_severity_experiments(
        experiment_raw,
        output_dir=tmp_path,
        high_risk_thresholds=(12,),
        feature_sets=("A",),
        scope="overall",
        min_normal_rows=2,
        max_iter=10,
    )

    assert set(metrics.status) == {"skipped"}
    assert metrics.reason.str.contains("single train class").all()
