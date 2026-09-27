import warnings

import joblib
import numpy as np
import pandas as pd
import pytest
from scipy.stats import ConstantInputWarning
from sklearn.dummy import DummyRegressor

from src.asset_features import SEVERITY_LEVELS
from src.industrial_data import PreparedRegressionTask, PreparedTask
from src.industrial_regression import (
    NonnegativeRegressor,
    predicted_severity,
    run_asset_score_suite,
    score_regression_metrics,
    severity_classification_metrics,
)


def _score_tasks(single_train_class: bool = False):
    dates = pd.to_datetime(
        [
            "2023-12-01",
            "2023-12-02",
            "2023-12-03",
            "2023-12-04",
            "2024-01-01",
            "2024-01-02",
            "2024-01-03",
            "2024-01-04",
            "2024-07-01",
            "2024-07-02",
            "2024-07-03",
            "2024-07-04",
        ]
    )
    points = [0, 3, 8, 12] * 3
    if single_train_class:
        points[:4] = [0, 0, 0, 0]
    frame = pd.DataFrame(
        {
            "transaction_date": dates,
            "label_end_date": dates,
            "machine_type": ["Press"] * 12,
            "asset_tag": ["A-1", "A-2"] * 6,
            "signal": np.arange(12, dtype=float),
            "failure_points": points,
        }
    )
    frame["severity_level"] = np.select(
        [
            frame.failure_points.eq(0),
            frame.failure_points.le(5),
            frame.failure_points.le(11),
        ],
        ["normal", "caution", "risk"],
        default="high_risk",
    )
    features = ("machine_type", "asset_tag", "signal")
    return (
        PreparedRegressionTask(
            frame,
            features,
            "failure_points",
            "asset",
            "current",
        ),
        PreparedTask(
            frame,
            features,
            "severity_level",
            "asset",
            "current",
            risk_definition="severity_4class",
        ),
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


def test_regression_metrics_skip_spearman_for_constant_prediction():
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        metrics = score_regression_metrics([0, 4, 8, 12], [6, 6, 6, 6])

    assert metrics["spearman"] is None
    assert not any(isinstance(item.message, ConstantInputWarning) for item in captured)


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


def test_asset_score_suite_writes_all_outputs(tmp_path):
    score_task, severity_task = _score_tasks()

    regression, severity = run_asset_score_suite(
        score_task,
        severity_task,
        output_dir=tmp_path,
        scope="overall",
        max_iter=10,
    )

    assert regression.query("selected_model == True").shape[0] == 2
    assert severity.query("selected_model == True").shape[0] == 2
    assert {
        "regression_metrics.csv",
        "severity_metrics.csv",
        "test_predictions.csv",
        "feature_importance.csv",
        "run_config.json",
    } <= {path.name for path in tmp_path.iterdir()}
    predictions = pd.read_csv(tmp_path / "test_predictions.csv")
    assert {
        "actual_failure_points",
        "predicted_failure_points",
        "actual_severity_level",
        "regression_severity_level",
        "predicted_severity_level",
        "prob_normal",
        "prob_caution",
        "prob_risk",
        "prob_high_risk",
    } <= set(predictions)


def test_asset_score_suite_reports_dummy_spearman_once(tmp_path, capsys):
    score_task, severity_task = _score_tasks()

    run_asset_score_suite(
        score_task,
        severity_task,
        output_dir=tmp_path,
        scope="overall",
        max_iter=10,
    )

    output = capsys.readouterr().out
    assert output.count("DummyRegressor Spearman") == 1


def test_asset_score_suite_records_single_train_class_as_skipped(tmp_path):
    score_task, severity_task = _score_tasks(single_train_class=True)

    _, severity = run_asset_score_suite(
        score_task,
        severity_task,
        output_dir=tmp_path,
        scope="overall",
        max_iter=10,
    )

    assert set(severity["status"]) == {"skipped"}
    assert "single train class" in severity.iloc[0]["reason"]
    predictions = pd.read_csv(tmp_path / "test_predictions.csv")
    assert len(predictions) == 4
    assert predictions["predicted_failure_points"].notna().all()
    assert predictions["predicted_severity_level"].isna().all()
    assert predictions[
        ["prob_normal", "prob_caution", "prob_risk", "prob_high_risk"]
    ].isna().all().all()


def test_saved_regression_predictor_clips_negative_predictions():
    raw = DummyRegressor(strategy="constant", constant=-1.5)
    raw.fit([[0.0], [1.0]], [0.0, 1.0])
    predictor = NonnegativeRegressor(raw)

    assert predictor.predict([[2.0]]).tolist() == [0.0]


def test_saved_asset_score_models_reproduce_predictions_and_class_order(tmp_path):
    score_task, severity_task = _score_tasks()
    run_asset_score_suite(
        score_task,
        severity_task,
        output_dir=tmp_path,
        scope="overall",
        max_iter=10,
    )
    regression_file = next((tmp_path / "models").glob("regression__*.joblib"))
    severity_file = next((tmp_path / "models").glob("severity__*.joblib"))
    regression_payload = joblib.load(regression_file)
    severity_payload = joblib.load(severity_file)
    rows = score_task.frame.loc[:, list(score_task.features)].head(3)
    test_rows = score_task.frame.loc[
        score_task.frame["transaction_date"].ge("2024-07-01")
    ]
    saved_predictions = pd.read_csv(tmp_path / "test_predictions.csv")

    assert np.allclose(
        regression_payload["pipeline"].predict(rows),
        regression_payload["verification_predictions"],
    )
    assert severity_payload["class_order"] == list(SEVERITY_LEVELS)
    assert severity_payload["pipeline"].classes_.tolist() == list(SEVERITY_LEVELS)
    assert np.allclose(
        severity_payload["pipeline"].predict_proba(rows),
        severity_payload["verification_probabilities"],
    )
    assert np.allclose(
        regression_payload["pipeline"].predict(
            test_rows.loc[:, list(score_task.features)]
        ),
        saved_predictions["predicted_failure_points"],
    )
    assert np.allclose(
        severity_payload["pipeline"].predict_proba(
            test_rows.loc[:, list(severity_task.features)]
        ),
        saved_predictions[
            ["prob_normal", "prob_caution", "prob_risk", "prob_high_risk"]
        ],
    )
