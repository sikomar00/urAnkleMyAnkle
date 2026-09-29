from dataclasses import replace

import joblib
import numpy as np
import pandas as pd
import pytest

from src.industrial_data import SENSOR_COLUMNS
from src.probabilistic_asset_risk import (
    choose_model_from_selection_metrics,
    fit_calibrated_part_model,
    predict_part_probabilities,
    run_probabilistic_asset_risk,
    select_part_model,
)
from src.probabilistic_risk_features import (
    ProbabilisticRiskFeatures,
    prepare_probabilistic_risk_features,
)


def raw_model_rows() -> pd.DataFrame:
    rows = []
    criticalities = ["A"] * 6 + ["B"] * 9 + ["C"] * 5
    for day_index, date in enumerate(pd.date_range("2023-01-01", periods=80)):
        sensor_values = {
            column: float(20 + sensor_index + day_index / 10)
            for sensor_index, column in enumerate(SENSOR_COLUMNS)
        }
        for part_index in range(20):
            rows.append(
                {
                    "transaction_date": date,
                    "machine_type": "Machine-1",
                    "asset_tag": "AST-1",
                    "plant_code": "Plant-1",
                    "part_no": f"P{part_index + 1:02d}",
                    "part_family": f"Family-{part_index // 4 + 1}",
                    "criticality": criticalities[part_index],
                    "breakdown_flag": int((day_index + part_index * 2) % 9 == 0),
                    **sensor_values,
                }
            )
    return pd.DataFrame(rows)


def prepared_model_rows() -> ProbabilisticRiskFeatures:
    return prepare_probabilistic_risk_features(
        raw_model_rows(),
        selection_start="2023-01-21",
        calibration_start="2023-02-10",
        test_start="2023-03-02",
    )


def test_choose_model_prioritizes_asset_score_over_top3():
    metrics = pd.DataFrame(
        [
            {
                "model_name": "logistic_regression",
                "score_mae": 2.0,
                "mean_high_risk_brier": 0.10,
                "part_average_precision": 0.90,
                "hit_rate_at_3": 0.95,
            },
            {
                "model_name": "hist_gradient_boosting",
                "score_mae": 1.0,
                "mean_high_risk_brier": 0.30,
                "part_average_precision": 0.20,
                "hit_rate_at_3": 0.20,
            },
            {
                "model_name": "random_forest",
                "score_mae": 1.0,
                "mean_high_risk_brier": 0.20,
                "part_average_precision": 0.10,
                "hit_rate_at_3": 0.10,
            },
        ]
    )

    selected = choose_model_from_selection_metrics(metrics)

    assert selected == "random_forest"


def test_choose_model_uses_fixed_order_after_metric_tie():
    metrics = pd.DataFrame(
        {
            "model_name": [
                "random_forest",
                "hist_gradient_boosting",
                "logistic_regression",
            ],
            "score_mae": [1.0, 1.0, 1.0],
            "mean_high_risk_brier": [0.2, 0.2, 0.2],
            "part_average_precision": [0.3, 0.3, 0.3],
            "hit_rate_at_3": [0.9, 0.1, 0.0],
        }
    )

    assert choose_model_from_selection_metrics(metrics) == "logistic_regression"


def test_select_fit_calibrate_predict_and_reload_are_reproducible(tmp_path):
    prepared = prepared_model_rows()
    selection = select_part_model(
        prepared,
        "A1",
        model_names=("logistic_regression",),
        max_iter=30,
        random_state=7,
    )
    assert selection.selected_model == "logistic_regression"
    assert set(selection.selection_metrics["split"]) == {"selection"}
    assert {
        "score_mae",
        "mean_high_risk_brier",
        "part_average_precision",
        "hit_rate_at_3",
    }.issubset(selection.selection_metrics.columns)

    fitted = fit_calibrated_part_model(prepared, selection)
    predictions = predict_part_probabilities(
        fitted, prepared.periods["test"], split="test"
    )
    assert len(predictions) == len(prepared.periods["test"])
    assert predictions["failure_probability"].between(0, 1).all()
    assert predictions["model_variant"].eq("A1").all()
    assert predictions["split"].eq("test").all()
    assert predictions.groupby(["transaction_date", "asset_tag"])[
        "probability_rank"
    ].nunique().eq(20).all()

    model_path = tmp_path / "a1.joblib"
    joblib.dump(fitted, model_path)
    loaded = joblib.load(model_path)
    reloaded = predict_part_probabilities(
        loaded, prepared.periods["test"], split="test"
    )
    assert reloaded["failure_probability"].to_numpy() == pytest.approx(
        predictions["failure_probability"].to_numpy()
    )


def test_refit_scaler_excludes_calibration_and_test_rows():
    prepared = prepared_model_rows()
    selection = select_part_model(
        prepared,
        "A1",
        model_names=("logistic_regression",),
        max_iter=30,
    )
    fitted = fit_calibrated_part_model(prepared, selection)

    numeric_features = [
        column
        for column in fitted.features
        if pd.api.types.is_numeric_dtype(
            pd.concat(
                [prepared.periods["train"], prepared.periods["selection"]]
            )[column]
        )
    ]
    expected_means = (
        pd.concat([prepared.periods["train"], prepared.periods["selection"]])[
            numeric_features
        ]
        .mean()
        .to_numpy()
    )
    scaler = fitted.estimator.named_steps["preprocess"].named_transformers_[
        "numeric"
    ].named_steps["scaler"]
    assert scaler.mean_ == pytest.approx(expected_means)


def test_test_labels_do_not_change_fitted_or_calibrated_probabilities():
    prepared = prepared_model_rows()
    selection = select_part_model(
        prepared, "A1", model_names=("logistic_regression",), max_iter=30
    )
    first = fit_calibrated_part_model(prepared, selection)
    first_probabilities = predict_part_probabilities(
        first, prepared.periods["test"], split="test"
    )["failure_probability"].to_numpy()

    changed_frame = prepared.frame.copy()
    changed_frame.loc[changed_frame["period"].eq("test"), "breakdown_flag"] = 1
    changed_periods = {
        name: changed_frame.loc[changed_frame["period"].eq(name)].copy()
        for name in prepared.periods
    }
    changed = replace(prepared, frame=changed_frame, periods=changed_periods)
    second = fit_calibrated_part_model(changed, selection)
    second_probabilities = predict_part_probabilities(
        second, prepared.periods["test"], split="test"
    )["failure_probability"].to_numpy()

    assert second_probabilities == pytest.approx(first_probabilities)


def test_select_and_calibration_reject_single_class_periods():
    prepared = prepared_model_rows()
    selection_periods = dict(prepared.periods)
    selection_periods["selection"] = selection_periods["selection"].assign(
        breakdown_flag=0
    )
    bad_selection = replace(prepared, periods=selection_periods)
    with pytest.raises(ValueError, match="선택 구간.*단일 클래스"):
        select_part_model(
            bad_selection, "A1", model_names=("logistic_regression",)
        )

    selection = select_part_model(
        prepared, "A1", model_names=("logistic_regression",), max_iter=30
    )
    calibration_periods = dict(prepared.periods)
    calibration_periods["calibration"] = calibration_periods["calibration"].assign(
        breakdown_flag=0
    )
    bad_calibration = replace(prepared, periods=calibration_periods)
    with pytest.raises(ValueError, match="보정 구간.*단일 클래스"):
        fit_calibrated_part_model(bad_calibration, selection)


def test_run_probabilistic_asset_risk_writes_all_required_artifacts(tmp_path):
    run = run_probabilistic_asset_risk(
        raw_model_rows(),
        output_dir=tmp_path / "probabilistic_asset_risk",
        selection_start="2023-01-21",
        calibration_start="2023-02-10",
        test_start="2023-03-02",
        model_names=("logistic_regression",),
        bootstrap_samples=50,
        max_iter=20,
        random_state=7,
    )

    required = {
        "part_probabilities.csv",
        "asset_risk_predictions.csv",
        "ranking_metrics.csv",
        "score_metrics.csv",
        "high_risk_metrics.csv",
        "severity_metrics.csv",
        "confusion_matrices.csv",
        "calibration_table.csv",
        "feature_importance.csv",
        "run_config.json",
    }
    assert required.issubset({path.name for path in run.output_dir.iterdir()})
    assert (run.output_dir / "models").is_dir()
    predictions = pd.read_csv(run.output_dir / "asset_risk_predictions.csv")
    assert {"A1", "A2", "B0"}.issubset(set(predictions["model_variant"]))
    assert set(predictions["localization_status"]) <= {
        "approved_on_validation",
        "insufficient_evidence",
    }
    assert set(predictions["top3_display_label"]) <= {
        "우선 점검 후보",
        "참고용 위험 순위",
    }
