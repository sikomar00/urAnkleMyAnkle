import joblib
import numpy as np
import pandas as pd

from src.family_diagnosis_experiments import (
    choose_family_thresholds,
    run_family_experiments,
    select_family_feature_set,
    select_family_model,
)
from src.family_features import FamilyDiagnosisFeatures


def _selection_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "split": "validation",
                "model": "logistic_regression",
                "feature_set": "A",
                "average_precision": 0.800,
                "f1": 0.70,
                "recall": 0.60,
                "precision": 0.50,
            },
            {
                "split": "validation",
                "model": "random_forest",
                "feature_set": "B",
                "average_precision": 0.805,
                "f1": 0.60,
                "recall": 0.80,
                "precision": 0.70,
            },
            {
                "split": "validation",
                "model": "hist_gradient_boosting",
                "feature_set": "C",
                "average_precision": 0.780,
                "f1": 0.95,
                "recall": 0.95,
                "precision": 0.95,
            },
            {
                "split": "test",
                "model": "hist_gradient_boosting",
                "feature_set": "C",
                "average_precision": 1.0,
                "f1": 1.0,
                "recall": 1.0,
                "precision": 1.0,
            },
        ]
    )


def test_selection_prefers_validation_ap_then_tolerance_tiebreakers():
    rows = _selection_rows()

    assert select_family_model(rows) == "logistic_regression"
    assert select_family_feature_set(rows) == "A"


def test_selection_ignores_test_metrics():
    rows = _selection_rows()
    changed = rows.copy()
    changed.loc[changed["split"].eq("test"), ["average_precision", "f1"]] = -1

    assert select_family_model(rows) == select_family_model(changed)
    assert select_family_feature_set(rows) == select_family_feature_set(changed)


def test_family_thresholds_return_four_fixed_policies():
    selected = {
        item.policy: item
        for item in choose_family_thresholds(
            [1, 0, 1, 0, 0, 1, 0, 0, 0, 0],
            [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
        )
    }

    assert set(selected) == {
        "f1",
        "min_precision_70",
        "min_precision_80",
        "top_10pct",
    }
    assert selected["min_precision_70"].status == "ok"
    assert selected["min_precision_80"].status == "ok"
    assert selected["top_10pct"].cutoff == 0.9


def test_family_thresholds_mark_unreachable_precision_unavailable():
    selected = {
        item.policy: item
        for item in choose_family_thresholds(
            [1, 0, 0, 0, 0],
            [0.5, 0.5, 0.5, 0.5, 0.5],
        )
    }

    assert selected["min_precision_70"].status == "unavailable"
    assert selected["min_precision_70"].cutoff is None
    assert selected["min_precision_80"].status == "unavailable"
    assert selected["min_precision_80"].cutoff is None


def _prepared_family_experiment(*, flip_test: bool = False) -> FamilyDiagnosisFeatures:
    dates = [
        *pd.date_range("2023-12-01", periods=20),
        *pd.date_range("2024-01-01", periods=20),
        *pd.date_range("2024-07-01", periods=20),
    ]
    rows = []
    for index, date in enumerate(dates):
        split_index = index % 20
        bearing = split_index % 2
        filter_affected = (split_index + 1) % 2
        filter_severe = int(split_index == 0)
        if flip_test and date >= pd.Timestamp("2024-07-01"):
            bearing = 1 - bearing
            filter_affected = 1 - filter_affected
        for family, affected, severe in (
            ("Bearing", bearing, bearing),
            ("Filter", filter_affected, filter_severe),
        ):
            rows.append(
                {
                    "transaction_date": date,
                    "label_end_date": date,
                    "machine_type": "Press",
                    "asset_tag": "A-1",
                    "part_family": family,
                    "signal": float(affected) + (split_index % 3) * 0.01,
                    "noise": float(split_index % 5),
                    "affected": affected,
                    "severe": severe,
                }
            )
    return FamilyDiagnosisFeatures(
        frame=pd.DataFrame(rows),
        feature_sets={"A": ("signal", "noise")},
    )


def test_runner_reuses_identical_target_and_skips_insufficient_positive_rows(
    tmp_path,
):
    result = run_family_experiments(
        _prepared_family_experiment(),
        output_dir=tmp_path,
        max_iter=5,
    )

    bearing_severe = result.metrics[
        result.metrics["part_family"].eq("Bearing")
        & result.metrics["target"].eq("severe")
    ]
    filter_severe = result.metrics[
        result.metrics["part_family"].eq("Filter")
        & result.metrics["target"].eq("severe")
    ]
    assert "identical_target" in set(bearing_severe["status"])
    assert "insufficient_positive_rows" in set(filter_severe["status"])


def test_runner_keeps_validation_selection_and_saved_predictions_reproducible(
    tmp_path,
):
    original = run_family_experiments(
        _prepared_family_experiment(),
        output_dir=tmp_path / "original",
        max_iter=5,
    )
    changed = run_family_experiments(
        _prepared_family_experiment(flip_test=True),
        output_dir=tmp_path / "changed",
        max_iter=5,
    )
    selection_columns = [
        "part_family",
        "target",
        "feature_set",
        "model",
        "threshold_policy",
        "probability_cutoff",
    ]
    original_selection = original.metrics.loc[
        original.metrics["selected_policy"].fillna(False), selection_columns
    ].reset_index(drop=True)
    changed_selection = changed.metrics.loc[
        changed.metrics["selected_policy"].fillna(False), selection_columns
    ].reset_index(drop=True)
    pd.testing.assert_frame_equal(original_selection, changed_selection)

    model_path = next(
        path
        for key, path in original.models.items()
        if key.startswith("Bearing__affected")
    )
    payload = joblib.load(model_path)
    test_rows = _prepared_family_experiment().frame
    test_rows = test_rows[
        test_rows["part_family"].eq("Bearing")
        & test_rows["transaction_date"].ge("2024-07-01")
    ]
    reloaded_scores = payload["pipeline"].predict_proba(
        test_rows[payload["features"]]
    )[:, 1]
    recorded = original.predictions[
        original.predictions["part_family"].eq("Bearing")
        & original.predictions["target"].eq("affected")
        & original.predictions["threshold_policy"].eq("f1")
    ]["risk_score"].to_numpy()
    assert np.allclose(reloaded_scores, recorded)
