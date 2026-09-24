import time

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

from src.industrial_data import PreparedTask
from src.industrial_training import (
    build_model,
    choose_threshold,
    choose_thresholds,
    classification_metrics,
    ranking_metrics,
    run_experiment_suite,
    validate_features,
)


def test_threshold_matches_original_search():
    rng = np.random.default_rng(42)
    for _ in range(30):
        y = rng.integers(0, 2, 100)
        scores = np.round(rng.random(100), 2)
        candidates = np.unique(np.r_[0.5, scores])
        expected = max(
            candidates,
            key=lambda t: (f1_score(y, scores >= t, zero_division=0), t),
        )
        assert choose_threshold(y, scores) == expected


def test_threshold_empty_and_single_class():
    assert choose_threshold([], []) == 0.5
    assert choose_threshold([0, 0], [0.1, 0.2]) == 0.5
    assert choose_threshold([1, 1], [0.1, 0.2]) == 0.5


def test_choose_thresholds_returns_three_policies():
    selected = {item.policy: item for item in choose_thresholds(
        [0, 0, 0, 1, 1], [0.1, 0.2, 0.3, 0.7, 0.9],
        min_precision=0.75, top_fraction=0.40,
    )}
    assert selected["f1"].status == "ok"
    assert selected["min_precision"].cutoff == 0.7
    assert selected["top_fraction"].cutoff == 0.7


def test_min_precision_reports_unavailable():
    selected = {item.policy: item for item in choose_thresholds(
        [1, 0, 0, 0], [0.1, 0.9, 0.8, 0.7], min_precision=0.80
    )}
    assert selected["min_precision"].cutoff is None
    assert selected["min_precision"].status == "unavailable"


def test_ranking_metrics_uses_exact_top_count():
    result = ranking_metrics([1, 0, 1, 0, 0], [0.9, 0.8, 0.7, 0.6, 0.5])
    assert result["precision_at_20pct"] == 1.0
    assert result["recall_at_20pct"] == 0.5
    assert result["lift_at_20pct"] == 2.5


def test_choose_thresholds_calculates_only_requested_policies():
    selected = choose_thresholds(
        [0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8], policies=("f1",)
    )
    assert [item.policy for item in selected] == ["f1"]


def test_min_precision_threshold_search_scales_to_many_unique_scores():
    size = 2_000
    labels = np.arange(size) % 7 == 0
    scores = np.linspace(0.0, 1.0, size, endpoint=False)
    started = time.perf_counter()
    choose_thresholds(labels, scores, policies=("min_precision",))
    assert time.perf_counter() - started < 0.5


def test_validate_features_rejects_every_target_prefix():
    with pytest.raises(ValueError, match="target_14d"):
        validate_features(["signal", "target_14d"], "target_7d")


def test_logistic_regression_scales_numeric_features():
    frame = pd.DataFrame({"signal": [0.0, 1_000.0, 2_000.0, 3_000.0]})
    model = build_model("logistic_regression", frame, max_iter=20, random_state=42)
    model.fit(frame, [0, 0, 1, 1])
    transformed = model.named_steps["preprocess"].transform(frame)
    assert np.allclose(np.mean(transformed, axis=0), 0.0, atol=1e-9)


def test_metrics_without_cutoff_keep_threshold_independent_scores():
    metrics = classification_metrics([0, 1], [0.2, 0.8], None)
    assert metrics["average_precision"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["precision"] is None
    assert metrics["true_positive"] is None


def _training_task(*, test_single_class: bool = False) -> PreparedTask:
    dates = pd.to_datetime([
        "2023-12-01", "2023-12-02", "2023-12-03", "2023-12-04",
        "2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04",
        "2024-07-01", "2024-07-02", "2024-07-03", "2024-07-04",
    ])
    labels = [0, 1] * 6
    if test_single_class:
        labels[-4:] = [0, 0, 0, 0]
    frame = pd.DataFrame({
        "transaction_date": dates,
        "label_end_date": dates,
        "machine_type": ["Press"] * 12,
        "asset_tag": ["A-1", "A-2"] * 6,
        "signal": np.arange(12, dtype=float),
        "target": labels,
    })
    return PreparedTask(
        frame, ("machine_type", "asset_tag", "signal"), "target", "asset", "current"
    )


def test_experiment_suite_writes_common_outputs(tmp_path):
    task = _training_task()
    metrics = run_experiment_suite([task], output_dir=tmp_path, scope="overall", max_iter=10)
    assert {
        "dummy_classifier", "logistic_regression", "random_forest",
        "hist_gradient_boosting",
    } <= set(metrics["model"])
    assert metrics.query("selected_model == True").shape[0] > 0
    for name in ("metrics.csv", "test_predictions.csv", "feature_importance.csv", "run_config.json"):
        assert (tmp_path / name).exists()


def test_suite_records_single_class_test_as_skipped(tmp_path):
    metrics = run_experiment_suite(
        [_training_task(test_single_class=True)],
        output_dir=tmp_path,
        scope="overall",
        max_iter=10,
    )
    assert set(metrics["status"]) == {"skipped"}
    assert "test" in metrics.iloc[0]["reason"]


def test_empty_task_writes_header_only_outputs(tmp_path):
    frame = pd.DataFrame(columns=[
        "transaction_date", "label_end_date", "machine_type", "asset_tag",
        "signal", "target",
    ])
    task = PreparedTask(
        frame, ("machine_type", "asset_tag", "signal"), "target", "asset", "current"
    )
    metrics = run_experiment_suite([task], output_dir=tmp_path, scope="overall")
    assert metrics.iloc[0]["status"] == "skipped"
    assert {"grain", "mode", "threshold_policy", "average_precision"} <= set(metrics)
    assert {"risk_score", "prediction"} <= set(pd.read_csv(tmp_path / "test_predictions.csv"))


def test_saved_selected_model_contains_reproducible_decision_settings(tmp_path):
    task = _training_task()
    run_experiment_suite([task], output_dir=tmp_path, scope="overall", max_iter=10)
    model_file = next((tmp_path / "models").glob("*__selected.joblib"))
    payload = joblib.load(model_file)
    rows = task.frame.loc[:, list(task.features)].head(3)
    actual = payload["pipeline"].predict_proba(rows)[:, 1]
    assert np.allclose(actual, payload["verification_scores"])
    assert payload["selected_model"] in {
        "logistic_regression", "random_forest", "hist_gradient_boosting"
    }
    assert set(payload["thresholds"]) == {"f1", "min_precision", "top_fraction"}
    assert payload["validation_start"] == "2024-01-01"
    assert payload["test_start"] == "2024-07-01"
