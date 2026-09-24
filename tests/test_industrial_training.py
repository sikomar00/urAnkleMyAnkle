import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.industrial_data import PreparedTask
from src.industrial_training import (
    choose_threshold, choose_thresholds, ranking_metrics, run_experiment_suite,
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


def test_experiment_suite_writes_common_outputs(tmp_path):
    dates = pd.to_datetime([
        "2023-12-01", "2023-12-02", "2023-12-03", "2023-12-04",
        "2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04",
        "2024-07-01", "2024-07-02", "2024-07-03", "2024-07-04",
    ])
    frame = pd.DataFrame({
        "transaction_date": dates, "label_end_date": dates,
        "machine_type": ["Press"] * 12, "asset_tag": ["A-1", "A-2"] * 6,
        "signal": np.arange(12, dtype=float), "target": [0, 1] * 6,
    })
    task = PreparedTask(frame, ("machine_type", "asset_tag", "signal"),
                        "target", "asset", "current")
    metrics = run_experiment_suite([task], output_dir=tmp_path, scope="overall", max_iter=10)
    assert {"logistic_regression", "random_forest", "hist_gradient_boosting"} <= set(metrics["model"])
    assert metrics.query("selected_model == True").shape[0] > 0
    for name in ("metrics.csv", "test_predictions.csv", "feature_importance.csv", "run_config.json"):
        assert (tmp_path / name).exists()
