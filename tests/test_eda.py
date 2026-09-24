import inspect

import pandas as pd
import pytest

from src import (
    asset_features,
    current_asset_model,
    current_part_model,
    forecast_asset_model,
    forecast_part_model,
    industrial_cli,
    industrial_data,
    industrial_training,
    part_features,
)
from src.eda import build_comparison, load_metric_outputs


def test_build_comparison_keeps_selected_test_rows_only():
    metrics = pd.DataFrame([
        {
            "grain": "asset",
            "mode": "current",
            "model": "random_forest",
            "selected_model": True,
            "split": "test",
            "threshold_policy": "f1",
            "average_precision": 0.4,
        },
        {
            "grain": "asset",
            "mode": "current",
            "model": "logistic_regression",
            "selected_model": False,
            "split": "test",
            "threshold_policy": "f1",
            "average_precision": 0.3,
        },
        {
            "grain": "asset",
            "mode": "current",
            "model": "random_forest",
            "selected_model": True,
            "split": "validation",
            "threshold_policy": "f1",
            "average_precision": 0.5,
        },
    ])

    result = build_comparison(metrics)

    assert len(result) == 1
    assert result.iloc[0]["model"] == "random_forest"


def test_load_metric_outputs_explains_how_to_create_missing_results(tmp_path):
    with pytest.raises(FileNotFoundError, match="current_asset_model.py"):
        load_metric_outputs([tmp_path / "missing.csv"])


def test_public_industrial_modules_have_korean_documentation():
    modules = [
        industrial_data,
        asset_features,
        part_features,
        industrial_training,
        industrial_cli,
        current_asset_model,
        forecast_asset_model,
        current_part_model,
        forecast_part_model,
    ]
    for module in modules:
        assert module.__doc__ and any("가" <= char <= "힣" for char in module.__doc__)
        for name, function in inspect.getmembers(module, inspect.isfunction):
            if function.__module__ == module.__name__ and not name.startswith("_"):
                assert function.__doc__, f"{module.__name__}.{name}에 docstring이 없습니다."
