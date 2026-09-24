import numpy as np
import pandas as pd
import pytest

from src.industrial_data import (
    PreparedTask,
    add_calendar_features,
    load_industrial_data,
    split_by_date,
)


def test_add_calendar_features_uses_transaction_date():
    frame = pd.DataFrame(
        {"transaction_date": pd.to_datetime(["2024-01-01", "2024-07-01"])}
    )

    result = add_calendar_features(frame)

    assert result["day_of_week"].tolist() == [0, 0]
    assert np.allclose(
        result["month_sin"], np.sin(2 * np.pi * np.array([1, 7]) / 12)
    )
    assert np.allclose(
        result["month_cos"], np.cos(2 * np.pi * np.array([1, 7]) / 12)
    )


def test_split_by_date_excludes_validation_label_crossing_test_boundary():
    frame = pd.DataFrame(
        {
            "transaction_date": pd.to_datetime(
                ["2023-12-20", "2024-01-01", "2024-06-25", "2024-07-01"]
            ),
            "label_end_date": pd.to_datetime(
                ["2023-12-27", "2024-01-08", "2024-07-02", "2024-07-08"]
            ),
        }
    )

    result = split_by_date(frame, "2024-01-01", "2024-07-01")

    assert result["train"]["transaction_date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2023-12-20"
    ]
    assert result["valid"]["transaction_date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2024-01-01"
    ]
    assert result["test"]["transaction_date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2024-07-01"
    ]


def test_prepared_task_rejects_feature_target_overlap():
    with pytest.raises(ValueError, match="Target"):
        PreparedTask(
            frame=pd.DataFrame({"target": [0]}),
            features=("target",),
            target="target",
            grain="asset",
            mode="current",
        )


def test_load_industrial_data_rejects_missing_columns(tmp_path):
    path = tmp_path / "missing.csv"
    pd.DataFrame({"transaction_date": ["2024-01-01"]}).to_csv(path, index=False)

    with pytest.raises(ValueError, match="필요한 컬럼"):
        load_industrial_data(path)
