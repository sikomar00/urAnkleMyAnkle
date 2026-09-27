import numpy as np
import pandas as pd
import pytest

from src.family_features import FAMILY_NAMES
from src.family_residual_features import (
    HEALTH_SENSOR_COLUMNS,
    RESIDUAL_FEATURES,
    OperatingResidualTransformer,
)
from src.industrial_data import add_calendar_features


@pytest.fixture
def family_rows() -> pd.DataFrame:
    rows = []
    for asset_tag, periods, offset in (("A-1", 40, 0.0), ("A-2", 20, 5.0)):
        for day, date in enumerate(pd.date_range("2023-10-01", periods=periods)):
            affected_family = "Bearing" if asset_tag == "A-1" and day == 5 else None
            load = 50.0 + day % 10
            rpm = 1_400.0 + day * 2
            power = 20.0 + load * 0.1
            sensor_values = {
                "temp_bearing_degC": 30.0 + 0.4 * load + offset + day % 3,
                "temp_motor_degC": 35.0 + 0.5 * load + offset + day % 4,
                "vibration_h_mms": 0.5 + 0.01 * rpm + offset * 0.01 + day % 2,
                "vibration_v_mms": 0.7 + 0.012 * rpm + offset * 0.01 + day % 3,
                "oil_pressure_bar": 3.0 + 0.02 * load + offset * 0.01 + day % 2,
                "load_pct": load,
                "shaft_rpm": rpm,
                "power_consumption_kw": power,
            }
            for family in FAMILY_NAMES:
                affected = int(family == affected_family)
                rows.append(
                    {
                        "transaction_date": date,
                        "label_end_date": date,
                        "machine_type": "Press",
                        "asset_tag": asset_tag,
                        "part_family": family,
                        "affected": affected,
                        "severe": affected,
                        **sensor_values,
                    }
                )

    for day, date in enumerate(pd.date_range("2024-01-01", periods=3)):
        for family in FAMILY_NAMES:
            rows.append(
                {
                    "transaction_date": date,
                    "label_end_date": date,
                    "machine_type": "Press",
                    "asset_tag": "A-1",
                    "part_family": family,
                    "affected": 0,
                    "severe": 0,
                    "temp_bearing_degC": 60.0 + day,
                    "temp_motor_degC": 70.0 + day,
                    "vibration_h_mms": 2.0 + day,
                    "vibration_v_mms": 2.5 + day,
                    "oil_pressure_bar": 4.0 + day,
                    "load_pct": 65.0 + day,
                    "shaft_rpm": 1_500.0 + day,
                    "power_consumption_kw": 27.0 + day,
                }
            )
    return add_calendar_features(pd.DataFrame(rows))


def test_fit_excludes_any_affected_day_and_selects_asset_or_global(family_rows):
    train = family_rows[family_rows["transaction_date"].lt("2024-01-01")]

    fitted = OperatingResidualTransformer(
        min_normal_rows=30, max_iter=5
    ).fit(train)
    artifacts = fitted.artifact_table()

    asset_one = artifacts[
        artifacts["asset_tag"].eq("A-1")
        & artifacts["sensor"].eq("temp_bearing_degC")
    ].iloc[0]
    asset_two = artifacts[
        artifacts["asset_tag"].eq("A-2")
        & artifacts["sensor"].eq("temp_bearing_degC")
    ].iloc[0]
    assert asset_one["normal_rows"] == 39
    assert asset_one["selected_scope"] == "asset_tag"
    assert asset_two["normal_rows"] == 20
    assert asset_two["selected_scope"] == "global"


def test_validation_and_test_values_do_not_change_fitted_residual_models(
    family_rows,
):
    train = family_rows.loc[family_rows["transaction_date"].lt("2024-01-01")]
    original = OperatingResidualTransformer(
        min_normal_rows=30, max_iter=5
    ).fit(train)
    changed = family_rows.copy()
    changed.loc[
        changed["transaction_date"].ge("2024-01-01"),
        HEALTH_SENSOR_COLUMNS,
    ] = 9999
    refit = OperatingResidualTransformer(min_normal_rows=30, max_iter=5).fit(
        changed.loc[changed["transaction_date"].lt("2024-01-01")]
    )

    pd.testing.assert_frame_equal(
        original.artifact_table(), refit.artifact_table()
    )


def test_unseen_asset_uses_global_model_and_residual_baseline(family_rows):
    train = family_rows.loc[family_rows["transaction_date"].lt("2024-01-01")]
    fitted = OperatingResidualTransformer(
        min_normal_rows=30, max_iter=5
    ).fit(train)
    unseen = family_rows.loc[
        family_rows["transaction_date"].eq(pd.Timestamp("2024-01-01"))
    ].copy()
    unseen["asset_tag"] = "A-NEW"

    transformed = fitted.transform(unseen)

    assert transformed[list(RESIDUAL_FEATURES)].shape[1] == len(RESIDUAL_FEATURES)
    assert transformed["temp_bearing_degC_residual"].notna().all()
    assert transformed["temp_bearing_degC_residual_robust_z"].notna().all()


def test_zero_residual_mad_and_std_produce_constant_zero_zscore(family_rows):
    constant = family_rows.copy()
    constant["temp_bearing_degC"] = 50.0
    train = constant.loc[constant["transaction_date"].lt("2024-01-01")]
    fitted = OperatingResidualTransformer(
        min_normal_rows=100, max_iter=5
    ).fit(train)
    global_row = fitted.artifact_table().loc[
        lambda frame: frame["asset_tag"].isna()
        & frame["sensor"].eq("temp_bearing_degC")
    ].iloc[0]

    transformed = fitted.transform(constant.iloc[:9])

    assert global_row["scale_method"] == "constant"
    assert np.allclose(
        transformed["temp_bearing_degC_residual_robust_z"], 0.0
    )
