"""
src/features.py 함수의 최소 동작 확인용 테스트.

실행:
    pytest
"""
import pandas as pd
from src.features import make_ai4i_features


def test_make_ai4i_features_adds_expected_columns():
    df = pd.DataFrame({
        "air_k": [300.0],
        "process_k": [310.0],
        "rpm": [1500],
        "torque": [40.0],
        "tool_wear": [10],
    })
    result = make_ai4i_features(df)
    for col in ["temp_diff", "power_w", "strain"]:
        assert col in result.columns
