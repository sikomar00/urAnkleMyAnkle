"""
src/features.py 함수의 최소 동작 확인용 테스트.

실행 (레포 루트에서):
    pytest

xfail 표시는 make_ai4i_features 구현이 끝나면 제거하세요.
제거 후 이 테스트가 통과해야 파생변수 로직이 계약대로 동작한다는
증거가 됩니다.
"""
import pandas as pd
import pytest

from src.features import make_ai4i_features


@pytest.mark.xfail(reason="make_ai4i_features 구현 전", strict=False)
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


@pytest.mark.xfail(reason="make_ai4i_features 구현 전", strict=False)
def test_make_ai4i_features_does_not_mutate_input():
    """원본 DataFrame을 건드리지 않아야 한다 (학습/추론 재현성)."""
    df = pd.DataFrame({
        "air_k": [300.0], "process_k": [310.0],
        "rpm": [1500], "torque": [40.0], "tool_wear": [10],
    })
    before = df.columns.tolist()
    make_ai4i_features(df)
    assert df.columns.tolist() == before
