"""기준 A·B가 학습 구간만 쓰는지 검증한다."""
import numpy as np
import pandas as pd
import pytest

from src.baselines import fit_constant_baseline, fit_group_rate_baseline, score_group_rate_baseline


def _frame():
    return pd.DataFrame({
        "asset_tag": ["A", "A", "A", "B", "B", "C"],
        "split": ["train", "train", "test", "train", "test", "test"],
        "y": [1, 0, 1, 0, 1, 1],
    })


def test_group_rate_uses_train_rows_only():
    frame = _frame()
    train, test = frame[frame.split.eq("train")], frame[frame.split.eq("test")]
    table = fit_group_rate_baseline(train, ["asset_tag"], "y")
    scores = score_group_rate_baseline(table, test, ["asset_tag"], fallback=fit_constant_baseline(train, "y"))

    # 평가 구간 정답을 모두 뒤집어도 점수는 그대로여야 한다.
    flipped = test.assign(y=1 - test.y)
    again = score_group_rate_baseline(table, flipped, ["asset_tag"], fallback=fit_constant_baseline(train, "y"))
    np.testing.assert_allclose(scores, again)
    # A: 학습 1/2, B: 학습 0/1, C: 학습 구간에 없음 → 기준 A(1/3)
    np.testing.assert_allclose(scores, [0.5, 0.0, 1 / 3])


def test_group_rate_table_counts_train_rows():
    frame = _frame()
    table = fit_group_rate_baseline(frame[frame.split.eq("train")], ["asset_tag"], "y")
    assert table.set_index("asset_tag").train_rows.to_dict() == {"A": 2, "B": 1}


def test_empty_train_is_rejected():
    with pytest.raises(ValueError):
        fit_constant_baseline(pd.DataFrame({"y": []}), "y")


def test_top_share_precision_splits_ties_by_expectation():
    from src.model_comparison import top_share_precision
    # 모든 점수가 같으면 양성 비율과 같다(행 순서에 좌우되지 않는다).
    assert top_share_precision([1, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0.3] * 10, 0.2) == pytest.approx(0.2)
    # 상위 2개(0.9, 0.8)가 모두 양성이면 1.0
    assert top_share_precision([1, 1, 0, 0], [0.9, 0.8, 0.1, 0.2], 0.5) == pytest.approx(1.0)
