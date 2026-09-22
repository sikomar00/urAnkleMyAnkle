import numpy as np
from sklearn.metrics import f1_score

from src.industrial_training import choose_threshold


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
