import numpy as np
import pandas as pd
import pytest

from src.data_huijae import build_classifier, checkpoint_dates, choose_threshold, classification_metrics


def test_preprocessing_fits_training_only_and_accepts_unknown_category():
    schema = {'numeric_features': ['sensor'], 'categorical_features': ['part']}
    train = pd.DataFrame({'sensor': [1., 2., 3., np.nan], 'part': ['a', 'b', 'a', 'b']})
    model = build_classifier(schema).fit(train, [0, 1, 0, 1])
    imputer = model['features'].named_transformers_['numeric']['impute']
    assert imputer.statistics_[0] == 2
    unseen = pd.DataFrame({'sensor': [10000., np.nan], 'part': ['new', 'a']})
    assert np.isfinite(model.predict_proba(unseen)).all()
    assert imputer.statistics_[0] == 2


def test_threshold_and_metrics_use_passed_validation_scores():
    y, score = np.array([0, 0, 1, 1]), np.array([.1, .2, .4, .6])
    threshold, curve = choose_threshold(y, score)
    assert .2 < threshold <= .4
    assert classification_metrics(y, score, threshold)['f1'] == 1
    assert len(curve) == 19


def test_sampled_backtest_keeps_last_day_and_daily_option():
    days = pd.date_range('2024-01-01', periods=31)
    assert list(checkpoint_dates(days, 14)) == list(days[[0, 14, 28, 30]])
    assert list(checkpoint_dates(days, 1)) == list(days)
    with pytest.raises(ValueError):
        checkpoint_dates(days, 0)
