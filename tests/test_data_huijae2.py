import json

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.data_huijae2 import (HISTORY, PART_KEYS, SENSORS, TARGET, daily_top_k,
                             feature_sets, prepare_rows, ranking_metrics,
                             run_test, run_validation)


def synthetic_raw():
    rows = []
    for asset_number, asset in enumerate(['M1', 'M2']):
        for day, date in enumerate(pd.date_range('2024-01-01', periods=160)):
            for part_number, part in enumerate(['P1', 'P2', 'P3']):
                rows.append(dict(asset_tag=asset, part_no=part, transaction_date=date,
                                 machine_type='test', plant_code='plant', part_description=part,
                                 part_family='bearing', criticality='A', uom='EA', unit_cost_inr=10.,
                                 qty_issued=0., issue_value_inr=0., wo_type=None,
                                 breakdown_flag=int((day + 3 * part_number + asset_number) % 17 == 0),
                                 **{sensor: 10. + np.sin(day / 10.) + asset_number for sensor in SENSORS}))
    return pd.DataFrame(rows)


def test_feature_groups_change_only_history_and_sensors():
    specs = feature_sets()
    a, b, c = [set(specs[name]['numeric'] + specs[name]['categorical']) for name in ['A', 'B', 'C']]
    assert not a.intersection(SENSORS)
    assert not b.intersection(HISTORY)
    assert c == a | b
    assert c - a == set(SENSORS)
    assert c - b == set(HISTORY)
    assert not c.intersection({TARGET, 'breakdown_flag', 'qty_issued', 'asset_tag', 'split'})


def test_history_is_separate_by_asset_and_future_cannot_change_training():
    raw = synthetic_raw()
    original = raw.copy(deep=True)
    splits, audit = prepare_rows(raw)
    pd.testing.assert_frame_equal(raw, original)
    train = splits['train']
    row = train.loc[train.asset_tag.eq('M1') & train.part_no.eq('P1')].iloc[0]
    history = raw.loc[raw.asset_tag.eq('M1') & raw.part_no.eq('P1') &
                      raw.transaction_date.between(row.transaction_date - pd.Timedelta(days=7),
                                                   row.transaction_date - pd.Timedelta(days=1))]
    assert row.breakdown_days_prev_7d == history.breakdown_flag.sum()
    changed = raw.copy()
    changed.loc[changed.asset_tag.eq('M2'), 'breakdown_flag'] = 0
    changed_splits, _ = prepare_rows(changed)
    pd.testing.assert_frame_equal(train.loc[train.asset_tag.eq('M1')].reset_index(drop=True),
                                  changed_splits['train'].loc[lambda f: f.asset_tag.eq('M1')].reset_index(drop=True))
    later = raw.transaction_date.ge(pd.Timestamp(audit['split_dates']['validation_start']))
    raw.loc[later, 'breakdown_flag'] = 1 - raw.loc[later, 'breakdown_flag']
    future_changed, _ = prepare_rows(raw)
    pd.testing.assert_frame_equal(train, future_changed['train'])
    assert train.target_window_end.max() < splits['validation'].transaction_date.min()


def test_daily_budget_ties_and_small_candidate_pool():
    frame = pd.DataFrame({'asset_tag': ['M2', 'M1', 'M3', 'M1'], 'part_no': ['P'] * 4,
                          'transaction_date': pd.to_datetime(['2024-01-01'] * 3 + ['2024-01-02']),
                          TARGET: [0, 1, 1, 1], 'score': [.8, .8, .2, .5]})
    daily, chosen = daily_top_k(frame, 2)
    assert daily.inspected.tolist() == [2, 1]
    assert daily.hits.tolist() == [1, 1]
    assert daily.precision_at_k.tolist() == [.5, 1.]
    # 정답을 바꾸거나 입력 순서를 섞어도 선택 결과는 같아야 합니다.
    altered = frame.sample(frac=1, random_state=1).assign(**{TARGET: 0})
    _, chosen_altered = daily_top_k(altered, 2)
    pd.testing.assert_frame_equal(chosen[PART_KEYS].reset_index(drop=True),
                                  chosen_altered[PART_KEYS].reset_index(drop=True))
    with pytest.raises(ValueError, match='중복'):
        daily_top_k(pd.concat([frame, frame.iloc[:1]]), 2)


def test_average_precision_is_not_trapezoid_area():
    frame = pd.DataFrame({TARGET: [0, 1], 'score': [.5, .5]})
    result = ranking_metrics(frame)
    assert result['average_precision'] == .5
    assert result['pr_auc_trapezoid'] == .75


def test_validation_does_not_predict_test_and_test_loads_frozen_models(tmp_path, monkeypatch):
    splits, audit = prepare_rows(synthetic_raw())
    run_dir = tmp_path / 'experiment'
    validation_only = {**splits, 'test': None}
    run_validation(validation_only, audit, tmp_path / 'source.csv', 'original', run_dir, 2)
    assert not (run_dir / 'test').exists()
    manifest = json.loads((run_dir / 'manifest.json').read_text(encoding='utf-8'))
    assert not manifest['test_evaluated']
    predictions = pd.read_csv(run_dir / 'validation/predictions.csv')
    assert predictions.groupby('experiment').size().tolist() == [len(splits['validation'])] * 3

    def forbid_refit(*args, **kwargs):
        raise AssertionError('시험 단계에서 재학습하면 안 됩니다.')

    monkeypatch.setattr(Pipeline, 'fit', forbid_refit)
    with pytest.raises(ValueError, match='원본'):
        run_test(splits, 'changed', run_dir)
    with pytest.raises(ValueError, match='점검 수'):
        run_test(splits, 'original', run_dir, 3)
    run_test(splits, 'original', run_dir)
    assert (run_dir / 'test/summary.md').exists()
    with pytest.raises(ValueError, match='이미'):
        run_test(splits, 'original', run_dir)
