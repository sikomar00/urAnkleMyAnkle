"""중요도 합산과 당일 정답, 날짜 분리, Train 전처리만 사용하는지 검사합니다."""
import numpy as np
import pandas as pd
import pytest

from src.huijae_example import (CATEGORICAL, SENSORS, TEMPORAL_NUMERIC, THRESHOLDS,
                                enhanced_sensor_frame, evaluate, failure_rates,
                                fit_improved_models, fit_models, prepare_machine_data,
                                run_current_day_models, machine_type_summary, split_by_date)


def sample():
    rows = []
    grades = ['A', 'A', 'A', 'B', 'C', 'C']
    failed = [[], [0, 3, 4], [0, 1, 3], [0, 1, 2, 3]]
    for day, date in enumerate(pd.date_range('2024-01-01', periods=100)):
        for asset in ['M1', 'M2']:
            for number, grade in enumerate(grades):
                rows.append(dict(asset_tag=asset, transaction_date=date, part_no=f'P{number}',
                                 criticality=grade, breakdown_flag=int(number in failed[day % 4]),
                                 machine_type='Type 1' if asset == 'M1' else 'Type 2',
                                 plant_code='Plant 1',
                                 **{sensor: float(day + index + (100 if asset == 'M2' else 0))
                                    for index, sensor in enumerate(SENSORS)}))
    return pd.DataFrame(rows)


def test_same_day_scores_threshold_boundaries_and_no_raw_mutation():
    raw = sample()
    original = raw.copy(deep=True)
    result = prepare_machine_data(raw)
    pd.testing.assert_frame_equal(raw, original)
    one = result.loc[result.asset_tag.eq('M1')]
    assert len(result) == 200
    assert one.severity_score.head(4).tolist() == [0, 7, 10, 14]
    assert one.failed_parts.head(4).tolist() == [0, 3, 3, 4]
    assert one[['failed_A', 'failed_B', 'failed_C']].iloc[1].tolist() == [1, 1, 1]
    assert one.severity_7.head(4).tolist() == [0, 1, 1, 1]
    assert one.severity_10.head(4).tolist() == [0, 0, 1, 1]
    for threshold in [11, 12, 13]:
        assert one[f'severity_{threshold}'].head(4).tolist() == [0, 0, 0, 1]
    assert one.severity_14.head(4).tolist() == [0, 0, 0, 1]
    assert one.iloc[-1].severity_score == 14  # 미래 기록 없는 마지막 날도 사용합니다.


@pytest.mark.parametrize('problem', ['duplicate', 'sensor', 'grade'])
def test_bad_input_is_not_silently_aggregated(problem):
    raw = sample()
    if problem == 'duplicate':
        raw = pd.concat([raw, raw.iloc[:1]])
    elif problem == 'sensor':
        raw.loc[0, SENSORS[0]] = 999
    else:
        raw.loc[0, 'criticality'] = 'D'
    with pytest.raises(ValueError):
        prepare_machine_data(raw)


def test_date_split_keeps_whole_dates_and_does_not_purge_future_days():
    machine = prepare_machine_data(sample().sample(frac=1, random_state=1))
    groups, summary = split_by_date(machine)
    assert summary.rows.tolist() == [140, 30, 30]
    assert sum(len(g) for g in groups.values()) == len(machine)
    assert groups['Train'].transaction_date.max() < groups['Validation'].transaction_date.min()
    assert groups['Validation'].transaction_date.max() < groups['Test'].transaction_date.min()


def test_train_only_imputation_scaling_and_separate_balanced_results():
    machine = prepare_machine_data(sample())
    groups, _ = split_by_date(machine)
    groups['Train'].loc[groups['Train'].index[:2], SENSORS[0]] = np.nan
    groups['Test'].loc[:, SENSORS] = 10000.
    results, predictions, transform = fit_models(groups, balanced_mode='all')
    expected = groups['Train'][SENSORS].median().to_numpy()
    np.testing.assert_allclose(transform['imputer'].statistics_, expected)
    np.testing.assert_allclose(transform['scaler'].mean_, groups['Train'][SENSORS].fillna(groups['Train'][SENSORS].median()).mean())
    assert transform.feature_names_in_.tolist() == SENSORS
    assert len(results) == 24  # 여섯 기준 × 두 모델 × Validation/Test입니다.
    for frame in predictions.values():
        assert frame.groupby('model_variant').size().to_dict() == {'balanced': 30, 'default': 30}


def test_high_accuracy_can_hide_zero_recall():
    result = evaluate(np.array([0] * 9 + [1]), np.zeros(10))
    assert result['Accuracy'] == result['always_normal_accuracy'] == .9
    assert result['Recall'] == result['F1'] == result['predicted_failure_rate'] == 0
    assert result['FN'] == 1 and result['TP'] == 0


def test_past_sensor_features_do_not_mix_assets_or_include_today():
    machine = prepare_machine_data(sample().sample(frac=1, random_state=2))
    name = SENSORS[0]
    a = machine.loc[machine.asset_tag.eq('M1')].sort_values('transaction_date').reset_index(drop=True)
    b = machine.loc[machine.asset_tag.eq('M2')].sort_values('transaction_date').reset_index(drop=True)
    assert pd.isna(a.loc[0, f'{name}_diff1'])
    assert pd.isna(b.loc[0, f'{name}_diff1'])
    assert a.loc[1, f'{name}_diff1'] == b.loc[1, f'{name}_diff1'] == 1
    assert a.loc[3, f'{name}_mean7'] == 1  # 이전 0·1·2의 평균입니다.
    assert a.loc[3, f'{name}_std7'] == 1
    assert a.loc[7, f'{name}_mean7'] == 3  # 현재 7은 평균에 포함되지 않습니다.
    assert a.loc[7, f'{name}_vs_mean30'] == 4
    assert b.loc[7, f'{name}_mean7'] == 103  # M1의 값이 M2의 과거에 섞이지 않습니다.
    assert len(TEMPORAL_NUMERIC) == 40


def test_temporal_pipeline_uses_allowed_columns_and_train_statistics():
    machine = prepare_machine_data(sample())
    groups, _ = split_by_date(machine)
    groups['Test'].loc[:, SENSORS] = 10000.
    results, predictions, pipelines = run_current_day_models(
        groups, TEMPORAL_NUMERIC, CATEGORICAL, 'expanded')
    assert len(results) == 12 and len(predictions) == 30 * 12
    assert CATEGORICAL == ['machine_type', 'plant_code']
    assert predictions.columns.tolist() == [
        'transaction_date', 'asset_tag', 'machine_type', 'plant_code',
        'severity_threshold', 'model', 'feature_set', 'actual_label',
        'predicted_label', 'predicted_probability']
    assert predictions.asset_tag.notna().all()
    model = pipelines[(7, '기본')]
    prep = model['preprocess']
    train_medians = groups['Train'][TEMPORAL_NUMERIC].median().to_numpy()
    np.testing.assert_allclose(prep.named_transformers_['numeric']['imputer'].statistics_, train_medians)
    assert prep.named_transformers_['categorical']['one_hot'].handle_unknown == 'ignore'
    assert set(prep.feature_names_in_) == set(TEMPORAL_NUMERIC + CATEGORICAL)
    assert 'asset_tag' not in prep.feature_names_in_
    assert prep.transformers_[1][2] == ['machine_type', 'plant_code']
    assert set(prep.feature_names_in_).isdisjoint({'severity_score', 'breakdown_flag',
                                                  'criticality', 'qty_issued', 'issue_value_inr'} |
                                                 {f'severity_{t}' for t in THRESHOLDS})
    report = machine_type_summary(groups, pipelines)
    assert len(report) == 12  # 설비 종류 2개 × 심각도 기준 6개입니다.
    assert report[['machine_type', 'score_threshold']].values.tolist() == [
        [kind, threshold] for kind in ['Type 1', 'Type 2'] for threshold in THRESHOLDS]
    assert report.status.eq('ok').all()
    for row in report.itertuples():
        group = groups['Train'].loc[groups['Train'].machine_type.eq(row.machine_type)]
        assert row.train_positive == group[f'severity_{row.score_threshold}'].sum()


def test_all_rates_are_checked_before_fitting_and_features_exclude_labels():
    machine = prepare_machine_data(sample())
    groups, _ = split_by_date(machine)
    rates = failure_rates(machine, groups)
    assert THRESHOLDS == [7, 10, 11, 12, 13, 14]
    assert len(rates) == 24
    assert set(rates.dataset) == {'전체', 'Train', 'Validation', 'Test'}
    assert rates.loc[rates.dataset.eq('전체'), 'failure_count'].tolist() == [150, 100, 50, 50, 50, 50]
    assert (rates.normal_count + rates.failure_count).eq(rates.dataset.map({'전체': 200, 'Train': 140,
                                                                            'Validation': 30, 'Test': 30})).all()
    features = enhanced_sensor_frame(machine)
    assert len(features.columns) == 12
    assert not set(features).intersection({'severity_score', 'breakdown_flag', 'criticality', 'wo_type'} |
                                         {f'severity_{t}' for t in THRESHOLDS})


def test_improved_thresholds_use_validation_not_test():
    machine = prepare_machine_data(sample())
    groups, _ = split_by_date(machine)
    comparison, predictions, choices = fit_improved_models(groups)
    assert len(choices) == 24  # 여섯 기준 × 두 입력 × 두 가중치입니다.
    assert len(comparison) == 96  # 두 분할 × 두 확률 기준까지 포함합니다.
    assert len(predictions) == 30 * 24 * 2
    assert choices.selection_split.eq('Validation').all()
    altered = {name: group.copy() for name, group in groups.items()}
    for threshold in THRESHOLDS:
        col = f'severity_{threshold}'
        altered['Test'][col] = 1 - altered['Test'][col]
    _, _, changed_choices = fit_improved_models(altered)
    pd.testing.assert_frame_equal(choices, changed_choices)
