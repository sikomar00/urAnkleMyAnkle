"""중요도 합산과 당일 정답, 날짜 분리, Train 전처리만 사용하는지 검사합니다."""
import numpy as np
import pandas as pd
import pytest

from src.huijae_example import SENSORS, evaluate, fit_models, prepare_machine_data, split_by_date


def sample():
    rows = []
    grades = ['A', 'A', 'A', 'B', 'C', 'C']
    failed = [[], [0, 3, 4], [0, 1, 3], [0, 1, 2, 3]]
    for day, date in enumerate(pd.date_range('2024-01-01', periods=100)):
        for asset in ['M1', 'M2']:
            for number, grade in enumerate(grades):
                rows.append(dict(asset_tag=asset, transaction_date=date, part_no=f'P{number}',
                                 criticality=grade, breakdown_flag=int(number in failed[day % 4]),
                                 **{sensor: float(day + index) for index, sensor in enumerate(SENSORS)}))
    return pd.DataFrame(rows)


def test_same_day_scores_threshold_boundaries_and_no_raw_mutation():
    raw = sample()
    original = raw.copy(deep=True)
    result = prepare_machine_data(raw)
    pd.testing.assert_frame_equal(raw, original)
    one = result.loc[result.asset_tag.eq('M1')]
    assert len(result) == 200
    assert one.severity_score.head(4).tolist() == [0, 7, 10, 14]
    assert one.severity_7.head(4).tolist() == [0, 1, 1, 1]
    assert one.severity_10.head(4).tolist() == [0, 0, 1, 1]
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
    assert len(results) == 12  # 세 기준 × 두 모델 × Validation/Test입니다.
    for frame in predictions.values():
        assert frame.groupby('model_variant').size().to_dict() == {'balanced': 30, 'default': 30}


def test_high_accuracy_can_hide_zero_recall():
    result = evaluate(np.array([0] * 9 + [1]), np.zeros(10))
    assert result['Accuracy'] == result['always_normal_accuracy'] == .9
    assert result['Recall'] == result['F1'] == result['predicted_failure_rate'] == 0
    assert result['FN'] == 1 and result['TP'] == 0
