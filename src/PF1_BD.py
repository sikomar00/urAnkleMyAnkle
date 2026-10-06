"""부품군 전체가 고장 표시된 설비의 당일 상태를 로지스틱 회귀로 분류합니다.

실행: python src/PF1_BD.py
다른 CSV: python src/PF1_BD.py --data "C:/path/to/data.csv"
"""

import argparse
import hashlib
import warnings
from pathlib import Path

import pandas as pd
from sklearn.exceptions import ConvergenceWarning

# 직접 실행할 때와 테스트에서 모듈로 불러올 때 모두 같은 기존 함수를 사용합니다.
if __package__:
    from .huijae_example import (CATEGORICAL, KEYS, TEMPORAL_NUMERIC,
                                 current_day_pipeline, evaluate, prepare_machine_data,
                                 split_by_date)
else:
    from huijae_example import (CATEGORICAL, KEYS, TEMPORAL_NUMERIC,
                                current_day_pipeline, evaluate, prepare_machine_data,
                                split_by_date)

ROOT = Path(__file__).resolve().parents[1]
TARGET = 'part_family_full_breakdown'


def file_hash(path):
    """실행 전후 원본 CSV가 바뀌지 않았는지 확인합니다."""
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def build_family_labels(raw):
    """세부 부품이 모두 고장 난 부품군이 하나라도 있는 설비·날짜에 1을 붙입니다."""
    required = KEYS + ['part_no', 'part_family', 'breakdown_flag']
    missing = sorted(set(required) - set(raw.columns))
    if missing:
        raise ValueError(f'필수 열이 없습니다: {missing}')
    parts = raw[required].copy()
    parts['transaction_date'] = pd.to_datetime(parts.transaction_date, errors='raise').dt.normalize()
    if parts[KEYS + ['part_no', 'part_family']].isna().any().any():
        raise ValueError('설비·날짜·부품·부품군에 빈값이 있습니다.')
    if parts.duplicated(KEYS + ['part_no']).any():
        raise ValueError('같은 설비·날짜·세부 부품이 중복되어 있습니다.')
    parts['breakdown_flag'] = pd.to_numeric(parts.breakdown_flag, errors='raise')
    if not parts.breakdown_flag.isin([0, 1]).all():
        raise ValueError('breakdown_flag는 0 또는 1이어야 합니다.')

    # 설비별 부품군의 전체 구성원을 정합니다. 특정 날짜에 빠진 세부 부품이 있으면 오류로 처리합니다.
    catalog = parts[['asset_tag', 'part_family', 'part_no']].drop_duplicates()
    if catalog.duplicated(['asset_tag', 'part_no']).any():
        raise ValueError('같은 설비의 세부 부품이 여러 부품군에 속합니다.')
    expected = catalog.groupby(['asset_tag', 'part_family']).part_no.nunique().rename('expected_parts')
    family = parts.groupby(KEYS + ['part_family'], as_index=False).agg(
        observed_parts=('part_no', 'nunique'), failed_parts=('breakdown_flag', 'sum'))
    family = family.join(expected, on=['asset_tag', 'part_family'], validate='many_to_one')
    expected_families = catalog.groupby('asset_tag').part_family.nunique()
    observed_families = family.groupby(KEYS).part_family.nunique()
    if not observed_families.eq(observed_families.index.get_level_values('asset_tag').map(expected_families)).all():
        raise ValueError('어떤 설비·날짜에는 부품군 전체가 누락되어 있습니다.')
    if not family.observed_parts.eq(family.expected_parts).all():
        raise ValueError('어떤 날짜에는 부품군의 세부 부품이 누락되어 전체 고장 여부를 판단할 수 없습니다.')
    family['all_parts_failed'] = family.failed_parts.eq(family.expected_parts).astype(int)
    label = family.groupby(KEYS, as_index=False).agg(
        **{TARGET: ('all_parts_failed', 'max')},
        fully_failed_families=('all_parts_failed', 'sum'))
    return label, family


def prepare_data(raw):
    """기존 설비·날짜 센서 표에 새로운 부품군 고장 정답만 결합합니다."""
    machine = prepare_machine_data(raw)
    label, family = build_family_labels(raw)
    machine = machine.merge(label, on=KEYS, how='left', validate='one_to_one')
    if machine[TARGET].isna().any():
        raise ValueError('일부 설비·날짜에 부품군 고장 정답이 없습니다.')
    return machine, family


def result_row(frame, probabilities, scope, name, model_name):
    """전체·설비 종류·개별 설비에 같은 방식으로 Test 성능을 계산합니다."""
    y = frame[TARGET].to_numpy()
    metrics = evaluate(y, probabilities, cutoff=.5)
    return {'구분': scope, '설비': name, '모델': model_name,
            '전체행': len(frame), '고장행': int(y.sum()), '고장비율': float(y.mean()),
            '예측고장행': int((probabilities >= .5).sum()), **metrics}


def run_models(groups):
    """전체 Train으로 학습한 모델을 전체와 각 설비의 Test 행에서 평가합니다."""
    columns = TEMPORAL_NUMERIC + CATEGORICAL
    forbidden = {'asset_tag', 'part_no', 'part_family', 'breakdown_flag', 'wo_type',
                 'criticality', 'severity_score', 'fully_failed_families', TARGET,
                 'qty_issued', 'issue_value_inr'}
    if set(columns) & forbidden or any(name.startswith('severity_') for name in columns):
        raise ValueError('정답 생성에 사용한 열이 모델 입력 X에 들어갔습니다.')
    if len(columns) != len(set(columns)):
        raise ValueError('모델 입력 X에 중복 열이 있습니다.')
    train, test = groups['Train'], groups['Test']
    if train[TARGET].nunique() != 2:
        raise ValueError('Train에 정상과 고장이 모두 있어야 합니다.')
    results, predictions = [], []
    for model_name, class_weight in [('기본', None), ('balanced', 'balanced')]:
        model = current_day_pipeline(TEMPORAL_NUMERIC, CATEGORICAL, class_weight)
        with warnings.catch_warnings():
            warnings.simplefilter('error', ConvergenceWarning)
            model.fit(train[columns], train[TARGET])
        trained_columns = list(model.named_steps['preprocess'].feature_names_in_)
        if trained_columns != columns:
            raise RuntimeError('실제 모델 입력 X가 지정한 변수 목록과 다릅니다.')
        probability = model.predict_proba(test[columns])[:, 1]
        results.append(result_row(test, probability, '전체', '전체 설비', model_name))
        # 종류별/개별 설비에서는 모델을 다시 학습하지 않고 해당 행만 골라 평가합니다.
        for scope, key in [('설비 종류', 'machine_type'), ('개별 설비', 'asset_tag')]:
            for name in sorted(test[key].dropna().unique()):
                mask = test[key].eq(name).to_numpy()
                results.append(result_row(test.loc[mask], probability[mask], scope, name, model_name))
        pred = test[['transaction_date', 'asset_tag', 'machine_type', 'plant_code', TARGET]].copy()
        pred['model'] = model_name
        pred['predicted_label'] = (probability >= .5).astype(int)
        pred['predicted_probability'] = probability
        predictions.append(pred.rename(columns={TARGET: 'actual_label'}))
    return pd.DataFrame(results), pd.concat(predictions, ignore_index=True), columns


def show_table(title, frame):
    """사진처럼 열 순서를 맞춰 터미널에 정렬된 표를 표시합니다."""
    columns = ['설비', '모델', '고장행', '전체행', '고장비율', 'Accuracy', 'Precision',
               'Recall', 'F1', 'PR_AUC', 'TN', 'FP', 'FN', 'TP']
    print(f'\n{title}\n')
    ordered = frame.assign(_model_order=frame['모델'].map({'기본': 0, 'balanced': 1}))
    ordered = ordered.sort_values(['설비', '_model_order'])
    print(ordered[columns].to_string(index=False, float_format=lambda value: f'{value:.6f}'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path,
                        default=Path.home() / 'Downloads' / 'synthetic_industrial_machine_data.csv',
                        help='원본 CSV 경로 (생략하면 Downloads의 CSV 사용)')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs' / 'pf1_bd',
                        help='결과 저장 폴더')
    args = parser.parse_args()
    source, output = args.data.resolve(), args.output_dir.resolve()
    if not source.is_file():
        parser.error(f'원본 CSV를 찾을 수 없습니다: {source} (--data로 경로를 지정할 수 있습니다)')
    if source == output / 'family_daily_status.csv':
        parser.error('원본 CSV를 결과 파일로 덮어쓸 수 없습니다.')
    before = file_hash(source)
    raw = pd.read_csv(source)
    machine, family = prepare_data(raw)
    groups, summary = split_by_date(machine)
    print(f'원본 {len(raw):,}행 → 설비×날짜 {len(machine):,}행')
    print(f'부품군 {raw.part_family.nunique()}개, 부품군별 세부 부품 수: '
          f'{family.expected_parts.min()}~{family.expected_parts.max()}개')
    print(f'전체 고장 {int(machine[TARGET].sum()):,}행 / {len(machine):,}행 '
          f'({machine[TARGET].mean():.2%})')
    print('\n날짜 구간\n', summary.to_string(index=False))
    print('\n구간별 고장 비율\n', pd.DataFrame([
        {'구간': name, '전체행': len(frame), '고장행': int(frame[TARGET].sum()),
         '고장비율': frame[TARGET].mean()} for name, frame in groups.items()
    ]).to_string(index=False, float_format=lambda value: f'{value:.4%}'))
    results, predictions, columns = run_models(groups)
    print(f'\n모델 입력 X ({len(columns)}개): {columns}')
    print('asset_tag와 part_family는 X에서 제외하고, 설비 구분과 정답 계산에만 사용했습니다.')
    for scope, title in [('전체', '전체 Test 결과'), ('설비 종류', '설비 종류별 Test 결과'),
                         ('개별 설비', '개별 설비별 Test 결과')]:
        show_table(title, results.loc[results['구분'].eq(scope)])
    output.mkdir(parents=True, exist_ok=True)
    ordered_results = results.assign(_scope_order=results['구분'].map(
        {'전체': 0, '설비 종류': 1, '개별 설비': 2}),
        _model_order=results['모델'].map({'기본': 0, 'balanced': 1}))
    ordered_results = ordered_results.sort_values(['_scope_order', '설비', '_model_order'])
    ordered_results.drop(columns=['_scope_order', '_model_order']).to_csv(
        output / 'logistic_results.csv', index=False, encoding='utf-8-sig')
    predictions.to_csv(output / 'logistic_predictions.csv', index=False, encoding='utf-8-sig')
    family.sort_values(KEYS + ['part_family']).to_csv(
        output / 'family_daily_status.csv', index=False, encoding='utf-8-sig')
    summary.to_csv(output / 'date_splits.csv', index=False, encoding='utf-8-sig')
    if file_hash(source) != before:
        raise RuntimeError('실행 중 원본 CSV가 변경됐습니다.')
    print(f'\n원본 변경 없음. 결과 저장 위치: {output}')


if __name__ == '__main__':
    main()
