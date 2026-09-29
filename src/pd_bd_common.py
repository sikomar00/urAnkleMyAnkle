"""설비 고장 정답 후보 다섯 실험에서 공통으로 쓰는 집계·학습·출력 코드."""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

try:
    from .huijae_example import (CATEGORICAL, KEYS, SENSORS, TEMPORAL_NUMERIC,
                                 current_day_pipeline, evaluate, prepare_machine_data,
                                 split_by_date)
except ImportError:
    from huijae_example import (CATEGORICAL, KEYS, SENSORS, TEMPORAL_NUMERIC,
                                current_day_pipeline, evaluate, prepare_machine_data,
                                split_by_date)


ROOT = Path(__file__).resolve().parents[1]
FEATURE_SETS = {
    '센서 8개': (SENSORS, []),
    '센서+과거변화+설비정보': (TEMPORAL_NUMERIC, CATEGORICAL),
}
SPLIT_ORDER = ['전체', 'Train', 'Validation', 'Test']
METRICS = ['Accuracy', 'Precision', 'Recall', 'F1', 'PR_AUC', 'TN', 'FP', 'FN', 'TP']


def file_hash(path):
    """원본 CSV가 실행 중 바뀌지 않았는지 확인하기 위한 값입니다."""
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def build_labels(raw, machine, experiment):
    """동일한 설비·날짜 표 위에 실험별 정답 0/1을 만듭니다."""
    data = machine.copy()
    labels = {}
    note = ''
    if experiment == 'CO':
        for threshold in [3, 4, 5, 6, 7]:
            code = f'failed_parts_ge_{threshold}'
            labels[code] = f'고장 부품 {threshold}개 이상'
            data[code] = data.failed_parts.ge(threshold).astype(int)
    elif experiment == 'A':
        rules = {
            'A_ge_1': ('A등급 1개 이상', data.failed_A.ge(1)),
            'A_ge_2': ('A등급 2개 이상', data.failed_A.ge(2)),
            'A_ge_1_and_total_ge_2': ('A등급 1개 이상 + 전체 2개 이상',
                                      data.failed_A.ge(1) & data.failed_parts.ge(2)),
            'A_ge_2_or_total_ge_4': ('A등급 2개 이상 또는 전체 4개 이상',
                                     data.failed_A.ge(2) | data.failed_parts.ge(4)),
        }
        for code, (description, mask) in rules.items():
            labels[code] = description
            data[code] = mask.astype(int)
    elif experiment == 'COST':
        if 'unit_cost_inr' not in raw:
            raise ValueError('원본에 unit_cost_inr 열이 없습니다.')
        parts = raw[KEYS + ['breakdown_flag', 'unit_cost_inr']].copy()
        parts.transaction_date = pd.to_datetime(parts.transaction_date, errors='raise').dt.normalize()
        cost = pd.to_numeric(parts.unit_cost_inr, errors='raise')
        flag = pd.to_numeric(parts.breakdown_flag, errors='raise')
        if cost.isna().any() or not np.isfinite(cost).all() or cost.lt(0).any():
            raise ValueError('unit_cost_inr에 빈값·음수·무한대가 있습니다.')
        parts['failed_unit_cost_inr'] = cost.where(flag.eq(1), 0)
        total = parts.groupby(KEYS, as_index=False).failed_unit_cost_inr.sum()
        data = data.merge(total, on=KEYS, how='left', validate='one_to_one')
        if data.failed_unit_cost_inr.isna().any():
            raise ValueError('일부 설비·날짜의 고장 부품 단가 합계가 없습니다.')
        for threshold in [1000, 2500, 5000, 10000, 20000]:
            code = f'failed_unit_cost_ge_{threshold}'
            labels[code] = f'고장 표시 부품 단가 합계 {threshold:,} INR 이상'
            data[code] = data.failed_unit_cost_inr.ge(threshold).astype(int)
        note = '금액은 실제 수리비가 아니라 그날 고장 표시된 부품의 unit_cost_inr 합계입니다.'
    elif experiment in {'START', 'CONSEC'}:
        data = data.sort_values(['asset_tag', 'transaction_date']).reset_index(drop=True)
        previous_date = data.groupby('asset_tag').transaction_date.shift(1)
        consecutive = data.transaction_date.sub(previous_date).dt.days.eq(1)
        first_days = int(previous_date.isna().sum())
        gaps = int((previous_date.notna() & ~consecutive).sum())
        note = (f'전날 기록 없음: 설비별 첫날 {first_days}행, 날짜 공백 {gaps}행. '
                '이 행은 정답을 알 수 없어 0으로 만들지 않고 학습·평가에서 제외합니다.')
        previous_parts = data.groupby('asset_tag').failed_parts.shift(1)
        if experiment == 'START':
            previous_severity = data.groupby('asset_tag').severity_score.shift(1)
            rules = {
                'start_parts_ge_3': ('전날 고장 부품 3개 미만 → 오늘 3개 이상',
                                     previous_parts.lt(3) & data.failed_parts.ge(3)),
                'start_severity_ge_10': ('전날 중요도 10점 미만 → 오늘 10점 이상',
                                         previous_severity.lt(10) & data.severity_score.ge(10)),
            }
        else:
            rules = {
                'consec_parts_ge_1': ('전날·오늘 고장 부품 모두 1개 이상',
                                      previous_parts.ge(1) & data.failed_parts.ge(1)),
                'consec_parts_ge_2': ('전날·오늘 고장 부품 모두 2개 이상',
                                      previous_parts.ge(2) & data.failed_parts.ge(2)),
            }
        for code, (description, mask) in rules.items():
            labels[code] = description
            data[code] = np.where(consecutive, mask.astype(int), np.nan)
        data = data.sort_values(['transaction_date', 'asset_tag']).reset_index(drop=True)
    else:
        raise ValueError(f'알 수 없는 실험: {experiment}')
    return data, labels, note


def split_and_rates(data, labels):
    """기존과 똑같은 날짜 경계로 나눈 뒤 정답을 모르는 행만 제외합니다."""
    all_groups, dates = split_by_date(data)
    valid = data.dropna(subset=list(labels)).copy()
    groups = {name: group.dropna(subset=list(labels)).copy() for name, group in all_groups.items()}
    for frame in [valid, *groups.values()]:
        for code in labels:
            frame[code] = frame[code].astype(int)
    if not groups['Train'].transaction_date.max() < groups['Validation'].transaction_date.min():
        raise ValueError('Train과 Validation 날짜가 겹칩니다.')
    if not groups['Validation'].transaction_date.max() < groups['Test'].transaction_date.min():
        raise ValueError('Validation과 Test 날짜가 겹칩니다.')
    rows = []
    for code, description in labels.items():
        for name, frame in [('전체', valid), *groups.items()]:
            count = int(frame[code].sum())
            rows.append({'criterion': code, 'description': description, 'dataset': name,
                         'rows': len(frame), 'failures': count,
                         'failure_rate': count / len(frame) if len(frame) else np.nan,
                         'always_normal_accuracy': 1 - count / len(frame) if len(frame) else np.nan})
    return groups, dates, pd.DataFrame(rows)


def model_pipeline(numeric_columns, categorical_columns, algorithm, weight):
    """두 알고리즘 모두 같은 Train 전처리를 거칩니다."""
    pipeline = current_day_pipeline(numeric_columns, categorical_columns, weight)
    if algorithm == '랜덤 포레스트':
        pipeline.set_params(classifier=RandomForestClassifier(
            n_estimators=100, random_state=42, class_weight=weight, n_jobs=-1))
    return pipeline


def undefined_reason(actual, predicted):
    """분모가 0이라 수치가 해석되지 않는 경우를 표에 남깁니다."""
    reasons = []
    if int(predicted.sum()) == 0:
        reasons.append('예측 고장 0건: Precision 해석 불가')
    if int(actual.sum()) == 0:
        reasons.append('실제 고장 0건: Recall 해석 불가')
    if int(actual.sum()) == len(actual):
        reasons.append('실제 정상 0건: PR-AUC 해석 불가')
    if actual.nunique() == 1:
        reasons.append('실제 정답 한 종류: PR-AUC 계산 불가')
    return '; '.join(reasons)


def result_metrics(actual, probabilities):
    """기존 중요도 점수 실험의 지표 계산 방식과 동일합니다."""
    score = evaluate(actual, probabilities, cutoff=.5)
    predicted = probabilities >= .5
    return {key: score[key] for key in METRICS} | {
        'predicted_failures': int(predicted.sum()), 'probability_cutoff': .5,
        'note': undefined_reason(actual, predicted)}


def run_models(groups, labels):
    """기준별·입력별·알고리즘별·가중치별로 학습하고 Test에서 평가합니다."""
    train, test = groups['Train'], groups['Test']
    forbidden = {'asset_tag', 'breakdown_flag', 'wo_type', 'criticality',
                 'severity_score', 'failed_parts', 'failed_A', 'failed_B', 'failed_C',
                 'failed_unit_cost_inr', 'qty_issued', 'issue_value_inr', *labels}
    results, asset_results = [], []
    for code, description in labels.items():
        actual = test[code]
        for feature_set, (numeric, categorical) in FEATURE_SETS.items():
            columns = numeric + categorical
            if len(columns) != len(set(columns)) or set(columns) & forbidden:
                raise ValueError(f'{feature_set}의 X에 정답 또는 중복 열이 들어갔습니다.')
            if 'asset_tag' in columns:
                raise ValueError('asset_tag는 X에 넣을 수 없습니다.')
            for algorithm in ['로지스틱 회귀', '랜덤 포레스트']:
                for variant, weight in [('기본', None), ('balanced', 'balanced')]:
                    common = {'criterion': code, 'description': description,
                              'feature_set': feature_set, 'algorithm': algorithm,
                              'variant': variant, 'test_rows': len(test),
                              'actual_failures': int(actual.sum()),
                              'failure_rate': float(actual.mean()),
                              'always_normal_accuracy': float(1 - actual.mean())}
                    if train[code].nunique() < 2:
                        status = '학습 불가: Train에 정상과 고장이 모두 있지 않음'
                        results.append(common | {key: np.nan for key in METRICS} |
                                       {'predicted_failures': np.nan, 'probability_cutoff': .5,
                                        'note': status, 'status': status})
                        for asset, frame in test.groupby('asset_tag', sort=True):
                            asset_results.append(common | {'asset_tag': asset,
                                'rows': len(frame), 'actual_failures': int(frame[code].sum()),
                                'failure_rate': float(frame[code].mean()),
                                'predicted_failures': np.nan,
                                **{key: np.nan for key in ['TP', 'FP', 'FN', 'TN', 'Precision', 'Recall', 'F1']},
                                'note': status, 'status': status})
                        continue
                    pipeline = model_pipeline(numeric, categorical, algorithm, weight)
                    pipeline.fit(train[columns], train[code])
                    if list(pipeline.named_steps['preprocess'].feature_names_in_) != columns:
                        raise RuntimeError('실제로 학습한 X와 허용 변수 목록이 다릅니다.')
                    probabilities = pipeline.predict_proba(test[columns])[:, 1]
                    metrics = result_metrics(actual, probabilities)
                    results.append(common | metrics | {'status': 'ok'})
                    for asset, frame in test.groupby('asset_tag', sort=True):
                        positions = test.index.get_indexer(frame.index)
                        subset = probabilities[positions]
                        one = result_metrics(frame[code], subset)
                        asset_results.append(common | {'asset_tag': asset,
                            'rows': len(frame), 'actual_failures': int(frame[code].sum()),
                            'failure_rate': float(frame[code].mean()),
                            'predicted_failures': one['predicted_failures'],
                            **{key: one[key] for key in ['TP', 'FP', 'FN', 'TN', 'Precision', 'Recall', 'F1']},
                            'note': one['note'], 'status': 'ok'})
                    print(f'완료: {description} / {feature_set} / {algorithm} / {variant}', flush=True)
    return pd.DataFrame(results), pd.DataFrame(asset_results)


def print_table(title, frame, columns):
    print(f'\n{title}\n')
    print(frame[columns].to_string(index=False, max_rows=None,
          float_format=lambda value: f'{value:.4f}'))


def interpret(results, rates):
    """점수와 정답 범위를 구분해 설명하며 현장 기준을 확정하지 않습니다."""
    valid = results.loc[results.status.eq('ok')]
    if not valid.empty:
        best = valid.sort_values('F1', ascending=False).iloc[0]
        print(f'\nTest F1 최고 조합: {best.description} / {best.feature_set} / '
              f'{best.algorithm} / {best.variant} / F1={best.F1:.3f}.')
        with_pr_auc = valid.dropna(subset=['PR_AUC'])
        if not with_pr_auc.empty:
            best_pr = with_pr_auc.sort_values('PR_AUC', ascending=False).iloc[0]
            print(f'Test PR-AUC 최고 조합: {best_pr.description} / {best_pr.feature_set} / '
                  f'{best_pr.algorithm} / {best_pr.variant} / PR-AUC={best_pr.PR_AUC:.3f}.')
    whole = rates.loc[rates.dataset.eq('전체')]
    widest = whole.sort_values('failure_rate', ascending=False).iloc[0]
    narrowest = whole.sort_values('failure_rate').iloc[0]
    print(f'이 실험에서 가장 넓은 기준: {widest.description} (고장일 {widest.failure_rate:.2%}).')
    print(f'이 실험에서 가장 좁은 기준: {narrowest.description} (고장일 {narrowest.failure_rate:.2%}).')
    broad = whole.loc[whole.failure_rate.ge(.40), 'description'].tolist()
    narrow = whole.loc[whole.failure_rate.le(.01), 'description'].tolist()
    print('고장일 비율 40% 이상인 넓은 후보:', ', '.join(broad) if broad else '없음')
    print('고장일 비율 1% 이하인 좁은 후보:', ', '.join(narrow) if narrow else '없음')
    print('40%·1%는 비교를 돕기 위한 표시 기준이며 실제 고장 정의가 아닙니다.')
    print('모델 점수가 높아도 실제 설비 고장 기준이 확정되는 것은 아닙니다.')
    print('이번 실험은 향후 예지보전용 정답 후보 비교이며, 미래 3일·7일 예측은 하지 않았습니다.')


def run_experiment(experiment):
    parser = argparse.ArgumentParser(description=f'{experiment}: 설비 고장 기준 후보 비교')
    parser.add_argument('--data', type=Path,
                        default=Path.home() / 'Downloads' / 'synthetic_industrial_machine_data.csv',
                        help='원본 CSV 경로. 생략하면 Downloads의 원본 CSV를 사용합니다.')
    parser.add_argument('--output-dir', type=Path,
                        default=ROOT / 'outputs' / f'pd_bd_{experiment.lower()}',
                        help='CSV 저장 폴더')
    args = parser.parse_args()
    source, output = args.data.resolve(), args.output_dir.resolve()
    if not source.is_file():
        parser.error(f'원본 CSV를 찾을 수 없습니다: {source}')
    if source == output or output in source.parents:
        parser.error('원본 CSV가 결과 저장 폴더 안에 있어 덮어쓰기 위험이 있습니다.')
    before = file_hash(source)
    raw = pd.read_csv(source)
    machine = prepare_machine_data(raw)
    data, labels, note = build_labels(raw, machine, experiment)
    groups, dates, rates = split_and_rates(data, labels)
    print(f'원본 {len(raw):,}행 → 설비×날짜 {len(machine):,}행')
    print('입력 ① 센서 8개, 입력 ② 센서·과거변화 56개 + 설비 종류·공장 2개')
    print('모델 입력에서 asset_tag와 고장 정답·집계값을 제외했습니다. 확률 판정 기준은 모두 0.5입니다.')
    if note:
        print(note)
    print_table('날짜 분할', dates, ['split', 'start', 'end', 'rows', 'dates'])
    print_table('1. 기준별 전체·Train·Validation·Test 고장률', rates,
                ['description', 'dataset', 'rows', 'failures', 'failure_rate',
                 'always_normal_accuracy'])
    results, asset_results = run_models(groups, labels)
    print_table('2. 기준 × 입력변수 구성 × 모델별 Test 결과', results,
                ['description', 'feature_set', 'algorithm', 'variant', 'test_rows',
                 'actual_failures', 'failure_rate', 'Accuracy', 'Precision', 'Recall',
                 'F1', 'PR_AUC', 'TN', 'FP', 'FN', 'TP', 'predicted_failures',
                 'probability_cutoff', 'status'])
    print_table('3. 개별 설비별 Test 결과', asset_results,
                ['description', 'feature_set', 'algorithm', 'variant', 'asset_tag',
                 'rows', 'actual_failures', 'failure_rate', 'predicted_failures',
                 'TP', 'FP', 'FN', 'TN', 'Precision', 'Recall', 'F1', 'note'])
    print_table('4. 전부 정상으로 예측할 때의 Accuracy와 실제 고장 비율',
                rates.loc[rates.dataset.eq('Test')],
                ['description', 'rows', 'failures', 'failure_rate', 'always_normal_accuracy'])
    interpret(results, rates)
    output.mkdir(parents=True, exist_ok=True)
    rates.to_csv(output / 'label_rates.csv', index=False, encoding='utf-8-sig')
    results.to_csv(output / 'overall_results.csv', index=False, encoding='utf-8-sig')
    asset_results.to_csv(output / 'asset_results.csv', index=False, encoding='utf-8-sig')
    dates.to_csv(output / 'date_splits.csv', index=False, encoding='utf-8-sig')
    if file_hash(source) != before:
        raise RuntimeError('실행 중 원본 CSV가 변경됐습니다.')
    print(f'원본 변경 없음. CSV 저장 위치: {output}')
