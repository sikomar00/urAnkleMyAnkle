"""부품·부품군 미래 고장 정답 후보 6개 실험의 공통 코드."""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, auc, confusion_matrix, f1_score,
                             mean_absolute_error, mean_squared_error,
                             precision_recall_curve, precision_recall_fscore_support,
                             precision_score, r2_score, recall_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from .data_huijae2 import (HISTORY, MODEL_PARAMS, PART_CATEGORICAL,
                               PART_NUMERIC)
    from .prepare_machine_data import PART_KEYS, SENSORS, prepare
except ImportError:
    from data_huijae2 import HISTORY, MODEL_PARAMS, PART_CATEGORICAL, PART_NUMERIC
    from prepare_machine_data import PART_KEYS, SENSORS, prepare


ROOT = Path(__file__).resolve().parents[1]
PART_NUMBERS = PART_NUMERIC + HISTORY + SENSORS
PART_CATEGORIES = PART_CATEGORICAL
FAMILY_NUMBERS = ['unit_cost_inr'] + HISTORY + SENSORS
FAMILY_CATEGORIES = ['part_family', 'machine_type', 'plant_code']
SPLITS = ['train', 'validation', 'test']
EXPERIMENTS = {
    'next_day': ('부품 다음 날 고장 표시', 1, 'binary', False),
    'within_3d': ('부품 향후 1~3일 고장 표시', 3, 'binary', False),
    'within_7d': ('부품 향후 1~7일 고장 표시', 7, 'binary', False),
    'count_7d': ('부품 향후 1~7일 고장 표시 일수', 7, 'count', False),
    'first_day_class': ('부품 향후 첫 고장 표시 시점', 7, 'class', False),
    'family_within_3d': ('부품군 향후 1~3일 고장 표시', 3, 'binary', True),
}


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def check_raw_dates(raw, family=False):
    """원본 순서와 관측 기간 중 빠진 달력 날짜를 먼저 확인합니다."""
    keys = ['asset_tag', 'part_family'] if family else ['asset_tag', 'part_no']
    date = pd.to_datetime(raw.transaction_date, errors='raise').dt.normalize()
    frame = raw[keys].copy()
    frame['transaction_date'] = date
    if family:
        frame = frame.drop_duplicates(keys + ['transaction_date'])
    backward = frame.groupby(keys, sort=False).transaction_date.diff().dt.days.lt(0)
    missing = 0
    for _, group in frame.groupby(keys, sort=False):
        days = pd.date_range(group.transaction_date.min(), group.transaction_date.max(), freq='D')
        missing += len(days) - group.transaction_date.nunique()
    return {'reverse_order_rows': int(backward.sum()), 'missing_calendar_days': int(missing),
            'groups': frame.groupby(keys).ngroups}


def make_family_frame(parts):
    """부품군 하루 중 세부 부품 하나라도 고장 표시되면 그날 부품군 표시를 1로 만듭니다."""
    keys = ['asset_tag', 'part_family', 'transaction_date']
    expected = parts.groupby(['asset_tag', 'part_family']).part_no.nunique()
    observed = parts.groupby(keys).part_no.nunique()
    expected_for_day = observed.index.droplevel('transaction_date').map(expected)
    if not observed.eq(expected_for_day).all():
        raise ValueError('일부 설비·부품군·날짜에 세부 부품이 빠져 고장 여부를 판단할 수 없습니다.')
    aggregation = {'breakdown_flag': 'max', 'unit_cost_inr': 'sum',
                   'machine_type': 'first', 'plant_code': 'first'}
    aggregation.update({sensor: 'first' for sensor in SENSORS})
    frame = parts.groupby(keys, as_index=False).agg(aggregation)
    frame = frame.sort_values(keys).reset_index(drop=True)
    frame['days_since_last_observed_breakdown'] = np.nan
    frame['has_prior_observed_breakdown'] = False
    frame['history_30d_complete'] = False
    for days in [7, 30]:
        frame[f'breakdown_days_prev_{days}d'] = np.nan
    for _, group in frame.groupby(['asset_tag', 'part_family'], sort=False):
        daily = group.set_index('transaction_date').reindex(
            pd.date_range(group.transaction_date.min(), group.transaction_date.max(), freq='D'))
        previous = daily.breakdown_flag.shift(1)
        for days in [7, 30]:
            values = previous.rolling(days, min_periods=days).sum()
            frame.loc[group.index, f'breakdown_days_prev_{days}d'] = values.reindex(
                group.transaction_date).to_numpy()
        prior_fault = pd.Series(daily.index, index=daily.index).where(
            daily.breakdown_flag.eq(1)).ffill().shift(1)
        elapsed = (pd.Series(daily.index, index=daily.index) - prior_fault).dt.days
        frame.loc[group.index, 'days_since_last_observed_breakdown'] = elapsed.reindex(
            group.transaction_date).to_numpy()
        frame.loc[group.index, 'has_prior_observed_breakdown'] = prior_fault.notna().reindex(
            group.transaction_date).to_numpy()
        frame.loc[group.index, 'history_30d_complete'] = previous.rolling(
            30, min_periods=30).count().eq(30).reindex(group.transaction_date).to_numpy()
    return frame


def future_labels(frame, keys, days, task):
    """현재 정상행을 고르기 전에 전체 시계열의 실제 달력 날짜로 미래 정답을 만듭니다."""
    labels = np.full(len(frame), np.nan)
    complete = np.zeros(len(frame), dtype=bool)
    missing_dates = 0
    for _, group in frame.groupby(keys, sort=False):
        ordered = group.sort_values('transaction_date')
        daily = ordered.set_index('transaction_date').breakdown_flag.reindex(
            pd.date_range(ordered.transaction_date.min(), ordered.transaction_date.max(), freq='D'))
        missing_dates += int(daily.isna().sum())
        future = np.column_stack([daily.shift(-step).to_numpy(dtype=float)
                                  for step in range(1, days + 1)])
        positions = daily.index.get_indexer(ordered.transaction_date)
        window = future[positions]
        valid = np.isfinite(window).all(axis=1)
        safe = np.nan_to_num(window, nan=0.0)
        if task == 'binary':
            value = safe.max(axis=1)
        elif task == 'count':
            value = safe.sum(axis=1)
        else:
            has_fault = safe.eq(1) if isinstance(safe, pd.DataFrame) else safe == 1
            first = has_fault.argmax(axis=1) + 1
            value = np.select([~has_fault.any(axis=1), first >= 4, first >= 2,
                               first == 1], [0, 1, 2, 3])
        labels[ordered.index] = np.where(valid, value, np.nan)
        complete[ordered.index] = valid
    return labels, complete, missing_dates


def prepare_experiment(raw, experiment):
    """기존 부품 C 실험의 전처리를 재사용하고 정답만 새로 만듭니다."""
    title, days, task, family = EXPERIMENTS[experiment]
    date_audit = check_raw_dates(raw, family=family)
    _, parts, _, audit = prepare(raw)
    frame = make_family_frame(parts) if family else parts.copy()
    keys = ['asset_tag', 'part_family'] if family else ['asset_tag', 'part_no']
    if frame.duplicated(keys + ['transaction_date']).any():
        raise ValueError('설비·부품(군)·날짜가 중복되어 있습니다.')
    labels, future_complete, missing_days = future_labels(frame, keys, days, task)
    if missing_days != date_audit['missing_calendar_days']:
        raise RuntimeError('날짜 누락 검사와 정답 생성의 누락 날짜 수가 다릅니다.')
    frame['target'] = labels
    frame['future_complete'] = future_complete
    train_end = pd.Timestamp(audit['split_dates']['train_end'])
    validation_end = pd.Timestamp(audit['split_dates']['validation_end'])
    test_end = pd.Timestamp(audit['split_dates']['test_end'])
    frame['split'] = np.select([frame.transaction_date.le(train_end),
                                frame.transaction_date.le(validation_end)],
                               ['train', 'validation'], default='test')
    boundary = frame.split.map({'train': train_end, 'validation': validation_end,
                                'test': test_end})
    frame['target_window_end'] = frame.transaction_date + pd.Timedelta(days=days)
    frame['within_split'] = frame.target_window_end.le(boundary)
    # 제외 이유는 서로 겹치지 않게 우선순위대로 표시합니다.
    frame['exclusion_reason'] = np.select(
        [frame.breakdown_flag.ne(0), ~frame.history_30d_complete.astype(bool),
         ~frame.future_complete, ~frame.within_split],
        ['오늘 고장 표시', '이전 30일 기록 부족', '미래 날짜 누락·기간 끝',
         '미래 정답이 다음 구간 침범'], default='사용')
    usable = frame.loc[frame.exclusion_reason.eq('사용')].copy()
    usable['target'] = usable.target.astype(int)
    if not usable.breakdown_flag.eq(0).all() or usable.target_window_end.gt(
            usable.split.map({'train': train_end, 'validation': validation_end,
                              'test': test_end})).any():
        raise RuntimeError('오늘 고장 행 또는 구간 밖 미래 정보가 학습 대상에 섞였습니다.')
    if family:
        numeric, categorical = FAMILY_NUMBERS, FAMILY_CATEGORIES
    else:
        numeric, categorical = PART_NUMBERS, PART_CATEGORIES
    forbidden = {'asset_tag', 'breakdown_flag', 'wo_type', 'qty_issued',
                 'issue_value_inr', 'target', 'target_breakdown_next_7d',
                 'future_complete', 'target_window_end'}
    columns = numeric + categorical
    if set(columns) & forbidden or len(columns) != len(set(columns)):
        raise ValueError('모델 입력에 미래 정보·정답·설비 ID 또는 중복 열이 있습니다.')
    splits = {name: usable.loc[usable.split.eq(name)].copy() for name in SPLITS}
    if any(group.empty for group in splits.values()):
        raise ValueError('어떤 날짜 구간에는 사용 가능한 행이 없습니다.')
    dates = pd.DataFrame([{'split': name, 'date_start': group.transaction_date.min().date(),
                           'date_end': group.transaction_date.max().date(), 'rows': len(group)}
                          for name, group in splits.items()])
    exclusions = frame.groupby(['split', 'exclusion_reason'], dropna=False).size().rename('rows').reset_index()
    return title, task, days, family, splits, dates, exclusions, date_audit, columns, numeric, categorical


def preprocessing(numeric, categorical):
    """기존 부품 C 모델처럼 Train에서만 결측·표준화·범주 변환을 학습합니다."""
    numeric_pipe = Pipeline([('imputer', SimpleImputer(strategy='median', add_indicator=True)),
                             ('scaler', StandardScaler())])
    categorical_pipe = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                                 ('encoder', OneHotEncoder(handle_unknown='ignore'))])
    return ColumnTransformer([('numeric', numeric_pipe, numeric),
                              ('categorical', categorical_pipe, categorical)])


def make_model(name, numeric, categorical):
    transform = preprocessing(numeric, categorical)
    if name == '로지스틱 회귀':
        estimator = LogisticRegression(**MODEL_PARAMS)
    elif name == '랜덤 포레스트':
        estimator = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    elif name == '학습 평균':
        estimator = DummyRegressor(strategy='mean')
    elif name == '릿지 회귀':
        estimator = Ridge(alpha=1.0)
    elif name == '랜덤 포레스트 회귀':
        estimator = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    else:
        raise ValueError(name)
    return Pipeline([('features', transform), ('model', estimator)])


def binary_metrics(actual, probability):
    prediction = (probability >= .5).astype(int)
    tn, fp, fn, tp = confusion_matrix(actual, prediction, labels=[0, 1]).ravel()
    if pd.Series(actual).nunique() == 2:
        precision, recall, _ = precision_recall_curve(actual, probability)
        pr_auc = float(auc(recall, precision))
    else:
        pr_auc = np.nan
    notes = []
    if prediction.sum() == 0:
        notes.append('예측 경보 0건: Precision 해석 불가')
    if np.sum(actual) == 0:
        notes.append('실제 고장 0건: Recall 해석 불가')
    return {'Accuracy': accuracy_score(actual, prediction),
            'Precision': precision_score(actual, prediction, zero_division=0),
            'Recall': recall_score(actual, prediction, zero_division=0),
            'F1': f1_score(actual, prediction, zero_division=0),
            'PR_AUC': pr_auc, 'TN': int(tn), 'FP': int(fp), 'FN': int(fn),
            'TP': int(tp), 'predicted_alerts': int(prediction.sum()),
            'probability_cutoff': .5, 'note': '; '.join(notes)}


def binary_run(splits, columns, numeric, categorical):
    train, test = splits['train'], splits['test']
    if train.target.nunique() != 2:
        raise ValueError('Train에 정답 0과 1이 모두 있어야 합니다.')
    results, assets, predictions = [], [], []
    for name in ['로지스틱 회귀', '랜덤 포레스트']:
        model = make_model(name, numeric, categorical)
        model.fit(train[columns], train.target)
        if list(model.named_steps['features'].feature_names_in_) != columns:
            raise RuntimeError('실제 학습 입력 목록이 허용 변수와 다릅니다.')
        probability = model.predict_proba(test[columns])[:, 1]
        metrics = binary_metrics(test.target.to_numpy(), probability)
        results.append({'model': name, 'rows': len(test), 'actual_failures': int(test.target.sum()),
                        'failure_rate': float(test.target.mean()),
                        'always_normal_accuracy': float(1 - test.target.mean()), **metrics})
        identity = ['transaction_date', 'asset_tag', 'part_family']
        if 'part_no' in test:
            identity.insert(2, 'part_no')
        pred = test[identity + ['target']].copy()
        pred['model'] = name
        pred['probability'] = probability
        pred['predicted_label'] = (probability >= .5).astype(int)
        predictions.append(pred)
        for asset, group in test.groupby('asset_tag', sort=True):
            position = test.index.get_indexer(group.index)
            one = binary_metrics(group.target.to_numpy(), probability[position])
            assets.append({'model': name, 'asset_tag': asset, 'rows': len(group),
                           'actual_failures': int(group.target.sum()),
                           'failure_rate': float(group.target.mean()), **one})
    return pd.DataFrame(results), pd.DataFrame(assets), pd.concat(predictions, ignore_index=True)


def count_run(splits, columns, numeric, categorical):
    train, test = splits['train'], splits['test']
    results, assets, predictions = [], [], []
    for name in ['학습 평균', '릿지 회귀', '랜덤 포레스트 회귀']:
        model = make_model(name, numeric, categorical)
        model.fit(train[columns], train.target)
        predicted = model.predict(test[columns])
        results.append({'model': name, 'rows': len(test),
                        'target_mean': float(test.target.mean()),
                        'zero_day_rate': float(test.target.eq(0).mean()),
                        'MAE': mean_absolute_error(test.target, predicted),
                        'RMSE': float(np.sqrt(mean_squared_error(test.target, predicted))),
                        'R2': r2_score(test.target, predicted)})
        pred = test[['transaction_date', 'asset_tag', 'part_no', 'target']].copy()
        pred['model'] = name
        pred['predicted_count'] = predicted
        predictions.append(pred)
        for asset, group in test.groupby('asset_tag', sort=True):
            one = predicted[test.index.get_indexer(group.index)]
            assets.append({'model': name, 'asset_tag': asset, 'rows': len(group),
                           'target_mean': float(group.target.mean()),
                           'zero_day_rate': float(group.target.eq(0).mean()),
                           'MAE': mean_absolute_error(group.target, one),
                           'RMSE': float(np.sqrt(mean_squared_error(group.target, one)))})
    return pd.DataFrame(results), pd.DataFrame(assets), pd.concat(predictions, ignore_index=True)


def class_run(splits, columns, numeric, categorical):
    train, test = splits['train'], splits['test']
    results, per_class, matrices, assets, predictions = [], [], [], [], []
    for name in ['로지스틱 회귀', '랜덤 포레스트']:
        model = make_model(name, numeric, categorical)
        model.fit(train[columns], train.target)
        predicted = model.predict(test[columns])
        results.append({'model': name, 'rows': len(test),
                        'Accuracy': accuracy_score(test.target, predicted),
                        'Macro_F1': f1_score(test.target, predicted, labels=[0, 1, 2, 3],
                                             average='macro', zero_division=0)})
        precision, recall, f1, support = precision_recall_fscore_support(
            test.target, predicted, labels=[0, 1, 2, 3], zero_division=0)
        for level in range(4):
            per_class.append({'model': name, 'class': level, 'actual_rows': int(support[level]),
                              'predicted_rows': int(np.sum(predicted == level)),
                              'Precision': precision[level], 'Recall': recall[level], 'F1': f1[level]})
        matrix = confusion_matrix(test.target, predicted, labels=[0, 1, 2, 3])
        for actual_level in range(4):
            for predicted_level in range(4):
                matrices.append({'model': name, 'actual_class': actual_level,
                                 'predicted_class': predicted_level,
                                 'rows': int(matrix[actual_level, predicted_level])})
        pred = test[['transaction_date', 'asset_tag', 'part_no', 'target']].copy()
        pred['model'] = name
        pred['predicted_class'] = predicted
        predictions.append(pred)
        for asset, group in test.groupby('asset_tag', sort=True):
            one = predicted[test.index.get_indexer(group.index)]
            row = {'model': name, 'asset_tag': asset, 'rows': len(group)}
            for level in range(4):
                row[f'actual_{level}'] = int(group.target.eq(level).sum())
                row[f'predicted_{level}'] = int(np.sum(one == level))
            assets.append(row)
    return tuple(pd.DataFrame(items) for items in [results, per_class, matrices,
                                                    assets, pd.concat(predictions, ignore_index=True)])


def show(title, frame, columns=None):
    print(f'\n{title}\n')
    print(frame[columns if columns else frame.columns.tolist()].to_string(
        index=False, max_rows=None, float_format=lambda value: f'{value:.4f}'))


def save(frame, path):
    frame.to_csv(path, index=False, encoding='utf-8-sig')


def run(experiment):
    title, days, task, family = EXPERIMENTS[experiment]
    parser = argparse.ArgumentParser(description=title)
    parser.add_argument('--data', type=Path,
                        default=Path.home() / 'Downloads' / 'synthetic_industrial_machine_data.csv',
                        help='원본 CSV 경로. 생략하면 Downloads 파일을 사용합니다.')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs' / f'pf_{experiment}',
                        help='결과 CSV 저장 폴더')
    args = parser.parse_args()
    source, output = args.data.resolve(), args.output_dir.resolve()
    if not source.is_file():
        parser.error(f'원본 CSV를 찾을 수 없습니다: {source}')
    if output in source.parents:
        parser.error('원본이 결과 저장 폴더 안에 있어 덮어쓸 수 있습니다.')
    before = digest(source)
    raw = pd.read_csv(source)
    (title, task, days, family, splits, dates, exclusions,
     date_audit, columns, numeric, categorical) = prepare_experiment(raw, experiment)
    print(f'{title}: 부품별 고장 표시를 예측합니다. 설비 전체 정지 예측은 아닙니다.')
    print('정답은 전체 날짜 기록에서 먼저 만들고, 오늘 고장 표시 0인 행만 남겼습니다.')
    print(f'원본 {len(raw):,}행 / 날짜 역순 {date_audit["reverse_order_rows"]:,}행 / '
          f'누락 달력 날짜 {date_audit["missing_calendar_days"]:,}일 / '
          f'시계열 {date_audit["groups"]:,}개')
    print(f'관측 단위: {"설비×부품군×날짜" if family else "설비×부품×날짜"}; '
          f'미래 정답: 실제 달력상 다음 {days}일')
    print(f'입력변수 ({len(columns)}개): {columns}')
    print('asset_tag·wo_type·당일/미래 고장 표시·미래 센서·미래 작업 정보는 X에서 제외했습니다.')
    print('결측·표준화·범주 변환은 Train에서만 학습합니다.')
    show('날짜별 사용 행', dates)
    show('제외 이유별 행 수', exclusions)
    rows = pd.DataFrame([{'split': name, 'rows': len(group),
                          'positive_rows': int(group.target.gt(0).sum()),
                          'positive_rate': float(group.target.gt(0).mean())}
                         for name, group in splits.items()])
    if task == 'binary':
        show('1. Train·Validation·Test 고장 표시 비율', rows)
        overall, assets, predictions = binary_run(splits, columns, numeric, categorical)
        show('2. 모델별 Test 결과', overall,
             ['model', 'rows', 'actual_failures', 'failure_rate', 'Accuracy',
              'Precision', 'Recall', 'F1', 'PR_AUC', 'TN', 'FP', 'FN', 'TP',
              'predicted_alerts', 'probability_cutoff'])
        show('3. 설비별 Test 결과', assets,
             ['model', 'asset_tag', 'rows', 'actual_failures', 'failure_rate',
              'TP', 'FP', 'FN', 'TN', 'Precision', 'Recall', 'F1', 'note'])
        show('4. 항상 정상 예측한 Accuracy', overall,
             ['model', 'actual_failures', 'failure_rate', 'always_normal_accuracy'])
    elif task == 'count':
        print('정답은 실제 수리 횟수가 아니라 향후 7일 동안 데이터에 표시된 고장 일수(0~7)입니다.')
        overall, assets, predictions = count_run(splits, columns, numeric, categorical)
        show('Test MAE·RMSE·R²와 고장 일수 분포', overall)
        show('설비별 MAE·RMSE', assets)
    else:
        print('단계 3은 오늘 확정 고장이 아니라 다음 날 첫 고장 표시입니다.')
        distribution = pd.DataFrame([{'split': name, 'class': level,
                                      'rows': int(group.target.eq(level).sum()),
                                      'rate': float(group.target.eq(level).mean())}
                                     for name, group in splits.items() for level in range(4)])
        show('네 단계의 실제 건수·비율', distribution)
        overall, per_class, matrices, assets, predictions = class_run(
            splits, columns, numeric, categorical)
        show('모델별 Accuracy·Macro F1', overall)
        show('각 단계 Precision·Recall·F1', per_class)
        for name in overall.model:
            matrix = matrices.loc[matrices.model.eq(name)].pivot(
                index='actual_class', columns='predicted_class', values='rows')
            print(f'\n{name} 4×4 혼동행렬 (행=실제, 열=예측)\n{matrix.to_string()}')
        show('설비별 실제·예측 단계 건수', assets)
    output.mkdir(parents=True, exist_ok=True)
    save(overall, output / 'overall_results.csv')
    save(assets, output / 'asset_results.csv')
    save(predictions, output / 'predictions.csv')
    save(dates, output / 'date_splits.csv')
    save(exclusions, output / 'exclusions.csv')
    save(rows, output / 'label_distribution.csv')
    if task == 'class':
        save(distribution, output / 'class_distribution.csv')
        save(per_class, output / 'per_class_results.csv')
        save(matrices, output / 'confusion_matrix.csv')
    if digest(source) != before:
        raise RuntimeError('실행 중 원본 CSV가 변경됐습니다.')
    print(f'원본 변경 없음. 결과 저장 위치: {output}')
    if family:
        print('부품군 3일 모델은 부품 3일 모델과 관측 단위·고장 비율이 달라 점수를 그대로 우열 비교할 수 없습니다.')
