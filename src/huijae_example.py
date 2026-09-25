"""당일 센서로 같은 날의 고장 심각도를 분류하는 로지스틱 회귀 비교입니다.

실행 예: python src/huijae_example.py --data "C:/data/synthetic_industrial_machine_data.csv"
필요 패키지: numpy, pandas, scikit-learn, matplotlib
미래 정답은 만들지 않습니다. 과거 센서 통계는 해당 설비의 이전 기록만 사용합니다.
"""

import argparse  # 명령창에서 CSV 경로와 실행 옵션을 받습니다.
import hashlib  # 실행 전후 원본 내용이 같은지 확인합니다.
import os  # 그래프 글꼴 캐시를 쓸 수 있는 폴더로 지정합니다.
import warnings  # 모델 학습이 충분히 끝나지 않으면 알려 줍니다.
from pathlib import Path  # Windows 파일 경로를 다룹니다.

import numpy as np  # 숫자 배열과 계산에 사용합니다.
import pandas as pd  # CSV를 표 형태로 읽고 가공합니다.
from sklearn.compose import ColumnTransformer  # 숫자형과 범주형에 서로 다른 전처리를 적용합니다.
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer  # 빈 센서값을 Train 중앙값으로 채웁니다.
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, auc, average_precision_score,
                             confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline  # 결측값 처리와 표준화를 순서대로 수행합니다.
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # 범주 변환과 숫자 크기 조정에 사용합니다.

# X는 이 목록으로만 선택합니다. 정답 관련 열이 추가되어도 입력에 섞이지 않습니다.
SENSORS = ['temp_bearing_degC', 'temp_motor_degC', 'vibration_h_mms',
           'vibration_v_mms', 'oil_pressure_bar', 'load_pct', 'shaft_rpm',
           'power_consumption_kw']
KEYS = ['asset_tag', 'transaction_date']
WEIGHTS = {'A': 4, 'B': 2, 'C': 1}
severity_thresholds = [7, 10, 11, 12, 13, 14]
THRESHOLDS = severity_thresholds
CATEGORICAL = ['machine_type', 'plant_code']  # 설비 번호는 모델 입력에 넣지 않습니다.
DIFF_FEATURES = [f'{sensor}_diff1' for sensor in SENSORS]
MEAN7_FEATURES = [f'{sensor}_mean7' for sensor in SENSORS]
STD7_FEATURES = [f'{sensor}_std7' for sensor in SENSORS]
VS_MEAN7_FEATURES = [f'{sensor}_vs_mean7' for sensor in SENSORS]
ANOMALY7_FEATURES = [f'{sensor}_anomaly7' for sensor in SENSORS]
VS_MEAN30_FEATURES = [f'{sensor}_vs_mean30' for sensor in SENSORS]
TEMPORAL_NUMERIC = (SENSORS + DIFF_FEATURES + MEAN7_FEATURES + STD7_FEATURES
                    + VS_MEAN7_FEATURES + ANOMALY7_FEATURES + VS_MEAN30_FEATURES)
ROOT = Path(__file__).resolve().parents[1]


def file_hash(path):
    """파일을 수정하지 않고 내용의 지문을 구합니다."""
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def prepare_machine_data(raw):
    """부품별 고장 점수를 합산하고, 중복 센서는 설비·날짜당 하나만 남깁니다."""
    required = KEYS + ['part_no', 'criticality', 'breakdown_flag',
                       'machine_type', 'plant_code'] + SENSORS
    missing = sorted(set(required) - set(raw.columns))
    if missing:
        raise ValueError(f'필수 열이 없습니다: {missing}')
    data = raw[required].copy()  # 원본 표에 계산용 열을 직접 추가하지 않습니다.
    data['transaction_date'] = pd.to_datetime(data.transaction_date, errors='raise').dt.normalize()
    if data[KEYS + ['part_no']].isna().any().any():
        raise ValueError('설비·날짜·부품 번호에 빈값이 있습니다.')
    if data.duplicated(KEYS + ['part_no']).any():
        raise ValueError('같은 설비·날짜·부품이 중복되어 심각도가 부풀려질 수 있습니다.')
    data['criticality'] = data.criticality.astype('string').str.strip().str.upper()
    if not data.criticality.isin(WEIGHTS).all():
        raise ValueError('criticality는 빈값 없이 A, B, C 중 하나여야 합니다.')
    data['breakdown_flag'] = pd.to_numeric(data.breakdown_flag, errors='raise')
    if not data.breakdown_flag.isin([0, 1]).all():
        raise ValueError('breakdown_flag는 0 또는 1이어야 합니다.')
    for column in SENSORS:
        data[column] = pd.to_numeric(data[column], errors='raise')
    if np.isinf(data[SENSORS].to_numpy(dtype=float)).any():
        raise ValueError('센서값에 무한대가 있습니다. 원본을 확인해 주세요.')
    # 빈 센서값은 나중에 Train 중앙값으로 채웁니다. 여기서 전체 평균을 쓰지 않습니다.
    sensor_counts = data.groupby(KEYS)[SENSORS + ['machine_type', 'plant_code']].nunique(dropna=False)
    if sensor_counts.gt(1).any().any():
        raise ValueError('같은 설비·날짜의 부품 행에서 센서값이 다릅니다. 임의로 첫 행을 고르지 않습니다.')
    # flag가 0이면 중요도와 관계없이 0점입니다. 1일 때만 4/2/1점을 더합니다.
    data['part_severity'] = data.criticality.map(WEIGHTS).astype(int) * data.breakdown_flag
    # 등급별 고장 부품 수를 남겨, 설비 점수가 어떻게 합산됐는지도 CSV에서 확인합니다.
    for grade in WEIGHTS:
        data[f'failed_{grade}'] = (data.criticality.eq(grade) & data.breakdown_flag.eq(1)).astype(int)
    severity = data.groupby(KEYS, as_index=False).agg(
        severity_score=('part_severity', 'sum'),
        failed_A=('failed_A', 'sum'), failed_B=('failed_B', 'sum'), failed_C=('failed_C', 'sum'))
    severity['failed_parts'] = severity[['failed_A', 'failed_B', 'failed_C']].sum(axis=1)
    if not severity.severity_score.eq(4 * severity.failed_A + 2 * severity.failed_B + severity.failed_C).all():
        raise ValueError('부품 등급별 고장 수와 설비 심각도 점수가 맞지 않습니다.')
    sensors = data[KEYS + SENSORS + ['machine_type', 'plant_code']].drop_duplicates(KEYS)
    machine = sensors.merge(severity, on=KEYS, how='inner', validate='one_to_one')
    # 여섯 정답 모두 동일한 설비·날짜 센서 행을 기준으로 만듭니다.
    for threshold in THRESHOLDS:
        machine[f'severity_{threshold}'] = machine.severity_score.ge(threshold).astype(int)
    machine = machine.sort_values(['asset_tag', 'transaction_date']).reset_index(drop=True)
    # 설비별로 따로 계산하므로 다른 설비의 전일 행이 이어지지 않습니다.
    for sensor in SENSORS:
        current = machine.groupby('asset_tag', sort=False)[sensor]
        previous = current.shift(1)
        machine[f'{sensor}_diff1'] = machine[sensor] - previous
        # 이전 7개 관측치만 사용하며 현재 날짜는 shift(1)로 제외합니다.
        machine[f'{sensor}_mean7'] = current.transform(
            lambda series: series.shift(1).rolling(window=7, min_periods=3).mean())
        machine[f'{sensor}_std7'] = current.transform(
            lambda series: series.shift(1).rolling(window=7, min_periods=3).std())
        # 현재 값이 이전 7개 관측치의 평균보다 얼마나 높거나 낮은지 계산합니다.
        machine[f'{sensor}_vs_mean7'] = machine[sensor] - machine[f'{sensor}_mean7']
        # 이상 점수는 평균에서 벗어난 크기를 과거 표준편차로 나눈 값입니다.
        # 과거 값의 표준편차가 0이면 나눌 수 없으므로 결측으로 남겨 Train 중앙값으로 채웁니다.
        past_std = machine[f'{sensor}_std7'].replace(0, np.nan)
        machine[f'{sensor}_anomaly7'] = machine[f'{sensor}_vs_mean7'].abs() / past_std
        past_30_mean = current.transform(
            lambda series: series.shift(1).rolling(window=30, min_periods=7).mean())
        machine[f'{sensor}_vs_mean30'] = machine[sensor] - past_30_mean
    return machine.sort_values(['transaction_date', 'asset_tag']).reset_index(drop=True)


def split_by_date(machine):
    """행을 무작위로 섞지 않고 고유 날짜를 70%·15%·15%로 나눕니다."""
    dates = pd.DatetimeIndex(sorted(machine.transaction_date.unique()))
    first, second = int(len(dates) * .70), int(len(dates) * .85)
    if not 0 < first < second < len(dates):
        raise ValueError('세 구간으로 나눌 날짜가 부족합니다.')
    groups = {
        'Train': machine.loc[machine.transaction_date.le(dates[first - 1])].copy(),
        'Validation': machine.loc[machine.transaction_date.between(dates[first], dates[second - 1])].copy(),
        'Test': machine.loc[machine.transaction_date.ge(dates[second])].copy(),
    }
    # 당일 분류이므로 미래 7일 정답을 위한 경계 제외는 하지 않습니다.
    summary = pd.DataFrame([{'split': name, 'start': group.transaction_date.min().date(),
                             'end': group.transaction_date.max().date(), 'rows': len(group),
                             'dates': group.transaction_date.nunique()} for name, group in groups.items()])
    return groups, summary


def evaluate(y_true, probabilities, cutoff=.5):
    """지정한 확률 기준으로 분류하고 동일한 성능값을 계산합니다."""
    predicted = (probabilities >= cutoff).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    both_classes = pd.Series(y_true).nunique() == 2
    if both_classes:
        precision, recall, _ = precision_recall_curve(y_true, probabilities)
        pr_auc = float(auc(recall, precision))  # PR 곡선의 사다리꼴 면적입니다.
        ap = float(average_precision_score(y_true, probabilities))  # AP도 별도 기록합니다.
        roc_auc = float(roc_auc_score(y_true, probabilities))
    else:
        pr_auc = ap = roc_auc = np.nan  # 한 종류 정답만 있으면 비교 면적을 보고하지 않습니다.
    actual_rate = float(np.mean(y_true))
    return {'Accuracy': accuracy_score(y_true, predicted),
            'Precision': precision_score(y_true, predicted, zero_division=0),
            'Recall': recall_score(y_true, predicted, zero_division=0),
            'F1': f1_score(y_true, predicted, zero_division=0),
            'ROC_AUC': roc_auc, 'PR_AUC': pr_auc, 'AP': ap,
            'TN': int(tn), 'FP': int(fp), 'FN': int(fn), 'TP': int(tp),
            'actual_failure_rate': actual_rate, 'predicted_failure_rate': float(predicted.mean()),
            'always_normal_accuracy': 1 - actual_rate, 'rows': len(y_true)}


def failure_rates(machine, groups):
    """학습 전에 전체/Train/Validation/Test의 0·1 건수와 비율을 검증합니다."""
    tables = {'전체': machine, **groups}
    rows = []
    for dataset, frame in tables.items():
        for threshold in severity_thresholds:
            label = frame[f'severity_{threshold}']
            failure_count = int(label.sum())
            rows.append({'dataset': dataset, 'severity_threshold': threshold,
                         'normal_count': len(frame) - failure_count, 'failure_count': failure_count,
                         'normal_rate': (len(frame) - failure_count) / len(frame),
                         'failure_rate': failure_count / len(frame)})
    result = pd.DataFrame(rows)
    for dataset, group in result.groupby('dataset', sort=False):
        ordered = group.sort_values('severity_threshold')
        if not ordered.failure_count.is_monotonic_decreasing:
            raise ValueError(f'{dataset}: 심각도 기준이 높아지는데 고장 건수가 증가했습니다.')
        if not ordered.failure_rate.is_monotonic_decreasing:
            raise ValueError(f'{dataset}: 심각도 기준이 높아지는데 고장 비율이 증가했습니다.')
        if not (ordered.normal_count + ordered.failure_count).eq(len(tables[dataset])).all():
            raise ValueError(f'{dataset}: 정상과 고장 건수가 전체 행 수와 다릅니다.')
    return result


def enhanced_sensor_frame(frame):
    """현재 날짜 센서만으로 네 조합 변수를 만듭니다. 정답과 미래 값은 쓰지 않습니다."""
    x = frame[SENSORS].copy()
    x['bearing_motor_temp_gap'] = x.temp_bearing_degC - x.temp_motor_degC
    x['vibration_magnitude'] = np.hypot(x.vibration_h_mms, x.vibration_v_mms)
    x['bearing_vibration_product'] = x.temp_bearing_degC * x.vibration_magnitude
    x['motor_load_product'] = x.temp_motor_degC * x.load_pct
    return x


def choose_validation_cutoff(y_true, probabilities):
    """Validation의 F1만으로 후보 확률 기준을 고릅니다. Test 정답은 읽지 않습니다."""
    candidates = np.round(np.arange(.05, .951, .01), 2)
    scores = [f1_score(y_true, probabilities >= cutoff, zero_division=0) for cutoff in candidates]
    options = pd.DataFrame({'cutoff': candidates, 'validation_F1': scores})
    options['distance_from_half'] = abs(options.cutoff - .5)
    selected = options.sort_values(['validation_F1', 'distance_from_half', 'cutoff'],
                                   ascending=[False, True, True]).iloc[0]
    return float(selected.cutoff), float(selected.validation_F1)


def fit_improved_models(groups):
    """입력 두 종류 × 가중치 두 종류를 학습하고 각자 검증 기준을 고정해 평가합니다."""
    if groups['Train'][SENSORS].isna().all().any():
        raise ValueError('Train에 중앙값을 구할 수 없는 센서 열이 있습니다.')
    # original_features는 기존의 센서 8개입니다. enhanced_features에만 조합 4개를 더합니다.
    feature_sets = {
        'original_features': {name: group[SENSORS].copy() for name, group in groups.items()},
        'enhanced_features': {name: enhanced_sensor_frame(group) for name, group in groups.items()},
    }
    excluded = {'severity_score', 'breakdown_flag', 'criticality', 'wo_type'}
    excluded.update(f'severity_{threshold}' for threshold in severity_thresholds)
    for name, frames in feature_sets.items():
        if set(frames['Train'].columns) & excluded:
            raise ValueError(f'{name}에 정답 생성 열이 섞였습니다.')
    comparison, predictions, choices = [], [], []
    for feature_name, frames in feature_sets.items():
        # 결측 대체값과 평균·표준편차는 각 입력의 Train으로만 학습합니다.
        preprocess = Pipeline([('imputer', SimpleImputer(strategy='median')),
                               ('scaler', StandardScaler())])
        x = {'Train': preprocess.fit_transform(frames['Train'])}
        for split in ['Validation', 'Test']:
            x[split] = preprocess.transform(frames[split])
        for threshold in severity_thresholds:
            target = f'severity_{threshold}'
            y_train = groups['Train'][target]
            if y_train.nunique() != 2:
                raise ValueError(f'{target}: Train에 0과 1이 모두 있어야 합니다.')
            for variant, weight in [('default', None), ('balanced', 'balanced')]:
                model = LogisticRegression(max_iter=1000, random_state=42, class_weight=weight)
                with warnings.catch_warnings():
                    warnings.simplefilter('error', ConvergenceWarning)
                    model.fit(x['Train'], y_train)
                validation_score = model.predict_proba(x['Validation'])[:, 1]
                selected, validation_f1 = choose_validation_cutoff(groups['Validation'][target], validation_score)
                choices.append({'severity_threshold': threshold, 'feature_set': feature_name,
                                'model_variant': variant, 'selected_cutoff': selected,
                                'validation_F1_at_selected': validation_f1, 'selection_split': 'Validation',
                                'selection_metric': 'F1', 'train_start': groups['Train'].transaction_date.min(),
                                'train_end': groups['Train'].transaction_date.max(),
                                'validation_start': groups['Validation'].transaction_date.min(),
                                'validation_end': groups['Validation'].transaction_date.max(),
                                'test_start': groups['Test'].transaction_date.min(),
                                'test_end': groups['Test'].transaction_date.max()})
                for split in ['Validation', 'Test']:
                    score = validation_score if split == 'Validation' else model.predict_proba(x['Test'])[:, 1]
                    for cutoff_name, cutoff in [('fixed_0.5', .5), ('validation_F1', selected)]:
                        values = evaluate(groups[split][target], score, cutoff)
                        comparison.append({'severity_threshold': threshold, 'feature_set': feature_name,
                                           'model_variant': variant, 'cutoff_method': cutoff_name,
                                           'probability_cutoff': cutoff, 'split': split,
                                           'train_failure_rate': float(y_train.mean()), **values})
                        if split == 'Test':
                            frame = groups['Test'][KEYS].copy()
                            frame['severity_threshold'] = threshold
                            frame['feature_set'] = feature_name
                            frame['model_variant'] = variant
                            frame['cutoff_method'] = cutoff_name
                            frame['probability_cutoff'] = cutoff
                            frame['actual'] = groups['Test'][target].to_numpy()
                            frame['probability'] = score
                            frame['predicted'] = (score >= cutoff).astype(int)
                            predictions.append(frame)
    return pd.DataFrame(comparison), pd.concat(predictions, ignore_index=True), pd.DataFrame(choices)


def current_day_pipeline(numeric_columns, categorical_columns, class_weight=None):
    """Train에서만 학습할 결측 처리·표준화·범주 변환·분류기를 한데 묶습니다."""
    numeric = Pipeline([('imputer', SimpleImputer(strategy='median')),
                        ('scaler', StandardScaler())])
    transforms = [('numeric', numeric, numeric_columns)]
    if categorical_columns:
        categorical = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                                ('one_hot', OneHotEncoder(handle_unknown='ignore'))])
        transforms.append(('categorical', categorical, categorical_columns))
    prep = ColumnTransformer(transforms, remainder='drop')
    classifier = LogisticRegression(max_iter=2000, random_state=42, class_weight=class_weight)
    return Pipeline([('preprocess', prep), ('classifier', classifier)])


def run_current_day_models(groups, numeric_columns, categorical_columns, feature_set):
    """여섯 정답에 기본/보정 모델을 학습하고 Test 확률 0.5로만 평가합니다."""
    columns = numeric_columns + categorical_columns
    # 허용 목록을 명시하여 고장 표시·점수·출고량 등이 추가되어도 X에 들어가지 않습니다.
    forbidden = {'asset_tag', 'breakdown_flag', 'wo_type', 'criticality', 'severity_score',
                 'qty_issued', 'issue_value_inr', 'part_severity'}
    forbidden.update(f'severity_{threshold}' for threshold in THRESHOLDS)
    if len(columns) != len(set(columns)) or set(columns) & forbidden:
        raise ValueError('모델 입력에 중복 또는 고장 정답 관련 열이 있습니다.')
    if categorical_columns and categorical_columns != CATEGORICAL:
        raise ValueError('범주형 변수 목록이 요청한 설비 정보와 다릅니다.')
    train, validation, test = (groups[name] for name in ['Train', 'Validation', 'Test'])
    if not train.transaction_date.max() < validation.transaction_date.min() < test.transaction_date.min():
        raise ValueError('Train/Validation/Test 날짜가 겹치거나 순서가 바뀌었습니다.')
    if not validation.transaction_date.max() < test.transaction_date.min():
        raise ValueError('Validation과 Test 날짜가 겹칩니다.')
    x_train, x_test = train[columns], test[columns]
    results, predictions, pipelines = [], [], {}
    for threshold in THRESHOLDS:
        target = f'severity_{threshold}'
        y_train, y_test = train[target], test[target]
        if y_train.nunique() != 2:
            raise ValueError(f'{target}: Train에 0과 1이 모두 필요합니다.')
        for model_name, weight in [('기본', None), ('balanced', 'balanced')]:
            pipeline = current_day_pipeline(numeric_columns, categorical_columns, weight)
            with warnings.catch_warnings():
                warnings.simplefilter('error', ConvergenceWarning)
                pipeline.fit(x_train, y_train)  # 전처리와 회귀계수 모두 Train으로만 학습합니다.
            trained_columns = list(pipeline.named_steps['preprocess'].feature_names_in_)
            if trained_columns != columns or 'asset_tag' in trained_columns:
                raise RuntimeError('실제로 학습한 X 열에 asset_tag가 있거나 입력 목록과 다릅니다.')
            if threshold == THRESHOLDS[0] and model_name == '기본':
                print(f'{feature_set} 실제 모델 X 변수 ({len(trained_columns)}개): {trained_columns}', flush=True)
                print(f'{feature_set} X에 asset_tag 포함: {"asset_tag" in trained_columns}', flush=True)
            score = pipeline.predict_proba(x_test)[:, 1]
            metrics = evaluate(y_test, score, cutoff=.5)
            results.append({'severity_threshold': threshold, 'model': model_name,
                            'feature_set': feature_set, 'failure_rows': int(y_test.sum()),
                            'total_rows': len(test), 'failure_rate': float(y_test.mean()),
                            'predicted_failure_rows': int((score >= .5).sum()),
                            **metrics})
            # 식별자는 결과 파일에 남기되 위 X에는 전달하지 않습니다.
            frame = test[['transaction_date', 'asset_tag', 'machine_type', 'plant_code']].copy()
            frame['severity_threshold'] = threshold
            frame['model'] = model_name
            frame['feature_set'] = feature_set
            frame['actual_label'] = y_test.to_numpy()
            frame['predicted_label'] = (score >= .5).astype(int)
            frame['predicted_probability'] = score
            predictions.append(frame)
            pipelines[(threshold, model_name)] = pipeline
            print(f'{feature_set}: {threshold}점 {model_name} 학습·Test 평가 완료', flush=True)
    return pd.DataFrame(results), pd.concat(predictions, ignore_index=True), pipelines


def machine_type_summary(groups, pipelines):
    """설비별 점수로 만든 정답을 설비 종류별로 평가해 보기 좋은 표를 만듭니다.

    모델은 전체 Train으로 학습한 기본 모델을 사용합니다. 설비 종류마다 Validation
    자료의 F1으로만 확률 기준을 고른 뒤 같은 종류의 Test 자료에 적용합니다.
    """
    rows = []
    feature_columns = TEMPORAL_NUMERIC + CATEGORICAL
    machine_types = sorted(groups['Train'].machine_type.dropna().unique())
    for machine_type in machine_types:
        train = groups['Train'].loc[groups['Train'].machine_type.eq(machine_type)]
        validation = groups['Validation'].loc[groups['Validation'].machine_type.eq(machine_type)]
        test = groups['Test'].loc[groups['Test'].machine_type.eq(machine_type)]
        for threshold in THRESHOLDS:
            label = f'severity_{threshold}'
            train_positive = int(train[label].sum())
            valid_positive = int(validation[label].sum())
            test_positive = int(test[label].sum())
            if validation.empty or test.empty:
                rows.append({'machine_type': machine_type, 'score_threshold': threshold,
                             'probability_cutoff': np.nan, 'status': 'missing_split',
                             'train_positive': train_positive, 'valid_positive': valid_positive,
                             'test_positive': test_positive, 'Accuracy': np.nan,
                             'Precision': np.nan, 'Recall': np.nan, 'F1': np.nan})
                continue
            model = pipelines[(threshold, '기본')]
            valid_probability = model.predict_proba(validation[feature_columns])[:, 1]
            if validation[label].nunique() == 2:
                cutoff, _ = choose_validation_cutoff(validation[label], valid_probability)
                status = 'ok'
            else:
                cutoff = .5  # 검증에 한 종류 정답만 있으면 최적 기준을 고르지 않습니다.
                status = 'single_validation_class'
            test_probability = model.predict_proba(test[feature_columns])[:, 1]
            metrics = evaluate(test[label], test_probability, cutoff)
            rows.append({'machine_type': machine_type, 'score_threshold': threshold,
                         'probability_cutoff': cutoff, 'status': status,
                         'train_positive': train_positive, 'valid_positive': valid_positive,
                         'test_positive': test_positive, 'Accuracy': metrics['Accuracy'],
                         'Precision': metrics['Precision'], 'Recall': metrics['Recall'],
                         'F1': metrics['F1']})
    return pd.DataFrame(rows).sort_values(['machine_type', 'score_threshold']).reset_index(drop=True)


def draw_current_day_charts(results, output):
    """새 모델 12개의 혼동행렬과 기준별 Precision·Recall·F1·PR-AUC를 그립니다."""
    plt = setup_plotting()
    fig, axes = plt.subplots(len(THRESHOLDS), 2, figsize=(11, 23), constrained_layout=True)
    for row, threshold in enumerate(THRESHOLDS):
        for col, variant in enumerate(['기본', 'balanced']):
            record = results.loc[results.severity_threshold.eq(threshold) & results.model.eq(variant)].iloc[0]
            matrix = np.array([[record.TN, record.FP], [record.FN, record.TP]])
            ax = axes[row, col]
            ax.imshow(matrix, cmap='Blues', vmin=0, vmax=record.total_rows)
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, f'{matrix[i, j]:,}', ha='center', va='center', fontsize=15,
                            color='white' if matrix[i, j] > record.total_rows / 2 else '#172b4d')
            ax.set(title=f'{threshold}점 · {variant}', xticks=[0, 1], yticks=[0, 1],
                   xticklabels=['예측 0', '예측 1'], yticklabels=['실제 0', '실제 1'])
    fig.suptitle('당일 고장 상태: Test 혼동행렬 · 확률 기준 0.5', fontsize=17)
    fig.savefig(output / 'logistic_confusion_matrices.png', dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(17, 6), constrained_layout=True)
    metrics = ['Precision', 'Recall', 'F1', 'PR_AUC']
    colors = ['#2874a6', '#16a085', '#d68910', '#8e44ad']
    for ax, variant in zip(axes, ['기본', 'balanced']):
        subset = results.loc[results.model.eq(variant)].sort_values('severity_threshold')
        positions = np.arange(len(subset))
        for index, (metric, color) in enumerate(zip(metrics, colors)):
            bars = ax.bar(positions + (index - 1.5) * .19, subset[metric], width=.19,
                          label=metric, color=color)
            ax.bar_label(bars, fmt='%.2f', padding=2, fontsize=8)
        ax.set(title=variant, xticks=positions,
               xticklabels=[f'{value}점' for value in subset.severity_threshold],
               ylim=(0, 1.1), ylabel='성능 (0~1)')
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
        ax.legend(ncol=4, fontsize=8)
    fig.suptitle('설비 정보와 과거 센서 변화 포함 · Test 성능 · 확률 기준 0.5', fontsize=17)
    fig.savefig(output / 'logistic_metrics_comparison.png', dpi=150)
    plt.close(fig)


def fit_models(groups, balanced_mode='auto', imbalance_cutoff=.20):
    """Train으로만 전처리와 학습을 수행합니다. Validation/Test는 변환과 평가만 합니다."""
    if groups['Train'][SENSORS].isna().all().any():
        raise ValueError('Train에 전체가 빈 센서 열이 있어 중앙값을 구할 수 없습니다.')
    preprocess = Pipeline([('imputer', SimpleImputer(strategy='median')),
                           ('scaler', StandardScaler())])
    # X_train은 한 번만 학습 변환하고, 여섯 기준에 똑같은 배열을 사용합니다.
    x = {'Train': preprocess.fit_transform(groups['Train'][SENSORS])}
    for name in ['Validation', 'Test']:
        x[name] = preprocess.transform(groups[name][SENSORS])
    results, predictions = [], {}
    for threshold in THRESHOLDS:
        target = f'severity_{threshold}'
        y_train = groups['Train'][target]
        if y_train.nunique() != 2:
            raise ValueError(f'{target}: Train에 0과 1 정답이 모두 있어야 합니다.')
        positive_rate = float(y_train.mean())
        minority_rate = min(positive_rate, 1 - positive_rate)
        variants = [('default', None)]
        # 20%는 이번 비교의 명시적인 실험 규칙이며 보편적인 불균형 기준은 아닙니다.
        use_balanced = balanced_mode == 'all' or (balanced_mode == 'auto' and minority_rate < imbalance_cutoff)
        if use_balanced:
            variants.append(('balanced', 'balanced'))
        predicted_rows = []
        for variant, weight in variants:
            model = LogisticRegression(max_iter=1000, random_state=42, class_weight=weight)
            with warnings.catch_warnings():
                warnings.simplefilter('error', ConvergenceWarning)
                model.fit(x['Train'], y_train)
            for split in ['Validation', 'Test']:
                score = model.predict_proba(x[split])[:, 1]
                values = evaluate(groups[split][target], score)
                results.append(dict(threshold=threshold, model_variant=variant, split=split,
                                    probability_cutoff=.5, train_failure_rate=positive_rate, **values))
                if split == 'Test':
                    # 예측 CSV에는 추적용 점수를 담지만, 이 점수는 X에 넣지 않았습니다.
                    frame = groups[split][KEYS + ['severity_score']].copy()
                    frame['y_true'] = groups[split][target].to_numpy()
                    frame['probability'] = score
                    frame['y_pred'] = (score >= .5).astype(int)
                    frame['threshold'] = threshold
                    frame['model_variant'] = variant
                    frame['split'] = split
                    predicted_rows.append(frame)
            print(f'{threshold}점 / {variant}: 학습 완료 (Train 고장 비율 {positive_rate:.2%})')
        # 보정 모델이 있는 파일은 model_variant 열로 명확하게 구분되는 긴 표입니다.
        predictions[threshold] = pd.concat(predicted_rows, ignore_index=True)
    return pd.DataFrame(results), predictions, preprocess


def draw_charts(results, output):
    """기본 모델과 불균형 보정 모델을 구분하여 두 개의 PNG를 저장합니다."""
    cache = ROOT / '.cache' / 'huijae_example_matplotlib'
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str(cache))
    import matplotlib
    matplotlib.use('Agg')  # 별도 창을 띄우지 않고 이미지 파일로 저장합니다.
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    if Path('C:/Windows/Fonts/malgun.ttf').exists():
        font_manager.fontManager.addfont('C:/Windows/Fonts/malgun.ttf')
        plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False
    test = results.loc[results.split.eq('Test')].reset_index(drop=True)
    count = len(test)
    columns = 3 if count in [3, 5, 6] else 2
    rows = int(np.ceil(count / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(4.5 * columns, 4.0 * rows), squeeze=False,
                             constrained_layout=True)
    for ax, record in zip(axes.flat, test.to_dict('records')):
        matrix = np.array([[record['TN'], record['FP']], [record['FN'], record['TP']]])
        ax.imshow(matrix, cmap='Blues', vmin=0, vmax=record['rows'])
        for i in range(2):
            for j in range(2):
                name = [['TN', 'FP'], ['FN', 'TP']][i][j]
                ax.text(j, i, f'{name}\n{matrix[i, j]:,}', ha='center', va='center', fontsize=15,
                        color='white' if matrix[i, j] > record['rows'] * .5 else '#172b4d')
        label = '기본' if record['model_variant'] == 'default' else '불균형 보정'
        ax.set(title=f"{int(record['threshold'])}점 이상 · {label}", xlabel='예측 상태', ylabel='실제 상태',
               xticks=[0, 1], yticks=[0, 1], xticklabels=['0: 기준 미만', '1: 기준 이상'],
               yticklabels=['0: 기준 미만', '1: 기준 이상'])
    for ax in list(axes.flat)[count:]:
        ax.set_visible(False)
    fig.suptitle('당일 고장 상태 분류: Test 혼동행렬 (판정 기준 0.5)', fontsize=16)
    fig.savefig(output / 'logistic_confusion_matrices.png', dpi=160)
    plt.close(fig)

    variants = list(test.model_variant.unique())
    fig, axes = plt.subplots(1, len(variants), figsize=(8 * len(variants), 5.5), squeeze=False,
                             constrained_layout=True)
    metrics = ['Precision', 'Recall', 'F1', 'PR_AUC']
    colors = ['#2874a6', '#16a085', '#d68910', '#8e44ad']
    for ax, variant in zip(axes.flat, variants):
        subset = test.loc[test.model_variant.eq(variant)].sort_values('threshold')
        positions = np.arange(len(subset))
        for index, (metric, color) in enumerate(zip(metrics, colors)):
            bars = ax.bar(positions + (index - 1.5) * .19, subset[metric], width=.19, label=metric, color=color)
            ax.bar_label(bars, fmt='%.2f', fontsize=8, padding=3)
        ax.set(xticks=positions, xticklabels=[f'{v}점 이상' for v in subset.threshold],
               ylim=(0, 1.13), ylabel='성능 (0~1)', title='기본 모델' if variant == 'default' else '불균형 보정 모델 (별도 비교)')
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
        ax.legend(loc='upper center', ncol=4, fontsize=9)
    fig.suptitle('당일 고장 상태 분류: Test 성능 · PR-AUC는 사다리꼴 면적', fontsize=16)
    fig.savefig(output / 'logistic_threshold_comparison.png', dpi=160)
    plt.close(fig)


def setup_plotting():
    """Windows에서도 한글 그래프를 파일로 저장할 수 있게 준비합니다."""
    cache = ROOT / '.cache' / 'huijae_example_matplotlib'
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str(cache))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    if Path('C:/Windows/Fonts/malgun.ttf').exists():
        font_manager.fontManager.addfont('C:/Windows/Fonts/malgun.ttf')
        plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False
    return plt


def draw_failure_rates(rates, output):
    """전체 고장 비율과 세 날짜 구간의 비율을 그룹 막대그래프로 저장합니다."""
    plt = setup_plotting()
    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    locations = np.arange(len(severity_thresholds))
    colors = {'전체': '#173f73', 'Train': '#2874a6', 'Validation': '#16a085', 'Test': '#d68910'}
    for number, (dataset, color) in enumerate(colors.items()):
        frame = rates.loc[rates.dataset.eq(dataset)].sort_values('severity_threshold')
        positions = locations + (number - 1.5) * .19
        bars = ax.bar(positions, frame.failure_rate.to_numpy() * 100, width=.19,
                      label=dataset, color=color)
        for bar, value in zip(bars, frame.failure_rate):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + .35,
                    f'{value:.2%}', ha='center', va='bottom', fontsize=8, rotation=45)
    ax.set(xticks=locations, xticklabels=[f'{t}점' for t in severity_thresholds],
           ylabel='고장 1의 비율 (%)', xlabel='심각도 점수 기준',
           ylim=(0, max(rates.failure_rate) * 100 + 8),
           title='심각도 기준별 고장 비율: 전체와 날짜 구간')
    ax.grid(axis='y', alpha=.2)
    ax.set_axisbelow(True)
    ax.legend(ncol=4)
    fig.savefig(output / 'severity_failure_rates.png', dpi=160)
    plt.close(fig)


def draw_improved_charts(comparison, output):
    """Validation으로 고른 확률 기준의 Test 혼동행렬과 묶은 막대그래프입니다."""
    plt = setup_plotting()
    selected = comparison.loc[comparison.split.eq('Test') &
                              comparison.cutoff_method.eq('validation_F1')].copy()
    variants = [('original_features', 'default'), ('original_features', 'balanced'),
                ('enhanced_features', 'default'), ('enhanced_features', 'balanced')]
    fig, axes = plt.subplots(len(severity_thresholds), len(variants), figsize=(18, 25),
                             constrained_layout=True)
    for row_index, threshold in enumerate(severity_thresholds):
        for col_index, (feature, variant) in enumerate(variants):
            ax = axes[row_index, col_index]
            record = selected.loc[selected.severity_threshold.eq(threshold) &
                                  selected.feature_set.eq(feature) &
                                  selected.model_variant.eq(variant)].iloc[0]
            matrix = np.array([[record.TN, record.FP], [record.FN, record.TP]])
            ax.imshow(matrix, cmap='Blues', vmin=0, vmax=record.rows)
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, f'{matrix[i, j]:,}', ha='center', va='center', fontsize=14,
                            color='white' if matrix[i, j] > record.rows / 2 else '#172b4d')
            ax.set(title=f'{threshold}점 · {feature} · {variant}\n기준 {record.probability_cutoff:.2f}',
                   xticks=[0, 1], yticks=[0, 1], xticklabels=['예측 0', '예측 1'],
                   yticklabels=['실제 0', '실제 1'])
            if col_index == 0:
                ax.set_ylabel('실제 상태')
    fig.suptitle('Test 혼동행렬 · 확률 기준은 Validation F1으로 선택', fontsize=17)
    fig.savefig(output / 'logistic_improved_confusion_matrices.png', dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(17, 11), constrained_layout=True)
    measures = ['Precision', 'Recall', 'F1', 'PR_AUC']
    colors = ['#2874a6', '#16a085', '#d68910', '#8e44ad']
    for ax, (feature, variant) in zip(axes.flat, variants):
        frame = selected.loc[selected.feature_set.eq(feature) &
                             selected.model_variant.eq(variant)].sort_values('severity_threshold')
        positions = np.arange(len(frame))
        for index, (measure, color) in enumerate(zip(measures, colors)):
            bars = ax.bar(positions + (index - 1.5) * .19, frame[measure], width=.19,
                          color=color, label=measure)
            ax.bar_label(bars, fmt='%.2f', fontsize=7, padding=2)
        ax.set(title=f'{feature} · {variant}', xticks=positions,
               xticklabels=[f'{t}점' for t in frame.severity_threshold],
               ylim=(0, 1.1), ylabel='성능 (0~1)')
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
        ax.legend(ncol=4, fontsize=8)
    fig.suptitle('Test 성능 · Validation에서 고른 확률 기준 · PR-AUC는 사다리꼴 면적', fontsize=17)
    fig.savefig(output / 'logistic_improved_metrics.png', dpi=150)
    plt.close(fig)


def improved_interpretation(rates, comparison):
    """점수 기준과 성능의 차이를 출력하되 현장 기준으로 확정하지 않습니다."""
    overall = rates.loc[rates.dataset.eq('전체')].sort_values('severity_threshold')
    lines = ['추가 비교: 7·10·11·12·13·14점 당일 상태 분류',
             '심각도 기준이 높아질수록 전체 고장 1의 비율은 ' +
             ' → '.join(f'{int(row.severity_threshold)}점 {row.failure_rate:.2%}' for row in overall.itertuples()) + '로 감소합니다.',
             'A=4, B=2, C=1과 모든 점수 기준은 실험적으로 정했습니다. 실제 현장의 중대 고장 기준으로 확정하지 않습니다.',
             'original_features는 당일 센서 8개입니다. enhanced_features는 여기에 베어링·모터 온도 차, 진동 크기, 베어링 온도×진동 크기, 모터 온도×부하율을 더합니다.',
             'PR-AUC는 정밀도-재현율 곡선의 사다리꼴 면적입니다. 정답 비율이 다른 기준끼리 면적만으로 우열을 정하지 않습니다.']
    candidates = overall.loc[overall.severity_threshold.isin([11, 12, 13])]
    least_skewed = candidates.loc[(candidates.failure_rate - .5).abs().idxmin()]
    lines.append(f'11·12·13점 중 가장 덜 불균형한 기준은 {int(least_skewed.severity_threshold)}점 '
                 f'(전체 고장 {least_skewed.failure_rate:.2%})입니다. 세 기준 모두 정상 쪽이 더 많습니다.')
    fixed = comparison.loc[comparison.split.eq('Test') & comparison.feature_set.eq('original_features') &
                           comparison.model_variant.eq('default') & comparison.cutoff_method.eq('fixed_0.5')]
    lines.append('기존 8개 센서·기본 모델·확률 기준 0.5의 Test 결과:')
    for row in fixed.sort_values('severity_threshold').itertuples():
        lines.append(f'{row.severity_threshold}점: 실제 고장 {row.actual_failure_rate:.2%}, '
                     f'PR-AUC {row.PR_AUC:.3f}, 정확도 {row.Accuracy:.2%}, '
                     f'전부 정상 예측 정확도 {row.always_normal_accuracy:.2%}, '
                     f'정밀도 {row.Precision:.3f}, 재현율 {row.Recall:.3f}, F1 {row.F1:.3f}, '
                     f'FP {row.FP}, FN {row.FN}.')
    middle = fixed.loc[fixed.severity_threshold.isin([11, 12, 13])]
    if middle.F1.max() == 0:
        lines.append('기본 모델의 확률 기준 0.5에서는 11·12·13점 모두 고장을 한 건도 찾지 못해 '
                     '이 셋 중 상대적으로 분류하기 쉬운 기준을 고를 수 없습니다.')
    else:
        relative = middle.sort_values('F1', ascending=False).iloc[0]
        lines.append(f'이 설정에서 11·12·13점 중 F1이 상대적으로 높은 것은 '
                     f'{int(relative.severity_threshold)}점이지만, FP·FN과 고장 비율을 함께 보아야 합니다.')
    selected = comparison.loc[comparison.split.eq('Test') &
                              comparison.cutoff_method.eq('validation_F1') &
                              comparison.feature_set.eq('enhanced_features') &
                              comparison.model_variant.eq('balanced') &
                              comparison.severity_threshold.isin([11, 12, 13])]
    if not selected.empty:
        relative = selected.sort_values('F1', ascending=False).iloc[0]
        lines.append(f'추가 변수·balanced·Validation 확률 기준의 Test에서는 '
                     f'{int(relative.severity_threshold)}점이 11·12·13점 중 F1 {relative.F1:.3f}으로 상대적으로 높았습니다. '
                     f'정밀도 {relative.Precision:.3f}, 재현율 {relative.Recall:.3f}, '
                     f'오경보 FP {relative.FP}, 놓친 고장 FN {relative.FN}입니다. '
                     '다른 정답 정의와 서로 다른 고장 비율의 비교이므로 현장 기준 선택 근거로 단정하지 않습니다.')
    lines.append('검증 F1으로 고른 확률 기준의 성적은 logistic_improved_comparison.csv에서 '
                 'cutoff_method=validation_F1 행을 보세요. Test 결과로 기준을 고르거나 조정하지 않았습니다.')
    lines.append('이번 결과는 같은 날의 상태를 분류한 것입니다. 미래 고장 예측 결과가 아닙니다.')
    return '\n'.join(lines)


def interpretation(results):
    """정확도 착시와 기준별 차이를 수치로 설명하되 현장 기준을 자동 결정하지 않습니다."""
    lines = ['이번 결과는 당일 중대 고장 상태 분류이며 미래 고장 예측이 아닙니다.',
             '0은 해당 심각도 기준 미만입니다. 부품 고장이 전혀 없다는 뜻은 아닙니다.',
             'A=4, B=2, C=1과 7·10·11·12·13·14점은 실험 기준이며 실제 현장 기준으로 확정할 수 없습니다.',
             '여섯 기준은 서로 다른 정답을 정의하므로 Accuracy나 PR-AUC 하나로 최적 기준을 고를 수 없습니다.',
             'PR_AUC는 사다리꼴 면적이고 AP는 평균 정밀도입니다. 고장 비율도 함께 비교하세요.',
             '--- Test 기본 모델 ---']
    for row in results.loc[results.split.eq('Test') & results.model_variant.eq('default')].itertuples():
        lines.append(f'{row.threshold}점: 실제 고장 {row.actual_failure_rate:.2%}, 예측 고장 {row.predicted_failure_rate:.2%}, '
                     f'정확도 {row.Accuracy:.2%} / 전부 0으로 예측한 정확도 {row.always_normal_accuracy:.2%}. '
                     f'정밀도 {row.Precision:.3f}, 재현율 {row.Recall:.3f}, F1 {row.F1:.3f}, PR-AUC {row.PR_AUC:.3f}. '
                     f'놓친 고장 FN={row.FN}, 오경보 FP={row.FP}.')
        if row.predicted_failure_rate == 0:
            lines.append(f'  {row.threshold}점 모델은 고장으로 분류한 사례가 없습니다. 정확도가 높아도 고장 탐지에 유용하지 않습니다.')
    basic = results.loc[results.split.eq('Test') & results.model_variant.eq('default')]
    best_f1 = basic.sort_values('F1', ascending=False).iloc[0]
    lines.append(f'현재 0.5 설정에서 기본 모델 중 {int(best_f1.threshold)}점의 F1이 가장 높습니다. '
                 '이는 해당 실험의 고장 탐지 성적 비교이며, 가장 적절한 중대 고장 정의라는 뜻은 아닙니다.')
    for row in results.loc[results.split.eq('Test') & results.model_variant.eq('balanced')].itertuples():
        lines.append(f'{row.threshold}점 balanced: 정밀도 {row.Precision:.3f}, 재현율 {row.Recall:.3f}, '
                     f'F1 {row.F1:.3f}, PR-AUC {row.PR_AUC:.3f}, 놓친 고장 FN={row.FN}, 오경보 FP={row.FP}.')
    lines += ['7점은 상대적으로 넓은 상태를, 14점은 드문 높은 점수 상태를 고장으로 정의합니다. 다른 기준은 그 사이입니다.',
              '놓치는 고장과 오경보의 비용, 실제 정비 필요 여부를 확인해야 가장 적절한 현장 기준을 정할 수 있습니다.',
              'balanced 결과는 기본 결과와 별개입니다. 재현율 개선과 함께 오경보 증가를 확인해야 합니다.',
              '이번 비교에서는 Test 결과로 모델이나 0.5 판정 기준을 조정하지 않았습니다.']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--data',
        type=Path,
        default=Path.home() / 'Downloads' / 'synthetic_industrial_machine_data.csv',
        help='원본 CSV 경로 (생략하면 Downloads의 synthetic_industrial_machine_data.csv 사용)',
    )
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs', help='결과 저장 폴더')
    parser.add_argument('--balanced', choices=['auto', 'all', 'none'], default='auto',
                        help='auto: Train 소수 클래스 비율이 기준 미만인 경우만 추가 비교')
    parser.add_argument('--imbalance-cutoff', type=float, default=.20, help='auto의 소수 클래스 비율 기준 (기본 0.20)')
    args = parser.parse_args()
    if not 0 < args.imbalance_cutoff <= .5:
        parser.error('--imbalance-cutoff는 0 초과 0.5 이하여야 합니다.')
    source, output = args.data.resolve(), args.output_dir.resolve()
    expected_names = ['severity_failure_rates.csv', 'severity_failure_rates.png',
                      'machine_day_severity_scores.csv', 'logistic_machine_type_summary.csv',
                      'logistic_7_10_11_12_13_14_results.csv',
                      'logistic_7_10_11_12_13_14_predictions.csv',
                      'logistic_sensor_baseline_comparison.csv', 'logistic_metrics_comparison.png',
                      'logistic_improved_comparison.csv', 'logistic_improved_predictions.csv',
                      'validation_thresholds.csv', 'logistic_improved_confusion_matrices.png',
                      'logistic_improved_metrics.png', 'logistic_improved_analysis.txt',
                      'logistic_threshold_comparison.csv', 'logistic_confusion_matrices.png',
                      'logistic_threshold_comparison.png', 'logistic_analysis.txt',
                      'logistic_validation_comparison.csv', 'logistic_split_summary.csv',
                      'logistic_label_distribution.csv'] + [f'logistic_predictions_{t}.csv' for t in THRESHOLDS]
    if source in [output / name for name in expected_names]:
        raise ValueError('원본 CSV를 결과 파일로 덮어쓸 수 없습니다.')
    before = file_hash(source)
    raw = pd.read_csv(source)
    machine = prepare_machine_data(raw)
    print(f'원본 {len(raw):,}행 → 설비×날짜 {len(machine):,}행 / 10,950행 일치: {len(machine) == 10950}')
    part_counts = raw.groupby(KEYS).size()
    print(f'설비×날짜별 원본 부품 행 수: 최소 {part_counts.min()}, 최대 {part_counts.max()}')
    # 각 기준에서 0과 1이 얼마나 있는지 전체 자료 기준으로 출력합니다.
    distribution = pd.DataFrame([{'threshold': t, 'label': label,
                                  'count': int(machine[f'severity_{t}'].eq(label).sum()),
                                  'ratio': float(machine[f'severity_{t}'].eq(label).mean())}
                                 for t in THRESHOLDS for label in [0, 1]])
    print('\n기준별 전체 정답 분포\n', distribution.to_string(index=False))
    groups, split_summary = split_by_date(machine)
    print('\n날짜 기준 분리\n', split_summary.to_string(index=False))
    # 모델 학습 전 전체·Train·Validation·Test의 0/1 비율을 출력하고 감소 순서를 확인합니다.
    rates = failure_rates(machine, groups)
    print('\n모델 학습 전 심각도 기준별 정상/고장 건수와 비율\n',
          rates.to_string(index=False, float_format=lambda value: f'{value:.4%}'))
    print('확인: 네 자료 모두 점수 기준이 높아질수록 고장 건수와 비율이 증가하지 않습니다.')
    output.mkdir(parents=True, exist_ok=True)
    rates.to_csv(output / 'severity_failure_rates.csv', index=False, encoding='utf-8-sig')
    # 20개 부품을 합쳐 얻은 설비·날짜별 점수도 별도 CSV로 보관합니다.
    score_columns = ['transaction_date', 'asset_tag', 'machine_type', 'plant_code',
                     'failed_A', 'failed_B', 'failed_C', 'failed_parts', 'severity_score'] + [
                         f'severity_{threshold}' for threshold in THRESHOLDS]
    machine[score_columns].sort_values(['machine_type', 'asset_tag', 'transaction_date']).to_csv(
        output / 'machine_day_severity_scores.csv', index=False, encoding='utf-8-sig')
    draw_failure_rates(rates, output)
    results, predictions, _ = fit_models(groups, args.balanced, args.imbalance_cutoff)
    # 기존 비교와 별도로 네 가지 입력/가중치 조합을 모두 실행합니다.
    improved, improved_predictions, validation_choices = fit_improved_models(groups)
    # 요청한 비교 CSV에는 Test 성적만 넣고 Validation 성적은 별도 파일로 저장합니다.
    test = results.loc[results.split.eq('Test')].sort_values(['model_variant', 'threshold'], ascending=[False, True])
    test.to_csv(output / 'logistic_threshold_comparison.csv', index=False, encoding='utf-8-sig')
    results.loc[results.split.eq('Validation')].to_csv(output / 'logistic_validation_comparison.csv', index=False, encoding='utf-8-sig')
    distribution.to_csv(output / 'logistic_label_distribution.csv', index=False, encoding='utf-8-sig')
    split_summary.to_csv(output / 'logistic_split_summary.csv', index=False, encoding='utf-8-sig')
    for threshold, frame in predictions.items():
        frame.to_csv(output / f'logistic_predictions_{threshold}.csv', index=False, encoding='utf-8-sig')
    draw_charts(results, output)
    improved.to_csv(output / 'logistic_improved_comparison.csv', index=False, encoding='utf-8-sig')
    improved_predictions.to_csv(output / 'logistic_improved_predictions.csv', index=False, encoding='utf-8-sig')
    validation_choices.to_csv(output / 'validation_thresholds.csv', index=False, encoding='utf-8-sig')
    draw_improved_charts(improved, output)
    # 이전 센서 8개 모델을 같은 설정으로 다시 학습해 이번 모델과 공정하게 비교합니다.
    baseline_results, _, _ = run_current_day_models(groups, SENSORS, [], '센서 8개')
    current_results, current_predictions, current_pipelines = run_current_day_models(
        groups, TEMPORAL_NUMERIC, CATEGORICAL, '센서+과거변화+설비정보')
    current_results.to_csv(output / 'logistic_7_10_11_12_13_14_results.csv',
                           index=False, encoding='utf-8-sig')
    current_predictions.to_csv(output / 'logistic_7_10_11_12_13_14_predictions.csv',
                               index=False, encoding='utf-8-sig')
    compare_columns = ['severity_threshold', 'model', 'Accuracy', 'Precision', 'Recall',
                       'F1', 'PR_AUC', 'FP', 'FN']
    comparison = baseline_results[compare_columns].merge(
        current_results[compare_columns], on=['severity_threshold', 'model'],
        suffixes=('_sensor8', '_expanded'), validate='one_to_one')
    comparison['F1_change'] = comparison.F1_expanded - comparison.F1_sensor8
    comparison['PR_AUC_change'] = comparison.PR_AUC_expanded - comparison.PR_AUC_sensor8
    comparison.to_csv(output / 'logistic_sensor_baseline_comparison.csv',
                      index=False, encoding='utf-8-sig')
    type_report = machine_type_summary(groups, current_pipelines)
    type_report.to_csv(output / 'logistic_machine_type_summary.csv',
                       index=False, encoding='utf-8-sig')
    draw_current_day_charts(current_results, output)
    explanation = interpretation(results)
    (output / 'logistic_analysis.txt').write_text(explanation + '\n', encoding='utf-8')
    improved_explanation = improved_interpretation(rates, improved)
    (output / 'logistic_improved_analysis.txt').write_text(improved_explanation + '\n', encoding='utf-8')
    print('\nTest 결과 (default와 balanced는 별도 모델)\n', test.to_string(index=False, float_format=lambda v: f'{v:.4f}'))
    print('\n' + explanation)
    print('\n' + improved_explanation)
    print('\n이번 모델: 설비 정보·과거 센서 변화 포함 (Test, 확률 기준 0.5)\n',
          current_results[['severity_threshold', 'model', 'failure_rows', 'total_rows',
                           'failure_rate', 'predicted_failure_rows', 'Precision', 'Accuracy',
                           'F1', 'Recall', 'PR_AUC', 'TN', 'FP', 'FN', 'TP']].to_string(
                               index=False, float_format=lambda value: f'{value:.3f}'))
    print('\n기존 센서 8개 대비 성능 변화\n', comparison.to_string(
        index=False, float_format=lambda value: f'{value:.3f}'))
    print('\n설비 종류별 당일 고장 분류 (설비·날짜 점수를 기준으로 집계)')
    print('probability_cutoff는 설비 종류별 Validation F1에서만 선택했습니다.')
    print(type_report.to_string(index=False, float_format=lambda value: f'{value:.6f}'))
    if file_hash(source) != before:
        raise RuntimeError('실행 중 원본 내용이 변경됐습니다. 결과 사용 전 확인해 주세요.')
    print(f'\n원본 변경 없음. 결과 저장 위치: {output}')


if __name__ == '__main__':
    main()  # 이 파일을 실행했을 때만 위 작업들을 순서대로 수행합니다.
