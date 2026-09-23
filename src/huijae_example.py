"""당일 센서로 같은 날의 고장 심각도를 분류하는 로지스틱 회귀 비교입니다.

실행 예: python src/huijae_example.py --data "C:/data/synthetic_industrial_machine_data.csv"
필요 패키지: numpy, pandas, scikit-learn, matplotlib
미래 정답이나 과거 이동평균은 만들지 않습니다. 원본 CSV는 읽기만 합니다.
"""

import argparse  # 명령창에서 CSV 경로와 실행 옵션을 받습니다.
import hashlib  # 실행 전후 원본 내용이 같은지 확인합니다.
import os  # 그래프 글꼴 캐시를 쓸 수 있는 폴더로 지정합니다.
import warnings  # 모델 학습이 충분히 끝나지 않으면 알려 줍니다.
from pathlib import Path  # Windows 파일 경로를 다룹니다.

import numpy as np  # 숫자 배열과 계산에 사용합니다.
import pandas as pd  # CSV를 표 형태로 읽고 가공합니다.
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer  # 빈 센서값을 Train 중앙값으로 채웁니다.
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, auc, average_precision_score,
                             confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline  # 결측값 처리와 표준화를 순서대로 수행합니다.
from sklearn.preprocessing import StandardScaler  # 변수마다 다른 숫자 크기를 맞춥니다.

# X는 이 목록으로만 선택합니다. 정답 관련 열이 추가되어도 입력에 섞이지 않습니다.
SENSORS = ['temp_bearing_degC', 'temp_motor_degC', 'vibration_h_mms',
           'vibration_v_mms', 'oil_pressure_bar', 'load_pct', 'shaft_rpm',
           'power_consumption_kw']
KEYS = ['asset_tag', 'transaction_date']
WEIGHTS = {'A': 4, 'B': 2, 'C': 1}
THRESHOLDS = [7, 10, 14]
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
    required = KEYS + ['part_no', 'criticality', 'breakdown_flag'] + SENSORS
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
    sensor_counts = data.groupby(KEYS)[SENSORS].nunique(dropna=False)
    if sensor_counts.gt(1).any().any():
        raise ValueError('같은 설비·날짜의 부품 행에서 센서값이 다릅니다. 임의로 첫 행을 고르지 않습니다.')
    # flag가 0이면 중요도와 관계없이 0점입니다. 1일 때만 4/2/1점을 더합니다.
    data['part_severity'] = data.criticality.map(WEIGHTS).astype(int) * data.breakdown_flag
    severity = data.groupby(KEYS, as_index=False).agg(severity_score=('part_severity', 'sum'))
    sensors = data[KEYS + SENSORS].drop_duplicates(KEYS)
    machine = sensors.merge(severity, on=KEYS, how='inner', validate='one_to_one')
    # 서로 다른 세 정답을 만들지만, 동일한 센서 행을 세 모델에서 공유합니다.
    for threshold in THRESHOLDS:
        machine[f'severity_{threshold}'] = machine.severity_score.ge(threshold).astype(int)
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


def evaluate(y_true, probabilities):
    """모든 모델에 0.5 기준을 동일하게 적용하고 여러 성능값을 계산합니다."""
    predicted = (probabilities >= .5).astype(int)
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


def fit_models(groups, balanced_mode='auto', imbalance_cutoff=.20):
    """Train으로만 전처리와 학습을 수행합니다. Validation/Test는 변환과 평가만 합니다."""
    if groups['Train'][SENSORS].isna().all().any():
        raise ValueError('Train에 전체가 빈 센서 열이 있어 중앙값을 구할 수 없습니다.')
    preprocess = Pipeline([('imputer', SimpleImputer(strategy='median')),
                           ('scaler', StandardScaler())])
    # X_train은 한 번만 학습 변환하고, 세 기준에 똑같은 배열을 사용합니다.
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


def interpretation(results):
    """정확도 착시와 기준별 차이를 수치로 설명하되 현장 기준을 자동 결정하지 않습니다."""
    lines = ['이번 결과는 당일 중대 고장 상태 분류이며 미래 고장 예측이 아닙니다.',
             '0은 해당 심각도 기준 미만입니다. 부품 고장이 전혀 없다는 뜻은 아닙니다.',
             'A=4, B=2, C=1과 7·10·14점은 실험 기준이며 실제 현장 기준으로 확정할 수 없습니다.',
             '세 기준은 서로 다른 정답을 정의하므로 Accuracy나 PR-AUC 하나로 최적 기준을 고를 수 없습니다.',
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
    lines += ['7점은 상대적으로 넓은 상태를, 14점은 드문 높은 점수 상태를 고장으로 정의합니다. 10점은 그 사이입니다.',
              '놓치는 고장과 오경보의 비용, 실제 정비 필요 여부를 확인해야 가장 적절한 현장 기준을 정할 수 있습니다.',
              'balanced 결과는 기본 결과와 별개입니다. 재현율 개선과 함께 오경보 증가를 확인해야 합니다.',
              '이번 비교에서는 Test 결과로 모델이나 0.5 판정 기준을 조정하지 않았습니다.']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True, help='원본 CSV 경로')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs', help='결과 저장 폴더')
    parser.add_argument('--balanced', choices=['auto', 'all', 'none'], default='auto',
                        help='auto: Train 소수 클래스 비율이 기준 미만인 경우만 추가 비교')
    parser.add_argument('--imbalance-cutoff', type=float, default=.20, help='auto의 소수 클래스 비율 기준 (기본 0.20)')
    args = parser.parse_args()
    if not 0 < args.imbalance_cutoff <= .5:
        parser.error('--imbalance-cutoff는 0 초과 0.5 이하여야 합니다.')
    source, output = args.data.resolve(), args.output_dir.resolve()
    expected_names = ['logistic_threshold_comparison.csv', 'logistic_confusion_matrices.png',
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
    results, predictions, _ = fit_models(groups, args.balanced, args.imbalance_cutoff)
    output.mkdir(parents=True, exist_ok=True)
    # 요청한 비교 CSV에는 Test 성적만 넣고 Validation 성적은 별도 파일로 저장합니다.
    test = results.loc[results.split.eq('Test')].sort_values(['model_variant', 'threshold'], ascending=[False, True])
    test.to_csv(output / 'logistic_threshold_comparison.csv', index=False, encoding='utf-8-sig')
    results.loc[results.split.eq('Validation')].to_csv(output / 'logistic_validation_comparison.csv', index=False, encoding='utf-8-sig')
    distribution.to_csv(output / 'logistic_label_distribution.csv', index=False, encoding='utf-8-sig')
    split_summary.to_csv(output / 'logistic_split_summary.csv', index=False, encoding='utf-8-sig')
    for threshold, frame in predictions.items():
        frame.to_csv(output / f'logistic_predictions_{threshold}.csv', index=False, encoding='utf-8-sig')
    draw_charts(results, output)
    explanation = interpretation(results)
    (output / 'logistic_analysis.txt').write_text(explanation + '\n', encoding='utf-8')
    print('\nTest 결과 (default와 balanced는 별도 모델)\n', test.to_string(index=False, float_format=lambda v: f'{v:.4f}'))
    print('\n' + explanation)
    if file_hash(source) != before:
        raise RuntimeError('실행 중 원본 내용이 변경됐습니다. 결과 사용 전 확인해 주세요.')
    print(f'\n원본 변경 없음. 결과 저장 위치: {output}')


if __name__ == '__main__':
    main()  # 이 파일을 실행했을 때만 위 작업들을 순서대로 수행합니다.
