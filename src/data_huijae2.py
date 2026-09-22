"""부품 고장 예측 A/B/C 비교: 부품 이력과 센서의 추가 효과를 확인합니다.

검증 실행: python -m src.data_huijae2 --top-k 10
최종 시험: python -m src.data_huijae2 --stage test --run-dir <검증 결과 폴더>
검증 실행에서는 시험 구간의 예측과 성능을 계산하지 않습니다.
"""
import argparse
import json
import warnings
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# 모듈 실행과 편집기에서 이 파일을 직접 실행하는 방식을 모두 지원합니다.
if __package__:
    from .prepare_machine_data import PART_KEYS, SENSORS, digest, prepare
else:
    from prepare_machine_data import PART_KEYS, SENSORS, digest, prepare

ROOT = Path(__file__).resolve().parents[1]
TARGET = 'target_breakdown_next_7d'
PART_NUMERIC = ['unit_cost_inr']
PART_CATEGORICAL = ['part_no', 'part_family', 'criticality', 'uom']
HISTORY = ['breakdown_days_prev_7d', 'breakdown_days_prev_30d',
           'days_since_last_observed_breakdown', 'has_prior_observed_breakdown']
MODEL_PARAMS = dict(C=1.0, solver='lbfgs', penalty='l2', max_iter=2000,
                    class_weight=None, random_state=42)
LABELS = {'A': '부품 정보 + 고장 이력', 'B': '부품 정보 + 센서',
          'C': '부품 정보 + 고장 이력 + 센서'}


def feature_sets():
    # 설비 ID는 이력 묶기와 점검 대상 식별에만 씁니다. 모델 입력에는 넣지 않습니다.
    return {name: {'numeric': PART_NUMERIC + extra, 'categorical': PART_CATEGORICAL.copy()}
            for name, extra in [('A', HISTORY), ('B', SENSORS), ('C', HISTORY + SENSORS)]}


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def save_csv(frame, path):
    frame.to_csv(path, index=False, encoding='utf-8-sig')


def prepare_rows(raw):
    """검증된 전처리를 재사용하여 원본을 복사한 뒤 설비·부품·날짜별로 계산합니다.

    prepare 내부에서 설비+부품별 shift(1), 7/30일 합계, 마지막 표시 이후
    경과일, t+1~t+7 정답, 날짜순 70/15/15 분할과 경계 7일 제외를 수행합니다.
    B에도 A/C와 동일한 과거 30일 조건을 적용하여 비교 행을 일치시킵니다.
    """
    _, part, _, audit = prepare(raw)
    usable = part.loc[part.eligible_for_model.eq(True)].sort_values(PART_KEYS).reset_index(drop=True)
    if usable[PART_KEYS].duplicated().any() or not usable.breakdown_flag.eq(0).all():
        raise ValueError('비교 행에 중복 또는 현재 고장 표시가 있습니다.')
    if usable[TARGET].isna().any():
        raise ValueError('미래 정답을 확인할 수 없는 행이 포함됐습니다.')
    splits = {name: usable.loc[usable.split.eq(name)].reset_index(drop=True) for name in ['train', 'validation', 'test']}
    for name, group in splits.items():
        if group.empty:
            raise ValueError(f'{name} 구간에 비교 가능한 행이 없습니다.')
        boundary = pd.Timestamp(audit['split_dates'][f'{name}_end'])
        if not group.target_window_end.le(boundary).all():
            raise ValueError(f'{name} 구간의 미래 정답이 경계를 넘습니다.')
    # 시험 정답 분포는 검증 보고서에 기록하지 않습니다.
    overview = {name: {'rows': len(group), 'date_start': str(group.transaction_date.min().date()),
                      'date_end': str(group.transaction_date.max().date())}
                for name, group in splits.items()}
    return splits, {'raw_rows': len(raw), 'machine_rows': audit['machine_rows'],
                    'split_dates': audit['split_dates'], 'usable_rows': overview}


def make_model(spec):
    # 세 실험의 결측 처리·표준화·범주 변환·로지스틱 설정은 동일합니다.
    numeric = Pipeline([('imputer', SimpleImputer(strategy='median', add_indicator=True)),
                        ('scaler', StandardScaler())])
    categorical = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                            ('encoder', OneHotEncoder(handle_unknown='ignore'))])
    features = ColumnTransformer([('numeric', numeric, spec['numeric']),
                                  ('categorical', categorical, spec['categorical'])])
    return Pipeline([('features', features), ('model', LogisticRegression(**MODEL_PARAMS))])


def ranking_metrics(frame):
    y, score = frame[TARGET].astype(int), frame.score.to_numpy()
    if not np.isfinite(score).all():
        raise ValueError('예측 점수에 유효하지 않은 값이 있습니다.')
    if y.nunique() < 2:
        return dict(rows=len(y), positive_rate=float(y.mean()), pr_auc_trapezoid=None,
                    average_precision=None, roc_auc=None)
    precision, recall, _ = precision_recall_curve(y, score)
    # AP와 사다리꼴 적분 PR-AUC는 계산이 다르므로 이름을 구분해 저장합니다.
    return dict(rows=len(y), positive_rate=float(y.mean()),
                pr_auc_trapezoid=float(auc(recall, precision)),
                average_precision=float(average_precision_score(y, score)),
                roc_auc=float(roc_auc_score(y, score)))


def daily_top_k(frame, k):
    """매일 전체 설비에서 상위 k개 설비·부품을 골라 적중 수를 계산합니다."""
    if k < 1:
        raise ValueError('하루 점검 수는 1 이상이어야 합니다.')
    if frame.duplicated(PART_KEYS).any():
        raise ValueError('같은 날 같은 설비·부품이 중복됩니다.')
    # 점수 동점은 정답과 무관한 설비 ID, 부품 번호 순서로 결정합니다.
    ranked = frame.sort_values(['transaction_date', 'score', 'asset_tag', 'part_no'],
                               ascending=[True, False, True, True], kind='stable').copy()
    ranked['daily_rank'] = ranked.groupby('transaction_date').cumcount() + 1
    selected = ranked.loc[ranked.daily_rank.le(k)].copy()
    totals = ranked.groupby('transaction_date').agg(candidates=(TARGET, 'size'), positives=(TARGET, 'sum'))
    chosen = selected.groupby('transaction_date').agg(inspected=(TARGET, 'size'), hits=(TARGET, 'sum'))
    daily = totals.join(chosen)
    daily['precision_at_k'] = daily.hits / daily.inspected
    daily['recall_at_k'] = daily.hits / daily.positives.replace(0, np.nan)
    daily['random_expected_hits'] = daily.inspected * daily.positives / daily.candidates
    daily['random_expected_precision'] = daily.positives / daily.candidates
    return daily.reset_index(), selected


def summarize(frame, daily):
    result = ranking_metrics(frame)
    result.update(days=len(daily), inspections=int(daily.inspected.sum()), hits=int(daily.hits.sum()),
                  precision_at_k=float(daily.hits.sum() / daily.inspected.sum()),
                  mean_daily_precision_at_k=float(daily.precision_at_k.mean()),
                  mean_hits_per_day=float(daily.hits.mean()),
                  random_expected_precision=float(daily.random_expected_hits.sum() / daily.inspected.sum()))
    return result


def save_evaluation(predictions, folder, k, stage):
    if predictions[TARGET].nunique() != 2:
        raise ValueError('구간 전체의 순위 성능을 비교하려면 0과 1 정답이 모두 필요합니다.')
    metrics, monthly, daily_tables, selected_tables = [], [], [], []
    for name, frame in predictions.groupby('experiment', sort=True):
        daily, selected = daily_top_k(frame, k)
        metrics.append(dict(experiment=name, **summarize(frame, daily)))
        daily_tables.append(daily.assign(experiment=name))
        selected_tables.append(selected)
        for month, group in frame.groupby(frame.transaction_date.dt.to_period('M')):
            subset = daily.loc[daily.transaction_date.dt.to_period('M').eq(month)]
            monthly.append(dict(experiment=name, month=str(month), **summarize(group, subset)))
    metrics = pd.DataFrame(metrics)
    monthly = pd.DataFrame(monthly)
    daily = pd.concat(daily_tables, ignore_index=True)
    comparison = monthly.pivot(index='month', columns='experiment',
                               values=['pr_auc_trapezoid', 'average_precision', 'precision_at_k'])
    delta = pd.DataFrame({f'{metric}_C_minus_A': comparison[metric]['C'] - comparison[metric]['A']
                          for metric in ['pr_auc_trapezoid', 'average_precision', 'precision_at_k']})
    day_comparison = daily.pivot(index='transaction_date', columns='experiment', values='precision_at_k')
    day_comparison['C_minus_A'] = day_comparison.C - day_comparison.A
    for filename, frame in [('metrics.csv', metrics), ('monthly_metrics.csv', monthly),
                            ('monthly_C_minus_A.csv', delta.reset_index()),
                            ('daily_top_k.csv', daily), ('daily_C_minus_A.csv', day_comparison.reset_index()),
                            ('selected_parts.csv', pd.concat(selected_tables, ignore_index=True)),
                            ('predictions.csv', predictions)]:
        save_csv(frame, folder / filename)
    indexed = metrics.set_index('experiment')
    sensor_effect = {m: float(indexed.loc['C', m] - indexed.loc['A', m])
                     for m in ['pr_auc_trapezoid', 'average_precision', 'precision_at_k']}
    both = delta.average_precision_C_minus_A.gt(0) & delta.precision_at_k_C_minus_A.gt(0)
    result = {'stage': stage, 'top_k': k, 'C_minus_A': sensor_effect,
              'months_C_better_on_both_AP_and_top_k': int(both.sum()), 'months_evaluated': len(delta),
              'days_C_better_at_k': int(day_comparison.C_minus_A.gt(0).sum()),
              'days_C_tied_at_k': int(day_comparison.C_minus_A.eq(0).sum()),
              'days_C_worse_at_k': int(day_comparison.C_minus_A.lt(0).sum()),
              'note': 'Descriptive comparison only. Overlapping seven-day labels and repeated parts are correlated.'}
    write_json(folder / 'comparison.json', result)
    lines = [f'# A/B/C {stage} 비교', '', f'하루 점검 수: 전체 설비 합산 최대 {k}개.', '',
             '| 실험 | PR-AUC(사다리꼴) | AP | 상위 점검 적중률 | 하루 평균 적중 수 |',
             '| --- | ---: | ---: | ---: | ---: |']
    for row in metrics.itertuples():
        lines.append(f'| {row.experiment} {LABELS[row.experiment]} | {row.pr_auc_trapezoid:.4f} | '
                     f'{row.average_precision:.4f} | {row.precision_at_k:.2%} | {row.mean_hits_per_day:.2f} |')
    lines += ['', f'C − A: AP {sensor_effect["average_precision"]:+.4f}, '
              f'상위 점검 적중률 {100 * sensor_effect["precision_at_k"]:+.2f}%p.',
              f'월별 AP와 상위 점검 적중률이 모두 개선된 달: {int(both.sum())}/{len(delta)}.', '',
              '월별 결과는 개선의 일관성을 살피는 참고 자료입니다. 통계적 유의성이나 실제 고장의 인과관계를 입증하지 않습니다.',
              '같은 설비·부품이 여러 날 점검 대상으로 나올 수 있고 미래 7일 정답도 서로 겹칩니다. 적중 건수는 서로 다른 고장 사건 수가 아닙니다.',
              '이번 B/C의 센서는 당일 8개 값입니다. 출고량·센서 전일값·센서 이동평균은 넣지 않았습니다.',
              'A/B/C는 부품 정보를 공통으로 사용하므로 B 역시 순수 센서 단독 실험은 아닙니다.',
              '이전 data_huijae.py 작업에서 같은 시험 기간의 결과를 이미 본 이력이 있어 완전히 처음 보는 최종 평가 자료는 아닙니다.']
    if stage == 'validation':
        lines += ['이번 실행은 검증 자료만 평가했습니다. 시험 평가는 저장한 모델과 설정을 그대로 불러오는 별도 명령입니다.']
    (folder / 'summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(metrics[['experiment', 'pr_auc_trapezoid', 'average_precision', 'precision_at_k', 'mean_hits_per_day']].to_string(index=False), flush=True)
    return result


def run_validation(splits, audit, source, source_hash, run_dir, k):
    if k < 1:
        raise ValueError('하루 점검 수는 1 이상이어야 합니다.')
    run_dir.mkdir(parents=True, exist_ok=False)
    folder = run_dir / 'validation'
    folder.mkdir()
    specs = feature_sets()
    manifest = dict(source_path=str(source), source_sha256=source_hash, top_k=k,
                    features=specs, model_parameters=MODEL_PARAMS, audit=audit,
                    created_utc=datetime.now(timezone.utc).isoformat(),
                    versions={p: version(p) for p in ['numpy', 'pandas', 'scikit-learn', 'joblib']},
                    test_evaluated=False, prior_test_exposure='Previous data_huijae.py evaluated this same test period.')
    # 실행 설정은 성능을 계산하기 전에 저장합니다.
    write_json(run_dir / 'manifest.json', manifest)
    train, validation = splits['train'], splits['validation']
    if train[TARGET].nunique() != 2:
        raise ValueError('학습 구간에 0과 1 정답이 모두 있어야 합니다.')
    predictions, model_hashes = [], {}
    for name, spec in specs.items():
        columns = spec['numeric'] + spec['categorical']
        model = make_model(spec)
        with warnings.catch_warnings():
            warnings.simplefilter('error', ConvergenceWarning)
            model.fit(train[columns], train[TARGET].astype(int))
        path = run_dir / f'model_{name}.joblib'
        joblib.dump({'pipeline': model, 'columns': columns}, path)
        restored = joblib.load(path)
        score = restored['pipeline'].predict_proba(validation[columns])[:, 1]
        np.testing.assert_allclose(score[:50], model.predict_proba(validation[columns].head(50))[:, 1])
        predictions.append(validation[PART_KEYS + [TARGET]].assign(score=score, experiment=name))
        model_hashes[name] = digest(path)
        print(f'{name} fit complete: {len(train)} train / {len(validation)} validation rows', flush=True)
    manifest['model_sha256'] = model_hashes
    write_json(run_dir / 'manifest.json', manifest)
    save_evaluation(pd.concat(predictions, ignore_index=True), folder, k, 'validation')


def run_test(splits, source_hash, run_dir, k=None):
    manifest = json.loads((run_dir / 'manifest.json').read_text(encoding='utf-8'))
    if source_hash != manifest['source_sha256']:
        raise ValueError('검증 때 사용한 원본과 다릅니다.')
    if k is not None and k != manifest['top_k']:
        raise ValueError('시험에서 점검 수를 변경할 수 없습니다. 검증 때 정한 값을 사용하세요.')
    if manifest['features'] != feature_sets() or manifest['model_parameters'] != MODEL_PARAMS:
        raise ValueError('검증 때의 실험 설정과 코드가 다릅니다.')
    if (run_dir / 'test').exists():
        raise ValueError('이 실험의 시험 평가는 이미 있습니다. 기존 결과를 확인하세요.')
    predictions = []
    test = splits['test']
    for name in manifest['features']:
        path = run_dir / f'model_{name}.joblib'
        if digest(path) != manifest['model_sha256'][name]:
            raise ValueError(f'{name} 저장 모델이 검증 후 변경됐습니다.')
        artifact = joblib.load(path)
        score = artifact['pipeline'].predict_proba(test[artifact['columns']])[:, 1]
        predictions.append(test[PART_KEYS + [TARGET]].assign(score=score, experiment=name))
    folder = run_dir / 'test'
    folder.mkdir()
    save_evaluation(pd.concat(predictions, ignore_index=True), folder, manifest['top_k'], 'test')
    manifest['test_evaluated'] = True
    write_json(run_dir / 'manifest.json', manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, help='원본 CSV 경로. 기본값은 Downloads의 합성 데이터입니다.')
    parser.add_argument('--stage', choices=['validation', 'test'], default='validation')
    parser.add_argument('--top-k', type=int, default=None, help='전체 설비 합산 하루 점검 수. 검증 기본값 10.')
    parser.add_argument('--run-dir', type=Path, help='검증은 새 폴더, 시험은 기존 검증 결과 폴더를 지정합니다.')
    args = parser.parse_args()
    if args.top_k is not None and args.top_k < 1:
        parser.error('--top-k must be positive')
    if args.stage == 'test' and args.run_dir is None:
        parser.error('--stage test requires --run-dir')
    if args.stage == 'test':
        manifest = json.loads((args.run_dir / 'manifest.json').read_text(encoding='utf-8'))
        source = (args.input or Path(manifest['source_path'])).resolve()
    else:
        source = (args.input or Path.home() / 'Downloads/synthetic_industrial_machine_data.csv').resolve()
    source_hash = digest(source)
    raw = pd.read_csv(source, parse_dates=['transaction_date'])
    splits, audit = prepare_rows(raw)
    if digest(source) != source_hash:
        raise RuntimeError('읽는 동안 원본이 변경됐습니다.')
    run_dir = args.run_dir or ROOT / 'data/processed/industrial/ablation_results' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    if args.stage == 'validation':
        run_validation(splits, audit, source, source_hash, run_dir, args.top_k or 10)
    else:
        run_test(splits, source_hash, run_dir, args.top_k)
    if digest(source) != source_hash:
        raise RuntimeError('실행 중 원본이 변경됐습니다.')
    print(f'Results: {run_dir.resolve() / args.stage}', flush=True)


if __name__ == '__main__':
    main()
