"""산업기계 부품 로지스틱 회귀와 설비 전력 Prophet 실행.

실행: python -m src.data_huijae
전처리: python -m src.prepare_machine_data
"""
import argparse
import json
import logging
import warnings
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
KEYS = ['asset_tag', 'part_no', 'transaction_date']


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def save_csv(frame, path):
    frame.to_csv(path, index=False, encoding='utf-8-sig')


def classification_metrics(y, score, threshold):
    predicted = np.asarray(score) >= threshold
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return dict(rows=len(y), positive_rate=float(np.mean(y)), threshold=float(threshold),
                accuracy=float(accuracy_score(y, predicted)),
                precision=float(precision_score(y, predicted, zero_division=0)),
                recall=float(recall_score(y, predicted, zero_division=0)),
                f1=float(f1_score(y, predicted, zero_division=0)),
                roc_auc=float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else None,
                average_precision=float(average_precision_score(y, score)) if np.any(y) else None,
                brier=float(brier_score_loss(y, score)),
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))


def choose_threshold(y, score):
    # 판정 기준은 검증 구간으로만 정하며 테스트 정답은 사용하지 않습니다.
    table = pd.DataFrame([classification_metrics(y, score, t) for t in np.linspace(.05, .95, 19)])
    table['distance_from_half'] = abs(table.threshold - .5)
    best = table.sort_values(['f1', 'distance_from_half'], ascending=[False, True]).iloc[0]
    return float(best.threshold), table


def build_classifier(schema):
    numeric = Pipeline([('impute', SimpleImputer(strategy='median', add_indicator=True)),
                        ('scale', StandardScaler())])
    categorical = Pipeline([('impute', SimpleImputer(strategy='most_frequent')),
                            ('encode', OneHotEncoder(handle_unknown='ignore'))])
    transform = ColumnTransformer([('numeric', numeric, schema['numeric_features']),
                                   ('categorical', categorical, schema['categorical_features'])])
    return Pipeline([('features', transform),
                     ('model', LogisticRegression(max_iter=2000, random_state=42))])


def run_logistic(data_dir, output):
    schema = json.loads((data_dir / 'feature_schema.json').read_text(encoding='utf-8'))
    part = pd.read_csv(data_dir / 'part_df.csv', parse_dates=['transaction_date'])
    features = schema['numeric_features'] + schema['categorical_features']
    if set(features) & set(schema['excluded_columns']):
        raise ValueError('입력 허용 목록에 제외 열이 포함되어 있습니다.')
    target = schema['target']
    usable = part.loc[part.eligible_for_model.eq(True)].copy()
    groups = {s: usable.loc[usable.split.eq(s)] for s in ['train', 'validation', 'test']}
    for name, group in groups.items():
        if group.empty or group[target].nunique() != 2:
            raise ValueError(f'{name} 구간에 두 종류의 정답이 필요합니다.')
    for left, right in [('train', 'validation'), ('validation', 'test')]:
        if groups[left].transaction_date.max() + pd.Timedelta(days=7) >= groups[right].transaction_date.min():
            raise ValueError('미래 7일 정답이 다음 평가 구간에 겹칩니다.')
    model = build_classifier(schema)
    train = groups['train']
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        model.fit(train[features], train[target].astype(int))
    validation = groups['validation']
    threshold, curve = choose_threshold(validation[target].astype(int),
                                        model.predict_proba(validation[features])[:, 1])
    save_csv(curve, output / 'logistic_threshold_validation.csv')
    prevalence = float(train[target].mean())
    metrics, predictions = [], []
    for split in ['validation', 'test']:
        group = groups[split]
        y = group[target].astype(int)
        score = model.predict_proba(group[features])[:, 1]
        for name, values, cutoff in [('logistic', score, threshold),
                                      ('logistic_threshold_0.5', score, .5),
                                      ('train_prior_baseline', np.full(len(y), prevalence), .5)]:
            metrics.append(dict(model=name, split=split, **classification_metrics(y, values, cutoff)))
        result = group[KEYS + [target, 'split']].copy()
        result['score'] = score
        result['predicted_flag'] = (score >= threshold).astype(int)
        predictions.append(result)
    save_csv(pd.DataFrame(metrics), output / 'logistic_metrics.csv')
    save_csv(pd.concat(predictions), output / 'logistic_predictions.csv')
    artifact = dict(pipeline=model, threshold=threshold, features=features,
                    target=target, prediction_time=schema['prediction_time'])
    model_path = output / 'logistic.joblib'
    joblib.dump(artifact, model_path)
    restored = joblib.load(model_path)
    sample = groups['test'][features].head(50)
    np.testing.assert_allclose(model.predict_proba(sample), restored['pipeline'].predict_proba(sample))
    # 최근 날짜는 정답을 알 수 없어도 현재 정상이고 이력이 충분하면 점수를 계산합니다.
    latest = part.loc[part.transaction_date.eq(part.transaction_date.max()) &
                      part.breakdown_flag.eq(0) & part.history_30d_complete.eq(True)].copy()
    latest['score'] = model.predict_proba(latest[features])[:, 1]
    latest['predicted_flag'] = (latest.score >= threshold).astype(int)
    save_csv(latest[KEYS + ['score', 'predicted_flag']], output / 'part_latest_scores.csv')
    coefficient = pd.DataFrame({'feature': model['features'].get_feature_names_out(),
                                'coefficient': model['model'].coef_[0]})
    save_csv(coefficient.sort_values('coefficient', ascending=False), output / 'logistic_coefficients.csv')
    report = dict(threshold=threshold, train_rows=len(train), latest_rows=len(latest), metrics=metrics,
                  note='Synthetic flag prediction; uncalibrated scores, not proven physical failure probabilities.')
    write_json(output / 'logistic_report.json', report)
    print('Logistic finished:', json.dumps(report), flush=True)


def checkpoint_dates(series, stride):
    if stride < 1:
        raise ValueError('평가 날짜 간격은 1 이상이어야 합니다.')
    dates = pd.DatetimeIndex(sorted(pd.to_datetime(series).unique()))
    return dates[::stride].union(dates[-1:])


def fit_power(history):
    from prophet import Prophet
    # 미래의 센서 실측값은 쓰지 않고 날짜와 과거 전력만 사용합니다.
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True,
                    daily_seasonality=False, uncertainty_samples=0)
    model.fit(history[['ds', 'y']], seed=42)
    return model


def run_prophet(data_dir, output, stride):
    from prophet.serialize import model_to_json, model_from_json
    logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
    machine = pd.read_csv(data_dir / 'machine_df.csv', parse_dates=['transaction_date'])
    rows, future_rows = [], []
    model_dir = output / 'prophet_models'
    model_dir.mkdir(exist_ok=True)
    for number, (asset, group) in enumerate(machine.groupby('asset_tag', sort=True), start=1):
        group = group.sort_values('transaction_date').rename(columns={'transaction_date': 'ds', 'power_consumption_kw': 'y'})
        if group.ds.duplicated().any() or not group.ds.diff().dropna().eq(pd.Timedelta(days=1)).all():
            raise ValueError(f'{asset}: 다음 날 평가에는 중복 없는 연속 일 자료가 필요합니다.')
        for split in ['validation', 'test']:
            dates = checkpoint_dates(group.loc[group.split.eq(split), 'ds'], stride)
            for day in dates:
                history = group.loc[group.ds < day]
                if len(history) < 365:
                    raise ValueError('연간 패턴 학습에는 최소 365일이 필요합니다.')
                model = fit_power(history)
                prediction = float(model.predict(pd.DataFrame({'ds': [day]})).yhat.iloc[0])
                rows.append(dict(asset_tag=asset, split=split, ds=day, origin=history.ds.max(),
                                 train_rows=len(history), actual=float(group.loc[group.ds.eq(day), 'y'].iloc[0]),
                                 prophet=prediction, previous_day=float(history.y.iloc[-1]),
                                 previous_week=float(history.y.iloc[-7])))
            print(f'Prophet {number}/10 {asset} {split}: {len(dates)} checkpoints', flush=True)
        final_model = fit_power(group)
        day = group.ds.max() + pd.Timedelta(days=1)
        forecast = float(final_model.predict(pd.DataFrame({'ds': [day]})).yhat.iloc[0])
        # 모델 파일명은 데이터 문자열 대신 순번을 써서 경로 문자를 방지합니다.
        model_path = model_dir / f'asset_{number:02d}.json'
        serialized = model_to_json(final_model)
        model_path.write_text(serialized, encoding='utf-8')
        restored = model_from_json(serialized)
        np.testing.assert_allclose(restored.predict(pd.DataFrame({'ds': [day]})).yhat.iloc[0], forecast)
        future_rows.append(dict(asset_tag=asset, ds=day, forecast_kw=forecast,
                                trained_through=group.ds.max(), model_file=str(model_path.relative_to(output))))
    backtest = pd.DataFrame(rows)
    save_csv(backtest, output / 'power_backtest.csv')
    save_csv(pd.DataFrame(future_rows), output / 'power_next_day.csv')
    metrics = []
    pooled = backtest.assign(asset_tag='ALL')
    for (asset, split), group in pd.concat([backtest, pooled]).groupby(['asset_tag', 'split']):
        for name in ['prophet', 'previous_day', 'previous_week']:
            error = group[name] - group.actual
            metrics.append(dict(asset_tag=asset, split=split, model=name, rows=len(group),
                                mae_kw=float(error.abs().mean()), rmse_kw=float(np.sqrt((error ** 2).mean()))))
    save_csv(pd.DataFrame(metrics), output / 'power_metrics.csv')
    write_json(output / 'prophet_report.json', dict(stride_days=stride, backtest_rows=len(backtest),
               horizon_days=1, evaluation='Expanding history, sampled checkpoints plus split end; fixed settings. Earlier test observations become available at later origins.',
               pooled_metrics=[m for m in metrics if m['asset_tag'] == 'ALL']))
    print('Prophet finished:', [m for m in metrics if m['asset_tag'] == 'ALL'], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data/processed/industrial')
    parser.add_argument('--models', choices=['all', 'logistic', 'prophet'], default='all')
    parser.add_argument('--prophet-stride', type=int, default=14,
                        help='평가 날짜 간격. 14는 표본 평가, 1은 모든 날짜 평가입니다.')
    args = parser.parse_args()
    if args.prophet_stride < 1:
        parser.error('--prophet-stride must be positive')
    output = args.data_dir / 'model_results'
    output.mkdir(parents=True, exist_ok=True)
    if args.models in ['all', 'logistic']:
        run_logistic(args.data_dir, output)
    if args.models in ['all', 'prophet']:
        run_prophet(args.data_dir, output, args.prophet_stride)
    write_json(output / 'run_versions.json', {p: version(p) for p in ['numpy', 'pandas', 'scikit-learn', 'prophet', 'cmdstanpy']})


if __name__ == '__main__':
    main()
