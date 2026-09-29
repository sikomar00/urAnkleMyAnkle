"""산업기계 CSV 분리·과거 특징·7일 정답 생성. 원본 파일은 읽기만 합니다."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SENSORS = ['temp_bearing_degC', 'temp_motor_degC', 'vibration_h_mms',
           'vibration_v_mms', 'oil_pressure_bar', 'load_pct', 'shaft_rpm',
           'power_consumption_kw']
MACHINE_KEYS = ['asset_tag', 'transaction_date']
PART_KEYS = ['asset_tag', 'part_no', 'transaction_date']
MACHINE_INFO = ['machine_type', 'plant_code']
PART_INFO = ['part_description', 'part_family', 'criticality', 'uom', 'unit_cost_inr']


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def validate(raw):
    required = set(PART_KEYS + MACHINE_INFO + PART_INFO + SENSORS +
                   ['breakdown_flag', 'qty_issued', 'issue_value_inr', 'wo_type'])
    if required - set(raw):
        raise ValueError(f'필수 열 누락: {sorted(required - set(raw))}')
    if raw[PART_KEYS].isna().any().any() or raw.duplicated(PART_KEYS).any():
        raise ValueError('기계·부품·날짜 키가 비어 있거나 중복됩니다.')
    numeric = SENSORS + ['unit_cost_inr', 'qty_issued', 'issue_value_inr', 'breakdown_flag']
    if not np.isfinite(raw[numeric].to_numpy(dtype=float)).all():
        raise ValueError('숫자 열에 결측 또는 무한값이 있습니다.')
    if not raw.breakdown_flag.isin([0, 1]).all():
        raise ValueError('고장 표시는 0 또는 1이어야 합니다.')
    if raw[['qty_issued', 'unit_cost_inr', 'issue_value_inr']].lt(0).any().any():
        raise ValueError('출고량·단가·금액에 음수가 있습니다.')
    if not np.allclose(raw.unit_cost_inr * raw.qty_issued, raw.issue_value_inr):
        raise ValueError('단가 × 출고량과 출고 금액이 다릅니다.')
    same_day = raw.groupby(MACHINE_KEYS)[SENSORS + MACHINE_INFO].nunique(dropna=False)
    if same_day.gt(1).any().any():
        raise ValueError('같은 기계·날짜 안의 센서 또는 설비 정보가 다릅니다.')


def future_seven(series):
    """일 단위로 재색인한 시계열에서 t+1~t+7의 표시 여부를 계산합니다."""
    future = pd.concat([series.shift(-i) for i in range(1, 8)], axis=1)
    return future.max(axis=1).where(future.notna().all(axis=1)).astype('Int64')


def calendar_group(group):
    ordered = group.set_index('transaction_date').sort_index()
    return ordered, ordered.reindex(pd.date_range(ordered.index.min(), ordered.index.max(), freq='D'))


def assign_split(frame, dates):
    """공통 날짜 경계와 미래 7일 정답의 구간 중첩을 표시합니다."""
    train_end = dates[int(len(dates) * .70) - 1]
    validation_end = dates[int(len(dates) * .85) - 1]
    result = frame.copy()
    result['split'] = np.select(
        [result.transaction_date <= train_end, result.transaction_date <= validation_end],
        ['train', 'validation'], default='test')
    end = result['split'].map({'train': train_end, 'validation': validation_end, 'test': dates[-1]})
    result['target_window_end'] = result.transaction_date + pd.Timedelta(days=7)
    result['target_within_split'] = result.target_window_end <= end
    return result, {'train_end': str(train_end.date()),
                    'validation_start': str(dates[int(len(dates) * .70)].date()),
                    'validation_end': str(validation_end.date()),
                    'test_start': str(dates[int(len(dates) * .85)].date()),
                    'test_end': str(dates[-1].date())}


def prepare(raw):
    raw = raw.copy()
    raw['transaction_date'] = pd.to_datetime(raw.transaction_date, errors='raise')
    validate(raw)
    raw = raw.sort_values(PART_KEYS).reset_index(drop=True)
    dates = pd.DatetimeIndex(sorted(raw.transaction_date.unique()))
    if len(dates) < 30:
        raise ValueError('날짜순 분할에는 최소 30개 날짜가 필요합니다.')
    machines = raw[MACHINE_KEYS + MACHINE_INFO + SENSORS].drop_duplicates(MACHINE_KEYS)
    machine_groups = []
    machine_features = []
    for _, group in machines.groupby('asset_tag', sort=True):
        observed, daily = calendar_group(group)
        for col in SENSORS:
            for suffix, values in [('_lag_1d', daily[col].shift(1)),
                                   ('_mean_prev_7d', daily[col].shift(1).rolling(7, min_periods=7).mean())]:
                name = col + suffix
                observed[name] = values.reindex(observed.index)
                if name not in machine_features:
                    machine_features.append(name)
        machine_groups.append(observed.reset_index())
    machine_df = pd.concat(machine_groups, ignore_index=True)
    machine_df, split_info = assign_split(machine_df, dates)
    # 센서용 표의 경계 표시는 7일 분류에만 해당하므로 제거합니다.
    machine_df = machine_df.drop(columns=['target_window_end', 'target_within_split'])

    part_groups = []
    history_features = []
    for _, group in raw.groupby(['asset_tag', 'part_no'], sort=True):
        observed, daily = calendar_group(group)
        for days in [7, 30]:
            for source, prefix in [('breakdown_flag', 'breakdown_days'), ('qty_issued', 'qty_issued')]:
                name = f'{prefix}_prev_{days}d'
                observed[name] = daily[source].shift(1).rolling(days, min_periods=days).sum().reindex(observed.index)
                if name not in history_features:
                    history_features.append(name)
        last_fault = pd.Series(daily.index, index=daily.index).where(daily.breakdown_flag.eq(1)).ffill().shift(1)
        elapsed = (pd.Series(daily.index, index=daily.index) - last_fault).dt.days
        observed['days_since_last_observed_breakdown'] = elapsed.reindex(observed.index)
        observed['has_prior_observed_breakdown'] = last_fault.notna().reindex(observed.index)
        observed['history_30d_complete'] = daily.breakdown_flag.shift(1).rolling(30, min_periods=30).count().eq(30).reindex(observed.index)
        target = future_seven(daily.breakdown_flag).reindex(observed.index)
        observed['future_7d_complete'] = target.notna()
        observed['target_breakdown_next_7d'] = target.where(observed.breakdown_flag.eq(0))
        part_groups.append(observed.reset_index())
    part_df = pd.concat(part_groups, ignore_index=True)
    part_df, _ = assign_split(part_df, dates)
    part_df['eligible_for_target'] = part_df.breakdown_flag.eq(0) & part_df.future_7d_complete
    part_df['eligible_for_model'] = (part_df.eligible_for_target & part_df.target_within_split & part_df.history_30d_complete)
    # 기계·날짜별 센서 특징을 정확히 한 번씩 연결합니다.
    part_df = part_df.merge(machine_df[MACHINE_KEYS + machine_features], on=MACHINE_KEYS, how='left', validate='many_to_one')
    numeric_features = SENSORS + machine_features + ['unit_cost_inr'] + history_features + ['days_since_last_observed_breakdown', 'has_prior_observed_breakdown']
    categorical_features = ['asset_tag', 'machine_type', 'plant_code', 'part_no', 'part_family', 'criticality', 'uom']
    schema = {
        'prediction_time': 'End of day t; sensors through t; part history strictly before t.',
        'target': 'target_breakdown_next_7d',
        'model_row_filter': 'eligible_for_model == True',
        'numeric_features': numeric_features, 'categorical_features': categorical_features,
        'excluded_columns': [c for c in part_df if c not in numeric_features + categorical_features],
        'notes': ['Fit imputers, scalers and encoders on train only.',
                  'Missing days_since_last_observed_breakdown means no previously observed flag, not zero days.',
                  'History counts flag-days, not distinct failure events.',
                  'Use this allowlist, never every column except target.',
                  'Known-machine temporal evaluation; unseen-machine generalization is not evaluated.']}
    # 설비 단위 정답은 진단용으로만 계산하며 부품 모델 정답으로 쓰지 않습니다.
    equipment = raw.groupby(MACHINE_KEYS).breakdown_flag.max().reset_index()
    machine_targets = []
    for _, group in equipment.groupby('asset_tag'):
        obs, daily = calendar_group(group)
        y = future_seven(daily.breakdown_flag).reindex(obs.index)
        machine_targets.append(y[obs.breakdown_flag.eq(0) & y.notna()])
    y_machine = pd.concat(machine_targets)
    y_part = part_df.loc[part_df.eligible_for_target, 'target_breakdown_next_7d']
    split_report = {}
    for name, group in part_df.groupby('split', sort=False):
        use = group[group.eligible_for_model]
        split_report[name] = {'all_rows': len(group), 'model_rows': len(use),
                             'positive_rows': int(use.target_breakdown_next_7d.sum()),
                             'positive_rate': float(use.target_breakdown_next_7d.mean()) if len(use) else None}
    report = {'raw_rows': len(raw), 'machine_rows': len(machine_df), 'part_rows': len(part_df),
              'assets': raw.asset_tag.nunique(), 'parts': raw.part_no.nunique(), 'dates': len(dates),
              'sensor_inconsistent_groups': 0,
              'mixed_part_flag_machine_days': int(raw.groupby(MACHINE_KEYS).breakdown_flag.nunique().gt(1).sum()),
              'machine_flag_days': int(equipment.breakdown_flag.sum()),
              'machine_no_flag_days': int(equipment.breakdown_flag.eq(0).sum()),
              'machine_7d_eligible_rows': len(y_machine),
              'machine_7d_positive_rate': float(y_machine.mean()) if len(y_machine) else None,
              'part_7d_eligible_rows': len(y_part), 'part_7d_positive_rows': int(y_part.sum()),
              'part_7d_positive_rate': float(y_part.mean()) if len(y_part) else None,
              'missing_work_order_rows': int(raw.wo_type.isna().sum()),
              'split_dates': split_info, 'splits': split_report}
    return machine_df, part_df, schema, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path.home() / 'Downloads' / 'synthetic_industrial_machine_data.csv')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'data/processed/industrial')
    args = parser.parse_args()
    source = args.input.resolve()
    output = args.output_dir.resolve()
    outputs = [output / name for name in ['machine_df.csv', 'part_df.csv', 'feature_schema.json', 'quality_report.json']]
    if source in outputs:
        raise ValueError('원본 경로에 결과를 저장할 수 없습니다.')
    before = digest(source)
    raw = pd.read_csv(source, parse_dates=['transaction_date'])
    machine, part, schema, report = prepare(raw)
    output.mkdir(parents=True, exist_ok=True)
    machine.to_csv(outputs[0], index=False, encoding='utf-8-sig')
    part.to_csv(outputs[1], index=False, encoding='utf-8-sig')
    report['source_path'] = str(source)
    report['source_sha256_before'] = before
    report['source_sha256_after'] = digest(source)
    report['source_unchanged'] = before == report['source_sha256_after']
    for path, document in [(outputs[2], schema), (outputs[3], report)]:
        path.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding='utf-8')
    if not report['source_unchanged']:
        raise RuntimeError('처리 도중 원본이 변경되었습니다. 결과 사용 전 확인하세요.')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
