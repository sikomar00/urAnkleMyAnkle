"""dashboard_data.py 화면 ⑤ 로더(load_screen5_kpis/load_failure_trend) 검증."""
import os
from pathlib import Path

import pytest

root = Path(__file__).resolve().parents[1]
name = 'synthetic_industrial_machine_data.csv'
paths = ([Path(os.environ['MACHINE_DATA_PATH'])] if os.environ.get('MACHINE_DATA_PATH')
         else [root / 'data/raw' / name, Path.home() / 'Downloads' / name])
if not any(path.is_file() for path in paths):
    pytest.skip('화면 ⑤ 데이터 로더 검증용 원본 CSV가 없습니다.', allow_module_level=True)

from src import dashboard_data as dd


def test_screen5_kpis_share_values_with_screen1():
    s1, s5 = dd.load_screen1_kpis(), dd.load_screen5_kpis()
    assert s5['failure_machine_days'] == s1['failure_machine_days']
    assert s5['failure_rate_pct'] == pytest.approx(s1['failure_rate_pct'])
    assert s5['parts_issue_value_inr'] == pytest.approx(s1['parts_issue_value_inr'])
    assert s5['avg_power_kw'] == pytest.approx(s1['avg_power_kw'])


def test_failure_trend_shape_and_dtypes():
    trend = dd.load_failure_trend()
    assert len(trend) == 1095
    assert list(trend.columns) == ['transaction_date', 'failure_days', 'high_risk_pct']
    assert trend['transaction_date'].is_monotonic_increasing
    assert trend['failure_days'].dtype.kind == 'i'
    assert trend['high_risk_pct'].dtype.kind == 'f'
    assert (trend['high_risk_pct'] >= 0).all() and (trend['high_risk_pct'] <= 100).all()


def test_failure_trend_last_30_days_sum():
    trend = dd.load_failure_trend()
    assert trend.tail(30)['failure_days'].sum() == 211


def test_asset_sensor_series_shape_and_flags():
    series = dd.load_asset_sensor_series('AST-1041')
    assert len(series) == 1095
    assert list(series.columns) == [
        'transaction_date', 'temp_bearing_degC', 'temp_motor_degC',
        'vibration_h_mms', 'vibration_v_mms', 'oil_pressure_bar', 'load_pct',
        'shaft_rpm', 'power_consumption_kw', 'is_failure_day', 'is_high_risk_day',
    ]
    assert series['transaction_date'].is_monotonic_increasing
    assert series['is_failure_day'].dtype.kind == 'b'
    assert series['is_high_risk_day'].dtype.kind == 'b'
    assert int(series['is_failure_day'].sum()) == 738
    assert int(series['is_high_risk_day'].sum()) == 114
    # 위험 기준선 초과일은 고장 표시일의 부분집합이다.
    assert not (series['is_high_risk_day'] & ~series['is_failure_day']).any()


def test_asset_sensor_series_rejects_unknown_asset():
    with pytest.raises(ValueError):
        dd.load_asset_sensor_series('AST-9999')


def test_asset_parts_history_top10_and_ranking():
    parts = dd.load_asset_parts_history('AST-1041')
    assert len(parts) == 10
    assert list(parts.columns) == ['part_no', 'part_description', 'total_issue_value_inr']
    assert parts.iloc[0]['part_no'] == 'MRO-20031'
    assert int(parts.iloc[0]['total_issue_value_inr']) == 281600
    assert parts['total_issue_value_inr'].is_monotonic_decreasing


def test_asset_parts_history_rejects_unknown_asset():
    with pytest.raises(ValueError):
        dd.load_asset_parts_history('AST-9999')


def test_asset_peer_comparison_pairs_and_shape():
    r1 = dd.load_asset_peer_comparison('AST-1041')
    assert r1['peer_asset_tag'] == 'AST-1042'
    assert len(r1['asset_values']) == 1095
    assert len(r1['peer_values']) == 1095

    r2 = dd.load_asset_peer_comparison('AST-2031')
    assert r2['peer_asset_tag'] == 'AST-2017'


def test_asset_peer_comparison_rejects_unknown_asset():
    with pytest.raises(ValueError):
        dd.load_asset_peer_comparison('AST-9999')


def test_asset_failure_onset_trend_uses_most_recent_episode():
    trend = dd.load_asset_failure_onset_trend('AST-1041')
    assert trend['episode_start_date'] == '2024-12-24'
    assert trend['episode_start_date'] != '2022-01-10'
    assert trend['relative_days'] == [-7, -6, -5, -4, -3, -2, -1, 0]
    assert trend['temp_bearing_degC'] == [66.4, 68.3, 68.4, 69.4, 66.7, 63.3, 66.9, 69.1]
    assert len(trend['dates']) == 8


def test_asset_failure_onset_trend_rejects_unknown_asset():
    with pytest.raises(ValueError):
        dd.load_asset_failure_onset_trend('AST-9999')


def test_asset_family_diagnosis_has_nine_part_families():
    rows = dd.load_asset_family_diagnosis('AST-1041')
    assert len(rows) == 9
    families = {r['part_family'] for r in rows}
    assert families == {
        'Bearing', 'Coupling', 'Drive Belt', 'Electrical', 'Fastener',
        'Filter', 'Lubrication', 'Seal & Gasket', 'Sensor',
    }
    for r in rows:
        assert set(r) == {
            'part_family', 'model', 'support', 'positive_rate',
            'precision', 'recall', 'average_precision', 'roc_auc',
        }


def test_asset_family_diagnosis_bearing_values():
    rows = dd.load_asset_family_diagnosis('AST-1041')
    bearing = next(r for r in rows if r['part_family'] == 'Bearing')
    assert bearing['average_precision'] == pytest.approx(0.3919, abs=1e-4)
    assert bearing['roc_auc'] == pytest.approx(0.6555, abs=1e-4)


def test_asset_family_diagnosis_rejects_unknown_asset():
    with pytest.raises(ValueError):
        dd.load_asset_family_diagnosis('AST-9999')


def test_asset_family_diagnosis_sorted_by_average_precision_desc():
    rows = dd.load_asset_family_diagnosis('AST-1041')
    aps = [r['average_precision'] for r in rows]
    assert aps == sorted(aps, reverse=True)
    assert rows[0]['part_family'] == 'Bearing'


def test_family_feature_importance_top_feature_is_shaft_rpm():
    rows = dd.load_family_feature_importance('Bearing')
    assert len(rows) == 10
    assert rows[0]['feature'] == 'shaft_rpm'
    importances = [r['importance_mean'] for r in rows]
    assert importances == sorted(importances, reverse=True)


def test_family_feature_importance_rejects_unknown_family():
    with pytest.raises(ValueError):
        dd.load_family_feature_importance('NotAFamily')


def test_family_pr_curve_and_confusion_matches_displayed_metrics():
    result = dd.load_family_pr_curve_and_confusion('AST-1041', 'Bearing')
    c = result['confusion']
    precision = c['tp'] / (c['tp'] + c['fp'])
    recall = c['tp'] / (c['tp'] + c['fn'])
    assert precision == pytest.approx(0.4135, abs=1e-4)
    assert recall == pytest.approx(1.0, abs=1e-4)
    assert len(result['precision_curve']) == len(result['recall_curve'])


def test_family_pr_curve_and_confusion_rejects_unknown_inputs():
    with pytest.raises(ValueError):
        dd.load_family_pr_curve_and_confusion('AST-9999', 'Bearing')
    with pytest.raises(ValueError):
        dd.load_family_pr_curve_and_confusion('AST-1041', 'NotAFamily')


def test_family_recurrence_intervals_are_positive_day_counts():
    intervals = dd.load_family_recurrence_intervals('AST-1041', 'Bearing')
    assert len(intervals) > 0
    assert all(isinstance(v, int) and v > 0 for v in intervals)


def test_family_recurrence_intervals_rejects_unknown_inputs():
    with pytest.raises(ValueError):
        dd.load_family_recurrence_intervals('AST-9999', 'Bearing')
    with pytest.raises(ValueError):
        dd.load_family_recurrence_intervals('AST-1041', 'NotAFamily')


def test_priority_table_asc_direction_reverses_desc_and_reranks():
    desc_rows = dd.load_priority_table('grade', 'desc')
    asc_rows = dd.load_priority_table('grade', 'asc')
    assert [r['asset_tag'] for r in asc_rows] == [r['asset_tag'] for r in desc_rows][::-1]
    assert [r['rank'] for r in asc_rows] == list(range(1, len(asc_rows) + 1))
    assert asc_rows[0]['failure_points'] <= desc_rows[0]['failure_points']


def test_screen1_machine_status_covers_all_assets_sorted():
    rows = dd.load_screen1_machine_status()
    assert [r['asset_tag'] for r in rows] == sorted(r['asset_tag'] for r in rows)
    assert len(rows) == 10
    for r in rows:
        assert set(r) == {
            'asset_tag', 'machine_type', 'current_grade', 'risk_score', 'sparkline',
        }
        assert len(r['sparkline']) == 30


def test_screen1_machine_status_ast1041_latest_values():
    row = next(r for r in dd.load_screen1_machine_status() if r['asset_tag'] == 'AST-1041')
    assert row['current_grade'] == '경계'
    assert row['risk_score'] == pytest.approx(6.0)
    assert row['sparkline'][-1] == pytest.approx(6.0)


def test_screen1_power_by_machine_sorted_desc_and_covers_all_assets():
    rows = dd.load_screen1_power_by_machine()
    assert len(rows) == 10
    assert {r['asset_tag'] for r in rows} == set(dd.load_asset_list())
    values = [r['avg_power_kw'] for r in rows]
    assert values == sorted(values, reverse=True)
    assert rows[0]['asset_tag'] == 'AST-2031'
    assert rows[0]['avg_power_kw'] == pytest.approx(71.203, abs=1e-3)


def test_asset_failure_heatmap_shape_and_ast1041_total():
    heat = dd.load_asset_failure_heatmap()
    assert list(heat.columns) == ['asset_tag', 'period', 'failed_part_count']
    assert len(heat) == 370  # 10개 자산 × 37개월
    assert heat['period'].nunique() == 37
    assert heat.groupby('asset_tag')['failed_part_count'].sum()['AST-1041'] == 2071
    assert (heat['failed_part_count'] >= 0).all()


# ------------------------------------------------------------
# 필터바(공장·기계 종류·기계·기간) 적용
# ------------------------------------------------------------
def test_period_start_counts_reference_day_inclusive():
    import pandas as pd
    assert dd.period_start(30) == pd.Timestamp('2024-12-03')
    assert dd.period_start(None) is None


def test_filter_assets_by_plant_type_and_machine():
    assert dd.filter_assets(plant='CHN-02') == ['AST-2031', 'AST-3008', 'AST-3019']
    assert dd.filter_assets(plant='CHN-02', machine_type='Belt Conveyor') == ['AST-3008', 'AST-3019']
    assert dd.filter_assets(plant='CHN-02', machine='AST-1041') == []
    assert dd.filter_assets() == dd.load_asset_list()


def test_priority_table_scoped_to_plant():
    rows = dd.load_priority_table('grade', 'desc', assets=dd.filter_assets(plant='CHN-02'))
    assert [r['asset_tag'] for r in rows] == ['AST-2031', 'AST-3019', 'AST-3008']
    assert [r['rank'] for r in rows] == [1, 2, 3]


def test_sensor_series_scoped_to_last_30_days():
    import pandas as pd
    series = dd.load_asset_sensor_series('AST-2031', start=dd.period_start(30))
    assert len(series) == 30
    assert series['transaction_date'].min() == pd.Timestamp('2024-12-03')
    assert series['transaction_date'].max() == pd.Timestamp('2025-01-01')


def test_screen1_loaders_scoped_to_assets_and_period():
    chn = dd.filter_assets(plant='CHN-02')
    start = dd.period_start(30)
    kpis = dd.load_screen1_kpis(assets=chn, start=start)
    assert kpis['observed_machines'] == 3
    assert kpis['failure_machine_days'] <= dd.load_screen1_kpis(assets=chn)['failure_machine_days']
    assert [r['asset_tag'] for r in dd.load_screen1_machine_status(assets=chn)] == chn
    assert {r['asset_tag'] for r in dd.load_screen1_power_by_machine(assets=chn, start=start)} == set(chn)
    heat = dd.load_asset_failure_heatmap(assets=chn, start=start)
    assert sorted(heat['asset_tag'].unique()) == chn
    assert sorted(heat['period'].unique()) == ['2024-12', '2025-01']


def test_table_page_and_csv_scoped_to_filters():
    start = dd.period_start(30)
    page = dd.load_table_page('daily', None, 'asc', 0, assets=['AST-2031'], start=start)
    assert page['total_rows'] == 30
    csv_bytes, _ = dd.export_table_csv('daily', assets=['AST-2031'], start=start)
    assert csv_bytes.decode('utf-8-sig').strip().count('\n') == 30  # 헤더 + 30행



# ------------------------------------------------------------
# 화면 ③ 모델 비교 — 화면 수치는 src/model_comparison.py 산출 CSV와 같아야 한다
# ------------------------------------------------------------
import pandas as pd  # noqa: E402

COMPARISON = root / 'outputs/machine_risk/comparison.csv'
needs_comparison = pytest.mark.skipif(not COMPARISON.is_file(), reason='모델 비교 결과가 없습니다.')


@needs_comparison
def test_model_comparison_rows_follow_baseline_then_model_order():
    rows = dd.load_model_comparison('machine_risk', 12)
    assert [r['model'] for r in rows] == ['baseline_a', 'baseline_b', 'logistic_regression',
                                          'random_forest', 'hist_gradient_boosting']
    assert [r['model'] for r in dd.load_model_comparison('part_current')] == [
        'baseline_a', 'baseline_b', 'hist_gradient_boosting']


@needs_comparison
def test_machine_risk_lr_reproduces_existing_logistic_result():
    existing = pd.read_csv(root / 'outputs/logistic_7_10_11_12_13_14_results.csv')
    for threshold in (12, 13, 14):
        lr = next(r for r in dd.load_model_comparison('machine_risk', threshold) if r['model'] == 'logistic_regression')
        old = existing[existing.severity_threshold.eq(threshold) & existing.model.eq('기본')].iloc[0]
        # 같은 코드·데이터로 다시 학습한 값 — 수치 연산 차이만 허용한다(소수 넷째 자리).
        assert lr['average_precision'] == pytest.approx(old.AP, abs=1e-4)
        assert lr['roc_auc'] == pytest.approx(old.ROC_AUC, abs=1e-4)


@needs_comparison
def test_focus_model_is_production_rf_or_validation_choice():
    assert dd.load_focus_model('machine_risk', 12) == 'random_forest'
    assert dd.load_focus_model('part_within_7d') == 'random_forest'
    assert dd.load_focus_model('part_current') == 'hist_gradient_boosting'


@needs_comparison
def test_metrics_at_stored_cutoff_match_comparison_csv():
    for task, threshold in [('machine_risk', 12), ('machine_risk', 14), ('part_within_7d', None), ('part_current', None)]:
        for row in dd.load_model_comparison(task, threshold):
            at = dd.load_comparison_at_cutoff(task, row['model'], row['cutoff'], threshold)
            assert (at['tp'], at['fp'], at['fn'], at['tn']) == (row['tp'], row['fp'], row['fn'], row['tn'])
            assert at['alert_rate'] == pytest.approx(row['alert_rate'])


@needs_comparison
def test_baseline_b_tables_count_train_rows_only():
    machine = pd.read_csv(root / 'outputs/machine_risk/baselines.csv')
    assert machine.groupby('severity_threshold').train_rows.sum().unique().tolist() == [7660]  # Train 7,660행
    part_current = pd.read_csv(root / 'outputs/current_state/baselines.csv')
    assert part_current.train_rows.sum() == 145600  # 2023-12-31 이전 부품·일 행


@needs_comparison
def test_screen3_kpis_equal_comparison_csv_values():
    from src import wireframe_app as w

    def texts(node):
        children = getattr(node, 'children', None)
        if isinstance(children, (str, int, float)):
            yield str(children)
        for child in children if isinstance(children, (list, tuple)) else ([children] if children is not None else []):
            if not isinstance(child, (str, int, float)):
                yield from texts(child)
            else:
                yield str(child)

    layout = w.screen_3(seg_state={**w.DEFAULT_SEG, 'task': 0, 'threshold': 0})
    shown = list(texts(layout))
    rf = next(r for r in dd.load_model_comparison('machine_risk', 12) if r['model'] == 'random_forest')
    for value in (f"{rf['average_precision']:.3f}", f"{rf['ap_lift']:.2f}",
                  f"{rf['alert_rate'] * 100:.1f}", f"{rf['recall']:.3f}"):
        assert value in shown


@needs_comparison
def test_comparison_feature_importance_is_sorted_and_present():
    rows = dd.load_comparison_feature_importance('machine_risk', 12)
    assert len(rows) == 10
    values = [r['importance_mean'] for r in rows]
    assert values == sorted(values, reverse=True)
