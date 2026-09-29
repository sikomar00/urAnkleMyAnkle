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


def test_current_classification_metrics_has_prior_and_hist_gradient_boosting():
    metrics = dd.load_current_classification_metrics()
    assert set(metrics) == {'prior', 'hist_gradient_boosting'}
    for model_metrics in metrics.values():
        assert set(model_metrics) == set(dd.CLASSIFICATION_METRIC_COLUMNS)


def test_current_classification_metrics_hist_gradient_boosting_values():
    hgb = dd.load_current_classification_metrics()['hist_gradient_boosting']
    assert hgb['roc_auc'] == pytest.approx(0.6815, abs=1e-4)
    assert hgb['average_precision'] == pytest.approx(0.1534, abs=1e-4)


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


def test_current_classification_pr_curve_and_confusion_matches_metrics():
    result = dd.load_current_classification_pr_curve_and_confusion('hist_gradient_boosting')
    c = result['confusion']
    assert c == {'tn': 15079, 'fp': 18233, 'fn': 641, 'tp': 3047}
    precision = c['tp'] / (c['tp'] + c['fp'])
    recall = c['tp'] / (c['tp'] + c['fn'])
    assert precision == pytest.approx(0.14319, abs=1e-4)
    assert recall == pytest.approx(0.82619, abs=1e-4)
    assert len(result['precision_curve']) == len(result['recall_curve'])


def test_current_classification_pr_curve_and_confusion_rejects_unknown_model():
    with pytest.raises(ValueError):
        dd.load_current_classification_pr_curve_and_confusion('not_a_model')


def test_part_failure_metrics_has_both_models():
    metrics = dd.load_part_failure_metrics()
    assert set(metrics) == {'로지스틱 회귀', '랜덤 포레스트'}
    for m in metrics.values():
        assert set(m) == {'average_precision', 'precision', 'recall', 'f1'}
    lr = metrics['로지스틱 회귀']
    assert lr['average_precision'] == pytest.approx(0.61258, abs=1e-4)
    assert lr['precision'] == pytest.approx(0.59586, abs=1e-4)
    assert lr['recall'] == pytest.approx(0.60201, abs=1e-4)
    assert lr['f1'] == pytest.approx(0.59892, abs=1e-4)


def test_part_failure_selected_model_is_highest_average_precision():
    assert dd.load_part_failure_selected_model() == '로지스틱 회귀'


def test_part_failure_pr_curve_and_confusion_defaults_to_selected_model():
    result = dd.load_part_failure_pr_curve_and_confusion()
    assert result['model'] == '로지스틱 회귀'
    assert result['confusion'] == {'tn': 8185, 'fp': 5894, 'fn': 5745, 'tp': 8690}
    assert result['cutoff'] == pytest.approx(0.5)
    assert len(result['precision_curve']) == len(result['recall_curve'])


def test_part_failure_pr_curve_and_confusion_rejects_unknown_model():
    with pytest.raises(ValueError):
        dd.load_part_failure_pr_curve_and_confusion('not_a_model')


def test_current_classification_actual_rate_matches_test_positive_rate():
    rate = dd.load_current_classification_actual_rate('hist_gradient_boosting')
    assert rate == pytest.approx(0.099676, abs=1e-5)


def test_current_classification_actual_rate_rejects_unknown_model():
    with pytest.raises(ValueError):
        dd.load_current_classification_actual_rate('not_a_model')


def test_part_failure_actual_rate_defaults_to_selected_model():
    rate = dd.load_part_failure_actual_rate()
    assert rate == pytest.approx(0.5062425475205162, abs=1e-6)


def test_part_failure_actual_rate_rejects_unknown_model():
    with pytest.raises(ValueError):
        dd.load_part_failure_actual_rate('not_a_model')


def test_current_classification_at_threshold_recomputes_confusion():
    result = dd.load_current_classification_at_threshold('hist_gradient_boosting', 0.2)
    assert result['predicted_alerts'] == 111 + 549
    assert result['false_alarms'] == 549
    assert result['missed_failures'] == 3577
    assert result['total_rows'] == 37000
    assert result['precision'] == pytest.approx(111 / (111 + 549))
    assert result['recall'] == pytest.approx(111 / (111 + 3577))


def test_current_classification_at_threshold_rejects_unknown_model():
    with pytest.raises(ValueError):
        dd.load_current_classification_at_threshold('not_a_model', 0.5)


def test_part_failure_at_threshold_matches_default_cutoff_confusion():
    result = dd.load_part_failure_at_threshold('로지스틱 회귀', 0.5)
    assert result['predicted_alerts'] == 8690 + 5894
    assert result['false_alarms'] == 5894
    assert result['missed_failures'] == 5745
    assert result['total_rows'] == 28514


def test_part_failure_at_threshold_defaults_to_selected_model():
    result = dd.load_part_failure_at_threshold(None, 0.3)
    assert result['predicted_alerts'] == 14410 + 14037
    assert result['missed_failures'] == 25


def test_part_failure_at_threshold_rejects_unknown_model():
    with pytest.raises(ValueError):
        dd.load_part_failure_at_threshold('not_a_model', 0.5)
