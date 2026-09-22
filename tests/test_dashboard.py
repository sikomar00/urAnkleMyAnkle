"""로컬 원본 CSV가 있을 때 실행하는 대시보드 집계·연결 검증."""
import io
import os
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip('dash')
root = Path(__file__).resolve().parents[1]
name = 'synthetic_industrial_machine_data.csv'
paths = [Path(os.environ['MACHINE_DATA_PATH'])] if os.environ.get('MACHINE_DATA_PATH') else [root / 'data/raw' / name, Path.home() / 'Downloads' / name]
if not any(path.is_file() for path in paths):
    pytest.skip('대시보드 검증용 원본 CSV가 없습니다.', allow_module_level=True)

from src import dashboard as d


def test_sensor_aggregation_does_not_count_each_part():
    raw, daily = d.filtered_data('all', 'all', '2024-12-03', '2025-01-01')
    assert len(raw) == len(daily) * 20
    _, _, power, rows = d.overview_data(raw, daily)
    expected = raw.groupby(['asset_tag', 'transaction_date']).power_consumption_kw.first().groupby('transaction_date').mean()
    assert list(power.data[0].y) == pytest.approx(expected.tolist())
    for row in rows:
        actual = raw[raw.asset_tag.eq(row['asset'])].groupby('transaction_date').breakdown_flag.max().sum()
        assert row['days'] == actual


def test_reference_excludes_current_and_future_observations():
    date = pd.Timestamp('2024-08-01')
    history, prior, value, median, delta, percentile = d.sensor_context('AST-1041', date, 'vibration_h_mms', '30')
    assert history.transaction_date.max() == date
    assert prior.transaction_date.max() < date
    assert len(prior) == 30
    assert median == prior.vibration_h_mms.median()
    assert delta == value - median
    assert percentile == pytest.approx((prior.vibration_h_mms <= value).sum() / 30 * 100)


def test_part_selection_changes_fault_track_but_not_sensors():
    raw, daily = d.filtered_data('all', 'AST-1041', '2024-10-04', '2025-01-01')
    a = d.detail_figures(raw, daily, 'temp_bearing_degC', '90', None)[0]
    b = d.detail_figures(raw, daily, 'temp_bearing_degC', '90', 'MRO-10045')[0]
    assert list(a.data[0].y) == list(b.data[0].y)
    expected = raw[raw.part_no.eq('MRO-10045')].sort_values('transaction_date').breakdown_flag
    assert list(b.data[2].y) == expected.tolist()


def test_export_preserves_filtered_order_and_scope():
    scope = {'start': '2024-10-04', 'end': '2025-01-01', 'plant': 'all', 'machine': 'AST-1041'}
    rows = d.refresh_scope(**scope)[4][2:4][::-1]
    exported = d.build_export(rows, d.PART_COLUMNS, scope, 'parts')
    assert exported['content'].startswith('\ufeff')
    frame = pd.read_csv(io.StringIO(exported['content']))
    assert frame['부품 번호'].tolist() == [r['part'] for r in rows]
    assert set(frame['설비 필터']) == {'AST-1041'}
    assert frame['금액 (INR)'].sum() == sum(r['cost'] for r in rows)
    assert len(pd.read_csv(io.StringIO(d.build_export([], d.PART_COLUMNS, scope, 'parts')['content']))) == 0


def test_first_observation_and_empty_ranges():
    result = d.refresh_scope('all', 'AST-1041', '2022-01-03', '2022-01-03')
    detail = d.refresh_detail(result[6], 'oil_pressure_bar', '90', None)
    assert len(detail) == 13
    assert detail[8] == '비교 기록 없음'
    invalid = d.refresh_scope('all', 'all', '2025-01-01', '2022-01-03')
    assert len(invalid) == 9
    assert invalid[6]['empty'] is True
    assert invalid[3:6] == ([], [], [])


def test_layout_contains_unique_callback_targets():
    client = d.server.test_client()
    response = client.get('/_dash-layout')
    assert response.status_code == 200
    identities = []

    def walk(node):
        if isinstance(node, dict):
            if 'props' in node and 'id' in node['props']:
                identities.append(node['props']['id'])
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(response.get_json())
    assert len(identities) == len(set(identities))
    assert client.get('/assets/dashboard.css').status_code == 200
