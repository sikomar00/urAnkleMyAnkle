"""필터바(공장·기계 종류·기계·기간)가 화면 ①·②·④에 실제로 적용되는지 검증."""
import os
from pathlib import Path

import pytest

root = Path(__file__).resolve().parents[1]
name = 'synthetic_industrial_machine_data.csv'
paths = ([Path(os.environ['MACHINE_DATA_PATH'])] if os.environ.get('MACHINE_DATA_PATH')
         else [root / 'data/raw' / name, Path.home() / 'Downloads' / name])
if not any(path.is_file() for path in paths):
    pytest.skip('필터 검증용 원본 CSV가 없습니다.', allow_module_level=True)

from dash import dcc, html  # noqa: E402

from src import wireframe_app as w  # noqa: E402
from src.dashboard_data import load_asset_list, load_screen1_kpis  # noqa: E402
from src.ui.base import _period_dates, _period_index  # noqa: E402
from src.ui.shell import filter_bar  # noqa: E402

ALL_PERIOD = 3
LAST_30_DAYS = 0


def _filters(**overrides):
    return {**w.DEFAULT_FILTERS, **overrides}


def _walk(node):
    yield node
    children = getattr(node, 'children', None)
    if isinstance(children, (list, tuple)):
        for child in children:
            yield from _walk(child)
    elif children is not None:
        yield from _walk(children)



def _prop(node, name):
    """aria-* 는 파이썬 식별자가 아니라 to_plotly_json()으로 읽는다."""
    return node.to_plotly_json()['props'].get(name)


def _text(node):
    return ''.join(str(n) for n in _walk(node) if isinstance(n, (str, int, float)))


def test_plant_filter_limits_priority_table_rows():
    assets, start, end = w._filter_scope(_filters(plant='CHN-02', period_index=ALL_PERIOD))
    layout = w.screen_1(assets=assets, start=start, end=end)
    priority_table = next(n for n in _walk(layout) if isinstance(n, html.Table))
    body = next(n for n in _walk(priority_table) if isinstance(n, html.Tbody))
    tags = [_text(row.children[1]) for row in body.children]
    assert tags == ['AST-2031', 'AST-3019', 'AST-3008']


def test_last_30_days_limits_sensor_x_axis():
    filters = _filters(machine='AST-2031', period_index=LAST_30_DAYS)
    _, start, end = w._filter_scope(filters)
    layout = w.screen_2(asset_tag=w._focus_asset(filters), start=start, end=end)
    graph = next(n for n in _walk(layout) if isinstance(n, dcc.Graph) and n.id['type'] == 'smult-chart')
    xs = list(graph.figure.data[0].x)
    assert str(xs[0])[:10] == '2024-12-03'
    assert str(xs[-1])[:10] == '2025-01-01'


def test_focus_asset_is_priority_top_when_machine_is_all():
    assert w._focus_asset(_filters()) == 'AST-2031'
    assert w._focus_asset(_filters(plant='DHR-03')) == 'AST-4055'
    assert w._focus_asset(_filters(machine='AST-1041')) == 'AST-1041'
    # ③은 공장·기계 종류 필터를 적용하지 않는다.
    assert w._focus_asset(_filters(plant='DHR-03'), machine_only=True) == 'AST-2031'


def test_filter_options_exclude_empty_combinations():
    plants, types, machines = w.sync_filter_options(_filters(plant='CHN-02'))
    assert [o['value'] for o in types] == ['Belt Conveyor', 'Hydraulic Press']
    assert [o['value'] for o in machines] == ['AST-2031', 'AST-3008', 'AST-3019']
    assert len(plants) == 3


def test_removed_datasets_fall_back_to_raw():
    layout = w.screen_4(seg_state={**w.DEFAULT_SEG, 'dataset': 3})
    pressed = [n for n in _walk(layout) if isinstance(n, html.Button) and _prop(n, 'aria-pressed') == 'true']
    assert [_text(n) for n in pressed] == ['원자료']


def test_custom_date_range_bounds_both_ends():
    filters = _filters(period_index=None, start='2024-01-01', end='2024-03-31')
    assets, start, end = w._filter_scope(filters)
    assert assets == load_asset_list()
    assert (f'{start:%Y-%m-%d}', f'{end:%Y-%m-%d}') == ('2024-01-01', '2024-03-31')
    # 종료일이 실제로 잘려야 한다 — 같은 시작일에 종료일만 연 경우보다 적다.
    bounded = load_screen1_kpis(assets, start, end)
    open_ended = load_screen1_kpis(assets, start, None)
    assert bounded['high_risk_days'] == 134
    assert bounded['high_risk_days'] < open_ended['high_risk_days']
    # 기준일 스냅샷은 선택 기간의 마지막 관측일 기준이다.
    assert bounded['latest_high_risk_machines'] == 0


def test_period_button_and_date_boxes_agree():
    # 버튼이 눌린 상태면 날짜 칸은 그 버튼이 뜻하는 구간을 보여 준다.
    assert _period_dates(_filters(period_index=LAST_30_DAYS)) == ('2024-12-03', '2025-01-01')
    assert _period_dates(_filters(period_index=ALL_PERIOD)) == ('2022-01-03', '2025-01-01')
    assert _period_index(_filters(period_index=ALL_PERIOD)) == ALL_PERIOD
    # 날짜를 직접 지정하면 눌린 버튼이 없다.
    custom = _filters(period_index=None, start='2024-01-01', end='2024-03-31')
    assert _period_index(custom) is None
    assert _period_dates(custom) == ('2024-01-01', '2024-03-31')
    # 빈 칸은 그쪽 끝을 연다.
    _, start, end = w._filter_scope(_filters(period_index=None, start=None, end='2024-03-31'))
    assert start is None and f'{end:%Y-%m-%d}' == '2024-03-31'


def test_date_boxes_render_as_native_date_inputs():
    """dcc.Input의 type 목록에 'date'가 없다 — Dash를 올릴 때 깨지면 여기서 잡는다."""
    boxes = [n for n in _walk(filter_bar())
             if isinstance(n, dcc.Input) and n.id in ('start-date', 'end-date')]
    assert [n.id for n in boxes] == ['start-date', 'end-date']
    assert {n.type for n in boxes} == {'date'}
    assert {(n.min, n.max) for n in boxes} == {('2022-01-03', '2025-01-01')}



def test_priority_table_sorts_by_every_column():
    """열 제목이 보내는 열 id 전부가 정렬 키로 동작한다. 기본 순서·모르는 열은 우선순위 순."""
    from src.dashboard_data import PRIORITY_SORT_KEYS, load_priority_table
    from src.ui.screen1 import PRIO_COLS
    default = [r['asset_tag'] for r in load_priority_table()]
    for _label, _align, field in PRIO_COLS:
        assert field in PRIORITY_SORT_KEYS, field
        for direction in ('asc', 'desc'):
            values = [r[field] for r in load_priority_table(field, direction)]
            assert values == sorted(values, reverse=direction == 'desc'), (field, direction)
    assert [r['asset_tag'] for r in load_priority_table('없는열', 'asc')] == default
    assert [r['asset_tag'] for r in load_priority_table('plant_code', None)] == default


def test_sort_cycle_is_asc_then_desc_then_default():
    """같은 열을 누를 때마다 오름차순 → 내림차순 → 기본, 다른 열을 누르면 오름차순부터."""
    from src.ui.base import next_sort, table_sort
    store = {}
    seen = []
    for _ in range(4):
        store = next_sort(store, 'priority', 'plant_code')
        seen.append(table_sort(store, 'priority'))
    assert seen == [('plant_code', 'asc'), ('plant_code', 'desc'), (None, None), ('plant_code', 'asc')]
    store = next_sort(store, 'priority', 'asset_tag')
    assert table_sort(store, 'priority') == ('asset_tag', 'asc')
    # 다른 표의 정렬은 건드리지 않는다.
    store = next_sort(store, 'power', 'avg_power_kw')
    assert table_sort(store, 'priority') == ('asset_tag', 'asc')


def test_screen1_kpi_cards_explain_scope_with_help_icon_not_label():
    """"선택 기간" 라벨을 빼고 물음표 설명으로 내렸다 — 범위 설명이 사라지면 안 된다."""
    from src.ui.screen1 import KPI_HELP
    layout = w.screen_1()
    assert len(KPI_HELP) == 4 and '선택 기간' in ' '.join(KPI_HELP.values())
    icons = [n for n in _walk(layout)
             if getattr(n, 'className', None) and 'pf-help' in str(n.className)]
    assert len(icons) == 6          # KPI 4개 + 점검 우선순위·히트맵 설명 각 1개
    assert all(i.title and i.title == _prop(i, 'aria-label') for i in icons)
    # 라벨 줄에는 제목과 물음표만 남는다 — 예전 "선택 기간"·"기준일" note가 없어야 한다.
    cards = [n for n in _walk(layout) if getattr(n, 'className', '') == 'pf-card pf-kpi']
    assert len(cards) == 4
    for card in cards:
        kinds = [str(n.className).split()[0] for n in card.children[0].children]
        assert kinds == ['pf-kpi__label', 'pf-help'], kinds


def test_screen1_column_titles_are_sort_buttons_marking_only_the_active_column():
    store = {'priority': {'col': 'plant_code', 'direction': 'asc'},
             'power': {'col': 'avg_power_kw', 'direction': 'desc'}}
    layout = w.screen_1(table_sort=store)
    heads = [n for n in _walk(layout) if 'pf-th--sortable' in str(getattr(n, 'className', ''))]
    assert len(heads) == 8 + 2                    # 우선순위 8열 + 전력 2열
    sorted_heads = {h.children.id['col']: _prop(h, 'aria-sort') for h in heads if _prop(h, 'aria-sort') != 'none'}
    assert sorted_heads == {'plant_code': 'ascending', 'avg_power_kw': 'descending'}
    arrows = [h.children.children[1].children for h in heads]
    assert sorted(a for a in arrows if a) == ['▲', '▼']
    # 기준선을 넘은 기계의 행만 붉게 강조한다.
    rows = [n for n in _walk(layout) if isinstance(n, html.Tr) and getattr(n, 'className', None) == 'pf-tr--flagged']
    assert [_text(r.children[1]) for r in rows] == ['AST-2031']


def test_heatmap_units_keep_the_same_total_and_mark_partial_periods():
    """일·주·월·년 어느 단위로 봐도 센 건수 합계는 같고, 데이터가 일부만 있는 칸은 '부분'이다."""
    from src.dashboard_data import load_asset_failure_heatmap, load_period_coverage
    from src.ui.base import HEATMAP_UNITS
    totals, columns = set(), {}
    for unit in HEATMAP_UNITS:
        heat = load_asset_failure_heatmap(unit=unit)
        coverage = load_period_coverage(unit=unit)
        periods = sorted(heat['period'].unique())
        totals.add(int(heat['failed_part_count'].sum()))
        columns[unit] = len(periods)
        assert set(periods) == set(coverage), unit
        partial = [p for p in periods if coverage[p][0] < coverage[p][1]]
        # 원본은 2022-01-03에 시작해 2025-01-01에 끝난다 — 하루 단위에는 '부분'이 없다.
        assert partial == ([] if unit == 'day' else partial) and len(partial) <= 2, unit
    assert totals == {21636}
    assert columns == {'day': 1095, 'week': 157, 'month': 37, 'year': 4}


def test_heatmap_card_has_unit_buttons_and_reset():
    layout = w.screen_1()
    ids = [n.id for n in _walk(layout) if isinstance(getattr(n, 'id', None), dict)]
    assert [i['index'] for i in ids if i.get('group') == 'heat_unit'] == [0, 1, 2, 3]
    assert {'type': 'heat-reset-btn', 'index': 'screen1'} in ids


def test_sensor_outlier_bounds_follow_robust_normal_baseline():
    """정상 범위 = 학습 구간 정상일의 중앙값 ± 3 × 1.4826 × MAD(설비 이상 실험과 같은 기준)."""
    import pytest

    from src.dashboard_data import OUTLIER_Z, SENSOR_COLUMNS, load_sensor_outlier_bounds
    bounds = load_sensor_outlier_bounds('AST-2031')
    assert list(bounds) == list(SENSOR_COLUMNS)
    bearing = bounds['temp_bearing_degC']
    assert (bearing['median'], round(bearing['scale'], 5)) == (68.2, 2.37216)
    for b in bounds.values():
        assert b['upper'] - b['lower'] == pytest.approx(2 * OUTLIER_Z * b['scale'])
    with pytest.raises(ValueError):
        load_sensor_outlier_bounds('AST-9999')


def test_sensor_card_marks_outside_days_and_drops_old_note():
    filters = _filters(machine='AST-2031', period_index=LAST_30_DAYS)
    _, start, end = w._filter_scope(filters)
    layout = w.screen_2(asset_tag='AST-2031', start=start, end=end)
    text = _text(layout)
    assert '센서 8종 종합 지표' in text and '스몰 멀티플' not in text and 'x축 공유' not in text
    graph = next(n for n in _walk(layout) if isinstance(n, dcc.Graph) and n.id['type'] == 'smult-chart')
    dashed = [t for t in graph.figure.data if t.mode == 'lines' and t.line.dash == 'dash']
    markers = [t for t in graph.figure.data if t.mode == 'markers']
    assert len(dashed) == 2 * 8
    # 라벨 열의 "경계 밖 N일" 합계 = 차트에 찍힌 빨간 점 수(최근 30일 AST-2031: 16개).
    counts = [int(t.split('경계 밖 ')[1].rstrip('일')) for t in
              (_text(n) for n in _walk(layout) if isinstance(n, html.Span)) if t.startswith('경계 밖 ') and t != '경계 밖 없음']
    assert sum(len(t.x) for t in markers) == sum(counts) == 16


def test_report_export_works_with_custom_date_range():
    """날짜를 직접 지정하면 눌린 기간 버튼이 없다(period_index=None) — 보고서의 '기간' 칸이 날짜로 적힌다."""
    from src.ui.report import _report_context, build_report_pdf, build_report_xlsx
    custom = _filters(period_index=None, start='2024-01-01', end='2024-03-31')
    meta = dict(_report_context(custom, w.DEFAULT_SEG, 'mgr')['meta'])
    assert meta['기간'] == '직접 지정 (2024-01-01 ~ 2024-03-31)'
    preset = dict(_report_context(_filters(period_index=LAST_30_DAYS), w.DEFAULT_SEG, 'mgr')['meta'])
    assert preset['기간'] == f"최근 30일 ({' ~ '.join(_period_dates(_filters(period_index=LAST_30_DAYS)))})"
    assert build_report_xlsx(custom, w.DEFAULT_SEG, 'mgr')[:2] == b'PK'
    assert build_report_pdf(custom, w.DEFAULT_SEG, 'mgr')[:4] == b'%PDF'


def test_rate_change_shows_trend_glyph_with_words_not_color_only():
    from src.ui.screen1 import rate_change
    worse = rate_change(13.33, 9.74)
    assert _text(worse[0]) == '▲ 3.6%p 증가' and 'pf-delta--worse' in worse[0].className
    assert worse[1] == ' · 직전 같은 기간 9.7'
    better = rate_change(9.7, 13.3)
    assert _text(better[0]) == '▼ 3.6%p 감소' and 'pf-delta--better' in better[0].className
    assert rate_change(9.71, 9.68) == '직전 같은 기간과 같음 (9.7)'
