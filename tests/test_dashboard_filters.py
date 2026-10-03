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
    assert '데이터 조회 · 원자료' in _text(layout)


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



def test_priority_table_sorts_by_every_column_and_reverses():
    """열 머리 ▲▼가 보내는 정렬 축 전부가 동작하고, 오름차순은 내림차순의 역순이다."""
    from src.dashboard_data import PRIORITY_SORT_KEYS, load_priority_table
    from src.ui.screen1 import PRIO_COLS, PRIO_SORT_AXIS
    for _label, _align, field in PRIO_COLS:
        axis = PRIO_SORT_AXIS.get(field, field)
        assert axis in PRIORITY_SORT_KEYS, field
        desc = load_priority_table(axis, 'desc')
        asc = load_priority_table(axis, 'asc')
        assert [r['asset_tag'] for r in asc] == [r['asset_tag'] for r in desc][::-1], field
        assert [r['rank'] for r in asc] == list(range(1, 11)), field
    # 모르는 축은 기본(등급가중 고장점수)으로 떨어진다.
    assert load_priority_table('없는열') == load_priority_table('grade')


def test_screen1_kpi_cards_explain_scope_with_help_icon_not_label():
    """"선택 기간" 라벨을 빼고 물음표 설명으로 내렸다 — 범위 설명이 사라지면 안 된다."""
    from src.ui.screen1 import KPI_HELP
    layout = w.screen_1()
    icons = [n for n in _walk(layout)
             if getattr(n, 'className', None) and 'pf-help' in str(n.className)]
    assert len(icons) == len(KPI_HELP) == 4
    assert all(i.title and i.title == _prop(i, 'aria-label') for i in icons)
    assert '선택 기간' in ' '.join(KPI_HELP.values())
    # 라벨 줄에는 제목과 물음표만 남는다 — 예전 "선택 기간"·"기준일" note가 없어야 한다.
    cards = [n for n in _walk(layout) if getattr(n, 'className', '') == 'pf-card pf-kpi']
    assert len(cards) == 4
    for card in cards:
        kinds = [str(n.className).split()[0] for n in card.children[0].children]
        assert kinds == ['pf-kpi__label', 'pf-help'], kinds


def test_screen1_sort_buttons_mark_only_the_active_one():
    layout = w.screen_1(prio_sort={'sort_by': 'plant_code', 'direction': 'asc'}, power_sort='asc')
    buttons = [n for n in _walk(layout)
               if getattr(n, 'className', None) == 'pf-sort__btn']
    assert len(buttons) == 2 * 8 + 2          # 표 8열 × ▲▼ + 전력 ▲▼
    pressed = [b.id for b in buttons if _prop(b, 'aria-pressed') == 'true']
    assert pressed == [{'type': 'prio-sort-btn', 'index': 'plant_code', 'dir': 'asc'},
                       {'type': 'power-sort-btn', 'index': 'screen1', 'dir': 'asc'}]
