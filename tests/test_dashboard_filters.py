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


def _text(node):
    return ''.join(str(n) for n in _walk(node) if isinstance(n, (str, int, float)))


def test_plant_filter_limits_priority_table_rows():
    assets, start = w._filter_scope(_filters(plant='CHN-02', period_index=ALL_PERIOD))
    layout = w.screen_1(assets=assets, start=start)
    priority_table = next(n for n in _walk(layout) if isinstance(n, html.Table))
    body = next(n for n in _walk(priority_table) if isinstance(n, html.Tbody))
    tags = [_text(row.children[1]) for row in body.children]
    assert tags == ['AST-2031', 'AST-3019', 'AST-3008']


def test_last_30_days_limits_sensor_x_axis():
    filters = _filters(machine='AST-2031', period_index=LAST_30_DAYS)
    _, start = w._filter_scope(filters)
    layout = w.screen_2(asset_tag=w._focus_asset(filters), start=start)
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
