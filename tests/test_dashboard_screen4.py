"""화면 ④ 데이터 — 데이터셋·CSV를 카드 머리로 옮겨 표 아래 행 수 안내가 잘리지 않고, 표가 열 제목으로 정렬된다."""
import os
from pathlib import Path

import pytest

root = Path(__file__).resolve().parents[1]
name = 'synthetic_industrial_machine_data.csv'
paths = ([Path(os.environ['MACHINE_DATA_PATH'])] if os.environ.get('MACHINE_DATA_PATH')
         else [root / 'data/raw' / name, Path.home() / 'Downloads' / name])
if not any(path.is_file() for path in paths):
    pytest.skip('화면 ④ 검증용 원본 CSV가 없습니다.', allow_module_level=True)

from dash import html  # noqa: E402

from src.dashboard_data import load_data_dictionary  # noqa: E402
from src.ui.screen4 import dictionary_block, screen_4  # noqa: E402

ASSETS = root / 'assets'


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


def _classes(node):
    return (getattr(node, 'className', None) or '').split()


def test_dataset_switch_and_csv_live_in_the_data_card_header():
    layout = screen_4()
    rows = layout.children
    # 따로 있던 도구줄(32px)이 없어지고 데이터 조회 카드가 그만큼(32 + 간격 16) 커진다 — 572 + 16 + 340 = 928.
    assert [r.style['height'] for r in rows] == ['572px', '340px']
    data_card = next(n for n in _walk(rows[0]) if 'pf-card' in _classes(n))
    actions = next(n for n in _walk(data_card) if 'pf-card__actions' in _classes(n))
    assert '데이터셋' in _text(actions) and 'CSV 내보내기' in _text(actions)
    footer = next(n for n in _walk(data_card) if isinstance(getattr(n, 'id', None), dict)
                  and n.id.get('type') == 'dtable-footer')
    assert _text(footer).endswith('행 표시')


def test_dictionary_table_sorts_by_column_title():
    records = load_data_dictionary()

    def first_column(store):
        body = next(n for n in _walk(dictionary_block(store)) if isinstance(n, html.Tbody))
        return [_text(tr.children[0]) for tr in body.children]

    assert first_column({}) == [r['column'] for r in records]
    assert first_column({"dictionary": {"col": "column", "direction": "asc"}}) == sorted(r['column'] for r in records)
    # 결측률이 없는 열(wo_type은 '작업 없음' 범주라 결측 아님)은 방향과 무관하게 맨 뒤.
    assert first_column({"dictionary": {"col": "missing_pct", "direction": "desc"}})[-1] == 'wo_type'


def test_datatable_header_sort_uses_whole_cell_and_glyphs():
    css = (ASSETS / '03-app.css').read_text(encoding='utf-8')
    js = (ASSETS / '04-a11y.js').read_text(encoding='utf-8')
    assert '.pf-dtable .column-header--sort::after { content: ""; position: absolute; inset: 0; }' in css
    assert 'svg[data-icon="sort-up"])::before { content: "▲"; }' in css
    assert 'svg[data-icon="sort-down"])::before { content: "▼"; }' in css
    assert '.column-header--sort' in js and 'aria-sort' in js
