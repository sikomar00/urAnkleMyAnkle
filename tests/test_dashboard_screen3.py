"""화면 ③ 모델·예측 — 필터바 숨김, 성능 해석, 정렬되는 비교표, 혼동행렬·영향 변수 표기, 운영 판단."""
import os
from pathlib import Path

import pytest

root = Path(__file__).resolve().parents[1]
name = 'synthetic_industrial_machine_data.csv'
paths = ([Path(os.environ['MACHINE_DATA_PATH'])] if os.environ.get('MACHINE_DATA_PATH')
         else [root / 'data/raw' / name, Path.home() / 'Downloads' / name])
if not any(path.is_file() for path in paths):
    pytest.skip('화면 ③ 검증용 원본 CSV가 없습니다.', allow_module_level=True)

from dash import html, no_update  # noqa: E402

from src import wireframe_app as w  # noqa: E402
from src.dashboard_data import (  # noqa: E402
    load_asset_family_diagnosis, load_asset_list, load_comparison_feature_importance,
    load_family_feature_importance, load_family_pr_curve_and_confusion, load_model_comparison,
)
from src.ui.base import CANVAS_H, FILTERBAR_H, FOOTER_H, HEADER_H  # noqa: E402
from src.ui.screen3 import (  # noqa: E402
    feature_label, model_table_block, operational_verdict, screen_3, THRESHOLD_HELP,
)

FAMILY_TASK = {"task": 3, "threshold": 0}


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


def _first_column(layout):
    body = next(n for n in _walk(layout) if isinstance(n, html.Tbody))
    return [_text(tr.children[0]) for tr in body.children]


def test_filter_bar_height_goes_to_screen3_body_and_canvas_stays_1080():
    with_bar, without_bar = w.main_style(True), w.main_style(False)
    assert HEADER_H + FILTERBAR_H + int(with_bar['height'][:-2]) + FOOTER_H == CANVAS_H
    assert HEADER_H + int(without_bar['height'][:-2]) + FOOTER_H == CANVAS_H
    # 화면 ③ 본문 예산 = 984 = 도구줄 60 + KPI 96 + 해석 56 + 비교표 276 + 432 + 간격 16×4
    assert int(without_bar['height'][:-2]) - 20 == 60 + 96 + 56 + 276 + 432 + 16 * 4


def test_threshold_warning_line_is_gone_and_help_icon_explains_it():
    for seg_state in ({"task": 0, "threshold": 0}, {"task": 0, "threshold": 2}, {"task": 1, "threshold": 0}):
        layout = screen_3(seg_state)
        text = _text(layout)
        assert '학습 라벨 정의다' not in text and '아래 전체가 바뀐다' not in text
        helps = [n for n in _walk(layout) if 'pf-help' in _classes(n)]
        assert any(n.title == THRESHOLD_HELP for n in helps)


def test_performance_sentence_uses_comparison_numbers():
    layout = screen_3({"task": 0, "threshold": 0})
    insight = next(n for n in _walk(layout) if 'pf-insight' in _classes(n))
    text = _text(insight)
    rows = {r['model']: r for r in load_model_comparison('machine_risk', 12)}
    rf, base_b = rows['random_forest'], rows['baseline_b']
    assert f"AP는 {rf['average_precision']:.3f}" in text
    assert f"{rf['ap_lift']:.2f}배" in text
    assert f"{rf['recall']:.1%}를 잡고" in text
    assert f"기준 B(기계별 과거 비율)의 AP는 {base_b['average_precision']:.3f}" in text


def test_model_table_sorts_by_column_and_bolds_rank_metric_best():
    seg_state = {"task": 0, "threshold": 0}
    rows = load_model_comparison('machine_risk', 12)
    default = model_table_block({}, seg_state)
    assert _first_column(default) == [r['model_label'] for r in rows]
    asc = model_table_block({"model": {"col": "average_precision", "direction": "asc"}}, seg_state)
    assert _first_column(asc) == [r['model_label'] for r in sorted(rows, key=lambda r: r['average_precision'])]
    best = [n for n in _walk(default) if isinstance(n, html.Td) and 'pf-td--best' in _classes(n)]
    best_ap = max(r['average_precision'] for r in rows)
    assert f"{best_ap:.3f}" in [_text(n) for n in best]
    # 정밀도·재현율·정확도는 판정 기준에 따라 기준 A가 1등이 되므로 굵게 하지 않는다.
    assert "1.000" not in [_text(n) for n in best]


def test_family_table_is_sortable_and_keeps_selected_row():
    store = {"family": {"col": "average_precision", "direction": "asc"}}
    layout = screen_3(FAMILY_TASK, asset_tag='AST-2031', family='Coupling', table_sort=store)
    families = _first_column(layout)
    rows = load_asset_family_diagnosis('AST-2031')
    assert families == [r['part_family'] for r in sorted(rows, key=lambda r: r['average_precision'])]
    selected = [n for n in _walk(layout) if isinstance(n, html.Tr)
                and n.to_plotly_json()['props'].get('aria-selected') == 'true']
    assert [_text(n.children[0]) for n in selected] == ['Coupling']


def test_confusion_matrix_names_each_cell_with_status_tint():
    layout = screen_3({"task": 0, "threshold": 0})
    cells = {_text(n.children[0]): n for n in _walk(layout) if 'pf-cm-cell' in _classes(n)}
    assert {k: [c for c in _classes(v) if c.startswith('pf-cm-cell--')] for k, v in cells.items()} == {
        '맞게 잡음': ['pf-cm-cell--good'], '놓침': ['pf-cm-cell--critical'],
        '오경보': ['pf-cm-cell--warning'], '정상 통과': ['pf-cm-cell--good']}


def test_every_shown_feature_has_a_korean_label_and_raw_name_on_hover():
    features = set()
    for threshold in (12, 13, 14):
        features |= {r['feature'] for r in load_comparison_feature_importance('machine_risk', threshold)}
    for task in ('part_within_7d', 'part_current'):
        features |= {r['feature'] for r in load_comparison_feature_importance(task, None)}
    for family in {r['part_family'] for r in load_asset_family_diagnosis('AST-2031')}:
        features |= {r['feature'] for r in load_family_feature_importance(family)}
    untranslated = sorted(f for f in features if feature_label(f) == f or '_' in feature_label(f))
    assert untranslated == []
    assert feature_label('oil_pressure_bar_residual_robust_z_lag7') == '오일 압력 · 기대값 대비 잔차 · 강건 z · 7일 전 값'
    layout = screen_3({"task": 0, "threshold": 0})
    labelled = [n for n in _walk(layout) if isinstance(n, html.Span) and getattr(n, 'title', None) == 'power_consumption_kw']
    assert [_text(n) for n in labelled] == ['소비 전력']


@pytest.mark.parametrize("lift, precision, recall, verdict", [
    (1.19, 0.9, 0.9, "사용 불가"),
    (1.2, 0.8, 0.8, "운영 검토 가능"),
    (2.0, 0.8, 0.79, "점검 보조 사용 가능"),
    (2.0, 0.5, 0.3, "점검 보조 사용 가능"),
    (2.0, 0.49, 0.95, "탐색적 사용 가능"),
])
def test_operational_verdict_rules(lift, precision, recall, verdict):
    assert operational_verdict(lift, precision, recall)[0] == verdict


def test_user_example_verdict_sentence():
    verdict, _, sentence, reason = operational_verdict(1.46, 0.477, 0.984)
    assert verdict == "탐색적 사용 가능"
    assert sentence == ("고장 누락은 적지만 오경보가 많아 자동 부품 교체 판단에는 사용할 수 없습니다. "
                        "점검 후보를 좁히는 보조 수단으로만 사용합니다.")
    assert reason == "AP ÷ 실제 고장률 1.46배(1.2배 이상)이지만 경고 100건당 실제 고장 47.7건 — 점검 보조 기준 50건 미만"


def test_family_screen_replaces_recurrence_with_operation_card():
    layout = screen_3(FAMILY_TASK, asset_tag='AST-2031', family='Bearing')
    text = _text(layout)
    assert '재발 간격' not in text and '운영 판단' in text
    confusion = load_family_pr_curve_and_confusion('AST-2031', 'Bearing')['confusion']
    total = sum(confusion.values())
    positives = confusion['tp'] + confusion['fn']
    for label, value in [("실제 고장률 (%)", f"{positives / total * 100:.1f}"),
                         ("예측 경고율 (%)", f"{(confusion['tp'] + confusion['fp']) / total * 100:.1f}"),
                         ("실제 고장 100건당 탐지 건수", f"{confusion['tp'] / positives * 100:.1f}")]:
        tile = next(n for n in _walk(layout) if 'pf-metric' in _classes(n) and label in _text(n))
        assert _text(tile).endswith(value)
    assert '탐색적 사용 가능' in text


def test_family_machine_picker_moves_the_global_machine_filter():
    filters = {**w.DEFAULT_FILTERS}
    shown = w._focus_asset(filters, machine_only=True)
    # 화면이 다시 그려지며 드롭다운이 마운트될 때(지금 보이는 기계 그대로)는 아무것도 바꾸지 않는다.
    assert w.pick_family_asset([shown], filters) == (no_update, no_update, no_update)
    other = next(a for a in load_asset_list() if a != shown)
    assert w.pick_family_asset([other], filters) == (other, no_update, no_update)
    # 공장 필터 밖 기계를 고르면 write_filters가 기계 선택을 풀지 않도록 공장·기계 종류도 푼다.
    assert w.pick_family_asset(['AST-1041'], {**filters, 'plant': 'CHN-02'}) == ('AST-1041', None, None)
    layout = screen_3(FAMILY_TASK, asset_tag='AST-1041')
    picker = next(n for n in _walk(layout) if getattr(n, 'id', None) == {"type": "family-asset-dd", "index": "screen3"})
    assert picker.value == 'AST-1041' and len(picker.options) == len(load_asset_list())


def test_sort_callback_knows_every_sortable_table():
    assert set(w.SORT_TABLE_BUILDERS) == {"priority", "power", "model", "family"}
