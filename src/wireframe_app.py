"""
산업 설비 모니터링 Dash 대시보드 — 화면 4개(① 현황 · ② 기계 상세 · ③ 모델·예측 · ④ 데이터)

데이터는 합성 데이터 원본(data/raw/synthetic_industrial_machine_data.csv)과 outputs/의
모델 결과 파일을 src/dashboard_data.py로 읽는다. 대시보드는 모델을 다시 학습하지 않는다.

동작:
  · 필터바 — 공장 · 기계 종류 · 기계 · 기간(최근 30일/90일/1년/전체, 데이터 최신일 포함).
    화면별 적용 범위가 다르고, 적용되지 않는 화면에서는 해당 컨트롤을 비활성화한다.
  · ② 기계 상세 — 필터의 기계를 보여 준다. 기계가 전체면 점검 우선순위 1위 기계.
  · ③ 과제 · 위험 기준선 선택, 판정 임계값 슬라이더(저장된 예측 확률 재이진화)
  · ④ 데이터 조회 — 서버 측 정렬·페이지, 필터 적용 CSV 내보내기
  · 테마 전환(라이트 ↔ 다크), 보고서 내보내기(PDF · Excel — 독자별 섹션 골격)
  · 로그인·DB 없이 실행한다(로그인·계정·감사 로그는 2026-10-03 archive/legacy/로 옮김).

실행:
    python -m src.wireframe_app
    → http://127.0.0.1:8052
"""

import os
import sys
from datetime import datetime
from pathlib import Path

from dash import Dash, html, dcc, Input, Output, State, ALL, MATCH, ctx, no_update

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "src"

from .dashboard_data import (  # noqa: E402
    export_table_csv,
    filter_assets,
    load_asset_catalog,
    load_comparison_at_cutoff,
    load_comparison_pr_curve,
    load_focus_model,
    load_asset_failure_heatmap,
    load_asset_family_diagnosis,
    load_asset_parts_history,
    load_asset_sensor_series,
    load_data_reference_date,
    load_data_start_date,
    load_month_coverage,
    load_failure_trend,
    load_family_pr_curve_and_confusion,
    load_family_recurrence_intervals,
    load_screen1_machine_status,
    load_table_page,
    period_start,
)
from .ui.base import (  # noqa: E402
    _filter_scope, _focus_asset, _period_index, CANVAS_H, DEFAULT_AUDIENCE,
    DEFAULT_FILTERS, DEFAULT_PRIO_SORT, DEFAULT_SEG, empty_state, FILTERBAR_H, HEADER_H,
    INDEX_STRING, MARGIN, NO_DATA_MARK, PERIOD_PRESETS, REPORT_AUDIENCES,
)
from .ui.shell import (  # noqa: E402
    app_header, filter_bar, FILTER_ECHO_STYLE,
)
from .ui.figures import (  # noqa: E402
    _family_recur_figure, _heatmap_figure, _parts_figure, _pr_figure, _smult_figure, _spark_figure,
    _trend_figure,
)
from .ui.screen1 import (  # noqa: E402
    screen_1,
)
from .ui.screen2 import (  # noqa: E402
    screen_2,
)
from .ui.screen3 import (  # noqa: E402
    _confusion_body, _screen3_task_key, _screen3_threshold, _threshold_body, SCREEN3_FAMILY_INDEX,
    SCREEN3_TASK_INFO, screen_3,
)
from .ui.screen4 import (  # noqa: E402
    _dtable_footer_text, screen_4,
)
from .ui.report import (  # noqa: E402
    build_report_pdf, build_report_xlsx,
)

# screen_5()는 탭에서는 빠지지만 함수 자체는 지우지 않는다 — 헤더의 내보내기
# 드롭다운(REPORT_AUDIENCES/build_report_pdf/xlsx)이 화면⑤ 없이도 정상
# 동작하는지는 확인했지만, screen_5()가 그 보고서 구성의 참조 구현이라
# 남겨 둔다(요청: "함수 자체는 삭제하지 마라").
SCREEN_BUILDERS = {"1": screen_1, "2": screen_2, "3": screen_3, "4": screen_4}


# ============================================================
# 앱 조립
# ============================================================

app = Dash(__name__, assets_folder=str(Path(__file__).resolve().parents[1] / "assets"))
app.title = "설비 모니터링 대시보드"
app.index_string = INDEX_STRING

app.layout = html.Div(
    [
        # storage_type="local" → 새로고침·재접속 후에도 마지막 선택이 남는다.
        # (관리자가 매번 같은 공장·기간을 다시 고르던 문제)
        dcc.Store(id="filter-store", data=DEFAULT_FILTERS, storage_type="local"),
        dcc.Store(id="seg-store", data=DEFAULT_SEG, storage_type="local"),
        dcc.Store(id="prio-sort-store", data=DEFAULT_PRIO_SORT),
        # ③ "부품군 진단" 표 행 클릭이 바꾸는, 현재 드릴다운 중인 부품군.
        # None이면 screen_3()이 정렬 후 1행(average_precision 최댓값)으로 대체한다.
        dcc.Store(id="selected-family-store", data=None),
        dcc.Store(id="theme-store", data="light", storage_type="local"),
        # 1280px 미만에서만 보인다(03-app.css). 그 폭에서는 가로 스크롤로 본다.
        html.Div("이 대시보드는 폭 1280px 이상의 데스크톱 화면 전용이다",
                 className="pf-desktop-only pf-notice pf-notice--warning label-12", role="note"),
        app_header(),
        filter_bar(),
        html.Main(
            id="screen-content",
            style={"height": f"{CANVAS_H - HEADER_H - FILTERBAR_H}px",
                   "padding": f"{MARGIN}px", "overflow": "hidden"},
        ),
    ],
    # 폭 1280~1920px에서 12열 그리드가 늘고 준다(03-app.css .pf-app). 높이는 레이아웃 토큰 고정.
    id="root", className="pf-app",
    style={"minHeight": f"{CANVAS_H}px"},
)



# ------------------------------------------------------------
# 화면 렌더 — 탭 / 세그먼트 상태 / 보고서 대상이 바뀌면 다시 그린다
# ------------------------------------------------------------
@app.callback(
    Output("screen-content", "children"),
    Input("screen-tabs", "value"),
    Input("seg-store", "data"),
    Input("prio-sort-store", "data"),
    Input("filter-store", "data"),
    Input("selected-family-store", "data"),
)
def render_screen(active, seg_state, prio_sort, filters, selected_family):
    assets, start = _filter_scope(filters)
    if active in ("1", "2", "4") and not assets:
        return empty_state("조건에 맞는 기계 없음",
                           "선택한 공장·기계 종류·기계 조합에 해당하는 기계가 없다", 400)
    kwargs = {"seg_state": seg_state, "audience": DEFAULT_AUDIENCE}
    if active == "1":
        kwargs.update(prio_sort=prio_sort or DEFAULT_PRIO_SORT, assets=assets, start=start)
    if active == "2":
        kwargs.update(asset_tag=_focus_asset(filters), start=start)
    if active == "3":
        kwargs.update(asset_tag=_focus_asset(filters, machine_only=True), family=selected_family)
    if active == "4":
        kwargs.update(assets=assets, start=start)
    try:
        return SCREEN_BUILDERS[active](**kwargs)
    except Exception:
        app.server.logger.exception("화면 %s 렌더 실패", active)
        return html.Div("화면을 불러오지 못했습니다. 서버 로그를 확인해 주세요.")


# ②의 "‹ 이전 기계"/"다음 기계 ›" — 공장·기계 종류 필터 안에서 알파벳순으로
# 순환하고, 고른 기계를 필터의 기계 값으로 쓴다(write_filters가 이어서 저장).
# nav_btn()이 공용 ghost-btn과 다른 id 타입을 쓰므로 echo_action과 겹치지 않는다.
@app.callback(
    Output("machine-dd", "value", allow_duplicate=True),
    Input({"type": "machine-nav-btn", "index": ALL}, "n_clicks"),
    State("filter-store", "data"),
    prevent_initial_call=True,
)
def cycle_selected_asset(_clicks, filters):
    triggered = ctx.triggered_id
    if not triggered:
        return no_update
    # 화면②가 새로 마운트될 때마다 Dash가 이 패턴매칭 Input을 n_clicks=0(또는
    # None)인 채로 한 번 발화시킨다(prevent_initial_call은 앱 최초 실행만 막는다).
    # 그 발화를 클릭으로 세면 사용자의 진짜 첫 클릭이 무시된 것처럼 보이므로,
    # n_clicks가 실제로 올라간 경우에만 이동한다.
    if not ctx.triggered[0]["value"]:
        return no_update
    f = filters or DEFAULT_FILTERS
    assets = filter_assets(f.get("plant"), f.get("machine_type"))
    current_asset = _focus_asset(f)
    idx = assets.index(current_asset) if current_asset in assets else 0
    if triggered["index"] == "next":
        idx = (idx + 1) % len(assets)
    elif triggered["index"] == "prev":
        idx = (idx - 1) % len(assets)
    else:
        return no_update
    return assets[idx]


# ③ "부품군 진단" 표의 행 클릭 — family-row는 task_sel==1일 때만 DOM에
# 있으므로, 다른 과제·다른 화면을 보는 동안은 이 패턴매칭 Input에 매치될
# 대상이 없어 콜백 자체가 호출되지 않는다. cycle_selected_asset과 같은
# 이유로 n_clicks 전부가 falsy인 마운트 시 유령 발화를 무시한다.
@app.callback(
    Output("selected-family-store", "data"),
    Input({"type": "family-row", "index": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def select_family_row(n_clicks_list):
    if not n_clicks_list or not any(n_clicks_list):
        return no_update
    triggered = ctx.triggered_id
    if not triggered:
        return no_update
    return triggered["index"]


# ③ "판정 임계값 조정" 슬라이더 — 과제 0·2에서만 thr-metrics가 DOM에 있으므로
# (과제 1은 "해당 없음" 정적 문구) 이 패턴매칭 Output이 0개일 때 Dash가
# 콜백 자체를 호출하지 않는다. 모델을 다시 학습·평가하지 않고, 이미 저장된
# 예측 확률을 슬라이더 값으로 다시 이진화만 한다.
@app.callback(
    Output({"type": "thr-metrics", "index": ALL}, "children"),
    Output({"type": "cm-body", "index": ALL}, "children"),
    Input({"type": "thr-slider", "index": ALL}, "value"),
    State("seg-store", "data"),
    prevent_initial_call=True,
)
def recompute_threshold_metrics(slider_values, seg_state):
    """슬라이더 값으로 경보율·정밀도·재현율과 혼동행렬을 함께 다시 그린다."""
    if not slider_values or slider_values[0] is None:
        return no_update, no_update
    task_key = _screen3_task_key(seg_state)
    if task_key == "family":
        return no_update, no_update
    threshold = _screen3_threshold(task_key, seg_state)
    focus = load_focus_model(task_key, threshold)
    metrics = load_comparison_at_cutoff(task_key, focus, slider_values[0] / 100, threshold)
    positive_name = SCREEN3_TASK_INFO[task_key][2].format(thr=threshold)
    return [_threshold_body(metrics, positive_name)], [_confusion_body(metrics, positive_name)]


# ------------------------------------------------------------
# 테마 전환 — <html data-theme>만 바꾸면 00-tokens.css의 변수가 전부 따라 바뀐다
# ------------------------------------------------------------
@app.callback(
    Output("theme-store", "data"),
    Input("theme-btn", "n_clicks"),
    State("theme-store", "data"),
    prevent_initial_call=True,
)
def toggle_theme(_n, current):
    return "light" if (current or "light") == "dark" else "dark"


# 화면⑤ "고장·위험 추세" 차트 전용 재색칠 — render_screen과 무관한 별도
# 콜백이라 테마 전환이 화면①~④의 render_screen 재실행(및 그로 인한 화면④
# 표의 page_current/sort_by 리셋)을 유발하지 않는다. 패턴매칭 id라 화면⑤가
# DOM에 없으면(다른 화면을 보는 중) Dash가 이 콜백을 호출하지 않는다 —
# KPI_FAILRATE_ID와 동일 원리. screen-tabs도 Input으로 받아, 이미 다크
# 테마인 상태에서 화면⑤에 처음 탭 이동할 때도(테마 자체는 안 바뀌었으므로
# theme-store만으로는 못 잡는 경우) 곧바로 올바른 색으로 그린다.
@app.callback(
    Output({"type": "trend-chart", "index": ALL}, "figure"),
    Input("theme-store", "data"),
    Input("screen-tabs", "value"),
)
def recolor_trend_chart(theme, _active_tab):
    return [_trend_figure(load_failure_trend(), theme or "light")]


# 화면① "고장 표시 히트맵" 차트도 같은 방식으로 재색칠한다 — trend-chart와
# id 타입을 나눠야 두 콜백의 Output 매칭 개수가 서로 섞이지 않는다.
@app.callback(
    Output({"type": "heatmap-chart", "index": ALL}, "figure"),
    Input("theme-store", "data"),
    Input("screen-tabs", "value"),
    State("filter-store", "data"),
)
def recolor_heatmap_chart(theme, _active_tab, filters):
    assets, start = _filter_scope(filters)
    return [_heatmap_figure(load_asset_failure_heatmap(assets, start), load_month_coverage(start), theme or "light")]


# 화면① "기계 상태" 타일 10개의 스파크라인도 같은 방식으로 재색칠한다.
# load_screen1_machine_status()가 매번 asset_tag 오름차순으로 반환하므로,
# 이 순서가 tiles_grid가 실제로 그린 spark-chart 컴포넌트 순서와 항상 같다.
@app.callback(
    Output({"type": "spark-chart", "index": ALL}, "figure"),
    Input("theme-store", "data"),
    Input("screen-tabs", "value"),
    State("filter-store", "data"),
)
def recolor_spark_charts(theme, _active_tab, filters):
    assets, _ = _filter_scope(filters)
    rows = load_screen1_machine_status(assets)
    return [_spark_figure(r["sparkline"], theme or "light") for r in rows]


# 화면② 스몰 멀티플도 같은 방식으로 재색칠한다. trend-chart와 id 타입을 나눠
# 둬야 두 콜백의 Output 매칭 개수가 서로 섞이지 않는다. 기계·기간 전환은 이미
# render_screen이 화면②를 다시 그리므로 filter-store는 State로만 읽는다.
@app.callback(
    Output({"type": "smult-chart", "index": ALL}, "figure"),
    Input("theme-store", "data"),
    Input("screen-tabs", "value"),
    State("filter-store", "data"),
)
def recolor_smult_chart(theme, _active_tab, filters):
    _, start = _filter_scope(filters)
    return [_smult_figure(load_asset_sensor_series(_focus_asset(filters), start), theme or "light")]


# 화면② "부품 출고 이력" 막대차트도 같은 방식으로 재색칠한다. smult-chart와
# id 타입을 나눠야 두 콜백의 Output 매칭 개수가 서로 섞이지 않는다.
@app.callback(
    Output({"type": "parts-chart", "index": ALL}, "figure"),
    Input("theme-store", "data"),
    Input("screen-tabs", "value"),
    State("filter-store", "data"),
)
def recolor_parts_chart(theme, _active_tab, filters):
    _, start = _filter_scope(filters)
    return [_parts_figure(load_asset_parts_history(_focus_asset(filters), start), theme or "light")]


# ③ "부품군 진단" 드릴다운의 PR곡선·재발간격 차트도 같은 방식으로 재색칠한다.
# family-pr-chart/family-recur-chart는 task_sel==1일 때만 DOM에 있으므로
# 다른 과제·다른 화면에서는 호출 자체가 안 된다 — asset_tag/family 폴백은
# screen_3()의 폴백 로직과 동일하게 반복한다(그 함수 자체를 State로 참조할
# 수는 없으므로).
@app.callback(
    Output({"type": "family-pr-chart", "index": ALL}, "figure"),
    Input("theme-store", "data"),
    Input("screen-tabs", "value"),
    State("filter-store", "data"),
    State("selected-family-store", "data"),
)
def recolor_family_pr_chart(theme, _active_tab, filters, family):
    asset_tag = _focus_asset(filters, machine_only=True)
    valid_families = [r["part_family"] for r in load_asset_family_diagnosis(asset_tag)]
    family = family if family in valid_families else valid_families[0]
    pr_cm = load_family_pr_curve_and_confusion(asset_tag, family)
    c = pr_cm["confusion"]
    pr_cm = {**pr_cm, "positive_rate": (c["tp"] + c["fn"]) / max(1, sum(c.values()))}
    return [_pr_figure(pr_cm, theme or "light")]


@app.callback(
    Output({"type": "family-recur-chart", "index": ALL}, "figure"),
    Input("theme-store", "data"),
    Input("screen-tabs", "value"),
    State("filter-store", "data"),
    State("selected-family-store", "data"),
)
def recolor_family_recur_chart(theme, _active_tab, filters, family):
    asset_tag = _focus_asset(filters, machine_only=True)
    valid_families = [r["part_family"] for r in load_asset_family_diagnosis(asset_tag)]
    family = family if family in valid_families else valid_families[0]
    intervals = load_family_recurrence_intervals(asset_tag, family)
    return [_family_recur_figure(intervals, theme or "light")]


# ③ 과제 ①~③의 PR 곡선도 같은 방식으로 재색칠한다 — 과제·기준은 seg-store에서 읽는다.
@app.callback(
    Output({"type": "comparison-pr-chart", "index": ALL}, "figure"),
    Input("theme-store", "data"),
    Input("screen-tabs", "value"),
    State("seg-store", "data"),
)
def recolor_comparison_pr_chart(theme, _active_tab, seg_state):
    task_key = _screen3_task_key(seg_state)
    if task_key == "family":
        return no_update
    threshold = _screen3_threshold(task_key, seg_state)
    pr = load_comparison_pr_curve(task_key, load_focus_model(task_key, threshold), threshold)
    return [_pr_figure(pr, theme or "light", height=360)]


# 좁은 화면(1600px 미만)에서 헤더 내보내기 팝오버를 여닫는다 — 넓은 화면에서는 CSS가
# 항상 펼쳐 두므로 이 클래스가 영향을 주지 않는다.
app.clientside_callback(
    """function(n){ const open = (n || 0) % 2 === 1;
        return [open ? 'pf-export is-open' : 'pf-export', open ? 'true' : 'false']; }""",
    Output("export-group", "className"),
    Output("export-menu-btn", "aria-expanded"),
    Input("export-menu-btn", "n_clicks"),
)


# DESIGN.md §1.3 — Python 콜백으로 CSS 변수를 바꾸지 않고 <html data-theme>만 토글한다.
app.clientside_callback(
    "function(theme){ document.documentElement.setAttribute('data-theme', "
    "(theme === 'dark') ? 'dark' : 'light'); return window.dash_clientside.no_update; }",
    Output("theme-store", "data", allow_duplicate=True),
    Input("theme-store", "data"),
    prevent_initial_call="initial_duplicate",
)


# ------------------------------------------------------------
# 필터 쓰기 — 드롭다운 / 기간 / 초기화 → filter-store
# 첫 렌더에는 저장된 필터(localStorage)를 드롭다운에 되돌려 놓는다.
# ------------------------------------------------------------
@app.callback(
    Output("filter-store", "data"),
    Output("plant-dd", "value"),
    Output("machine-type-dd", "value"),
    Output("machine-dd", "value"),
    Input("plant-dd", "value"),
    Input("machine-type-dd", "value"),
    Input("machine-dd", "value"),
    Input({"type": "period-btn", "index": ALL}, "n_clicks"),
    Input("reset-btn", "n_clicks"),
    State("filter-store", "data"),
)
def write_filters(plant, machine_type, machine, _period_clicks, _reset_clicks, current):
    trigger = ctx.triggered_id
    current = current or DEFAULT_FILTERS
    new = dict(current)
    dd_values = (no_update, no_update, no_update)

    # 첫 렌더에 no_update를 돌려주면 filter-store만 입력으로 받는 콜백
    # (render_filters·sync_filter_options)의 첫 호출을 Dash가 건너뛴다 — 값을 그대로 다시 쓴다.
    if trigger is None:
        return new, new.get("plant"), new.get("machine_type"), new.get("machine")
    if trigger == "reset-btn":
        new = dict(DEFAULT_FILTERS)
        dd_values = (None, None, None)
    elif isinstance(trigger, dict) and trigger.get("type") == "period-btn":
        new["period_index"] = trigger["index"]
    else:
        new.update(plant=plant, machine_type=machine_type, machine=machine)
        # 공장·기계 종류를 바꿔 선택한 기계가 범위를 벗어나면 기계 선택을 푼다.
        if machine and machine not in filter_assets(plant, machine_type):
            new["machine"] = None
            dd_values = (no_update, no_update, None)

    if new == current:
        return no_update, *dd_values
    return new, *dd_values


# 공장·기계 종류가 서로의 선택지를, 둘이 함께 기계 선택지를 좁힌다 —
# 해당 기계가 없는 조합은 고를 수 없다.
@app.callback(
    Output("plant-dd", "options"),
    Output("machine-type-dd", "options"),
    Output("machine-dd", "options"),
    Input("filter-store", "data"),
)
def sync_filter_options(data):
    f = data or DEFAULT_FILTERS
    catalog = load_asset_catalog()
    by_type = catalog[catalog["machine_type"].eq(f["machine_type"])] if f.get("machine_type") else catalog
    by_plant = catalog[catalog["plant_code"].eq(f["plant"])] if f.get("plant") else catalog

    def options(values):
        return [{"label": v, "value": v} for v in values]

    return (options(sorted(by_type["plant_code"].unique())),
            options(sorted(by_plant["machine_type"].unique())),
            options(filter_assets(f.get("plant"), f.get("machine_type"))))


# 화면③ 모델 지표는 고정 평가 구간 전체 결과라 공장·기계 종류·기간 필터를
# 적용하지 않는다 — 해당 컨트롤을 끄고 이유를 적는다. 기계 필터는 부품군
# 진단 과제에서만 쓴다.
@app.callback(
    Output("plant-dd", "disabled"),
    Output("machine-type-dd", "disabled"),
    Output("machine-dd", "disabled"),
    Output({"type": "period-btn", "index": ALL}, "disabled"),
    Output("filter-scope-note", "children"),
    Output("filter-echo-wrap", "style"),
    Input("screen-tabs", "value"),
    Input("seg-store", "data"),
)
def apply_filter_scope(active, seg_state):
    period_count = len(PERIOD_PRESETS)
    if active != "3":
        return False, False, False, [False] * period_count, "", FILTER_ECHO_STYLE
    family_task = (seg_state or DEFAULT_SEG).get("task", 0) == SCREEN3_FAMILY_INDEX
    note_text = ("모델 지표는 평가 구간 전체 기준 · "
                 + ("부품군 진단은 기계 필터만 적용" if family_task else "필터 미적용"))
    return True, True, not family_task, [True] * period_count, note_text, {"display": "none"}


# ------------------------------------------------------------
# 필터 읽기 — filter-store → 기간 버튼 눌린 상태 + echo
# (prevent_initial_call 없음: localStorage에서 복원된 값도 첫 렌더에 반영된다)
# ------------------------------------------------------------
@app.callback(
    Output({"type": "period-btn", "index": ALL}, "aria-pressed"),
    Output("filter-echo", "children"),
    Input("filter-store", "data"),
)
def render_filters(data):
    data = data or DEFAULT_FILTERS
    pi = _period_index(data)
    pressed = ["true" if i == pi else "false" for i in range(len(PERIOD_PRESETS))]
    start = period_start(PERIOD_PRESETS[pi][1])
    # 공장·기계 종류·기계는 드롭다운이 이미 보여 주므로 기간 버튼이 뜻하는 실제 날짜만 적는다.
    start_text = load_data_start_date() if start is None else f"{start:%Y-%m-%d}"
    return pressed, f"{start_text} ~ {load_data_reference_date()}"


# ------------------------------------------------------------
# 세그먼티드 컨트롤 — ③ 과제 / ③ 위험 기준선 / ④ 데이터셋
# 선택이 바뀌면 seg-store가 갱신되고, render_screen이 화면을 다시 그린다.
# ------------------------------------------------------------
@app.callback(
    Output("seg-store", "data"),
    Input({"type": "seg-btn", "group": ALL, "index": ALL}, "n_clicks"),
    State("seg-store", "data"),
    prevent_initial_call=True,
)
def write_seg(_clicks, current):
    # 화면이 다시 그려지면 버튼이 n_clicks=0으로 재생성되며 이 콜백이 한 번
    # 더 불린다 — 값이 0인 호출은 무시한다(무한 루프 방지).
    trig = ctx.triggered[0] if ctx.triggered else None
    if not trig or not trig.get("value"):
        return no_update
    tid = ctx.triggered_id
    if not isinstance(tid, dict):
        return no_update
    new = dict(current or DEFAULT_SEG)
    new[tid["group"]] = tid["index"]
    return new


# ------------------------------------------------------------
# 기능 미구현 버튼 — 클릭은 되고, 무엇이 없는지 그대로 알려준다
# ------------------------------------------------------------
@app.callback(
    Output("action-echo", "children"),
    Input({"type": "ghost-btn", "index": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def echo_action(_clicks):
    trig = ctx.triggered[0] if ctx.triggered else None
    if not trig or not trig.get("value"):
        return no_update
    tid = ctx.triggered_id
    label = tid["index"] if isinstance(tid, dict) else str(tid)
    return f"'{label}' 클릭됨 — 동작 미구현 ({NO_DATA_MARK})"


# ------------------------------------------------------------
# ① KPI "기준일 고위험 기계" 드릴다운 — 클릭할 때마다 "점검 우선순위" 표 정렬을
# 등급가중 고장점수 ↔ 기준선 초과 사이로 토글한다. 화면을 새로 만들지 않고
# 이미 있는 표를 재사용해 원인(어느 공장·기계·부품)까지 이어지게 한다.
# ------------------------------------------------------------
@app.callback(
    Output("prio-sort-store", "data"),
    Output("action-echo", "children", allow_duplicate=True),
    Input({"type": "kpi-drill", "index": ALL}, "n_clicks"),
    State("prio-sort-store", "data"),
    prevent_initial_call=True,
)
def drill_failrate_to_priority(n_clicks_list, current):
    # ALL 패턴이라 리스트로 온다 — ①이 화면에 없을 땐 빈 리스트, 있을 땐 [n].
    # 화면 재렌더로 타일이 다시 만들어질 때의 n_clicks=0도 함께 무시한다.
    if not n_clicks_list or not any(n_clicks_list):
        return no_update, no_update
    current = current or DEFAULT_PRIO_SORT
    new_sort_by = "grade" if current.get("sort_by") == "threshold" else "threshold"
    msg = ("'기준일 고위험 기계' 클릭 → 점검 우선순위를 기준선 초과 기준으로 정렬"
           if new_sort_by == "threshold" else
           "'기준일 고위험 기계' 다시 클릭 → 점검 우선순위를 등급가중 고장점수 기준으로 복귀")
    new_state = {"sort_by": new_sort_by, "direction": current.get("direction", "desc")}
    return new_state, msg


# 점검 우선순위 표의 오름차순 ↔ 내림차순 토글 — sort_by(등급가중 고장점수 ↔
# 기준선 초과)는 그대로 두고 방향만 뒤집는다. drill_failrate_to_priority와
# 같은 prio-sort-store를 쓰므로 화면 전환·테마 전환에도 함께 유지된다.
@app.callback(
    Output("prio-sort-store", "data", allow_duplicate=True),
    Input({"type": "prio-dir-btn", "index": ALL}, "n_clicks"),
    State("prio-sort-store", "data"),
    prevent_initial_call=True,
)
def toggle_priority_direction(n_clicks_list, current):
    if not n_clicks_list or not any(n_clicks_list):
        return no_update
    current = current or DEFAULT_PRIO_SORT
    new_direction = "asc" if current.get("direction", "desc") == "desc" else "desc"
    return {"sort_by": current.get("sort_by", "grade"), "direction": new_direction}


# ------------------------------------------------------------
# 보고서 내보내기 — 대상(독자)에 따라 섹션 구성이 달라진다
# ------------------------------------------------------------
@app.callback(
    Output("report-download", "data"),
    Output("action-echo", "children", allow_duplicate=True),
    Input("export-run-btn", "n_clicks"),
    State("export-dd", "value"),
    State("filter-store", "data"),
    State("seg-store", "data"),
    prevent_initial_call=True,
)
def export_report(_run, export_value, filters, seg_state):
    fmt, audience = (export_value or f"pdf:{DEFAULT_AUDIENCE}").split(":")
    aud = REPORT_AUDIENCES[audience]
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    base = f"설비모니터링_보고서_{audience}_{stamp}"
    try:
        if fmt == "pdf":
            payload, name, mime = build_report_pdf(filters, seg_state, audience), f"{base}.pdf", "application/pdf"
        else:
            payload, name, mime = (
                build_report_xlsx(filters, seg_state, audience), f"{base}.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    except Exception as exc:  # noqa: BLE001 — 실패 이유를 화면에 그대로 보여준다
        app.server.logger.exception("보고서 내보내기 실패")
        return no_update, f"내보내기 실패 — {type(exc).__name__}"
    return (dcc.send_bytes(lambda b: b.write(payload), name, type=mime),
            f"{aud['label']} 보고서 내보냄 — {name} (섹션 {len(aud['sections'])}개 · 값은 {NO_DATA_MARK})")


# ------------------------------------------------------------
# 화면 ④ 데이터 조회 표 — 서버 측 정렬·페이지네이션 + CSV 내보내기
# ------------------------------------------------------------
# page_current와 sort_by를 각각 다른 콜백(A/B 체인)으로 나눴더니, 정렬을
# 바꾼 순간 A가 옛 page_current로 먼저 실행되고 B가 뒤늦게 0으로 되돌리는
# 경쟁 상태가 Playwright 검증에서 실제로 확인됐다(2페이지에서 정렬해도
# page_current가 2에 머무름) — 콜백 하나로 합쳐 트리거를 직접 판별한다.
@app.callback(
    Output({"type": "dtable", "index": MATCH}, "data"),
    Output({"type": "dtable", "index": MATCH}, "page_count"),
    Output({"type": "dtable", "index": MATCH}, "page_current"),
    Output({"type": "dtable-footer", "index": MATCH}, "children"),
    Input({"type": "dtable", "index": MATCH}, "page_current"),
    Input({"type": "dtable", "index": MATCH}, "sort_by"),
    State("filter-store", "data"),
    prevent_initial_call=True,
)
def update_dtable_page(page_current, sort_by, filters):
    dataset_key = ctx.triggered_id["index"]
    sort_col = sort_by[0]["column_id"] if sort_by else None
    sort_dir = sort_by[0]["direction"] if sort_by else "asc"
    triggered_prop = ctx.triggered[0]["prop_id"].rsplit(".", 1)[-1] if ctx.triggered else None
    page = 0 if triggered_prop == "sort_by" else (page_current or 0)
    assets, start = _filter_scope(filters)
    result = load_table_page(dataset_key, sort_col, sort_dir, page, assets=assets, start=start)
    footer_text = _dtable_footer_text(result["total_rows"], result["page"], 16)
    return result["data"], result["page_count"], result["page"], footer_text


@app.callback(
    Output("table-download", "data"),
    Input({"type": "csv-export-btn", "index": ALL}, "n_clicks"),
    State("filter-store", "data"),
    prevent_initial_call=True,
)
def export_dtable_csv(_clicks, filters):
    # 화면이 다시 그려지면 버튼이 n_clicks=0으로 재생성되며 이 콜백이 한 번
    # 더 불린다 — write_seg와 동일하게 값이 0인 호출은 무시한다.
    trig = ctx.triggered[0] if ctx.triggered else None
    if not trig or not trig.get("value"):
        return no_update
    dataset_key = ctx.triggered_id["index"]
    if dataset_key not in ("raw", "daily"):
        return no_update
    try:
        assets, start = _filter_scope(filters)
        csv_bytes, filename = export_table_csv(dataset_key, assets=assets, start=start)
    except Exception:
        app.server.logger.exception("CSV 내보내기 실패: %s", dataset_key)
        return no_update
    return dcc.send_bytes(lambda buf: buf.write(csv_bytes), filename, type="text/csv")


if __name__ == "__main__":
    # 최신 Dash(2.17+)는 app.run, 이전 버전은 app.run_server를 쓴다.
    # 이전 실행본이 8050 포트에 남아 오래된 시간 기록을 만들 수 있어 새 포트를 사용한다.
    # 같은 사내·가정 네트워크의 다른 기기도 접속할 수 있도록 모든 네트워크 인터페이스에서 받는다.
    # 외부 인터넷 공개는 별도의 방화벽·공유기·HTTPS 설정이 필요하므로 여기서 자동으로 열지 않는다.
    app.run(host=os.environ.get("DASHBOARD_HOST", "0.0.0.0"), port=int(os.environ.get("DASHBOARD_PORT", "8052")), debug=False)
