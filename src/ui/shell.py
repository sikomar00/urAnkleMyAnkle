"""대시보드 틀 — 헤더(AppHeader)와 필터바(FilterBar)."""

from dash import dcc, html

from ..dashboard_data import load_data_reference_date
from .base import (
    CODE_12, DEFAULT_AUDIENCE, filter_dropdown, FILTERBAR_H, HEADER_H, LABEL_12, MACHINE_OPTIONS,
    MACHINE_TYPE_OPTIONS, NOTE_12, period_toggle, PLANT_OPTIONS, REPORT_AUDIENCES, SCREENS,
)


# ============================================================
# 공통 컴포넌트: AppHeader / FilterBar
# ============================================================

def app_header():
    """좌: 시스템명·기준일 / 중앙: 화면 탭 4개(dcc.Tabs) / 우: 내보내기·테마 토글·계정 메뉴."""
    left = html.Div(
        [html.H1("산업 기계 센서 기반 고장 위험 예측", className="pf-header__brand title-16"),
         html.Div([html.Span("데이터 기준일", className=LABEL_12),
                   html.Span(load_data_reference_date(), className=f"{CODE_12} pf-muted")],
                  style={"display": "flex", "alignItems": "center", "gap": "6px", "whiteSpace": "nowrap"}),
         html.Span("합성 데이터 · 교육용", className="pf-chip-synthetic micro-11")],
        style={"display": "flex", "alignItems": "center", "gap": "12px", "minWidth": "0"},
    )
    center = dcc.Tabs(
        id="screen-tabs", value="1",
        children=[dcc.Tab(label=label, value=tid) for tid, label in SCREENS],
        style={"height": "56px"},
    )
    export_options = [
        {"label": f"{fmt_label} · {aud['label']}", "value": f"{fmt_value}:{aud_key}"}
        for fmt_label, fmt_value in [("PDF", "pdf"), ("Excel", "xlsx")]
        for aud_key, aud in REPORT_AUDIENCES.items()
    ]
    right = html.Div(
        # 1600px 미만에서는 내보내기 드롭다운을 이 버튼 뒤 팝오버로 접는다(03-app.css).
        # 계정 메뉴는 접지 않는다.
        [html.Button("보고서 내보내기", id="export-menu-btn", n_clicks=0,
                     className="pf-btn label-12 pf-export-toggle",
                     **{"aria-expanded": "false", "aria-controls": "export-group"}),
         html.Div([
             dcc.Dropdown(
                 id="export-dd",
                 options=export_options,
                 value=f"pdf:{DEFAULT_AUDIENCE}", clearable=False,
                 style={"width": "220px"},
             ),
             html.Button("내보내기", id="export-run-btn", n_clicks=0, className="pf-btn label-12"),
         ], id="export-group", className="pf-export"),
         html.Button("", id="theme-btn", n_clicks=0, title="테마 전환", className="pf-btn body-14",
                     style={"width": "32px", "padding": "0"}, **{"aria-label": "테마 전환"}),
         dcc.Download(id="report-download"),
         dcc.Download(id="table-download")],
        style={"display": "flex", "alignItems": "center", "justifyContent": "flex-end", "gap": "8px",
               # overflow:hidden을 여기 두면 "내보내기" 드롭다운 팝업까지 잘라버릴 수 있다.
               "minWidth": "0", "position": "relative"},
    )
    return html.Header(
        [left, center, right],
        # 3열 그리드(좌 · 탭 · 우) — 열 구성은 화면 폭에 따라 03-app.css(.pf-header--grid)가 정한다.
        className="pf-header pf-header--grid",
        style={"height": f"{HEADER_H}px",
               # 드롭다운 팝업이 main(카드들) 밑에 깔리지 않도록 헤더를 항상 위에 둔다.
               "position": "relative", "zIndex": 30},
    )


FILTER_ECHO_STYLE = {"display": "flex", "alignItems": "center", "gap": "6px", "flexShrink": "0"}


def filter_bar():
    """공장 / 기계 종류 / 기계 / 기간 / 초기화 · 우측 현재 필터 상태와 적용 범위 안내.
    dcc.Store(id="filter-store")가 화면 전환과 무관하게 값을 들고 있어 화면을
    옮겨도 선택이 유지된다."""
    return html.Section(
        [filter_dropdown("plant-dd", "공장", PLANT_OPTIONS, 120),
         filter_dropdown("machine-type-dd", "기계 종류", MACHINE_TYPE_OPTIONS, 150),
         filter_dropdown("machine-dd", "기계", MACHINE_OPTIONS, 130),
         period_toggle(),
         html.Button("초기화", id="reset-btn", n_clicks=0, className="pf-btn label-12",
                     style={"flexShrink": "0"}),
         html.Div(style={"flexGrow": "1"}),
         # 화면 ③에서는 적용 기간 대신 필터 적용 범위 안내를 보여 준다(apply_filter_scope).
         html.Span(id="filter-scope-note", className=NOTE_12,
                   style={"whiteSpace": "nowrap", "overflow": "hidden", "textOverflow": "ellipsis", "minWidth": "0"}),
         html.Div([html.Span("적용 기간", className=LABEL_12, style={"whiteSpace": "nowrap"}),
                   html.Span(id="filter-echo", className=f"{CODE_12} pf-muted", style={"whiteSpace": "nowrap"})],
                  id="filter-echo-wrap", style=FILTER_ECHO_STYLE),
         html.Span("", id="action-echo", className=NOTE_12,
                   style={"maxWidth": "208px", "minWidth": "0", "textAlign": "right",
                          "whiteSpace": "nowrap", "overflow": "hidden", "textOverflow": "ellipsis"})],
        className="pf-filterbar", **{"aria-label": "조회 조건"},
        style={"height": f"{FILTERBAR_H}px", "gap": "10px",
               # overflow:hidden을 쓰면 드롭다운 팝업까지 clip된다. 줄바꿈만 막는다.
               "flexWrap": "nowrap",
               # app_header()와 같은 이유 — 필터바를 main보다 항상 위에 둔다.
               "position": "relative", "zIndex": 20},
    )
