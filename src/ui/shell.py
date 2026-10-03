"""대시보드 틀 — 헤더(AppHeader)와 필터바(FilterBar)."""

from dash import dcc, html

from ..dashboard_data import load_data_reference_date, load_data_start_date
from .base import (
    date_range_inputs, DEFAULT_AUDIENCE, filter_dropdown, FILTERBAR_H, FOOTER_H, HEADER_H,
    MACHINE_OPTIONS, MACHINE_TYPE_OPTIONS, MARGIN, NOTE_12, period_toggle, PLANT_OPTIONS,
    REPORT_AUDIENCES, SCREENS,
)


# ============================================================
# 공통 컴포넌트: AppHeader / FilterBar
# ============================================================

def app_header():
    """좌: 시스템명 / 중앙: 화면 탭 4개(dcc.Tabs) / 우: 내보내기·테마 토글.
    프로젝트명·데이터 기간·합성 데이터 고지는 조작 대상이 아니라서 page_footer()로 내렸다."""
    left = html.Div(
        html.H1("산업 기계 센서 기반 고장 위험 예측", className="pf-header__brand title-16"),
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


FILTERBAR_STYLE = {"height": f"{FILTERBAR_H}px", "gap": "10px",
                   # overflow:hidden을 쓰면 드롭다운 팝업까지 clip된다. 줄바꿈만 막는다.
                   "flexWrap": "nowrap",
                   # app_header()와 같은 이유 — 필터바를 main보다 항상 위에 둔다.
                   "position": "relative", "zIndex": 20}


def filter_bar():
    """공장 / 기계 종류 / 기계 / 기간 / 초기화 · 우측 조작 결과 안내.
    dcc.Store(id="filter-store")가 화면 전환과 무관하게 값을 들고 있어 화면을
    옮겨도 선택이 유지된다. 화면 ③(모델 지표는 평가 구간 전체 기준)에서는 쓰는 필터가 없어
    필터바를 숨긴다 — wireframe_app의 clientside 콜백이 style을 바꾼다."""
    return html.Section(
        [filter_dropdown("plant-dd", "공장", PLANT_OPTIONS, 120),
         filter_dropdown("machine-type-dd", "기계 종류", MACHINE_TYPE_OPTIONS, 150),
         filter_dropdown("machine-dd", "기계", MACHINE_OPTIONS, 130),
         period_toggle(),
         date_range_inputs(),
         html.Button("초기화", id="reset-btn", n_clicks=0, className="pf-btn label-12",
                     style={"flexShrink": "0"}),
         html.Div(style={"flexGrow": "1"}),
         html.Span("", id="action-echo", className=NOTE_12,
                   style={"maxWidth": "208px", "minWidth": "0", "textAlign": "right",
                          "whiteSpace": "nowrap", "overflow": "hidden", "textOverflow": "ellipsis"})],
        id="filter-bar", className="pf-filterbar", **{"aria-label": "조회 조건"},
        style=FILTERBAR_STYLE,
    )


def page_footer():
    """화면 하단 오른쪽 안내 한 줄 — 프로젝트명·전체 데이터 기간·합성 데이터 고지.

    조작 대상이 아니라 배경 정보이므로 타원 칩이 아니라 평문으로 둔다. <footer>는
    보조기술에 contentinfo 영역으로 잡혀 따로 이름을 붙이지 않아도 된다."""
    return html.Footer(
        f"LS-JumpUp 프로젝트 · 전체 데이터 기간 {load_data_start_date()} ~ {load_data_reference_date()}"
        " · 이 프로젝트는 실측 데이터가 아닌 학습용 인공합성 데이터로 진행하였습니다.",
        className="micro-11 pf-muted",
        style={"height": f"{FOOTER_H}px", "padding": f"0 {MARGIN}px", "display": "flex",
               "alignItems": "center", "justifyContent": "flex-end", "whiteSpace": "nowrap",
               "overflow": "hidden", "textOverflow": "ellipsis"},
    )
