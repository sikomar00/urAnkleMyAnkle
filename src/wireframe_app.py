"""
산업 설비 모니터링 Dash 대시보드 — 와이어프레임 스켈레톤

이 파일은 최종 대시보드가 아니라 레이아웃 뼈대다. 카드 경계 · 제목 · 치수 주석만
있고 실제 차트 렌더링, 더미 데이터, 색상 팔레트는 없다. Plantfloor 디자인 시스템의
그레이스케일 토큰(surface/ink/border)과 레이아웃 토큰(그리드/행 높이)만 그대로
가져왔다. 카드 내부 자리(placeholder)를 실제 dcc.Graph, dash_table.DataTable 등으로
바꿔 끼우면 된다.

실행:
    pip install dash
    python wireframe_app.py
    → http://127.0.0.1:8050
"""

from dash import Dash, html, dcc, Input, Output, State, ALL, ctx, no_update

# ============================================================
# 디자인 토큰 (Plantfloor, 그레이스케일만 — 계열/상태/강조 색은 쓰지 않는다)
# ============================================================

PAGE = "#f9f9f7"        # surface-page
CARD = "#fcfcfb"        # surface-card
SUNK = "#f0efec"        # surface-sunken (플레이스홀더 채움)
RAISED = "#ffffff"      # surface-raised
INK = "#0b0b0b"         # ink
INK2 = "#52514e"        # ink-secondary
MUTED = "#68665f"       # ink-muted
HAIR = "#e1e0d9"        # border-hairline
CTRL = "#898781"        # border-control (점선 테두리)

SANS = ("Pretendard, 'Pretendard Variable', system-ui, -apple-system, "
        "'Segoe UI', 'Malgun Gothic', 'Apple SD Gothic Neo', sans-serif")
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, 'Liberation Mono', monospace"

TITLE_14 = {"margin": "0", "fontSize": "14px", "lineHeight": "20px",
            "fontWeight": "600", "color": INK, "whiteSpace": "nowrap"}
LABEL_12 = {"fontSize": "12px", "lineHeight": "16px", "fontWeight": "500", "color": INK2}
MICRO_11 = {"fontSize": "11px", "lineHeight": "14px", "fontWeight": "500", "color": MUTED}
NUM_12 = {"fontFamily": MONO, "fontSize": "12px", "lineHeight": "16px",
          "fontWeight": "500", "color": MUTED, "whiteSpace": "nowrap"}

# 레이아웃 토큰
CANVAS_W, CANVAS_H = 1920, 1080
HEADER_H, FILTERBAR_H = 56, 56
MARGIN, GUTTER = 20, 16
ROW_KPI, ROW_MAIN, ROW_SUB = 96, 460, 340

SCREENS = [
    ("1", "① 현황"),
    ("2", "② 기계 상세"),
    ("3", "③ 모델·예측"),
    ("4", "④ 데이터"),
    ("5", "⑤ 보고서 요약"),
]

# ------------------------------------------------------------
# 필터바 상호작용 — 이 옵션 목록은 전부 임시(placeholder)다.
# 실제 데이터가 붙으면 공장/기계 종류/기계 옵션은 데이터셋의 고유값으로
# 교체해야 한다(PUN-01 외 3곳만 Plantfloor 디자인 시스템 README에서 확인됨,
# 기계 종류 5종은 이름이 정해지지 않아 "종류 1..5"로 임시 표기).
# ------------------------------------------------------------
PLANT_OPTIONS = ["PUN-01", "PUN-02", "PUN-03"]          # 확인됨 (디자인 시스템 README)
MACHINE_TYPE_OPTIONS = [f"종류 {i+1}" for i in range(5)]  # 미확인 — 실제 이름으로 교체 필요
MACHINE_OPTIONS = [f"기계 {i+1:03d}" for i in range(20)]  # 미확인 — 실제 기계 ID로 교체 필요
PERIOD_PRESETS = ["프리셋 1", "프리셋 2", "프리셋 3", "전체"]

DEFAULT_FILTERS = {
    "plant": None, "machine_type": None, "machine": None, "period_index": 3,
}


# ============================================================
# 공용 프리미티브
# ============================================================

def dim(w, h):
    """카드/타일 우측 상단 치수 주석."""
    return html.Span(f"{w} × {h}", style=NUM_12)


def note(text, style=None):
    """작은 보조 설명(micro-11)."""
    s = dict(MICRO_11)
    if style:
        s.update(style)
    return html.Span(text, style=s)


def slot(label, w, h, sub=None):
    """아직 채워지지 않은 영역 — 점선 테두리 + 라벨 + 치수."""
    width_style = {"width": f"{w}px"} if isinstance(w, (int, float)) else {"flexGrow": "1", "minWidth": "0"}
    w_txt = w if isinstance(w, (int, float)) else "가변"
    if h < 34:
        return html.Div(
            note(f"{label} · {w_txt}×{h}"),
            style={**width_style, "height": f"{h}px", "flexShrink": "0", "boxSizing": "border-box",
                   "border": f"1px dashed {CTRL}", "background": SUNK, "display": "flex",
                   "alignItems": "center", "justifyContent": "center", "padding": "0 6px",
                   "overflow": "hidden"},
        )
    children = [html.Span(label, style={**LABEL_12, "textAlign": "center"}),
                html.Span(f"{w_txt} × {h}", style=NUM_12)]
    if sub:
        children.append(html.Span(sub, style={**MICRO_11, "textAlign": "center"}))
    return html.Div(
        children,
        style={**width_style, "height": f"{h}px", "flexShrink": "0", "boxSizing": "border-box",
               "border": f"1px dashed {CTRL}", "background": SUNK, "display": "flex",
               "flexDirection": "column", "alignItems": "center", "justifyContent": "center",
               "gap": "2px", "padding": "4px 8px", "overflow": "hidden"},
    )


def card(title, w, h, body, right=None):
    """카드 = 1px 테두리 + 라운드, 그림자 없음. 제목 줄 36px 고정."""
    header_right = [right] if right else []
    header_right.append(dim(w, h))
    return html.Section(
        [
            html.Div(
                [html.H3(title, style=TITLE_14),
                 html.Div(header_right, style={"display": "flex", "alignItems": "center", "gap": "12px",
                                                "minWidth": "0"})],
                style={"height": "36px", "flexShrink": "0", "display": "flex",
                       "alignItems": "flex-start", "justifyContent": "space-between", "gap": "12px"},
            ),
            html.Div(body, style={"width": f"{w - 32}px", "height": f"{h - 68}px",
                                   "display": "flex", "flexDirection": "column", "position": "relative"}),
        ],
        style={"width": f"{w}px", "height": f"{h}px", "flexShrink": "0", "boxSizing": "border-box",
               "background": CARD, "outline": f"1px solid {HAIR}", "outlineOffset": "-1px",
               "borderRadius": "4px", "padding": "16px", "display": "flex", "flexDirection": "column",
               "overflow": "hidden"},
    )


def row(height, children, gap=16, width=1880):
    return html.Div(children, style={"width": f"{width}px", "height": f"{height}px",
                                      "display": "flex", "gap": f"{gap}px"})


def col(width, height, children, gap=16):
    return html.Div(children, style={"width": f"{width}px", "height": f"{height}px",
                                      "flexShrink": "0", "display": "flex",
                                      "flexDirection": "column", "gap": f"{gap}px"})


def hstack(children, gap=8, style=None):
    s = {"display": "flex", "gap": f"{gap}px"}
    if style:
        s.update(style)
    return html.Div(children, style=s)


def tile(label, w=300, h=96, sub="값 · value-28"):
    return html.Div(
        [
            html.Div([html.Span(label, style=LABEL_12), dim(w, h)],
                      style={"height": "16px", "display": "flex", "justifyContent": "space-between", "gap": "8px"}),
            slot(sub, 168, 32),
        ],
        style={"width": f"{w}px", "height": f"{h}px", "flexShrink": "0", "boxSizing": "border-box",
               "background": CARD, "outline": f"1px solid {HAIR}", "outlineOffset": "-1px",
               "borderRadius": "4px", "padding": "16px", "display": "flex",
               "flexDirection": "column", "gap": "8px"},
    )


def bar(pct=45, h=8):
    """더미 숫자가 아니라 '여기 값이 온다'는 자리 표시일 뿐 — 길이에 의미 없음."""
    return html.Div(style={"width": f"{pct}%", "height": f"{h}px", "background": SUNK, "borderRadius": "2px"})


def btn(label, w=None, pressed=None, extra_style=None):
    style = {"height": "32px", "flexShrink": "0", "boxSizing": "border-box", "padding": "0 12px",
             "borderRadius": "2px", "background": RAISED if pressed else "transparent",
             "fontFamily": "inherit", "fontSize": "13px", "lineHeight": "18px",
             "fontWeight": "600" if pressed else "500", "color": INK if pressed else INK2,
             "whiteSpace": "nowrap", "cursor": "default",
             "border": f"1px solid {CTRL}" if pressed else f"1px solid {HAIR}"}
    if w:
        style["width"] = f"{w}px"
    if extra_style:
        style.update(extra_style)
    return html.Button(label, style=style, disabled=True)


def seg(label_text, options, sel=0):
    buttons = [btn(o, pressed=(i == sel), extra_style={"borderRadius": "0", "marginLeft": "-1px"})
               for i, o in enumerate(options)]
    return html.Div(
        [html.Span(label_text, style={**LABEL_12, "whiteSpace": "nowrap"}),
         html.Div(buttons, style={"display": "flex", "paddingLeft": "1px"})],
        style={"display": "flex", "alignItems": "center", "gap": "8px"},
    )


def filter_dropdown(dd_id, label, options, w):
    """실제로 선택 가능한 dcc.Dropdown. react-select 내부 스타일이라 와이어프레임
    토큰과 100% 같은 룩은 아니다 — 테두리색/글꼴만 최대한 맞췄다."""
    return html.Div(
        [html.Span(label, style={**LABEL_12, "whiteSpace": "nowrap"}),
         dcc.Dropdown(
             id=dd_id, options=[{"label": o, "value": o} for o in options],
             value=None, placeholder="전체", clearable=True,
             style={"width": f"{w}px", "fontFamily": "inherit", "fontSize": "13px"},
         )],
        style={"display": "flex", "alignItems": "center", "gap": "6px"},
    )


def period_toggle():
    """기간 프리셋 — 실제로 클릭되고 선택 상태가 바뀌는 버튼 4개."""
    buttons = [
        html.Button(label, id={"type": "period-btn", "index": i}, n_clicks=0,
                    style=period_btn_style(i == DEFAULT_FILTERS["period_index"]))
        for i, label in enumerate(PERIOD_PRESETS)
    ]
    return html.Div(
        [html.Span("기간", style={**LABEL_12, "whiteSpace": "nowrap"}),
         html.Div(buttons, id="period-btn-group", style={"display": "flex", "paddingLeft": "1px"})],
        style={"display": "flex", "alignItems": "center", "gap": "8px"},
    )


def period_btn_style(pressed):
    return {"height": "32px", "boxSizing": "border-box", "padding": "0 12px",
            "borderRadius": "0", "marginLeft": "-1px",
            "background": RAISED if pressed else "transparent",
            "fontFamily": "inherit", "fontSize": "13px", "lineHeight": "18px",
            "fontWeight": "600" if pressed else "500", "color": INK if pressed else INK2,
            "whiteSpace": "nowrap", "cursor": "pointer",
            "border": f"1px solid {CTRL}" if pressed else f"1px solid {HAIR}"}


def table_placeholder(cols, nrows, row_h, head_h=32, first_idx=False, sort_col=None, width=None, cell_h=6):
    """cols: [(라벨, 폭px, 정렬)] — 셀 내용은 실제 값이 아니라 자리 막대."""
    thead = html.Tr(
        [html.Th(f"{l}{' ▼' if i == sort_col else (' ▲▼' if sort_col is not None else '')}",
                 style={"width": f"{w}px", "boxSizing": "border-box", "padding": "0 8px",
                        "textAlign": a, **LABEL_12, "whiteSpace": "nowrap", "overflow": "hidden",
                        "borderBottom": f"1px solid {CTRL}"})
         for i, (l, w, a) in enumerate(cols)],
        style={"height": f"{head_h}px", "background": CARD},
    )
    body_rows = []
    for r in range(nrows):
        cells = []
        for i, (l, w, a) in enumerate(cols):
            if first_idx and i == 0:
                cell = html.Span(str(r + 1), style={"fontFamily": MONO, "fontSize": "12px", "color": MUTED})
            else:
                just = "flex-end" if a == "right" else ("center" if a == "center" else "flex-start")
                pct = 45 if a != "left" else 70
                cell = html.Div(bar(pct, cell_h), style={"display": "flex", "justifyContent": just})
            cells.append(html.Td(cell, style={"boxSizing": "border-box", "padding": "0 8px",
                                               "textAlign": a, "borderBottom": f"1px solid {HAIR}"}))
        body_rows.append(html.Tr(cells, style={"height": f"{row_h}px"}))
    tw = width or sum(c[1] for c in cols)
    return html.Table(
        [html.Thead(thead), html.Tbody(body_rows)],
        style={"width": f"{tw}px", "tableLayout": "fixed", "borderCollapse": "collapse", "flexShrink": "0"},
    )


def empty_state(title, reason, w, h):
    """확정되지 않은 값은 0/—이 아니라 이유가 적힌 빈 상태로 표시한다."""
    return html.Div(
        [note("EmptyState"),
         html.Span(title, style={"margin": "0", "fontSize": "14px", "lineHeight": "20px",
                                  "fontWeight": "600", "color": INK}),
         html.Span(reason, style={"fontSize": "13px", "lineHeight": "18px", "color": INK2, "textAlign": "center"}),
         html.Span(f"{w} × {h}", style=NUM_12)],
        style={"width": f"{w}px", "height": f"{h}px", "boxSizing": "border-box",
               "border": f"1px dashed {CTRL}", "borderRadius": "4px", "display": "flex",
               "flexDirection": "column", "alignItems": "center", "justifyContent": "center",
               "gap": "6px", "padding": "16px"},
    )


# ============================================================
# 공통 컴포넌트: AppHeader / FilterBar
# ============================================================

def app_header():
    """좌: 시스템명·기준일 / 중앙: 화면 탭 4개(dcc.Tabs) / 우: 경보·배지·토글·내보내기."""
    tab_style = {"height": "56px", "boxSizing": "border-box", "padding": "0 16px", "display": "flex",
                 "alignItems": "center", "fontSize": "14px", "lineHeight": "20px", "fontWeight": "500",
                 "color": INK2, "border": "none", "borderBottom": "2px solid transparent", "background": "none"}
    tab_selected_style = {**tab_style, "fontWeight": "600", "color": INK, "borderBottom": f"2px solid {INK}"}

    left = html.Div(
        [slot("시스템명 · title-16", 220, 28),
         html.Div([html.Span("데이터 기준일", style=LABEL_12), slot("YYYY-MM-DD · mono", 132, 24)],
                  style={"display": "flex", "alignItems": "center", "gap": "6px"}),
         note("AppHeader 1920 × 56")],
        style={"display": "flex", "alignItems": "center", "gap": "16px", "minWidth": "0"},
    )
    center = dcc.Tabs(
        id="screen-tabs", value="1",
        children=[dcc.Tab(label=label, value=tid, style=tab_style, selected_style=tab_selected_style)
                  for tid, label in SCREENS],
        style={"height": "56px"},
    )
    right = html.Div(
        [html.Div([html.Span("경보", style=LABEL_12), slot("카운터", 56, 24)],
                   style={"display": "flex", "alignItems": "center", "gap": "6px"}),
         html.Span("합성 데이터 · 교육용",
                    style={"height": "24px", "boxSizing": "border-box", "padding": "0 8px",
                           "border": f"1px solid {CTRL}", "borderRadius": "2px", "display": "inline-flex",
                           "alignItems": "center", "fontSize": "11px", "lineHeight": "14px",
                           "fontWeight": "500", "color": INK, "whiteSpace": "nowrap"}),
         btn("테마 전환"), btn("보고서 내보내기")],
        style={"display": "flex", "alignItems": "center", "justifyContent": "flex-end", "gap": "12px",
               "minWidth": "0"},
    )
    return html.Header(
        [left, center, right],
        style={"width": f"{CANVAS_W}px", "height": f"{HEADER_H}px", "boxSizing": "border-box",
               "padding": f"0 {MARGIN}px", "background": CARD, "borderBottom": f"1px solid {HAIR}",
               "display": "grid", "gridTemplateColumns": "minmax(0, 1fr) auto minmax(0, 1fr)",
               "alignItems": "center", "gap": "16px", "fontFamily": SANS, "color": INK},
    )


def filter_bar():
    """공장 / 기계 종류 / 기계 / 기간 프리셋 / 기간 직접지정 / 초기화 · 우측 현재 필터 상태 echo.
    dcc.Store(id="filter-store")가 화면 전환과 무관하게 값을 들고 있어 스펙의
    "화면 이동 시 상태 유지"를 충족한다. 실제 슬라이스 행 수는 데이터가 붙어야
    나오므로, 지금은 선택된 필터 조합을 그대로 보여주는 것으로 상호작용만 증명한다."""
    return html.Div(
        [filter_dropdown("plant-dd", "공장", PLANT_OPTIONS, 140),
         filter_dropdown("machine-type-dd", "기계 종류", MACHINE_TYPE_OPTIONS, 140),
         filter_dropdown("machine-dd", "기계", MACHINE_OPTIONS, 180),
         period_toggle(),
         html.Div([html.Span("직접 지정", style={**LABEL_12, "whiteSpace": "nowrap"}),
                   slot("시작일 – 종료일", 240, 32)],
                  style={"display": "flex", "alignItems": "center", "gap": "6px"}),
         html.Button("초기화", id="reset-btn", n_clicks=0,
                     style={"height": "32px", "boxSizing": "border-box", "padding": "0 12px",
                            "border": f"1px solid {HAIR}", "borderRadius": "2px", "background": "transparent",
                            "fontFamily": "inherit", "fontSize": "13px", "fontWeight": "500", "color": INK2,
                            "cursor": "pointer"}),
         html.Div(style={"flexGrow": "1"}),
         note("FilterBar 1920 × 56 · dcc.Store로 화면 이동 시 상태 유지"),
         html.Div([html.Span("현재 필터 (임시 echo — 실제 슬라이스 행 수는 데이터 연결 후)",
                              style={**LABEL_12, "whiteSpace": "nowrap"}),
                   html.Span(id="filter-echo", style={**NUM_12, "minWidth": "220px"})],
                  style={"display": "flex", "alignItems": "center", "gap": "6px"})],
        style={"width": f"{CANVAS_W}px", "height": f"{FILTERBAR_H}px", "boxSizing": "border-box",
               "padding": f"0 {MARGIN}px", "background": CARD, "borderBottom": f"1px solid {HAIR}",
               "display": "flex", "alignItems": "center", "gap": "16px", "fontFamily": SANS, "color": INK},
    )


# ============================================================
# 화면 ① 현황 — 행 96 / 460 / 340
# ============================================================

def screen_1():
    kpi_labels = ["관측 기계 (대)", "고장 표시 기계·일", "위험 기준선 초과 기계 (대)",
                  "평균 소비 전력 (kW)", "최고 베어링 온도 (°C)", "부품 출고 금액 (INR)"]
    row_a = row(ROW_KPI, [tile(k) for k in kpi_labels])

    prio_cols = [("순위", 48, "right"), ("대상", 200, "left"), ("종류", 110, "left"), ("공장", 100, "left"),
                 ("등급가중 고장점수", 150, "right"), ("기준선 초과", 110, "center"),
                 ("최근 고장 표시일", 150, "left"), ("당일 고장 표시 부품 수", 190, "right")]
    prio = card("점검 우선순위", 1090, ROW_MAIN,
                table_placeholder(prio_cols, 11, 32, head_h=32, first_idx=True, sort_col=4),
                right=note("대상 = 엔티티 무관 (현재 기계 행만) · 헤더 고정 · 정렬 · 행 클릭 → ②"))

    def machine_tile():
        return html.Div(
            [html.Div([slot("기계 태그", 96, 22), slot("종류", 96, 18)],
                      style={"width": "96px", "flexShrink": "0", "display": "flex",
                             "flexDirection": "column", "gap": "4px"}),
             slot("스파크라인", "가변", 32),
             html.Div([slot("현재값", 88, 22), slot("상태 배지", 88, 18)],
                      style={"width": "88px", "flexShrink": "0", "display": "flex",
                             "flexDirection": "column", "gap": "4px"})],
            style={"width": "367px", "height": "72px", "flexShrink": "0", "boxSizing": "border-box",
                   "outline": f"1px solid {HAIR}", "outlineOffset": "-1px", "borderRadius": "2px",
                   "background": CARD, "padding": "10px 12px", "display": "flex", "alignItems": "center",
                   "gap": "8px"},
        )
    tiles_grid = html.Div([machine_tile() for _ in range(10)],
                           style={"display": "grid", "gridTemplateColumns": "repeat(2, minmax(0, 1fr))",
                                  "gridTemplateRows": "repeat(5, 72px)", "columnGap": "8px", "rowGap": "8px",
                                  "width": "742px", "height": "392px"})
    status = card("기계 상태", 774, ROW_MAIN, tiles_grid, right=note("타일 367 × 72 · 2열 × 5행 · 간격 8"))
    row_b = row(ROW_MAIN, [prio, status])

    ylabels = html.Div([slot(f"기계 {i+1}", 112, 25) for i in range(10)],
                        style={"width": "112px", "flexShrink": "0", "display": "flex", "flexDirection": "column"})
    heatgrid = html.Div(
        [html.Div(style={"height": "25px", "boxSizing": "border-box", "borderBottom": f"1px dashed {HAIR}"})
         for _ in range(10)],
        style={"flexGrow": "1", "minWidth": "0", "boxSizing": "border-box", "border": f"1px dashed {CTRL}",
               "background": SUNK, "position": "relative"},
    )
    legend = html.Div(
        [html.Div([html.Div(style={"width": "14px", "height": "14px", "border": f"1px solid {CTRL}",
                                    "boxSizing": "border-box"}),
                   html.Span(f"구간 {i+1}", style=MICRO_11)],
                  style={"display": "flex", "alignItems": "center", "gap": "6px"})
         for i in range(5)],
        style={"width": "96px", "flexShrink": "0", "display": "flex", "flexDirection": "column", "gap": "6px"},
    )
    heat_body = html.Div(
        [hstack([ylabels, heatgrid,
                 html.Div([html.Span("범례 (부품 수)", style=LABEL_12), legend], style={"width": "96px",
                           "flexShrink": "0", "display": "flex", "flexDirection": "column", "gap": "6px"})],
                8, {"height": "250px"}),
         hstack([html.Div(style={"width": "112px"}), slot("x축 · 날짜", "가변", 22)], 8, {"height": "22px"})],
    )
    heat = card("고장 표시 히트맵", 1248, ROW_SUB, heat_body,
                right=note("셀 = 그날 고장 표시된 부품 수 · 5구간"))

    prows = html.Div(
        [html.Div([html.Div(bar(70, 6), style={"width": "88px", "flexShrink": "0"}),
                   slot("막대 (두께 ≤ 24)" if i == 0 else "", "가변", 16),
                   html.Div(bar(70, 6), style={"width": "48px", "flexShrink": "0", "display": "flex",
                                                "justifyContent": "flex-end"})],
                  style={"height": "25px", "display": "flex", "alignItems": "center", "gap": "8px"})
         for i in range(10)],
    )
    power_body = html.Div([prows,
                            hstack([html.Div(style={"width": "96px"}), slot("x축 (kW)", "가변", 22)], 0,
                                   {"height": "22px"})])
    power = card("기계별 평균 소비 전력 (kW)", 616, ROW_SUB, power_body, right=note("10개 · 내림차순"))
    row_c = row(ROW_SUB, [heat, power])

    return html.Div([row_a, row_b, row_c],
                     style={"display": "flex", "flexDirection": "column", "gap": f"{GUTTER}px"})


# ============================================================
# 화면 ② 기계 상세 — 행 88 / 520 / 288
# ============================================================

def screen_2():
    def metric(label, w):
        return html.Div([html.Span(label, style={**LABEL_12, "whiteSpace": "nowrap"}), slot("값", w, 26)],
                         style={"width": f"{w}px", "display": "flex", "flexDirection": "column", "gap": "4px"})

    strip = html.Section(
        [btn("‹ 이전 기계"),
         html.Div([slot("기계 태그 · value-20 mono", 220, 26), slot("종류 · 공장", 220, 22)],
                  style={"display": "flex", "flexDirection": "column", "gap": "4px"}),
         btn("다음 기계 ›"),
         html.Div(style={"flexGrow": "1"}),
         metric("현재 등급", 120), metric("위험도", 120), metric("최근 고장 표시일", 140),
         metric("고장 표시 일수 (일)", 140), metric("평균 소비 전력 (kW)", 140),
         dim(1880, 88)],
        style={"width": "1880px", "height": "88px", "boxSizing": "border-box", "background": CARD,
               "outline": f"1px solid {HAIR}", "outlineOffset": "-1px", "borderRadius": "4px",
               "padding": "16px", "display": "flex", "alignItems": "center", "gap": "16px"},
    )

    sensors = ["베어링 온도 (°C)", "모터 온도 (°C)", "수평 진동 [단위]", "수직 진동 [단위]",
               "오일 압력 [단위]", "부하율 (%)", "회전 속도 [단위]", "소비 전력 (kW)"]
    panels = [html.Div([html.Div(s, style={"width": "160px", "flexShrink": "0", "display": "flex",
                                            "alignItems": "center", **LABEL_12}),
                        slot(f"패널 {i+1}", "가변", 48)],
                       style={"height": "48px", "display": "flex", "gap": "8px", "flexShrink": "0"})
              for i, s in enumerate(sensors)]
    band_note = html.Div("고장 표시일 세로 밴드 (자리 표시)",
                          style={**MICRO_11, "position": "absolute", "left": "804px", "top": "2px",
                                 "background": CARD, "padding": "0 4px"})
    band = html.Div(style={"position": "absolute", "left": "760px", "top": "0", "width": "36px",
                            "height": "412px", "boxSizing": "border-box",
                            "borderLeft": f"1px dashed {INK2}", "borderRight": f"1px dashed {INK2}"})
    sm_body = html.Div(
        [html.Div([*panels, band, band_note],
                  style={"display": "flex", "flexDirection": "column", "gap": "4px", "height": "412px",
                         "position": "relative"}),
         html.Div([html.Div([note("8 × 48 + 7 × 4 + 40 = 452")],
                            style={"width": "160px", "flexShrink": "0", "display": "flex",
                                   "alignItems": "center"}),
                   slot("공유 x축 밴드 · 날짜", "가변", 40)],
                  style={"display": "flex", "gap": "8px", "height": "40px"})],
    )
    smult = card("센서 8종 스몰 멀티플", 1248, 520, sm_body,
                 right=note("x축 공유 · 패널 48 + 간격 4"))

    anom = card("이상 점수 추이", 616, 144, slot("라인 + 임계선", 584, 76), right=note("임계선 포함"))
    clus = card("군집 위치", 616, 144, slot("산점도 · 군집 1 / 2 / 3", 584, 76), right=note("선택 기계 표시"))
    parts = card("부품 출고 이력", 616, 200, slot("가로 막대", 584, 132))
    rightcol = col(616, 520, [anom, clus, parts])
    row_b = row(520, [smult, rightcol])

    pre = card("고장 직전 센서 변화", 932, 288,
               html.Div([slot("센서 추이 · t=0 정렬", 900, 198), slot("x축 · t−7 … t", 900, 22)]),
               right=note("고장 표시 시점 t=0 · t−7 ~ t"))
    peer = card("동종 기계 대비", 932, 288,
                html.Div([slot("같은 종류 기계 센서 분포 + 선택 기계 위치", 900, 198), slot("축 · 센서", 900, 22)]),
                right=note("같은 종류 기계만"))
    row_c = row(288, [pre, peer])

    return html.Div([strip, row_b, row_c],
                     style={"display": "flex", "flexDirection": "column", "gap": f"{GUTTER}px"})


# ============================================================
# 화면 ③ 모델·예측 — 툴바 32 / 행 96 / 460 / 272 (마지막 행이 남는 높이 흡수)
# ============================================================

def screen_3():
    toolbar = html.Div(
        [html.Div([seg("과제", ["현재 고장 표시 분류", "7일 사전 예측", "다음 관측일 전력"], sel=0),
                   html.Div([note("툴바 1880 × 32"),
                             seg("위험 기준선 (등급가중 고장점수 · 선택 시 모델 교체)", ["12", "13", "14"], sel=0),
                             slot("값 추가 여지", 96, 32)],
                            style={"display": "flex", "alignItems": "center", "gap": "12px"})],
                  style={"height": "32px", "display": "flex", "alignItems": "center",
                         "justifyContent": "space-between", "gap": "16px"}),
         html.Div([html.Span([k, html.Span(" [ ]", style={"fontFamily": MONO})], style={"whiteSpace": "nowrap"})
                   for k in ["그레인", "모집단", "평가 구간", "제외 규칙", "기준선 조건"]] +
                  [note("· 메타 줄 16")],
                  style={"height": "16px", "display": "flex", "alignItems": "center", "gap": "16px", **LABEL_12})],
        style={"width": "1880px", "height": "52px", "display": "flex", "flexDirection": "column",
               "gap": "4px", "flexShrink": "0"},
    )

    row_a = row(ROW_KPI, [tile(f"[분류 지표 {i+1}]", sub="값 또는 빈 상태") for i in range(6)])

    metric_cols = [("모델", 182, "left")] + [(f"[지표 {i+1}]", 93, "right") for i in range(6)]
    model_rows = []
    for name in ["기준 A", "기준 B", "RandomForest"]:
        cells = [html.Td(
            html.Div([html.Span(style={"width": "10px", "height": "10px", "border": f"1px solid {CTRL}",
                                        "boxSizing": "border-box"}),
                      html.Span(name, style={"fontSize": "13px", "color": INK})],
                     style={"display": "flex", "alignItems": "center", "gap": "8px"}),
            style={"padding": "0 8px", "borderBottom": f"1px solid {HAIR}"})]
        cells += [html.Td(html.Div(bar(45, 8), style={"display": "flex", "justifyContent": "flex-end"}),
                          style={"padding": "0 8px", "borderBottom": f"1px solid {HAIR}"}) for _ in range(6)]
        model_rows.append(html.Tr(cells, style={"height": "32px"}))
    model_rows.append(html.Tr(
        html.Td("+ 레지스트리 모델 행 (가변 · 32px/행)", colSpan=7,
                style={"padding": "0 8px", "border": f"1px dashed {CTRL}", "background": SUNK, **LABEL_12}),
        style={"height": "32px"}))
    mthead = html.Tr([html.Th(l, style={"width": f"{w}px", "padding": "0 8px", "textAlign": a,
                                         **LABEL_12, "whiteSpace": "nowrap", "borderBottom": f"1px solid {CTRL}"})
                       for l, w, a in metric_cols], style={"height": "32px"})
    mtable = html.Table([html.Thead(mthead), html.Tbody(model_rows)],
                         style={"width": "740px", "tableLayout": "fixed", "borderCollapse": "collapse"})
    mcomp_body = html.Div([mtable, html.Div(style={"flexGrow": "1"}),
                           note("가시 한도: 헤더 32 + 모델 10행 × 32 = 352 / 392"),
                           note("기계 종류 → 필터바 · 위험 기준선 → 툴바 (표의 축 아님)")])
    mcomp = card("모델 비교", 774, ROW_MAIN, mcomp_body, right=note("행 = 모델 (레지스트리) · 열 = 지표"))

    pr_body = html.Div([slot("범례 · 모델 n + 무작위 기준선", 584, 20),
                        hstack([slot("y축 · 정밀도", 40, 342), slot("PR 곡선 영역", "가변", 342)], 0),
                        hstack([html.Div(style={"width": "40px"}), slot("x축 · 재현율", "가변", 22)], 0)])
    prc = card("PR 곡선", 616, ROW_MAIN, pr_body, right=note("모델 수만큼 + 무작위 기준선"))

    def cell(t):
        return slot(t, 169, 160)
    cm_body = html.Div([
        hstack([html.Div(style={"width": "72px"}),
                html.Div("예측 고장 표시 있음", style={"width": "169px", "textAlign": "center", **LABEL_12}),
                html.Div("예측 고장 표시 없음", style={"width": "169px", "textAlign": "center", **LABEL_12})],
               8, {"height": "24px", "alignItems": "center"}),
        hstack([html.Div("실제 있음", style={"width": "72px", "display": "flex", "alignItems": "center", **LABEL_12}),
                cell("실제 있음 · 예측 있음"), cell("실제 있음 · 예측 없음")], 8, {"height": "160px"}),
        hstack([html.Div("실제 없음", style={"width": "72px", "display": "flex", "alignItems": "center", **LABEL_12}),
                cell("실제 없음 · 예측 있음"), cell("실제 없음 · 예측 없음")], 8, {"height": "160px"}),
        html.Div([html.Span("판정 임계값", style={**LABEL_12, "whiteSpace": "nowrap"}),
                  slot("값 · 하단 슬라이더와 연동", 220, 24)],
                 style={"height": "32px", "display": "flex", "alignItems": "center", "gap": "8px"}),
    ])
    cmx = card("혼동행렬", 458, ROW_MAIN, cm_body, right=note("2 × 2"))
    row_b = row(ROW_MAIN, [mcomp, prc, cmx])

    frows = html.Div(
        [html.Div([html.Div(bar(70, 5), style={"width": "160px", "flexShrink": "0"}),
                   html.Div(style={"flexGrow": "1", "height": "11px", "boxSizing": "border-box",
                                    "border": f"1px dashed {CTRL}", "background": SUNK}),
                   html.Div(bar(80, 5), style={"width": "48px", "flexShrink": "0"})],
                  style={"height": "17px", "display": "flex", "alignItems": "center", "gap": "8px",
                         "flexShrink": "0"})
         for _ in range(12)],
    )
    feat = card("주요 영향 변수 상위 12", 774, 272, frows, right=note("12행 × 17 · 직접 값 라벨"))

    slider = html.Div(
        [html.Label("판정 임계값 (즉시 재계산)", htmlFor="thr-slider", style={**LABEL_12, "whiteSpace": "nowrap"}),
         dcc.Slider(id="thr-slider", min=0, max=100, value=50, marks=None,
                    tooltip={"placement": "bottom"})],
        style={"display": "flex", "alignItems": "center", "gap": "8px", "width": "260px"},
    )
    thr_body = html.Div([hstack([slot("y축", 40, 182), slot("위험도 분포 히스토그램 + 임계값 세로선", "가변", 182)], 0),
                         hstack([html.Div(style={"width": "40px"}), slot("x축 · 위험도", "가변", 22)], 0)])
    thr = card("판정 임계값 조정", 616, 272, thr_body, right=slider)

    lead = card("선행 경보 일수 분포", 458, 272,
                empty_state("이 과제에는 해당 없음", "7일 사전 예측 과제에서만 활성", 426, 204))
    row_c = row(272, [feat, thr, lead])

    return html.Div([toolbar, row_a, row_b, row_c],
                     style={"display": "flex", "flexDirection": "column", "gap": f"{GUTTER}px"})


# ============================================================
# 화면 ④ 데이터 — 툴바 32 / 524 / 340
# ============================================================

def screen_4():
    toolbar = html.Div(
        [seg("데이터셋", ["원자료", "기계·일 집계", "부품 출고", "모델 입력 피처"], sel=0),
         html.Div([note("툴바 1880 × 32"), btn("CSV 내보내기")],
                  style={"display": "flex", "alignItems": "center", "gap": "12px"})],
        style={"width": "1880px", "height": "32px", "flexShrink": "0", "display": "flex",
               "alignItems": "center", "justifyContent": "space-between"},
    )

    dcols = [(f"[열 {i+1}]", 184, "left" if i < 2 else "right") for i in range(10)]
    dt = table_placeholder(dcols, 16, 24, head_h=32, width=1840, cell_h=6)
    scrollbar = html.Div(style={"width": "8px", "height": "384px", "marginTop": "32px",
                                 "boxSizing": "border-box", "border": f"1px dashed {CTRL}",
                                 "background": SUNK})
    footer = html.Div(
        [html.Div([slot("전체 행 수 · 표시 범위", 220, 24),
                   note("헤더 32 (sticky) · 열 머리: 정렬 ▲▼ + 필터 ›"),
                   note("본문 16행 × 24 = 384 · 이 본문만 세로 스크롤")],
                  style={"display": "flex", "alignItems": "center", "gap": "16px"}),
         html.Div([btn("‹"), slot("페이지 번호", 160, 32), btn("›")],
                  style={"display": "flex", "gap": "4px"})],
        style={"height": "32px", "display": "flex", "alignItems": "center", "justifyContent": "space-between"},
    )
    dtable_body = html.Div([hstack([html.Div(dt, style={"width": "1840px", "height": "416px", "overflow": "hidden"}),
                                    scrollbar], 0, {"height": "416px"}),
                            html.Div(style={"height": "8px"}), footer])
    dtable = card("데이터 조회", 1880, 524, dtable_body,
                  right=note("헤더 고정 · 열 정렬·필터 · 페이지네이션"))

    dict_cols = [("컬럼명", 130, "left"), ("타입", 64, "left"), ("단위", 56, "left"),
                 ("결측률 (%)", 72, "right"), ("설명", 120, "left")]
    def dict_half():
        return html.Div(table_placeholder(dict_cols, 11, 22, head_h=24, width=442, cell_h=6),
                         style={"width": "442px"})
    ddict = card("데이터 사전 (22열)", 932, ROW_SUB, hstack([dict_half(), dict_half()], 16),
                 right=note("11행 × 2단 · 행 22"))

    qual = card("품질 요약", 932, 162, hstack([slot(f"[품질 항목 {i+1}]", 213, 94) for i in range(4)], 16))
    src = card("출처 · 라이선스 · 합성 데이터 한계", 932, 162,
               hstack([slot("출처", 288, 94), slot("라이선스", 288, 94), slot("합성 데이터 한계", 292, 94)], 16))
    row_c = row(ROW_SUB, [ddict, col(932, ROW_SUB, [qual, src])])

    return html.Div([toolbar, row(524, [dtable]), row_c],
                     style={"display": "flex", "flexDirection": "column", "gap": f"{GUTTER}px"})


# ============================================================
# 화면 ⑤ 보고서 요약 — 경영진 / 최종 보고서용 집계 화면 (①②와 분리된 대상 독자)
# 개별 기계 행이 없다 — 조직 전체 집계·추세만. "보고서 내보내기"가
# 이 화면을 기준으로 PDF/PPT를 만든다고 가정한다.
# ============================================================

def screen_5():
    kpi_labels = ["관측 기간 고장 표시 총 건수", "위험 기준선 초과 기계 비율 (%)", "총 부품 출고 금액 (INR)",
                  "평균 소비 전력 (kW)", "전기간 대비 증감 (건)", "모델 최종 평가 지표"]
    row_a = row(ROW_KPI, [tile(k, sub="값 또는 빈 상태") for k in kpi_labels])

    trend_body = html.Div([slot("범례 · 고장 표시 건수 / 위험 기준선 초과 비율", 584, 20),
                           hstack([slot("y축", 40, 342), slot("기간별 추세 라인 (일/주 단위 집계)", "가변", 342)], 0),
                           hstack([html.Div(style={"width": "40px"}), slot("x축 · 기간", "가변", 22)], 0)])
    trend = card("고장·위험 추세", 1248, ROW_MAIN, trend_body,
                 right=note("개별 기계 아님 — 조직 전체 집계"))
    dist_body = html.Div([hstack([slot("y축", 40, 342),
                                   slot("공장별 · 기계 종류별 고장 비중 (가로 막대, 집계)", "가변", 342)], 0),
                          hstack([html.Div(style={"width": "40px"}), slot("x축 · 비중 (%)", "가변", 22)], 0)])
    dist = card("리스크 분포", 616, ROW_MAIN, dist_body, right=note("공장 3 / 기계 종류 5 집계"))
    row_b = row(ROW_MAIN, [trend, dist])

    summary_lines = html.Div(
        [slot(f"요약 문장 {i+1}", "가변", 24) for i in range(9)],
        style={"display": "flex", "flexDirection": "column", "gap": "4px", "overflow": "hidden"},
    )
    summary_card = card("이번 기간 요약", 932, ROW_SUB, summary_lines,
                        right=note("자동 생성 문구 자리 · 9줄 한도"))
    caveat_top = empty_state("모델 최종 평가 전", "평가 구간·지표는 평가 완료 후 기재", 900, 78)
    caveat_bot = html.Div(
        html.Span("합성 데이터 · 교육용 — 이 보고서의 모든 수치는 실제 설비 이력이 아니다 "
                  "(헤더 배지와 동일 문구를 보고서 산출물에도 유지)",
                  style={**LABEL_12, "textAlign": "center"}),
        style={"width": "900px", "height": "78px", "boxSizing": "border-box", "border": f"1px dashed {CTRL}",
               "background": SUNK, "display": "flex", "alignItems": "center", "justifyContent": "center",
               "padding": "8px 16px"},
    )
    caveat_card = card("데이터·모델 신뢰도 고지", 932, ROW_SUB,
                       html.Div([caveat_top, caveat_bot],
                                style={"display": "flex", "flexDirection": "column", "gap": "8px"}))
    row_c = row(ROW_SUB, [summary_card, caveat_card])

    return html.Div([row_a, row_b, row_c],
                     style={"display": "flex", "flexDirection": "column", "gap": f"{GUTTER}px"})


SCREEN_BUILDERS = {"1": screen_1, "2": screen_2, "3": screen_3, "4": screen_4, "5": screen_5}


# ============================================================
# 앱 조립
# ============================================================

app = Dash(__name__)
app.title = "설비 모니터링 대시보드 — 와이어프레임"

app.layout = html.Div(
    [
        dcc.Store(id="filter-store", data=DEFAULT_FILTERS),
        app_header(),
        filter_bar(),
        html.Main(
            id="screen-content",
            style={"width": f"{CANVAS_W}px", "height": f"{CANVAS_H - HEADER_H - FILTERBAR_H}px",
                   "boxSizing": "border-box", "padding": f"{MARGIN}px", "overflow": "hidden"},
        ),
    ],
    style={"width": f"{CANVAS_W}px", "minHeight": f"{CANVAS_H}px", "background": PAGE, "color": INK,
           "fontFamily": SANS, "margin": "0 auto"},
)


@app.callback(Output("screen-content", "children"), Input("screen-tabs", "value"))
def render_screen(active):
    return SCREEN_BUILDERS[active]()


@app.callback(
    Output("filter-store", "data"),
    Output({"type": "period-btn", "index": ALL}, "style"),
    Output("filter-echo", "children"),
    Output("plant-dd", "value"),
    Output("machine-type-dd", "value"),
    Output("machine-dd", "value"),
    Input("plant-dd", "value"),
    Input("machine-type-dd", "value"),
    Input("machine-dd", "value"),
    Input({"type": "period-btn", "index": ALL}, "n_clicks"),
    Input("reset-btn", "n_clicks"),
    State("filter-store", "data"),
    prevent_initial_call=True,
)
def update_filters(plant, machine_type, machine, _period_clicks, _reset_clicks, current):
    """드롭다운 3개 + 기간 프리셋 버튼 + 초기화, 전부 dcc.Store 하나로 모인다.
    '화면 이동 시 상태 유지'는 이 store가 app.layout 최상단에 한 번만 있고
    화면 콘텐츠(screen-content)만 다시 그려지기 때문에 그냥 따라온다.
    아직 실제 데이터가 없어 필터가 뭔가를 걸러내지는 않는다 — 선택값이
    잡히는 것만 filter-echo로 보여준다."""
    trigger = ctx.triggered_id
    new = dict(current)
    dd_reset = (no_update, no_update, no_update)

    if trigger == "reset-btn":
        new = dict(DEFAULT_FILTERS)
        dd_reset = (None, None, None)
    elif isinstance(trigger, dict) and trigger.get("type") == "period-btn":
        new["period_index"] = trigger["index"]
    else:
        new["plant"] = plant
        new["machine_type"] = machine_type
        new["machine"] = machine

    styles = [period_btn_style(i == new["period_index"]) for i in range(len(PERIOD_PRESETS))]
    echo = (f"공장={new['plant'] or '전체'} · 종류={new['machine_type'] or '전체'} · "
            f"기계={new['machine'] or '전체'} · 기간={PERIOD_PRESETS[new['period_index']]}")
    return new, styles, echo, *dd_reset


if __name__ == "__main__":
    # 최신 Dash(2.17+)는 app.run, 이전 버전은 app.run_server를 쓴다.
    app.run(debug=True)
