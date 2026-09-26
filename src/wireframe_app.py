"""
산업 설비 모니터링 Dash 대시보드 — 와이어프레임 스켈레톤

이 파일은 최종 대시보드가 아니라 레이아웃 뼈대다. 카드 경계 · 제목 · 치수 주석만
있고 실제 차트 렌더링, 더미 데이터, 색상 팔레트는 없다. Plantfloor 디자인 시스템의
그레이스케일 토큰(surface/ink/border)과 레이아웃 토큰(그리드/행 높이)만 그대로
가져왔다. 카드 내부 자리(placeholder)를 실제 dcc.Graph, dash_table.DataTable 등으로
바꿔 끼우면 된다.

이번 판에서 실제로 동작하는 것 (데이터 없이도 되는 것만):
  · 화면 탭 5개
  · 필터바 드롭다운 3개 + 기간 프리셋 + 초기화 → localStorage에 유지
  · 테마 전환 (라이트 ↔ 다크, 그레이스케일 그대로 · CSS 변수 교체)
  · 세그먼티드 컨트롤 3종 — ③ 과제 / ③ 위험 기준선 / ④ 데이터셋
      ③ 과제를 바꾸면 행 A 타일 구성(6×300 ↔ 4×458)과
        '선행 경보 일수 분포' 카드가 실제로 갈린다
      ③ 위험 기준선에는 "필터가 아니라 학습 라벨 정의" 경고를 상시 노출
  · 보고서 내보내기 — 대상(독자)별로 섹션 구성이 다른 PDF / Excel 생성
  · 나머지 버튼은 클릭되며, 무엇이 미구현인지 하단 action-echo에 표시

아직 안 되는 것 (데이터 연결이 선행돼야 하는 것):
  · 필터가 실제로 행을 걸러내지 않는다 (선택값만 잡힌다)
  · 기간 프리셋이 "지난 7일" 같은 실제 날짜 범위가 아니다 — 타임스탬프
    컬럼이 붙어야 PERIOD_PRESETS를 timedelta로 바꿀 수 있다
  · 보고서의 값 칸은 전부 "데이터 미연결"이다 (가짜 숫자를 넣지 않는다)
  · ③ 행 B(모델 비교·PR 곡선·혼동행렬)는 분류 과제 기준 고정 —
    전력 회귀 과제용 변형은 아직 없다

실행:
    pip install dash reportlab openpyxl
    python wireframe_app.py
    → http://127.0.0.1:8050
"""

import io
from datetime import datetime

from dash import Dash, html, dcc, Input, Output, State, ALL, ctx, no_update

# ============================================================
# 디자인 토큰 (Plantfloor, 그레이스케일만 — 계열/상태/강조 색은 쓰지 않는다)
#
# 토큰을 hex 리터럴이 아니라 CSS 변수 참조로 둔다. 인라인 style에 들어가는
# 문자열이 var(--pf-*)이므로, 루트 div의 className만 theme-light ↔ theme-dark로
# 바꾸면 트리를 다시 그리지 않고도 전체 팔레트가 갈린다. "테마 전환"이 실제로
# 동작하는 건 이 구조 때문이다. (계열/상태/강조 색은 와이어프레임 단계에서
# 여전히 쓰지 않는다 — 다크도 그레이스케일 대응값일 뿐이다.)
# ============================================================

PAGE = "var(--pf-page)"      # surface-page
CARD = "var(--pf-card)"      # surface-card
SUNK = "var(--pf-sunken)"    # surface-sunken (플레이스홀더 채움)
RAISED = "var(--pf-raised)"  # surface-raised
INK = "var(--pf-ink)"        # ink
INK2 = "var(--pf-ink-2)"     # ink-secondary
MUTED = "var(--pf-muted)"    # ink-muted
HAIR = "var(--pf-hair)"      # border-hairline
CTRL = "var(--pf-ctrl)"      # border-control (점선 테두리)

# 테마별 실제 값. 라이트는 Plantfloor 원본 값, 다크는 같은 역할의 그레이스케일
# 대응값이다(대비 뒤집기 — 새 색을 도입하지 않는다).
THEME_CSS = """
.theme-light {
  --pf-page:#f9f9f7; --pf-card:#fcfcfb; --pf-sunken:#f0efec; --pf-raised:#ffffff;
  --pf-ink:#0b0b0b; --pf-ink-2:#52514e; --pf-muted:#68665f;
  --pf-hair:#e1e0d9; --pf-ctrl:#898781;
  --pf-dd-bg:#ffffff; --pf-dd-ink:#0b0b0b;
}
.theme-dark {
  --pf-page:#131311; --pf-card:#1b1b19; --pf-sunken:#262622; --pf-raised:#2f2f2b;
  --pf-ink:#f4f4f1; --pf-ink-2:#bdbcb6; --pf-muted:#9d9b95;
  --pf-hair:#34342f; --pf-ctrl:#6f6d68;
  --pf-dd-bg:#1b1b19; --pf-dd-ink:#f4f4f1;
}
/* dcc.Dropdown은 자체 DOM(.dash-dropdown-*)이라 인라인 토큰이 닿지 않는다.
   Dash 4 기준 클래스명이며, Dash를 올리면 이 블록만 다시 맞추면 된다. */
.theme-dark .dash-dropdown,
.theme-dark .dash-dropdown-content,
.theme-dark .dash-dropdown-search {
  background:var(--pf-dd-bg) !important; color:var(--pf-dd-ink) !important;
  border-color:var(--pf-ctrl) !important;
}
.theme-dark .dash-dropdown-value,
.theme-dark .dash-dropdown-value-item,
.theme-dark .dash-dropdown-option,
.theme-dark .dash-options-list-option { color:var(--pf-dd-ink) !important; }
.theme-dark .dash-dropdown-placeholder { color:var(--pf-ink-2) !important; }
.theme-dark .dash-dropdown-option:hover,
.theme-dark .dash-options-list-option.selected { background:var(--pf-sunken) !important; }
.theme-dark .dash-dropdown-trigger-icon { color:var(--pf-ink-2) !important; }

/* dcc.Tabs는 컨테이너를 100% 폭으로 잡고 탭 5개를 균등 분배(flex:1)한다.
   그대로 두면 "⑤ 보고서 요약"처럼 긴 라벨이 잘린다. 바깥 컨테이너는
   내용 폭으로, 각 탭은 자기 글자 폭으로 되돌린다. */
#screen-tabs-parent, #screen-tabs { width:max-content !important; }
#screen-tabs { flex-wrap:nowrap !important; }
#screen-tabs .tab { flex:0 0 auto !important; width:auto !important; }

/* 캔버스가 1920 고정폭이라 넓은 화면에서는 좌우 여백이 생긴다 —
   다크에서 그 여백이 흰색으로 남지 않도록 body에도 같은 배경을 준다. */
html, body { margin:0; background:var(--pf-page); }
"""

INDEX_STRING = """<!DOCTYPE html>
<html>
  <head>
    {%metas%}<title>{%title%}</title>{%favicon%}{%css%}
    <style>__THEME_CSS__</style>
  </head>
  <body>{%app_entry%}<footer>{%config%}{%scripts%}{%renderer%}</footer></body>
</html>
""".replace("__THEME_CSS__", THEME_CSS)

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

# ------------------------------------------------------------
# 보고서 대상(독자)별 산출물 구성
#
# 인증·역할 관리 시스템이 없으므로 "로그인 사용자의 직책"을 알 방법이 없다.
# 그래서 역할을 추론하지 않고 헤더에서 직접 고르게 한다. 인증이 붙으면
# report-audience-dd의 기본값을 세션 역할로 채우고 드롭다운을 숨기면 된다.
#
# sections = 해당 독자의 보고서에 들어가는 섹션. 화면 ①~⑤ 중 어디서 오는지
# 명시해 둔다 — 운영(①②)과 최종보고(⑤)의 분리가 이 표에서 실제로 갈린다.
# ------------------------------------------------------------
REPORT_AUDIENCES = {
    "mgr": {
        "label": "공장 관리자 (운영)",
        "purpose": "당장 무엇을 점검할지 결정",
        "grain": "개별 기계 · 기계·일 단위",
        "sections": [
            ("① 현황", "KPI 6종", ["관측 기계 (대)", "고장 표시 기계·일", "위험 기준선 초과 기계 (대)",
                                   "평균 소비 전력 (kW)", "최고 베어링 온도 (°C)", "부품 출고 금액 (INR)"]),
            ("① 현황", "점검 우선순위 (기계 행)", ["순위", "대상", "종류", "공장", "등급가중 고장점수",
                                                  "기준선 초과", "최근 고장 표시일", "당일 고장 표시 부품 수"]),
            ("② 기계 상세", "상위 기계 센서 요약", ["기계 태그", "현재 등급", "위험도", "최근 고장 표시일",
                                                   "고장 표시 일수 (일)", "평균 소비 전력 (kW)"]),
            ("① 현황", "고장 표시 히트맵 (기계 × 날짜)", ["기계", "날짜", "고장 표시 부품 수"]),
        ],
        "excludes": "모델 학습 지표·PR 곡선·혼동행렬은 넣지 않는다 (판단에 불필요).",
    },
    "exec": {
        "label": "경영진 (최종 보고)",
        "purpose": "기간 성과·리스크 총량 파악",
        "grain": "조직 전체 집계 — 개별 기계 행 없음",
        "sections": [
            ("⑤ 보고서 요약", "집계 KPI 6종", ["관측 기간 고장 표시 총 건수", "위험 기준선 초과 기계 비율 (%)",
                                               "총 부품 출고 금액 (INR)", "평균 소비 전력 (kW)",
                                               "전기간 대비 증감 (건)", "모델 최종 평가 지표"]),
            ("⑤ 보고서 요약", "고장·위험 추세", ["기간", "고장 표시 건수", "위험 기준선 초과 비율 (%)"]),
            ("⑤ 보고서 요약", "리스크 분포", ["공장", "기계 종류", "고장 비중 (%)"]),
            ("⑤ 보고서 요약", "이번 기간 요약 문장", ["요약"]),
            ("⑤ 보고서 요약", "데이터·모델 신뢰도 고지", ["고지"]),
        ],
        "excludes": "기계 태그·센서 원계열·데이터 사전은 넣지 않는다 (의사결정 단위가 아님).",
    },
    "analyst": {
        "label": "설비·데이터 분석가",
        "purpose": "모델 타당성과 데이터 품질 검증",
        "grain": "모델 · 피처 · 컬럼 단위",
        "sections": [
            ("③ 모델·예측", "과제 설정", ["과제", "위험 기준선", "그레인", "모집단", "평가 구간",
                                          "제외 규칙", "기준선 조건"]),
            ("③ 모델·예측", "모델 비교", ["모델", "[지표 1]", "[지표 2]", "[지표 3]",
                                          "[지표 4]", "[지표 5]", "[지표 6]"]),
            ("③ 모델·예측", "주요 영향 변수 상위 12", ["순위", "변수", "영향도"]),
            ("④ 데이터", "데이터 사전 (22열)", ["컬럼명", "타입", "단위", "결측률 (%)", "설명"]),
            ("④ 데이터", "품질 요약 · 출처 · 합성 데이터 한계", ["항목", "내용"]),
        ],
        "excludes": "요약 문장·경영 지표는 넣지 않는다 (해석은 독자가 한다).",
    },
}
DEFAULT_AUDIENCE = "mgr"

# 데이터가 아직 붙지 않았다는 사실을 산출물에도 그대로 적는다 — 가짜 숫자를
# 채우지 않는다는 와이어프레임 원칙을 내보내기에도 유지한다.
NO_DATA_MARK = "데이터 미연결"

# ------------------------------------------------------------
# 세그먼티드 컨트롤 상태 — ③ 과제 / ③ 위험 기준선 / ④ 데이터셋
# 화면 탭을 옮겼다 돌아와도 선택이 유지되도록 seg-store에 모아둔다.
# ------------------------------------------------------------
SEG_GROUPS = {
    "task": ["현재 고장 표시 분류", "7일 사전 예측", "다음 관측일 전력"],
    "threshold": ["12", "13", "14"],
    "dataset": ["원자료", "기계·일 집계", "부품 출고", "모델 입력 피처"],
}
DEFAULT_SEG = {"task": 0, "threshold": 0, "dataset": 0}


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


def btn_style(pressed=False, w=None, extra=None):
    style = {"height": "32px", "flexShrink": "0", "boxSizing": "border-box", "padding": "0 12px",
             "borderRadius": "2px", "background": RAISED if pressed else "transparent",
             "fontFamily": "inherit", "fontSize": "13px", "lineHeight": "18px",
             "fontWeight": "600" if pressed else "500", "color": INK if pressed else INK2,
             "whiteSpace": "nowrap", "cursor": "pointer",
             "border": f"1px solid {CTRL}" if pressed else f"1px solid {HAIR}"}
    if w:
        style["width"] = f"{w}px"
    if extra:
        style.update(extra)
    return style


def btn(label, w=None, pressed=None, extra_style=None, bid=None):
    """와이어프레임의 모든 버튼은 클릭된다. 기능이 아직 없는 버튼은
    {"type": "ghost-btn"} 패턴을 달아, 눌렀을 때 하단 action-echo에
    '무엇이 눌렸고 무엇이 아직 없는지'를 그대로 표시한다. 가짜로 동작하는
    척하지 않는다 — 상호작용만 증명한다."""
    return html.Button(
        label, id={"type": "ghost-btn", "index": bid or label}, n_clicks=0,
        style=btn_style(pressed, w, extra_style),
    )


def seg(group, label_text, options, sel=0):
    """세그먼티드 컨트롤 — 실제로 선택이 바뀐다. group은 seg-store의 키다.
    ③ 과제 / ③ 위험 기준선 / ④ 데이터셋이 각각 하나의 group."""
    buttons = [
        html.Button(o, id={"type": "seg-btn", "group": group, "index": i}, n_clicks=0,
                    style=btn_style(i == sel, extra={"borderRadius": "0", "marginLeft": "-1px"}))
        for i, o in enumerate(options)
    ]
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
             # 새로고침·재접속 후에도 마지막 선택이 남는다. filter-store도
             # storage_type="local"이라 둘이 같은 값을 들고 복원된다.
             persistence=True, persistence_type="local",
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
    tab_style = {"height": "56px", "boxSizing": "border-box", "padding": "0 11px", "display": "flex",
                 "alignItems": "center", "whiteSpace": "nowrap", "flexShrink": "0",
                 "fontSize": "14px", "lineHeight": "20px", "fontWeight": "500",
                 "color": INK2, "border": "none", "borderBottom": "2px solid transparent", "background": "none"}
    tab_selected_style = {**tab_style, "fontWeight": "600", "color": INK, "borderBottom": f"2px solid {INK}"}

    left = html.Div(
        [slot("시스템명 · title-16", 180, 28),
         html.Div([html.Span("데이터 기준일", style=LABEL_12), slot("YYYY-MM-DD", 108, 24)],
                  style={"display": "flex", "alignItems": "center", "gap": "6px"}),
         note("1920 × 56")],
        style={"display": "flex", "alignItems": "center", "gap": "12px", "minWidth": "0"},
    )
    center = dcc.Tabs(
        id="screen-tabs", value="1",
        children=[dcc.Tab(label=label, value=tid, style=tab_style, selected_style=tab_selected_style)
                  for tid, label in SCREENS],
        style={"height": "56px"},
    )
    right = html.Div(
        [html.Div([html.Span("경보", style=LABEL_12), slot("카운터", 40, 24)],
                   style={"display": "flex", "alignItems": "center", "gap": "4px", "flexShrink": "0"}),
         html.Span("합성 데이터 · 교육용",
                    style={"height": "24px", "boxSizing": "border-box", "padding": "0 8px",
                           "border": f"1px solid {CTRL}", "borderRadius": "2px", "display": "inline-flex",
                           "alignItems": "center", "fontSize": "11px", "lineHeight": "14px",
                           "fontWeight": "500", "color": INK, "whiteSpace": "nowrap"}),
         # 인증·역할 관리 시스템이 없으므로 보고서 대상을 여기서 직접 고른다.
         html.Div(
             [html.Span("대상", style={**LABEL_12, "whiteSpace": "nowrap"}),
              dcc.Dropdown(
                  id="report-audience-dd",
                  options=[{"label": v["label"], "value": k} for k, v in REPORT_AUDIENCES.items()],
                  value=DEFAULT_AUDIENCE, clearable=False,
                  style={"width": "152px", "fontFamily": "inherit", "fontSize": "13px"},
              )],
             style={"display": "flex", "alignItems": "center", "gap": "4px", "flexShrink": "0"},
         ),
         html.Button("PDF", id="export-pdf-btn", n_clicks=0, style=btn_style(w=52)),
         html.Button("Excel", id="export-xlsx-btn", n_clicks=0, style=btn_style(w=60)),
         html.Button("테마 전환", id="theme-btn", n_clicks=0, style=btn_style(w=104)),
         dcc.Download(id="report-download")],
        style={"display": "flex", "alignItems": "center", "justifyContent": "flex-end", "gap": "8px",
               "minWidth": "0", "overflow": "hidden"},
    )
    return html.Header(
        [left, center, right],
        style={"width": f"{CANVAS_W}px", "height": f"{HEADER_H}px", "boxSizing": "border-box",
               "padding": f"0 {MARGIN}px", "background": CARD, "borderBottom": f"1px solid {HAIR}",
               # center를 max-content로 못박아야 탭 5개가 잘리지 않는다
               # (auto면 1fr 두 열이 먼저 자리를 가져가 탭이 깎인다).
               "display": "grid", "gridTemplateColumns": "minmax(0, 1fr) max-content minmax(0, 1fr)",
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
                   slot("시작일 – 종료일", 130, 32)],
                  style={"display": "flex", "alignItems": "center", "gap": "6px", "flexShrink": "0"}),
         html.Button("초기화", id="reset-btn", n_clicks=0,
                     style={"height": "32px", "boxSizing": "border-box", "padding": "0 12px",
                            "border": f"1px solid {HAIR}", "borderRadius": "2px", "background": "transparent",
                            "fontFamily": "inherit", "fontSize": "13px", "fontWeight": "500", "color": INK2,
                            "cursor": "pointer", "whiteSpace": "nowrap", "flexShrink": "0"}),
         html.Div(style={"flexGrow": "1"}),
         # storage_type="local" 이므로 새로고침·재접속 후에도 마지막 선택이 남는다.
         note("1920 × 56", {"whiteSpace": "nowrap", "flexShrink": "0"}),
         html.Div([html.Span("현재 필터", style={**LABEL_12, "whiteSpace": "nowrap"}),
                   html.Span(id="filter-echo", style={**NUM_12, "whiteSpace": "nowrap"})],
                  style={"display": "flex", "alignItems": "center", "gap": "6px", "flexShrink": "0"}),
         html.Span("선택값은 localStorage에 유지됨", id="action-echo",
                   style={**MICRO_11, "width": "208px", "textAlign": "right", "flexShrink": "0",
                          "whiteSpace": "nowrap", "overflow": "hidden", "textOverflow": "ellipsis"})],
        style={"width": f"{CANVAS_W}px", "height": f"{FILTERBAR_H}px", "boxSizing": "border-box",
               "padding": f"0 {MARGIN}px", "background": CARD, "borderBottom": f"1px solid {HAIR}",
               "display": "flex", "alignItems": "center", "gap": "10px", "fontFamily": SANS, "color": INK,
               "overflow": "hidden"},
    )


# ============================================================
# 화면 ① 현황 — 행 96 / 460 / 340
# ============================================================

def screen_1(seg_state=None, audience=DEFAULT_AUDIENCE):
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

def screen_2(seg_state=None, audience=DEFAULT_AUDIENCE):
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

def screen_3(seg_state=None, audience=DEFAULT_AUDIENCE):
    seg_state = seg_state or DEFAULT_SEG
    thr_sel = seg_state.get("threshold", 0)
    thr_changed = thr_sel != DEFAULT_SEG["threshold"]

    # 위험 기준선은 표의 축이 아니라 '라벨 정의'다 — 바꾸면 정답이 바뀌므로
    # 모델을 다시 학습/교체해야 한다. 관리자가 필터로 착각하고 눌렀다가
    # 화면 전체 수치가 갈리는 것이 원래 문제였다. 경고를 컨트롤 바로 아래에
    # 상시 노출하고, 기본값에서 벗어나면 문구를 강조로 바꾼다.
    warn_text = ("주의 — 위험 기준선은 필터가 아니라 학습 라벨 정의다. "
                 f"값을 바꾸면 해당 기준선으로 학습된 모델로 교체되고 아래 모든 지표가 다시 계산된다."
                 + (f"  현재 기본값({SEG_GROUPS['threshold'][DEFAULT_SEG['threshold']]}) 아님 "
                    f"→ {SEG_GROUPS['threshold'][thr_sel]} 기준 모델" if thr_changed else ""))
    warn_line = html.Div(
        html.Span(warn_text,
                  style={"fontSize": "11px", "lineHeight": "16px",
                         "fontWeight": "600" if thr_changed else "500",
                         "color": INK if thr_changed else MUTED, "whiteSpace": "nowrap",
                         "boxSizing": "border-box", "padding": "0 6px",
                         "border": f"1px solid {CTRL if thr_changed else HAIR}", "borderRadius": "2px"}),
        style={"height": "16px", "display": "flex", "alignItems": "center", "justifyContent": "flex-end"},
    )

    toolbar = html.Div(
        [html.Div([seg("task", "과제", SEG_GROUPS["task"], sel=seg_state.get("task", 0)),
                   html.Div([note("툴바 1880 × 32"),
                             seg("threshold", "위험 기준선 (등급가중 고장점수)",
                                 SEG_GROUPS["threshold"], sel=thr_sel),
                             slot("값 추가 여지", 96, 32)],
                            style={"display": "flex", "alignItems": "center", "gap": "12px"})],
                  style={"height": "32px", "display": "flex", "alignItems": "center",
                         "justifyContent": "space-between", "gap": "16px"}),
         warn_line,
         html.Div([html.Span([k, html.Span(" [ ]", style={"fontFamily": MONO})], style={"whiteSpace": "nowrap"})
                   for k in ["그레인", "모집단", "평가 구간", "제외 규칙", "기준선 조건"]] +
                  [note("· 메타 줄 16")],
                  style={"height": "16px", "display": "flex", "alignItems": "center", "gap": "16px", **LABEL_12})],
        style={"width": "1880px", "height": "72px", "display": "flex", "flexDirection": "column",
               "gap": "4px", "flexShrink": "0"},
    )

    # 과제에 따라 행 A가 갈린다 (원래 스펙의 '③ 행A 변형'):
    #   분류 2종 → 지표 타일 6개 × 300 / 전력 회귀 → 지표 타일 4개 × 458
    task_sel = seg_state.get("task", 0)
    if task_sel == 2:
        row_a = row(ROW_KPI, [tile(f"[회귀 지표 {i+1}]", w=458, sub="값 또는 빈 상태") for i in range(4)])
    else:
        kind = "분류" if task_sel == 0 else "사전예측"
        row_a = row(ROW_KPI, [tile(f"[{kind} 지표 {i+1}]", sub="값 또는 빈 상태") for i in range(6)])

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
                  style={"height": "15px", "display": "flex", "alignItems": "center", "gap": "8px",
                         "flexShrink": "0"})
         for _ in range(12)],
    )
    feat = card("주요 영향 변수 상위 12", 774, 252, frows, right=note("12행 × 15 · 직접 값 라벨"))

    slider = html.Div(
        [html.Label("판정 임계값 (즉시 재계산)", htmlFor="thr-slider", style={**LABEL_12, "whiteSpace": "nowrap"}),
         dcc.Slider(id="thr-slider", min=0, max=100, value=50, marks=None,
                    tooltip={"placement": "bottom"})],
        style={"display": "flex", "alignItems": "center", "gap": "8px", "width": "260px"},
    )
    thr_body = html.Div([hstack([slot("y축", 40, 162), slot("위험도 분포 히스토그램 + 임계값 세로선", "가변", 162)], 0),
                         hstack([html.Div(style={"width": "40px"}), slot("x축 · 위험도", "가변", 22)], 0)])
    thr = card("판정 임계값 조정", 616, 252, thr_body, right=slider)

    # 과제 세그먼트가 실제로 동작하므로 이 카드도 실제로 갈린다
    # (원래 카드 문구가 "7일 사전 예측 과제에서만 활성"이었는데, 선택이
    #  고정이라 영원히 빈 상태였다).
    if task_sel == 1:
        lead_body = html.Div([slot("선행 경보 일수 히스토그램", 426, 162), slot("x축 · 선행 일수", 426, 22)])
    else:
        lead_body = empty_state("이 과제에는 해당 없음", "7일 사전 예측 과제에서만 활성", 426, 184)
    lead = card("선행 경보 일수 분포", 458, 252, lead_body)
    row_c = row(252, [feat, thr, lead])

    return html.Div([toolbar, row_a, row_b, row_c],
                     style={"display": "flex", "flexDirection": "column", "gap": f"{GUTTER}px"})


# ============================================================
# 화면 ④ 데이터 — 툴바 32 / 524 / 340
# ============================================================

def screen_4(seg_state=None, audience=DEFAULT_AUDIENCE):
    seg_state = seg_state or DEFAULT_SEG
    toolbar = html.Div(
        [seg("dataset", "데이터셋", SEG_GROUPS["dataset"], sel=seg_state.get("dataset", 0)),
         html.Div([note("툴바 1880 × 32"), btn("CSV 내보내기", bid="csv-export")],
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
    dtable = card(f"데이터 조회 · {SEG_GROUPS['dataset'][seg_state.get('dataset', 0)]}", 1880, 524, dtable_body,
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

def screen_5(seg_state=None, audience=DEFAULT_AUDIENCE):
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
    aud = REPORT_AUDIENCES.get(audience, REPORT_AUDIENCES[DEFAULT_AUDIENCE])
    summary_card = card("이번 기간 요약", 932, ROW_SUB, summary_lines,
                        right=note(f"자동 생성 문구 자리 · 9줄 한도 · 현재 내보내기 대상: "
                                   f"{aud['label']} (섹션 {len(aud['sections'])}개)"))
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
# 보고서 내보내기 (PDF / Excel)
#
# 데이터가 아직 붙지 않았으므로 산출물에 가짜 숫자를 넣지 않는다. 대신
# "이 독자에게는 어떤 섹션·어떤 열이 들어가는지"가 확정된 보고서 골격을
# 내보낸다. 값 칸은 전부 NO_DATA_MARK다. 데이터가 붙으면 _section_rows()만
# 실제 쿼리 결과로 바꾸면 되고, 레이아웃·섹션 구성은 그대로 쓴다.
# ============================================================

CAVEAT = ("합성 데이터 · 교육용 — 이 보고서의 모든 수치는 실제 설비 이력이 아니다. "
          "운영 판단의 단독 근거로 쓰지 않는다.")

# ------------------------------------------------------------
# PDF 한글 폰트
#
# reportlab 내장 CID 폰트(HYSMyeongJo-Medium)는 쓰지 않는다. 폰트를 PDF에
# 임베드하지 않고 뷰어의 Adobe 한국어 폰트팩에 의존하기 때문에, 그 팩이 없는
# 뷰어(크롬 내장 뷰어, poppler, 미리보기 등)에서는 글자가 통째로 사라진다.
# 실제로 그 증상을 확인했다. 그래서 시스템에 있는 한글 TrueType을 찾아
# 임베드한다. TrueType만 된다 — Noto Sans CJK 같은 OTF/CFF 계열은 reportlab이
# 거부한다("postscript outlines are not supported").
#
# 하나도 못 찾으면 PDF를 만들지 않고 실패로 알린다. 글자 없는 PDF를
# 성공인 척 내보내지 않는다.
# ------------------------------------------------------------
KOREAN_TTF_CANDIDATES = [
    # Windows
    r"C:\Windows\Fonts\malgun.ttf",           # 맑은 고딕
    r"C:\Windows\Fonts\malgunsl.ttf",
    r"C:\Windows\Fonts\gulim.ttc",            # 굴림
    r"C:\Windows\Fonts\batang.ttc",           # 바탕
    # macOS
    "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
    "/Library/Fonts/AppleGothic.ttf",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    # Linux (sudo apt-get install fonts-nanum)
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf",
    "/usr/share/fonts/truetype/nanum/NanumMyeongjo.ttf",
    "/usr/share/fonts/nanum/NanumGothic.ttf",
    # 이 파일 옆에 폰트를 직접 두는 경우
    "NanumGothic.ttf", "malgun.ttf",
]
PDF_FONT_NAME = "KoreanBody"
_pdf_font_cache = {}


def resolve_pdf_font():
    """등록에 성공한 (폰트명, 경로)를 돌려준다. 못 찾으면 (None, None)."""
    if _pdf_font_cache:
        return _pdf_font_cache["name"], _pdf_font_cache["path"]
    import os
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    for path in KOREAN_TTF_CANDIDATES:
        if not os.path.exists(path):
            continue
        # .ttc는 한 파일에 여러 폰트가 들어 있어 인덱스를 훑어야 한다
        indices = (0, 1, 2, 3) if path.lower().endswith(".ttc") else (None,)
        for idx in indices:
            try:
                kwargs = {} if idx is None else {"subfontIndex": idx}
                pdfmetrics.registerFont(TTFont(PDF_FONT_NAME, path, **kwargs))
                # 한글 글리프가 실제로 있는지 확인 — 라틴 전용 폰트를 거른다
                if pdfmetrics.getFont(PDF_FONT_NAME).stringWidth("설비 모니터링", 10) <= 0:
                    raise ValueError("한글 글리프 없음")
            except Exception:
                continue
            _pdf_font_cache.update(name=PDF_FONT_NAME, path=path)
            return PDF_FONT_NAME, path
    return None, None


class PdfFontMissing(RuntimeError):
    pass


def _report_context(filters, seg_state, audience):
    filters = filters or DEFAULT_FILTERS
    seg_state = seg_state or DEFAULT_SEG
    aud = REPORT_AUDIENCES.get(audience, REPORT_AUDIENCES[DEFAULT_AUDIENCE])
    pi = filters.get("period_index", DEFAULT_FILTERS["period_index"])
    return {
        "aud": aud,
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "meta": [
            ("보고서 대상", aud["label"]),
            ("보고 목적", aud["purpose"]),
            ("집계 단위", aud["grain"]),
            ("생성 시각", datetime.now().strftime("%Y-%m-%d %H:%M")),
            ("데이터 기준일", NO_DATA_MARK),
            ("공장", filters.get("plant") or "전체"),
            ("기계 종류", filters.get("machine_type") or "전체"),
            ("기계", filters.get("machine") or "전체"),
            ("기간", PERIOD_PRESETS[pi] if 0 <= pi < len(PERIOD_PRESETS) else "전체"),
            ("모델 과제", SEG_GROUPS["task"][seg_state.get("task", 0)]),
            ("위험 기준선", SEG_GROUPS["threshold"][seg_state.get("threshold", 0)]),
            ("제외 섹션", aud["excludes"]),
        ],
    }


def esc(s):
    """Paragraph는 미니 HTML을 파싱하므로 & < > 를 이스케이프한다."""
    from xml.sax.saxutils import escape
    return escape(str(s))


def _section_rows(cols):
    """데이터 연결 전까지 값 칸은 미연결 표시 1행. 연결 후 이 함수만 교체한다."""
    return [[NO_DATA_MARK] * len(cols)]


def build_report_pdf(filters, seg_state, audience):
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle)

    FONT, font_path = resolve_pdf_font()
    if not FONT:
        raise PdfFontMissing(
            "PDF에 임베드할 한글 TrueType 폰트를 찾지 못했다. "
            "Windows/macOS는 기본 폰트가 잡히고, Linux는 "
            "`sudo apt-get install fonts-nanum` 후 다시 시도하거나 "
            "NanumGothic.ttf를 이 파일 옆에 두면 된다. (Excel 내보내기는 영향 없음)")

    ctxd = _report_context(filters, seg_state, audience)
    aud = ctxd["aud"]

    h1 = ParagraphStyle("h1", fontName=FONT, fontSize=17, leading=22, spaceAfter=2, alignment=TA_LEFT)
    h2 = ParagraphStyle("h2", fontName=FONT, fontSize=11.5, leading=15, spaceBefore=9, spaceAfter=3)
    body = ParagraphStyle("body", fontName=FONT, fontSize=8.5, leading=12)
    small = ParagraphStyle("small", fontName=FONT, fontSize=7.5, leading=10.5,
                           textColor=colors.HexColor("#68665f"))

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=landscape(A4),
        leftMargin=14 * mm, rightMargin=14 * mm, topMargin=12 * mm, bottomMargin=12 * mm,
        title=f"설비 모니터링 보고서 — {aud['label']}", author="설비 모니터링 대시보드 (와이어프레임)",
    )

    grid = TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), FONT),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c9c8c2")),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f0efec")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("TEXTCOLOR", (0, 1), (-1, -1), colors.HexColor("#68665f")),
    ])

    flow = [
        Paragraph("설비 모니터링 보고서", h1),
        Paragraph(esc(f"{aud['label']} · {aud['purpose']} · 생성 {ctxd['generated']}"), small),
        Spacer(1, 6),
        Paragraph(esc(CAVEAT), small),
        Spacer(1, 8),
        Paragraph("보고 조건", h2),
    ]

    meta_tbl = Table([[Paragraph(esc(k), body), Paragraph(esc(v), body)] for k, v in ctxd["meta"]],
                     colWidths=[34 * mm, 205 * mm])
    meta_tbl.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), FONT),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c9c8c2")),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f0efec")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    flow.append(meta_tbl)

    flow.append(Paragraph(f"섹션 {len(aud['sections'])}개", h2))
    for i, (src, name, cols) in enumerate(aud["sections"], 1):
        flow.append(Paragraph(f"{esc(i)}. {esc(name)}　<font size=7.5>[{esc(src)}]</font>", h2))
        data = [[Paragraph(esc(c), body) for c in cols]] + [[Paragraph(esc(c), body) for c in r]
                                                        for r in _section_rows(cols)]
        avail = 239 * mm
        cw = [avail / len(cols)] * len(cols)
        t = Table(data, colWidths=cw, repeatRows=1)
        t.setStyle(grid)
        flow.append(t)

    flow.append(Spacer(1, 8))
    flow.append(Paragraph(
        f"값 칸이 '{NO_DATA_MARK}'인 것은 데이터 소스가 아직 연결되지 않았기 때문이다. "
        "섹션 구성과 열 정의는 확정본이므로, 데이터 연결 후 값만 채우면 된다.  "
        f"(임베드 폰트: {esc(font_path)})", small))

    doc.build(flow)
    return buf.getvalue()


def _safe_sheet(name, used):
    bad = set('[]:*?/\\')
    s = "".join(("·" if ch in bad else ch) for ch in name)[:31] or "sheet"
    base, n = s, 2
    while s in used:
        suffix = f"_{n}"
        s = base[:31 - len(suffix)] + suffix
        n += 1
    used.add(s)
    return s


def build_report_xlsx(filters, seg_state, audience):
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter

    ctxd = _report_context(filters, seg_state, audience)
    aud = ctxd["aud"]

    thin = Side(style="thin", color="C9C8C2")
    edge = Border(left=thin, right=thin, top=thin, bottom=thin)
    head_fill = PatternFill("solid", fgColor="F0EFEC")
    bold = Font(bold=True, size=10)
    muted = Font(size=10, color="68665F")

    wb = Workbook()
    used = set()

    ws = wb.active
    ws.title = _safe_sheet("표지·보고조건", used)
    ws["A1"] = "설비 모니터링 보고서"
    ws["A1"].font = Font(bold=True, size=14)
    ws["A2"] = f"{aud['label']} · {aud['purpose']}"
    ws["A2"].font = muted
    ws["A3"] = CAVEAT
    ws["A3"].font = muted
    r = 5
    ws.cell(r, 1, "항목").font = bold
    ws.cell(r, 2, "값").font = bold
    for c in (1, 2):
        ws.cell(r, c).fill = head_fill
        ws.cell(r, c).border = edge
    for k, v in ctxd["meta"]:
        r += 1
        ws.cell(r, 1, k).border = edge
        cell = ws.cell(r, 2, str(v))
        cell.border = edge
        cell.alignment = Alignment(wrap_text=True, vertical="top")
    r += 2
    ws.cell(r, 1, "섹션 목록").font = bold
    for i, (src, name, cols) in enumerate(aud["sections"], 1):
        r += 1
        ws.cell(r, 1, f"{i}. {name}")
        ws.cell(r, 2, f"[{src}] · 열 {len(cols)}개")
        ws.cell(r, 2).font = muted
    ws.column_dimensions["A"].width = 28
    ws.column_dimensions["B"].width = 96
    ws.freeze_panes = "A6"

    for i, (src, name, cols) in enumerate(aud["sections"], 1):
        s = wb.create_sheet(_safe_sheet(f"{i}·{name}", used))
        s["A1"] = f"{name}  [{src}]"
        s["A1"].font = Font(bold=True, size=11)
        s["A2"] = f"{NO_DATA_MARK} — 열 정의는 확정본, 값은 데이터 연결 후 채운다."
        s["A2"].font = muted
        for c, label in enumerate(cols, 1):
            cell = s.cell(4, c, label)
            cell.font = bold
            cell.fill = head_fill
            cell.border = edge
            s.column_dimensions[get_column_letter(c)].width = max(14, min(30, len(label) + 6))
        for rr, rowvals in enumerate(_section_rows(cols), start=5):
            for c, v in enumerate(rowvals, 1):
                cell = s.cell(rr, c, v)
                cell.border = edge
                cell.font = muted
        s.freeze_panes = "A5"

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


# ============================================================
# 앱 조립
# ============================================================

app = Dash(__name__)
app.title = "설비 모니터링 대시보드 — 와이어프레임"
app.index_string = INDEX_STRING

app.layout = html.Div(
    [
        # storage_type="local" → 새로고침·재접속 후에도 마지막 선택이 남는다.
        # (관리자가 매번 같은 공장·기간을 다시 고르던 문제)
        dcc.Store(id="filter-store", data=DEFAULT_FILTERS, storage_type="local"),
        dcc.Store(id="seg-store", data=DEFAULT_SEG, storage_type="local"),
        dcc.Store(id="theme-store", data="light", storage_type="local"),
        app_header(),
        filter_bar(),
        html.Main(
            id="screen-content",
            style={"width": f"{CANVAS_W}px", "height": f"{CANVAS_H - HEADER_H - FILTERBAR_H}px",
                   "boxSizing": "border-box", "padding": f"{MARGIN}px", "overflow": "hidden"},
        ),
    ],
    id="root", className="theme-light",
    style={"width": f"{CANVAS_W}px", "minHeight": f"{CANVAS_H}px", "background": PAGE, "color": INK,
           "fontFamily": SANS, "margin": "0 auto"},
)


# ------------------------------------------------------------
# 화면 렌더 — 탭 / 세그먼트 상태 / 보고서 대상이 바뀌면 다시 그린다
# ------------------------------------------------------------
@app.callback(
    Output("screen-content", "children"),
    Input("screen-tabs", "value"),
    Input("seg-store", "data"),
    Input("report-audience-dd", "value"),
)
def render_screen(active, seg_state, audience):
    return SCREEN_BUILDERS[active](seg_state=seg_state, audience=audience or DEFAULT_AUDIENCE)


# ------------------------------------------------------------
# 테마 전환 — 루트 className만 갈아끼우면 CSS 변수가 전부 따라 바뀐다
# ------------------------------------------------------------
@app.callback(
    Output("theme-store", "data"),
    Input("theme-btn", "n_clicks"),
    State("theme-store", "data"),
    prevent_initial_call=True,
)
def toggle_theme(_n, current):
    return "light" if (current or "light") == "dark" else "dark"


@app.callback(
    Output("root", "className"),
    Output("theme-btn", "children"),
    Input("theme-store", "data"),
)
def apply_theme(theme):
    theme = theme if theme in ("light", "dark") else "light"
    return f"theme-{theme}", f"테마 · {'다크' if theme == 'dark' else '라이트'}"


# <html>에도 같은 클래스를 얹는다. #root는 1920 고정폭이라 넓은 화면에서
# 좌우 여백(body)이 남는데, 변수가 html에 있어야 그 여백까지 테마를 따라간다.
app.clientside_callback(
    "function(theme){ document.documentElement.className = 'theme-' + "
    "((theme === 'dark') ? 'dark' : 'light'); return window.dash_clientside.no_update; }",
    Output("theme-store", "data", allow_duplicate=True),
    Input("theme-store", "data"),
    prevent_initial_call="initial_duplicate",
)


# ------------------------------------------------------------
# 필터 쓰기 — 드롭다운 / 기간 프리셋 / 초기화 → filter-store
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
    prevent_initial_call=True,
)
def write_filters(plant, machine_type, machine, _period_clicks, _reset_clicks, current):
    trigger = ctx.triggered_id
    new = dict(current or DEFAULT_FILTERS)
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

    return new, *dd_reset


# ------------------------------------------------------------
# 필터 읽기 — filter-store → 기간 버튼 눌린 상태 + echo
# (prevent_initial_call 없음: localStorage에서 복원된 값도 첫 렌더에 반영된다)
# ------------------------------------------------------------
@app.callback(
    Output({"type": "period-btn", "index": ALL}, "style"),
    Output("filter-echo", "children"),
    Input("filter-store", "data"),
)
def render_filters(data):
    data = data or DEFAULT_FILTERS
    pi = data.get("period_index", DEFAULT_FILTERS["period_index"])
    if not 0 <= pi < len(PERIOD_PRESETS):
        pi = DEFAULT_FILTERS["period_index"]
    styles = [period_btn_style(i == pi) for i in range(len(PERIOD_PRESETS))]
    echo = (f"공장={data.get('plant') or '전체'} · 종류={data.get('machine_type') or '전체'} · "
            f"기계={data.get('machine') or '전체'} · 기간={PERIOD_PRESETS[pi]}")
    return styles, echo


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
# 보고서 내보내기 — 대상(독자)에 따라 섹션 구성이 달라진다
# ------------------------------------------------------------
@app.callback(
    Output("report-download", "data"),
    Output("action-echo", "children", allow_duplicate=True),
    Input("export-pdf-btn", "n_clicks"),
    Input("export-xlsx-btn", "n_clicks"),
    State("report-audience-dd", "value"),
    State("filter-store", "data"),
    State("seg-store", "data"),
    prevent_initial_call=True,
)
def export_report(_pdf, _xlsx, audience, filters, seg_state):
    audience = audience or DEFAULT_AUDIENCE
    aud = REPORT_AUDIENCES[audience]
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    base = f"설비모니터링_보고서_{audience}_{stamp}"
    try:
        if ctx.triggered_id == "export-pdf-btn":
            payload, name, mime = build_report_pdf(filters, seg_state, audience), f"{base}.pdf", "application/pdf"
        else:
            payload, name, mime = (
                build_report_xlsx(filters, seg_state, audience), f"{base}.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    except Exception as exc:  # noqa: BLE001 — 실패 이유를 화면에 그대로 보여준다
        return no_update, f"내보내기 실패 — {type(exc).__name__}: {exc}"
    return (dcc.send_bytes(lambda b: b.write(payload), name, type=mime),
            f"{aud['label']} 보고서 내보냄 — {name} (섹션 {len(aud['sections'])}개 · 값은 {NO_DATA_MARK})")


if __name__ == "__main__":
    # 최신 Dash(2.17+)는 app.run, 이전 버전은 app.run_server를 쓴다.
    app.run(debug=True)
