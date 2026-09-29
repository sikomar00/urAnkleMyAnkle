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
import sys
from datetime import datetime, timedelta
from pathlib import Path

import plotly.graph_objects as go
from dash import Dash, html, dcc, dash_table, Input, Output, State, ALL, MATCH, ctx, no_update
from plotly.subplots import make_subplots

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "src"

from .dashboard_data import (  # noqa: E402
    CLASSIFICATION_METRIC_COLUMNS,
    export_table_csv,
    load_asset_detail_kpis,
    load_asset_failure_onset_trend,
    load_asset_family_diagnosis,
    load_asset_list,
    load_asset_parts_history,
    load_asset_peer_comparison,
    load_asset_sensor_series,
    load_current_classification_metrics,
    load_data_dictionary,
    load_data_quality_summary,
    load_data_reference_date,
    load_failure_trend,
    load_priority_table,
    load_screen1_kpis,
    load_screen5_kpis,
    load_source_info,
    load_table_page,
)

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

/* 드롭다운 팝업이 카드 밑에 깔리는 문제 방지 (Dash 버전 호환).
   이 환경(Dash 4.x)의 dcc.Dropdown은 position:fixed + z-index:500인
   팝업(.dash-dropdown-content)을 쓰지만, 설치된 Dash 버전이 다르면
   z-index 없는 position:absolute짜리 구버전 react-select 팝업
   (.Select-menu-outer 등)을 쓸 수 있다 — 그 경우 헤더/필터바가
   DOM에서 main보다 먼저 나와 z-index 없이는 뒤에 그려지는 카드가
   위로 깔린다. 두 세대 모두 항상 최상단에 오도록 강제한다. */
.dash-dropdown-content, .Select-menu-outer { z-index: 1000 !important; }
.Select-menu-outer { position: absolute !important; }
"""

# go.Figure는 var(--pf-*) CSS 변수를 안정적으로 못 읽으므로, Plotly 차트
# 전용으로만 THEME_CSS의 라이트/다크 hex 값을 그대로 복사해 둔다 — 새 색상
# 토큰이 아니라 기존 값의 사본이다.
FIGURE_COLORS = {
    "light": {"ink": "#0b0b0b", "ink2": "#52514e", "muted": "#68665f",
              "hair": "#e1e0d9", "card": "#fcfcfb"},
    "dark": {"ink": "#f4f4f1", "ink2": "#bdbcb6", "muted": "#9d9b95",
             "hair": "#34342f", "card": "#1b1b19"},
}

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
# 필터바 상호작용 — 확인된 실제 값(2026-09-27 기준
# synthetic_industrial_machine_data.csv 고유값).
# ------------------------------------------------------------
PLANT_OPTIONS = ["CHN-02", "DHR-03", "PUN-01"]
MACHINE_TYPE_OPTIONS = ["Belt Conveyor", "CNC Lathe", "EOT Crane",
                        "Hydraulic Press", "Screw Compressor"]
MACHINE_OPTIONS = load_asset_list()
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
    "task": ["현재 고장 표시 분류", "부품군 진단", "단독 부품 이상 탐지"],
    "threshold": ["12", "13", "14"],
    "dataset": ["원자료", "기계·일 집계", "부품 출고", "모델 입력 피처"],
}
DEFAULT_SEG = {"task": 0, "threshold": 0, "dataset": 0}

# ① 화면의 "점검 우선순위" 표 정렬 축. KPI "종합 고장율" 타일을 클릭하면
# "grade"(등급가중 고장점수, 기본) → "threshold"(기준선 초과)로 바뀐다.
DEFAULT_PRIO_SORT = "grade"

# ①에만 있는 타일이라 패턴 매칭 id를 쓴다 — 이유는 tile() 호출부 주석 참고.
KPI_FAILRATE_ID = {"type": "kpi-drill", "index": "failrate"}


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


def tile(label, w=300, h=96, sub="값 · value-28", tid=None):
    """tid를 주면 클릭 가능한 KPI 타일이 된다 — html.Div도 n_clicks를 받으므로
    버튼으로 바꾸지 않아도 된다. 클릭 가능하다는 걸 시각적으로도 드러내려고
    커서와 살짝 진한 테두리를 준다."""
    style = {"width": f"{w}px", "height": f"{h}px", "flexShrink": "0", "boxSizing": "border-box",
             "background": CARD, "outline": f"1px solid {CTRL if tid else HAIR}", "outlineOffset": "-1px",
             "borderRadius": "4px", "padding": "16px", "display": "flex",
             "flexDirection": "column", "gap": "8px",
             "cursor": "pointer" if tid else "default"}
    kwargs = {"id": tid, "n_clicks": 0} if tid else {}
    return html.Div(
        [
            html.Div([html.Span(label, style=LABEL_12), dim(w, h)],
                      style={"height": "16px", "display": "flex", "justifyContent": "space-between", "gap": "8px"}),
            slot(sub, 168, 32),
        ],
        style=style, **kwargs,
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
         html.Div([html.Span("데이터 기준일", style=LABEL_12),
                   html.Span(load_data_reference_date(), style=NUM_12)],
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
         dcc.Download(id="report-download"),
         dcc.Download(id="table-download")],
        style={"display": "flex", "alignItems": "center", "justifyContent": "flex-end", "gap": "8px",
               # overflow:hidden을 여기 두면 "보고서 대상" 드롭다운 팝업까지
               # 잘라버릴 수 있다(설치된 Dash 버전에 따라 팝업이 position:fixed가
               # 아니라 absolute일 수 있음) — 가로 폭은 이미 자식 실측으로
               # 맞춰뒀으니 clip 없이도 넘치지 않는다.
               "minWidth": "0"},
    )
    return html.Header(
        [left, center, right],
        style={"width": f"{CANVAS_W}px", "height": f"{HEADER_H}px", "boxSizing": "border-box",
               "padding": f"0 {MARGIN}px", "background": CARD, "borderBottom": f"1px solid {HAIR}",
               # center를 max-content로 못박아야 탭 5개가 잘리지 않는다
               # (auto면 1fr 두 열이 먼저 자리를 가져가 탭이 깎인다).
               "display": "grid", "gridTemplateColumns": "minmax(0, 1fr) max-content minmax(0, 1fr)",
               "alignItems": "center", "gap": "16px", "fontFamily": SANS, "color": INK,
               # 헤더/필터바가 DOM에서 main보다 먼저 나오는데, 설치된 Dash
               # 버전에 따라 드롭다운 팝업이 z-index 없는 position:absolute로
               # 뜨는 구버전 컴포넌트를 쓸 수도 있다 — 그 경우 position:static인
               # 두 요소는 "나중에 그려지는 쪽이 위" 규칙을 따르므로 main(카드들)이
               # 위로 깔린다. 헤더 자체에 명시적 position+z-index를 줘서 어느
               # Dash 버전이든 항상 main보다 위에 그려지게 만든다.
               "position": "relative", "zIndex": 30},
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
               # overflow:hidden을 쓰면 자식(드롭다운 팝업 메뉴)까지 clip돼서
               # 목록이 잘리거나 다른 카드 밑에 깔린 것처럼 보인다. 가로 폭은
               # 이미 자식 폭 실측으로 1920px에 맞춰뒀으니(각 자식 flexShrink:0)
               # overflow는 넣지 않고 줄바꿈만 막는다.
               "flexWrap": "nowrap",
               # app_header()와 같은 이유 — 설치된 Dash 버전이 옛 방식(z-index
               # 없는 position:absolute) 드롭다운을 쓸 경우를 대비해 필터바
               # 자체를 main보다 항상 위에 오도록 고정한다.
               "position": "relative", "zIndex": 20},
    )


# ============================================================
# 화면 ① 현황 — 행 96 / 460 / 340
# ============================================================

def kpi_value_tile(label, value_text, w=300, h=96, tid=None):
    """tile()과 같은 치수·토큰을 쓰지만, slot()의 'w×h' 주석 대신 실제 값
    문자열을 그대로 보여준다. tile()/slot()은 화면 ③·④에서도 쓰는 공용
    와이어프레임 자리표시자라 여기서는 건드리지 않고, 화면 ①에서만 쓰는
    이 함수로 실제 값 표시를 대신한다."""
    style = {"width": f"{w}px", "height": f"{h}px", "flexShrink": "0", "boxSizing": "border-box",
             "background": CARD, "outline": f"1px solid {CTRL if tid else HAIR}", "outlineOffset": "-1px",
             "borderRadius": "4px", "padding": "16px", "display": "flex",
             "flexDirection": "column", "gap": "8px",
             "cursor": "pointer" if tid else "default"}
    kwargs = {"id": tid, "n_clicks": 0} if tid else {}
    return html.Div(
        [html.Div([html.Span(label, style=LABEL_12), dim(w, h)],
                   style={"height": "16px", "display": "flex", "justifyContent": "space-between", "gap": "8px"}),
         html.Div(html.Span(value_text, style={"fontFamily": MONO, "fontSize": "22px",
                                                 "fontWeight": "600", "color": INK}),
                  style={"height": "32px", "display": "flex", "alignItems": "center"})],
        style=style, **kwargs,
    )


def _trend_figure(df, theme):
    """화면 ⑤ "고장·위험 추세" 카드용 이중 y축 라인 차트. 새 색상 토큰 없이
    FIGURE_COLORS(THEME_CSS 값의 사본)만 써서 그레이스케일로 그린다."""
    colors = FIGURE_COLORS.get(theme, FIGURE_COLORS["light"])
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=df["transaction_date"], y=df["failure_days"],
                              name="고장 표시 건수", mode="lines",
                              line=dict(color=colors["ink"], width=1.6)),
                  secondary_y=False)
    fig.add_trace(go.Scatter(x=df["transaction_date"], y=df["high_risk_pct"],
                              name="위험 기준선 초과 비율 (%)", mode="lines",
                              line=dict(color=colors["ink2"], width=1.6, dash="dash")),
                  secondary_y=True)
    fig.update_layout(
        paper_bgcolor=colors["card"], plot_bgcolor=colors["card"],
        font=dict(color=colors["muted"], size=11),
        margin=dict(l=48, r=48, t=8, b=32),
        legend=dict(orientation="h", y=1.14, x=0),
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=False, color=colors["muted"])
    fig.update_yaxes(title_text="건수", gridcolor=colors["hair"], color=colors["muted"],
                      secondary_y=False)
    fig.update_yaxes(title_text="%", showgrid=False, color=colors["muted"],
                      secondary_y=True)
    return fig


# 화면 ② 스몰 멀티플 — (라벨, _daily() 컬럼) 쌍. 순서는 SENSOR_COLUMNS와 같다.
SMULT_SENSORS = [("베어링 온도 (°C)", "temp_bearing_degC"),
                 ("모터 온도 (°C)", "temp_motor_degC"),
                 ("수평 진동 (mm/s)", "vibration_h_mms"),
                 ("수직 진동 (mm/s)", "vibration_v_mms"),
                 ("오일 압력 (bar)", "oil_pressure_bar"),
                 ("부하율 (%)", "load_pct"),
                 ("회전 속도 (rpm)", "shaft_rpm"),
                 ("소비 전력 (kW)", "power_consumption_kw")]
# 카드 본문 452 = 패널 48×8 + 간격 4×7 + 공유 x축 40. 왼쪽 라벨 열과 차트 내부
# 패널이 같은 치수를 써야 1:1로 정렬되므로 상수로 묶어 둔다.
SMULT_PANEL_H, SMULT_GAP, SMULT_AXIS_H = 48, 4, 40
SMULT_PLOT_H = SMULT_PANEL_H * len(SMULT_SENSORS) + SMULT_GAP * (len(SMULT_SENSORS) - 1)
SMULT_BODY_H = SMULT_PLOT_H + SMULT_AXIS_H


def _true_runs(flags):
    """불리언 시리즈의 연속 True 구간을 (시작 인덱스, 끝 인덱스) 목록으로
    묶는다 — 하루당 shape 하나씩 만들지 않기 위해."""
    flags = flags.to_numpy()
    runs, start = [], None
    for i, on in enumerate(flags):
        if on and start is None:
            start = i
        elif not on and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(flags) - 1))
    return runs


def _band_shapes(dates, flags, color):
    """위험 구간을 8개 패널 전체를 관통하는 세로 밴드(shape)로 만든다.

    날짜를 numpy datetime64 그대로 넘기면 Plotly가 나노초 정수로 직렬화해
    ISO 문자열인 trace x축과 어긋나므로(축이 NaN이 된다) isoformat 문자열로
    넘긴다. 또 하루짜리 구간은 x0 == x1이면 폭 0이라 보이지 않으므로
    앞뒤로 반나절씩 넓혀 날짜 눈금 가운데에 오게 한다."""
    half = timedelta(hours=12)
    return [dict(type="rect", xref="x", yref="paper", y0=0, y1=1,
                 x0=(dates.iloc[a] - half).isoformat(),
                 x1=(dates.iloc[b] + half).isoformat(),
                 fillcolor=color, opacity=0.18, line_width=0, layer="below")
            for a, b in _true_runs(flags)]


def _smult_figure(df, theme):
    """화면 ② "센서 8종 스몰 멀티플" — 센서 8종 스파크라인(축·격자·범례 없음,
    맨 아래 공유 x축만) + 위험 기준선 초과일 세로 밴드. 밴드는 yref="paper"라
    8개 패널과 그 사이 간격까지 하나로 관통한다."""
    colors = FIGURE_COLORS.get(theme, FIGURE_COLORS["light"])
    fig = make_subplots(rows=len(SMULT_SENSORS), cols=1, shared_xaxes=True,
                        vertical_spacing=SMULT_GAP / SMULT_PLOT_H)
    for r, (_, column) in enumerate(SMULT_SENSORS, start=1):
        fig.add_trace(go.Scatter(x=df["transaction_date"], y=df[column], mode="lines",
                                  line=dict(color=colors["ink"], width=0.8),
                                  showlegend=False, hoverinfo="skip"),
                      row=r, col=1)
    # 밴드는 shapes로 한 번에 넘긴다 — 구간마다 add_vrect()를 호출하면 호출마다
    # figure 전체를 다시 검증해 2분이 넘게 걸린다(일괄 할당은 30ms 수준).
    fig.update_layout(
        height=SMULT_BODY_H, margin=dict(l=0, r=0, t=0, b=SMULT_AXIS_H),
        paper_bgcolor=colors["card"], plot_bgcolor=colors["card"],
        font=dict(color=colors["muted"], size=11),
        shapes=_band_shapes(df["transaction_date"], df["is_high_risk_day"], colors["muted"]),
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_xaxes(visible=True, showgrid=False, color=colors["muted"],
                      row=len(SMULT_SENSORS), col=1)
    return fig


def _parts_figure(df, theme):
    """화면 ② "부품 출고 이력" — 부품별 출고 금액 상위 10개 가로 막대.
    금액 내림차순이 위로 오도록 autorange="reversed"."""
    colors = FIGURE_COLORS.get(theme, FIGURE_COLORS["light"])
    fig = go.Figure(go.Bar(
        x=df["total_issue_value_inr"], y=df["part_no"], orientation="h",
        marker_color=colors["ink"],
        text=[f"{v:,.0f}" for v in df["total_issue_value_inr"]],
        textposition="outside", textfont=dict(color=colors["muted"], size=11),
        cliponaxis=False,
        customdata=df["part_description"],
        hovertemplate="%{y} · %{customdata}<br>%{text} INR<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed", showgrid=False, tickfont=dict(size=11))
    fig.update_xaxes(visible=False, range=[0, df["total_issue_value_inr"].max() * 1.35])
    fig.update_layout(
        height=132, margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor=colors["card"], plot_bgcolor=colors["card"],
        font=dict(size=11, color=colors["muted"]), bargap=0.28, showlegend=False,
    )
    return fig


def _hex_to_rgba(hex_color, alpha):
    """FIGURE_COLORS의 hex 값을 박스플롯 fillcolor용 rgba 문자열로 바꾼다 —
    새 색상 값을 만드는 게 아니라 기존 hex를 그대로 반투명하게 쓰는 것뿐이다."""
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{alpha})"


def _peer_figure(comparison, theme):
    """화면 ② "동종 기계 대비" — 선택 자산 vs 동종 나머지 1대의 베어링 온도
    분포 박스플롯 2개. 선택 기계는 진한 테두리·채움, 동종 기계는 옅게 그려
    어느 쪽이 선택 기계인지 시각적으로 드러낸다."""
    colors = FIGURE_COLORS.get(theme, FIGURE_COLORS["light"])
    fig = go.Figure()
    fig.add_trace(go.Box(
        y=comparison["asset_values"], name=f"{comparison['asset_tag']} (선택)",
        line=dict(color=colors["ink"], width=2),
        fillcolor=_hex_to_rgba(colors["ink"], 0.55),
        marker_color=colors["ink"], boxpoints=False,
    ))
    fig.add_trace(go.Box(
        y=comparison["peer_values"], name=f"{comparison['peer_asset_tag']} (동종)",
        line=dict(color=colors["muted"], width=1),
        fillcolor=_hex_to_rgba(colors["muted"], 0.12),
        marker_color=colors["muted"], boxpoints=False,
    ))
    fig.update_yaxes(gridcolor=colors["hair"], tickfont=dict(size=11), color=colors["muted"])
    fig.update_xaxes(tickfont=dict(size=11), color=colors["muted"])
    fig.update_layout(
        height=220, margin=dict(l=32, r=8, t=4, b=22),
        paper_bgcolor=colors["card"], plot_bgcolor=colors["card"],
        font=dict(size=11, color=colors["muted"]), showlegend=False,
    )
    return fig


def _pre_figure(trend, theme):
    """화면 ② "고장 직전 센서 변화" — 가장 최근 위험 기준선 에피소드의
    t-7~t 베어링 온도 추이. x축은 실제 날짜 대신 상대 위치(t-7..t)로
    표시하고, 실제 날짜는 호버 툴팁에서만 보여준다."""
    colors = FIGURE_COLORS.get(theme, FIGURE_COLORS["light"])
    labels = [f"t{d}" if d != 0 else "t" for d in trend["relative_days"]]
    fig = go.Figure(go.Scatter(
        x=labels, y=trend["temp_bearing_degC"], mode="lines+markers",
        line=dict(color=colors["ink"], width=1.6), marker=dict(size=5, color=colors["ink"]),
        customdata=trend["dates"],
        hovertemplate="%{x} · %{customdata}<br>%{y}°C<extra></extra>",
    ))
    fig.add_vline(x="t", line=dict(color=colors["muted"], width=1, dash="dash"))
    fig.update_yaxes(gridcolor=colors["hair"], tickfont=dict(size=11), color=colors["muted"])
    fig.update_xaxes(showgrid=False, tickfont=dict(size=11), color=colors["muted"])
    fig.update_layout(
        height=220, margin=dict(l=32, r=8, t=4, b=22),
        paper_bgcolor=colors["card"], plot_bgcolor=colors["card"],
        font=dict(size=11, color=colors["muted"]), showlegend=False,
    )
    return fig


def priority_table(cols, records, row_h, head_h=32, sort_col=None):
    """table_placeholder()와 같은 헤더/셀 스타일을 쓰되, 자리표시 막대 대신
    load_priority_table()이 만든 실제 자산별 값을 채운다. table_placeholder()
    자체는 화면 ③·④에서도 쓰므로 건드리지 않는다."""
    thead = html.Tr(
        [html.Th(f"{l}{' ▼' if i == sort_col else (' ▲▼' if sort_col is not None else '')}",
                 style={"width": f"{w}px", "boxSizing": "border-box", "padding": "0 8px",
                        "textAlign": a, **LABEL_12, "whiteSpace": "nowrap", "overflow": "hidden",
                        "borderBottom": f"1px solid {CTRL}"})
         for i, (l, w, a) in enumerate(cols)],
        style={"height": f"{head_h}px", "background": CARD},
    )
    field_order = ["rank", "asset_tag", "machine_type", "plant_code", "failure_points",
                   "threshold_exceeded", "last_failure_date", "failed_part_count"]
    body_rows = []
    for record in records:
        cells = []
        for (l, w, a), field in zip(cols, field_order):
            value = record[field]
            if field == "failure_points":
                text = f"{value:,.0f}"
            elif field == "threshold_exceeded":
                text = "예" if value else "아니오"
            elif field == "last_failure_date":
                text = value or "—"
            else:
                text = str(value)
            cell_style = {"fontFamily": MONO, "fontSize": "12px", "color": MUTED} if field == "rank" else NUM_12
            cells.append(html.Td(html.Span(text, style=cell_style),
                                  style={"boxSizing": "border-box", "padding": "0 8px",
                                         "textAlign": a, "borderBottom": f"1px solid {HAIR}"}))
        body_rows.append(html.Tr(cells, style={"height": f"{row_h}px"}))
    tw = sum(c[1] for c in cols)
    return html.Table(
        [html.Thead(thead), html.Tbody(body_rows)],
        style={"width": f"{tw}px", "tableLayout": "fixed", "borderCollapse": "collapse", "flexShrink": "0"},
    )


def screen_1(seg_state=None, audience=DEFAULT_AUDIENCE, prio_sort=None):
    # 절충안: 6타일 중 "위험 기준선 초과 기계 (대)" 한 자리만 종합 고장율(%)로
    # 바꾼다. 그 지표는 원래 절대 건수(분자)만 보여줘서 "전체 대비 얼마나
    # 심각한가"를 암산해야 했는데, 비율로 바꾸면 그 계산이 필요 없어진다.
    # 나머지 5개(관측 기계·고장 표시 기계·일·전력·온도·부품 금액)는 그대로 —
    # 전부 원자료 수준 지표라 "지금 뭐가 몇 대냐"에 즉답하는 역할을 유지한다.
    #
    # 정의(확정): 종합 고장율(%) = 위험 기준선 초과 기계 수 ÷ 전체 관측 기계 수 × 100
    # — ③에서 고른 위험 기준선(등급가중 고장점수 12/13/14)을 그대로 물려받는다.
    #
    # 이 값은 "전체 중 몇 %가 위험선을 넘었나"라는 집계 하나뿐이라, 그것만으로는
    # "어느 공장·기계·부품이 원인인가"를 알 수 없다 — 그래서 새 카드를 더
    # 만드는 대신, 클릭하면 아래 "점검 우선순위" 표가 기준선 초과 기준으로
    # 다시 정렬되게 연결한다. 새 화면 공간을 쓰지 않고 이미 있는 표를 재사용.
    prio_sort = prio_sort or DEFAULT_PRIO_SORT
    # kpi-failrate-tile은 화면 ①에만 존재하고 ②~⑤로 넘어가면 DOM에서 사라진다.
    # 일반 문자열 id로 Input을 걸면 그 화면들에서 "ID not found in layout"
    # 콘솔 경고가 뜬다 — 패턴 매칭 id({"type":...})를 쓰면 Dash가 "지금 이
    # id를 가진 컴포넌트가 0개일 수 있다"를 정상 상태로 취급해 경고가 안 뜬다.
    kpis = load_screen1_kpis()
    kpi_specs = [
        ("관측 기계 (대)", f"{kpis['observed_machines']:,}", None),
        ("고장 표시 기계·일", f"{kpis['failure_machine_days']:,}", None),
        ("종합 고장율 (%)", f"{kpis['failure_rate_pct']:.1f}%", KPI_FAILRATE_ID),
        ("평균 소비 전력 (kW)", f"{kpis['avg_power_kw']:,.2f}", None),
        ("최고 베어링 온도 (°C)", f"{kpis['max_bearing_temp']:.1f}", None),
        ("부품 출고 금액 (누적, INR)", f"{kpis['parts_issue_value_inr']:,.0f}", None),
    ]
    row_a = row(ROW_KPI, [kpi_value_tile(label, value_text, tid=tid) for label, value_text, tid in kpi_specs])

    prio_cols = [("순위", 48, "right"), ("대상", 200, "left"), ("종류", 110, "left"), ("공장", 100, "left"),
                 ("등급가중 고장점수", 150, "right"), ("기준선 초과", 110, "center"),
                 ("최근 고장 표시일", 150, "left"), ("당일 고장 표시 부품 수", 190, "right")]
    sort_idx = 5 if prio_sort == "threshold" else 4
    sort_hint = ("정렬: 기준선 초과 (KPI '종합 고장율' 클릭으로 이동함)" if prio_sort == "threshold"
                 else "정렬: 등급가중 고장점수 (기본)")
    priority_records = load_priority_table(prio_sort)
    prio = card("점검 우선순위", 1090, ROW_MAIN,
                priority_table(prio_cols, priority_records, 32, head_h=32, sort_col=sort_idx),
                right=note(f"대상 = 엔티티 무관 (현재 기계 행만) · 헤더 고정 · 행 클릭 → ② · {sort_hint}"))

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

def screen_2(seg_state=None, audience=DEFAULT_AUDIENCE, asset_tag=None):
    def value_box(text, w, h=26, mono=False):
        """slot()의 점선 테두리/치수 주석 대신 실제 값을 그대로 보여준다 —
        strip 전용, screen_2() 안에서만 쓰인다."""
        return html.Div(
            html.Span(text, style={"fontFamily": MONO if mono else "inherit", "fontSize": "13px",
                                    "color": INK, "whiteSpace": "nowrap", "overflow": "hidden",
                                    "textOverflow": "ellipsis"}),
            style={"width": f"{w}px", "height": f"{h}px", "boxSizing": "border-box",
                   "display": "flex", "alignItems": "center", "overflow": "hidden"},
        )

    def metric(label, w, value_text):
        return html.Div([html.Span(label, style={**LABEL_12, "whiteSpace": "nowrap"}),
                          value_box(value_text, w)],
                         style={"width": f"{w}px", "display": "flex", "flexDirection": "column", "gap": "4px"})

    def nav_btn(label, index):
        """공용 btn()은 id 타입이 항상 "ghost-btn"이라 전역 echo_action이
        같이 반응해 '미구현' 문구를 띄운다. 여기서는 실제 기계 전환 콜백만
        반응하도록 다른 id 타입을 쓴다."""
        return html.Button(label, id={"type": "machine-nav-btn", "index": index},
                            n_clicks=0, style=btn_style())

    assets = load_asset_list()
    asset_tag = asset_tag if asset_tag in assets else assets[0]
    detail = load_asset_detail_kpis(asset_tag)

    strip = html.Section(
        [nav_btn("‹ 이전 기계", "prev"),
         html.Div([value_box(detail["asset_tag"], 220, mono=True),
                   value_box(f"{detail['machine_type']} · {detail['plant_code']}", 220, h=22)],
                  style={"display": "flex", "flexDirection": "column", "gap": "4px"}),
         nav_btn("다음 기계 ›", "next"),
         html.Div(style={"flexGrow": "1"}),
         metric("현재 등급", 120, detail["current_grade"]),
         metric("위험도", 120, f"{detail['risk_score']:,.0f}"),
         metric("최근 고장 표시일", 140, detail["last_failure_date"] or "—"),
         metric("고장 표시 일수 (일)", 140, f"{detail['failure_days_count']:,}"),
         metric("평균 소비 전력 (kW)", 140, f"{detail['avg_power_30d_kw']:,.2f}"),
         dim(1880, 88)],
        style={"width": "1880px", "height": "88px", "boxSizing": "border-box", "background": CARD,
               "outline": f"1px solid {HAIR}", "outlineOffset": "-1px", "borderRadius": "4px",
               "padding": "16px", "display": "flex", "alignItems": "center", "gap": "16px"},
    )

    # 라벨 열은 HTML로 두고 오른쪽 차트만 Plotly로 그린다. 패널 높이·간격이
    # _smult_figure()의 subplot 치수(SMULT_*)와 같아야 1:1로 정렬된다.
    sensor_labels = html.Div(
        [html.Div(label, style={"height": f"{SMULT_PANEL_H}px", "flexShrink": "0",
                                 "display": "flex", "alignItems": "center", **LABEL_12})
         for label, _ in SMULT_SENSORS]
        + [html.Div(note("8 × 48 + 7 × 4 + 40 = 452"),
                    style={"height": f"{SMULT_AXIS_H}px", "flexShrink": "0", "display": "flex",
                           "alignItems": "center"})],
        style={"width": "160px", "flexShrink": "0", "display": "flex", "flexDirection": "column",
               "gap": f"{SMULT_GAP}px"},
    )
    sm_body = hstack(
        [sensor_labels,
         dcc.Graph(id={"type": "smult-chart", "index": "screen2"},
                   figure=_smult_figure(load_asset_sensor_series(asset_tag), "light"),
                   config={"displayModeBar": False, "responsive": True},
                   style={"flex": "1 1 auto", "minHeight": "0", "minWidth": "0"})],
        8, style={"height": f"{SMULT_BODY_H}px"},
    )
    smult = card("센서 8종 스몰 멀티플", 1248, 520, sm_body,
                 right=note("x축 공유 · 밴드 = 위험 기준선 초과일"))

    anom = card("이상 점수 추이", 616, 144, slot("라인 + 임계선", 584, 76), right=note("임계선 포함"))
    clus = card("군집 위치", 616, 144, slot("산점도 · 군집 1 / 2 / 3", 584, 76), right=note("선택 기계 표시"))
    parts = card("부품 출고 이력", 616, 200,
                 dcc.Graph(id={"type": "parts-chart", "index": "screen2"},
                           figure=_parts_figure(load_asset_parts_history(asset_tag), "light"),
                           config={"displayModeBar": False, "responsive": True},
                           style={"flex": "1 1 auto", "minHeight": "0", "minWidth": "0"}))
    rightcol = col(616, 520, [anom, clus, parts])
    row_b = row(520, [smult, rightcol])

    pre = card("고장 직전 센서 변화", 932, 288,
               dcc.Graph(id={"type": "pre-chart", "index": "screen2"},
                         figure=_pre_figure(load_asset_failure_onset_trend(asset_tag), "light"),
                         config={"displayModeBar": False, "responsive": True},
                         style={"flex": "1 1 auto", "minHeight": "0", "minWidth": "0"}),
               right=note("고장 표시 시점 t=0 · t−7 ~ t"))
    peer = card("동종 기계 대비", 932, 288,
                dcc.Graph(id={"type": "peer-chart", "index": "screen2"},
                          figure=_peer_figure(load_asset_peer_comparison(asset_tag), "light"),
                          config={"displayModeBar": False, "responsive": True},
                          style={"flex": "1 1 auto", "minHeight": "0", "minWidth": "0"}),
                right=note("동일 종류 나머지 1대 대비"))
    row_c = row(288, [pre, peer])

    return html.Div([strip, row_b, row_c],
                     style={"display": "flex", "flexDirection": "column", "gap": f"{GUTTER}px"})


# ============================================================
# 화면 ③ 모델·예측 — 툴바 32 / 행 96 / 460 / 272 (마지막 행이 남는 높이 흡수)
# ============================================================

def screen_3(seg_state=None, audience=DEFAULT_AUDIENCE, asset_tag=None):
    seg_state = seg_state or DEFAULT_SEG
    thr_sel = seg_state.get("threshold", 0)
    thr_changed = thr_sel != DEFAULT_SEG["threshold"]
    task_sel = seg_state.get("task", 0)
    assets = load_asset_list()
    asset_tag = asset_tag if asset_tag in assets else assets[0]

    if task_sel == 0:
        # 이 과제(부품 단위 당일 고장 분류)에는 위험 기준선이 적용되지
        # 않는다 — 설비 단위 위험도 분류 과제(별도)에서만 쓰이므로, 세그먼트
        # 값과 무관하게 항상 이 안내를 상시 노출한다. "기본값 아님" 강조
        # 개념 자체가 이 과제엔 의미가 없어 항상 비강조 스타일로 고정한다.
        warn_text = ("위험 기준선(12/13/14)은 이 과제(부품 단위 당일 고장 분류)에는 "
                     "적용되지 않는다 — 설비 단위 위험도 분류 과제(별도)에서만 쓰인다.")
        warn_emphasis = False
    else:
        # 위험 기준선은 표의 축이 아니라 '라벨 정의'다 — 바꾸면 정답이 바뀌므로
        # 모델을 다시 학습/교체해야 한다. 관리자가 필터로 착각하고 눌렀다가
        # 화면 전체 수치가 갈리는 것이 원래 문제였다. 경고를 컨트롤 바로 아래에
        # 상시 노출하고, 기본값에서 벗어나면 문구를 강조로 바꾼다.
        warn_text = ("주의 — 위험 기준선은 필터가 아니라 학습 라벨 정의다. "
                     f"값을 바꾸면 해당 기준선으로 학습된 모델로 교체되고 아래 모든 지표가 다시 계산된다."
                     + (f"  현재 기본값({SEG_GROUPS['threshold'][DEFAULT_SEG['threshold']]}) 아님 "
                        f"→ {SEG_GROUPS['threshold'][thr_sel]} 기준 모델" if thr_changed else ""))
        warn_emphasis = thr_changed
    warn_line = html.Div(
        html.Span(warn_text,
                  style={"fontSize": "11px", "lineHeight": "16px",
                         "fontWeight": "600" if warn_emphasis else "500",
                         "color": INK if warn_emphasis else MUTED, "whiteSpace": "nowrap",
                         "boxSizing": "border-box", "padding": "0 6px",
                         "border": f"1px solid {CTRL if warn_emphasis else HAIR}", "borderRadius": "2px"}),
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
    #   과제 0(현재 고장 표시 분류)은 실제 값(hist_gradient_boosting 기준) 타일 6개.
    #   과제 1(부품군 진단)은 KPI 타일 행 자체를 만들지 않는다 — mcomp 카드
    #   자리에 부품군별 표가 대신 들어간다.
    #   과제 2(단독 부품 이상 탐지)는 아직 자리표시자 그대로(타일 4개 × 458).
    kpi_labels = ["정확도", "정밀도", "재현율", "F1", "ROC-AUC", "평균정밀도(AP)"]
    if task_sel == 2:
        row_a = row(ROW_KPI, [tile(f"[지표 {i+1}]", w=458, sub="값 또는 빈 상태") for i in range(4)])
    elif task_sel == 0:
        current_metrics = load_current_classification_metrics()["hist_gradient_boosting"]
        row_a = row(ROW_KPI, [
            kpi_value_tile(label, f"{current_metrics[col]:.3f}")
            for label, col in zip(kpi_labels, CLASSIFICATION_METRIC_COLUMNS)
        ])
    else:
        row_a = html.Div()

    if task_sel == 1:
        # 부품군 진단 — mcomp 카드 자리에 선택 자산의 부품군 9종별 표를 넣는다
        # (기존 '모델 비교' 표와는 열 구성 자체가 달라 별도로 만든다).
        family_rows = load_asset_family_diagnosis(asset_tag)
        family_cols = [
            ("부품군", 130, "left"), ("모델", 150, "left"), ("표본 수", 64, "right"),
            ("양성률(%)", 80, "right"), ("정밀도", 82, "right"), ("재현율", 82, "right"),
            ("AP", 72, "right"), ("ROC-AUC", 82, "right"),
        ]
        def family_cell(content, align="left"):
            # boxSizing 없이는 padding이 지정한 width 위에 더해져 8열이 카드
            # 폭(742px)을 넘기고 overflow:hidden에 잘린다(기존 '모델 비교'
            # 표에도 있던 문제이나, 그 표는 이번 범위 밖이라 손대지 않는다).
            return html.Td(content, style={"width": "auto", "padding": "0 8px", "textAlign": align,
                                            "boxSizing": "border-box",
                                            "borderBottom": f"1px solid {HAIR}"})
        family_body_rows = [
            html.Tr([
                family_cell(html.Span(fr["part_family"], style={"fontSize": "13px", "color": INK})),
                family_cell(html.Span(fr["model"], style={"fontSize": "13px", "color": INK})),
                family_cell(str(fr["support"]), align="right"),
                family_cell(f"{fr['positive_rate'] * 100:.1f}%", align="right"),
                family_cell(f"{fr['precision']:.3f}", align="right"),
                family_cell(f"{fr['recall']:.3f}", align="right"),
                family_cell(f"{fr['average_precision']:.3f}", align="right"),
                family_cell(f"{fr['roc_auc']:.3f}", align="right"),
            ], style={"height": "32px"})
            for fr in family_rows
        ]
        family_thead = html.Tr([html.Th(l, style={"width": f"{w}px", "padding": "0 8px", "textAlign": a,
                                                    "boxSizing": "border-box",
                                                    **LABEL_12, "whiteSpace": "nowrap",
                                                    "borderBottom": f"1px solid {CTRL}"})
                                 for l, w, a in family_cols], style={"height": "32px"})
        family_table = html.Table([html.Thead(family_thead), html.Tbody(family_body_rows)],
                                   style={"width": "742px", "tableLayout": "fixed", "borderCollapse": "collapse"})
        mcomp_body = html.Div([family_table])
        mcomp = card("부품군 진단", 774, ROW_MAIN, mcomp_body, right=note("행 = 부품군 · 열 = 지표"))
    else:
        metric_cols = [("모델", 182, "left")] + [(f"[지표 {i+1}]", 93, "right") for i in range(6)]
        model_rows = []
        if task_sel == 0:
            comparison_metrics = load_current_classification_metrics()
            model_names = ["prior", "hist_gradient_boosting"]
        else:
            comparison_metrics = None
            model_names = ["기준 A", "기준 B", "RandomForest"]
        for name in model_names:
            cells = [html.Td(
                html.Div([html.Span(style={"width": "10px", "height": "10px", "border": f"1px solid {CTRL}",
                                            "boxSizing": "border-box"}),
                          html.Span(name, style={"fontSize": "13px", "color": INK})],
                         style={"display": "flex", "alignItems": "center", "gap": "8px"}),
                style={"padding": "0 8px", "borderBottom": f"1px solid {HAIR}"})]
            if comparison_metrics is not None:
                row_pct = [comparison_metrics[name][col] * 100 for col in CLASSIFICATION_METRIC_COLUMNS]
            else:
                row_pct = [45] * 6
            cells += [html.Td(html.Div(bar(pct, 8), style={"display": "flex", "justifyContent": "flex-end"}),
                              style={"padding": "0 8px", "borderBottom": f"1px solid {HAIR}"}) for pct in row_pct]
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

    if task_sel == 1:
        pr_body = empty_state("이 과제에는 해당 없음", "부품군 진단 과제에서는 표시하지 않음", 584, 392)
    else:
        pr_body = html.Div([slot("범례 · 모델 n + 무작위 기준선", 584, 20),
                            hstack([slot("y축 · 정밀도", 40, 342), slot("PR 곡선 영역", "가변", 342)], 0),
                            hstack([html.Div(style={"width": "40px"}), slot("x축 · 재현율", "가변", 22)], 0)])
    prc = card("PR 곡선", 616, ROW_MAIN, pr_body, right=note("모델 수만큼 + 무작위 기준선"))

    def cell(t):
        return slot(t, 169, 160)
    if task_sel == 1:
        cm_body = empty_state("이 과제에는 해당 없음", "부품군 진단 과제에서는 표시하지 않음", 426, 392)
    else:
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

    if task_sel == 1:
        feat_body = empty_state("이 과제에는 해당 없음", "부품군 진단 과제에서는 표시하지 않음", 742, 184)
    else:
        feat_body = html.Div(
            [html.Div([html.Div(bar(70, 5), style={"width": "160px", "flexShrink": "0"}),
                       html.Div(style={"flexGrow": "1", "height": "11px", "boxSizing": "border-box",
                                        "border": f"1px dashed {CTRL}", "background": SUNK}),
                       html.Div(bar(80, 5), style={"width": "48px", "flexShrink": "0"})],
                      style={"height": "15px", "display": "flex", "alignItems": "center", "gap": "8px",
                             "flexShrink": "0"})
             for _ in range(12)],
        )
    feat = card("주요 영향 변수 상위 12", 774, 252, feat_body, right=note("12행 × 15 · 직접 값 라벨"))

    slider = html.Div(
        [html.Label("판정 임계값 (즉시 재계산)", htmlFor="thr-slider", style={**LABEL_12, "whiteSpace": "nowrap"}),
         dcc.Slider(id="thr-slider", min=0, max=100, value=50, marks=None,
                    tooltip={"placement": "bottom"})],
        style={"display": "flex", "alignItems": "center", "gap": "8px", "width": "260px"},
    )
    if task_sel == 1:
        thr_body = empty_state("이 과제에는 해당 없음", "부품군 진단 과제에서는 표시하지 않음", 584, 184)
    else:
        thr_body = html.Div([hstack([slot("y축", 40, 162), slot("위험도 분포 히스토그램 + 임계값 세로선", "가변", 162)], 0),
                             hstack([html.Div(style={"width": "40px"}), slot("x축 · 위험도", "가변", 22)], 0)])
    thr = card("판정 임계값 조정", 616, 252, thr_body, right=slider)

    # 선행 경보 지표는 아직 어떤 과제에도 연결되지 않았다(과거에는 "7일
    # 사전 예측" 과제에서만 조건부로 활성화했으나, 그 과제 자체가 "부품군
    # 진단"으로 바뀌면서 대상이 없어졌다).
    lead_body = empty_state("이 과제에는 해당 없음",
                             "선행 경보 지표가 아직 어느 과제에도 연결되지 않음", 426, 184)
    lead = card("선행 경보 일수 분포", 458, 252, lead_body)
    row_c = row(252, [feat, thr, lead])

    return html.Div([toolbar, row_a, row_b, row_c],
                     style={"display": "flex", "flexDirection": "column", "gap": f"{GUTTER}px"})


# ============================================================
# 화면 ④ 데이터 — 툴바 32 / 524 / 340
# ============================================================

# SEG_GROUPS["dataset"] 인덱스 → dashboard_data의 dataset 키. 2(부품 출고)·
# 3(모델 입력 피처)은 아직 연결되지 않아 매핑하지 않는다.
DATASET_KEY_BY_SEG_INDEX = {0: "raw", 1: "daily"}


def _dtable_footer_text(total_rows, page, page_size=16):
    """"전체 행 수 · 표시 범위" 푸터 문구 — screen_4 초기 렌더와 정렬/페이지
    콜백이 공유한다."""
    if total_rows == 0:
        return "0행"
    start = page * page_size + 1
    end = min((page + 1) * page_size, total_rows)
    return f"{total_rows:,}행 중 {start:,}–{end:,}행 표시"


def screen_4(seg_state=None, audience=DEFAULT_AUDIENCE):
    seg_state = seg_state or DEFAULT_SEG
    dataset_index = seg_state.get("dataset", 0)
    dataset_key = DATASET_KEY_BY_SEG_INDEX.get(dataset_index)

    def csv_export_button(key, disabled):
        style = btn_style(extra={"opacity": "0.4", "cursor": "not-allowed"} if disabled else None)
        return html.Button(
            "CSV 내보내기", id={"type": "csv-export-btn", "index": key or "unavailable"},
            n_clicks=0, disabled=disabled, style=style,
        )

    if dataset_key is None:
        dtable_body = html.Div(
            "이 데이터셋은 아직 연결되지 않았습니다",
            style={"height": "456px", "display": "flex", "alignItems": "center",
                   "justifyContent": "center", **LABEL_12},
        )
        csv_btn = csv_export_button(None, disabled=True)
    else:
        page0 = load_table_page(dataset_key, None, "asc", 0)
        columns_prop = [{"name": c["label"], "id": c["id"]} for c in page0["columns"]]
        right_align_ids = [c["id"] for c in page0["columns"] if c["align"] == "right"]
        table = dash_table.DataTable(
            id={"type": "dtable", "index": dataset_key},
            columns=columns_prop,
            data=page0["data"],
            page_action="custom", page_current=0, page_count=page0["page_count"], page_size=16,
            sort_action="custom", sort_mode="single", sort_by=[],
            filter_action="none",
            # fixed_columns(앞 2열 고정)를 시도했으나 헤더 셀이 어긋나(빈 헤더가
            # 섞여 나옴) 포기하고 가로 스크롤만 남긴다(사용자 승인 — 이 문제에
            # 시간을 더 쓰지 않음).
            style_table={"overflowX": "auto", "width": "1840px"},
            style_header={"backgroundColor": CARD, "fontFamily": SANS, "fontSize": "12px",
                          "fontWeight": "500", "color": INK2, "borderBottom": f"1px solid {CTRL}",
                          "height": "32px", "minHeight": "32px", "maxHeight": "32px"},
            style_cell={"backgroundColor": CARD, "border": "none", "borderBottom": f"1px solid {HAIR}",
                        "padding": "0 8px", "height": "24px", "minHeight": "24px", "maxHeight": "24px",
                        "lineHeight": "16px", "fontFamily": SANS, "fontSize": "12px", "textAlign": "left"},
            style_data={"backgroundColor": CARD},
            style_cell_conditional=[
                {"if": {"column_id": cid}, "textAlign": "right", "fontFamily": MONO}
                for cid in right_align_ids
            ],
            # 내장 페이저 padding·여백을 줄여 카드 높이(524px) 예산에 맞춘다
            # (정정 #2 — 행 높이·page_size는 그대로 두고 페이저만 압축).
            css=[
                {"selector": ".previous-next-container", "rule": "padding:2px 0; margin:0;"},
                {"selector": ".previous-next-container button", "rule": "padding:2px 4px; margin:0 2px;"},
                {"selector": ".page-number, .current-page-container",
                 "rule": f"font-family:{MONO}; font-size:11px; color:{INK2}; margin:0 2px;"},
                # 진단으로 특정한 진짜 원인: dash_table 기본 번들 스타일시트의
                # ".dash-spreadsheet-inner tr { height:30px; min-height:30px; }"가
                # style_cell의 24px 지정을 무시하고 행 높이를 30px로 고정한다
                # (.dash-cell-value/-container 문제가 아님). 같은 선택자로
                # 이 표에만 재정의한다.
                {"selector": ".dash-spreadsheet-inner tr",
                 "rule": "height:24px; min-height:24px;"},
            ],
        )
        footer = html.Div(
            html.Span(_dtable_footer_text(page0["total_rows"], 0, 16),
                      id={"type": "dtable-footer", "index": dataset_key}, style=LABEL_12),
            style={"height": "20px", "display": "flex", "alignItems": "center"},
        )
        dtable_body = html.Div([table, footer])
        csv_btn = csv_export_button(dataset_key, disabled=False)

    toolbar = html.Div(
        [seg("dataset", "데이터셋", SEG_GROUPS["dataset"], sel=dataset_index),
         html.Div([csv_btn], style={"display": "flex", "alignItems": "center", "gap": "12px"})],
        style={"width": "1880px", "height": "32px", "flexShrink": "0", "display": "flex",
               "alignItems": "center", "justifyContent": "space-between"},
    )

    dtable = card(f"데이터 조회 · {SEG_GROUPS['dataset'][dataset_index]}", 1880, 524, dtable_body,
                  right=note("헤더 고정 · 서버 측 정렬·페이지네이션"))

    dict_cols = [("컬럼명", 130, "left"), ("타입", 64, "left"), ("단위", 56, "left"),
                 ("결측률 (%)", 72, "right"), ("설명", 120, "left")]

    def dict_table(records):
        """table_placeholder()와 같은 헤더 스타일이되, "설명" 칸만 말줄임+title
        툴팁을 쓴다 — screen_4 전용, table_placeholder() 자체는 건드리지 않는다."""
        thead = html.Tr(
            [html.Th(l, style={"width": f"{w}px", "boxSizing": "border-box", "padding": "0 8px",
                                "textAlign": a, **LABEL_12, "whiteSpace": "nowrap", "overflow": "hidden",
                                "borderBottom": f"1px solid {CTRL}"})
             for l, w, a in dict_cols],
            style={"height": "24px", "background": CARD},
        )
        body_rows = []
        for record in records:
            missing_pct = record["missing_pct"]
            missing_text = "—" if missing_pct is None else f"{missing_pct:.2f}"
            values = [record["column"], record["dtype_label"], record["unit"],
                      missing_text, record["description"]]
            cells = []
            for (l, w, a), text in zip(dict_cols, values):
                if l in ("컬럼명", "설명"):
                    span_style = {**NUM_12, "whiteSpace": "nowrap", "overflow": "hidden",
                                  "textOverflow": "ellipsis", "display": "block"}
                    span = html.Span(text, style=span_style, title=text)
                else:
                    span = html.Span(text, style=NUM_12)
                cells.append(html.Td(span, style={"boxSizing": "border-box", "padding": "0 8px",
                                                   "textAlign": a, "borderBottom": f"1px solid {HAIR}"}))
            body_rows.append(html.Tr(cells, style={"height": "22px"}))
        tw = sum(c[1] for c in dict_cols)
        return html.Table([html.Thead(thead), html.Tbody(body_rows)],
                           style={"width": f"{tw}px", "tableLayout": "fixed",
                                  "borderCollapse": "collapse", "flexShrink": "0"})

    def dict_half(records):
        return html.Div(dict_table(records), style={"width": "442px"})

    dict_rows = load_data_dictionary()
    ddict = card("데이터 사전 (22열)", 932, ROW_SUB,
                 hstack([dict_half(dict_rows[:11]), dict_half(dict_rows[11:])], 16),
                 right=note("11행 × 2단 · 행 22"))

    def info_box(text, w, h=94):
        """slot()의 점선 테두리 대신 실제 문장을 보여준다 — screen_4 전용."""
        return html.Div(
            html.Span(text, style={"fontSize": "12px", "lineHeight": "16px", "color": INK,
                                    "whiteSpace": "pre-line", "wordBreak": "keep-all"}),
            style={"width": f"{w}px", "height": f"{h}px", "boxSizing": "border-box", "overflow": "hidden"},
        )

    q = load_data_quality_summary()
    period_tile = html.Div(
        [html.Span(f"기간: {q['period_days']:,}일",
                    style={"fontSize": "12px", "lineHeight": "16px", "color": INK}),
         html.Span(f"{q['period_start']} ~ {q['period_end']}",
                    style={"fontSize": "12px", "lineHeight": "16px", "color": INK, "whiteSpace": "nowrap"})],
        style={"width": "213px", "height": "94px", "boxSizing": "border-box", "overflow": "hidden",
               "display": "flex", "flexDirection": "column"},
    )
    quality_tiles = [
        info_box(f"중복: 복합키(날짜·기계·부품) 중복 {q['composite_key_duplicates']:,}건 · "
                 f"완전 중복 {q['full_duplicates']:,}건", 213),
        info_box(f"wo_type: '작업 없음' 범주 {q['wo_type_blank_count']:,}행 ({q['wo_type_blank_pct']:.2f}%) · "
                 f"결측 아님", 213),
        period_tile,
        info_box(f"센서값 반복: 기계×날짜 {q['group_count']:,}개 그룹, 그룹당 부품 행 {q['rows_per_group']:,}개에 "
                 f"센서값 동일 반복", 213),
    ]
    qual = card("품질 요약", 932, 162, hstack(quality_tiles, 16))

    s = load_source_info()
    source_lines = "\n".join([s["source_name"], f"{s['row_count']:,}행 × {s['col_count']}열",
                               s["access_date_note"]])
    src = card("출처 · 라이선스 · 합성 데이터 한계", 932, 162,
               hstack([info_box(source_lines, 288), info_box(s["license"], 288),
                       info_box(s["limitations"], 292)], 16))
    row_c = row(ROW_SUB, [ddict, col(932, ROW_SUB, [qual, src])])

    return html.Div([toolbar, row(524, [dtable]), row_c],
                     style={"display": "flex", "flexDirection": "column", "gap": f"{GUTTER}px"})


# ============================================================
# 화면 ⑤ 보고서 요약 — 경영진 / 최종 보고서용 집계 화면 (①②와 분리된 대상 독자)
# 개별 기계 행이 없다 — 조직 전체 집계·추세만. "보고서 내보내기"가
# 이 화면을 기준으로 PDF/PPT를 만든다고 가정한다.
# ============================================================

def screen_5(seg_state=None, audience=DEFAULT_AUDIENCE):
    kpis5 = load_screen5_kpis()
    kpi_specs = [
        ("관측 기간 고장 표시 총 건수", f"{kpis5['failure_machine_days']:,}"),
        ("위험 기준선 초과 기계 비율 (%)", f"{kpis5['failure_rate_pct']:.1f}%"),
        ("총 부품 출고 금액 (INR)", f"{kpis5['parts_issue_value_inr']:,.0f}"),
        ("평균 소비 전력 (kW)", f"{kpis5['avg_power_kw']:,.2f}"),
    ]
    placeholder_labels = ["전기간 대비 증감 (건)", "모델 최종 평가 지표"]
    row_a = row(ROW_KPI,
        [kpi_value_tile(label, value_text) for label, value_text in kpi_specs] +
        [tile(label, sub="값 또는 빈 상태") for label in placeholder_labels])

    trend_body = dcc.Graph(
        id={"type": "trend-chart", "index": "screen5"},
        figure=_trend_figure(load_failure_trend(), "light"),
        config={"displayModeBar": False, "responsive": True},
        style={"flex": "1 1 auto", "minHeight": "0", "minWidth": "0"},
    )
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
        dcc.Store(id="prio-sort-store", data=DEFAULT_PRIO_SORT),
        # ②의 "‹ 이전 기계"/"다음 기계 ›"가 바꾸는, 현재 상세를 보고 있는 기계.
        # prio-sort-store와 동일하게 세션 메모리(storage_type 미지정)로 둔다.
        dcc.Store(id="selected-asset-store", data=load_asset_list()[0]),
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
    Input("prio-sort-store", "data"),
    Input("selected-asset-store", "data"),
)
def render_screen(active, seg_state, audience, prio_sort, selected_asset):
    kwargs = {"seg_state": seg_state, "audience": audience or DEFAULT_AUDIENCE}
    if active == "1":
        kwargs["prio_sort"] = prio_sort or DEFAULT_PRIO_SORT
    if active == "2":
        kwargs["asset_tag"] = selected_asset
    if active == "3":
        kwargs["asset_tag"] = selected_asset
    return SCREEN_BUILDERS[active](**kwargs)


# ②의 "‹ 이전 기계"/"다음 기계 ›" — 알파벳순으로 순환 이동한다. nav_btn()이
# 공용 ghost-btn과 다른 id 타입을 쓰므로 echo_action과 겹치지 않는다.
@app.callback(
    Output("selected-asset-store", "data"),
    Input({"type": "machine-nav-btn", "index": ALL}, "n_clicks"),
    State("selected-asset-store", "data"),
    prevent_initial_call=True,
)
def cycle_selected_asset(_clicks, current_asset):
    triggered = ctx.triggered_id
    if not triggered:
        return no_update
    # 화면②가 새로 마운트될 때마다 Dash가 이 패턴매칭 Input을 n_clicks=0(또는
    # None)인 채로 한 번 발화시킨다(prevent_initial_call은 앱 최초 실행만 막는다).
    # 그 발화를 클릭으로 세면 사용자의 진짜 첫 클릭이 무시된 것처럼 보이므로,
    # n_clicks가 실제로 올라간 경우에만 이동한다.
    if not ctx.triggered[0]["value"]:
        return no_update
    assets = load_asset_list()
    idx = assets.index(current_asset) if current_asset in assets else 0
    if triggered["index"] == "next":
        idx = (idx + 1) % len(assets)
    elif triggered["index"] == "prev":
        idx = (idx - 1) % len(assets)
    else:
        return no_update
    return assets[idx]


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


# 화면② 스몰 멀티플도 같은 방식으로 재색칠한다. trend-chart와 id 타입을 나눠
# 둬야 두 콜백의 Output 매칭 개수가 서로 섞이지 않는다. 기계 전환은 이미
# render_screen이 화면②를 다시 그리므로 selected-asset-store는 State로만 읽는다.
@app.callback(
    Output({"type": "smult-chart", "index": ALL}, "figure"),
    Input("theme-store", "data"),
    Input("screen-tabs", "value"),
    State("selected-asset-store", "data"),
)
def recolor_smult_chart(theme, _active_tab, asset_tag):
    assets = load_asset_list()
    asset_tag = asset_tag if asset_tag in assets else assets[0]
    return [_smult_figure(load_asset_sensor_series(asset_tag), theme or "light")]


# 화면② "부품 출고 이력" 막대차트도 같은 방식으로 재색칠한다. smult-chart와
# id 타입을 나눠야 두 콜백의 Output 매칭 개수가 서로 섞이지 않는다.
@app.callback(
    Output({"type": "parts-chart", "index": ALL}, "figure"),
    Input("theme-store", "data"),
    Input("screen-tabs", "value"),
    State("selected-asset-store", "data"),
)
def recolor_parts_chart(theme, _active_tab, asset_tag):
    assets = load_asset_list()
    asset_tag = asset_tag if asset_tag in assets else assets[0]
    return [_parts_figure(load_asset_parts_history(asset_tag), theme or "light")]


# 화면② "동종 기계 대비" 박스플롯도 같은 방식으로 재색칠한다. smult-chart/
# parts-chart와 id 타입을 나눠야 세 콜백의 Output 매칭 개수가 서로 섞이지 않는다.
@app.callback(
    Output({"type": "peer-chart", "index": ALL}, "figure"),
    Input("theme-store", "data"),
    Input("screen-tabs", "value"),
    State("selected-asset-store", "data"),
)
def recolor_peer_chart(theme, _active_tab, asset_tag):
    assets = load_asset_list()
    asset_tag = asset_tag if asset_tag in assets else assets[0]
    return [_peer_figure(load_asset_peer_comparison(asset_tag), theme or "light")]


# 화면② "고장 직전 센서 변화" 차트도 같은 방식으로 재색칠한다. smult-chart/
# parts-chart/peer-chart와 id 타입을 나눠야 네 콜백의 Output 매칭 개수가
# 서로 섞이지 않는다.
@app.callback(
    Output({"type": "pre-chart", "index": ALL}, "figure"),
    Input("theme-store", "data"),
    Input("screen-tabs", "value"),
    State("selected-asset-store", "data"),
)
def recolor_pre_chart(theme, _active_tab, asset_tag):
    assets = load_asset_list()
    asset_tag = asset_tag if asset_tag in assets else assets[0]
    return [_pre_figure(load_asset_failure_onset_trend(asset_tag), theme or "light")]


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
# ① KPI "종합 고장율" 드릴다운 — 클릭할 때마다 "점검 우선순위" 표 정렬을
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
    new = "grade" if (current or DEFAULT_PRIO_SORT) == "threshold" else "threshold"
    msg = ("'종합 고장율' 클릭 → 점검 우선순위를 기준선 초과 기준으로 정렬"
           if new == "threshold" else
           "'종합 고장율' 다시 클릭 → 점검 우선순위를 등급가중 고장점수 기준으로 복귀")
    return new, msg


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
    prevent_initial_call=True,
)
def update_dtable_page(page_current, sort_by):
    dataset_key = ctx.triggered_id["index"]
    sort_col = sort_by[0]["column_id"] if sort_by else None
    sort_dir = sort_by[0]["direction"] if sort_by else "asc"
    triggered_prop = ctx.triggered[0]["prop_id"].rsplit(".", 1)[-1] if ctx.triggered else None
    page = 0 if triggered_prop == "sort_by" else (page_current or 0)
    result = load_table_page(dataset_key, sort_col, sort_dir, page)
    footer_text = _dtable_footer_text(result["total_rows"], result["page"], 16)
    return result["data"], result["page_count"], result["page"], footer_text


@app.callback(
    Output("table-download", "data"),
    Input({"type": "csv-export-btn", "index": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def export_dtable_csv(_clicks):
    # 화면이 다시 그려지면 버튼이 n_clicks=0으로 재생성되며 이 콜백이 한 번
    # 더 불린다 — write_seg와 동일하게 값이 0인 호출은 무시한다.
    trig = ctx.triggered[0] if ctx.triggered else None
    if not trig or not trig.get("value"):
        return no_update
    dataset_key = ctx.triggered_id["index"]
    if dataset_key not in ("raw", "daily"):
        return no_update
    try:
        csv_bytes, filename = export_table_csv(dataset_key)
    except Exception:  # noqa: BLE001 — 다운로드만 조용히 건너뛴다
        return no_update
    return dcc.send_bytes(lambda buf: buf.write(csv_bytes), filename, type="text/csv")


if __name__ == "__main__":
    # 최신 Dash(2.17+)는 app.run, 이전 버전은 app.run_server를 쓴다.
    app.run(debug=True)
