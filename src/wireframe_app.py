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
  · 운영 모드는 로그인·역할·감사 로그(MySQL)를 쓴다. 데모 모드는 DB·로그인 없이 읽기 전용.

실행:
    python -m src.wireframe_app           # 운영 모드 (.env의 MACHINE_DATABASE_URL 필요)
    python -m src.wireframe_app --demo    # 데모 모드 (DASHBOARD_MODE=demo와 같다)
    → http://127.0.0.1:8052
"""

import io
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import plotly.graph_objects as go
from dash import Dash, html, dcc, dash_table, Input, Output, State, ALL, MATCH, ctx, no_update
from plotly.subplots import make_subplots
from flask import got_request_exception, session

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "src"

from .dashboard_data import (  # noqa: E402
    CLASSIFICATION_METRIC_COLUMNS,
    export_table_csv,
    filter_assets,
    load_asset_catalog,
    load_asset_detail_kpis,
    load_asset_failure_heatmap,
    load_asset_family_diagnosis,
    load_asset_list,
    load_asset_parts_history,
    load_asset_sensor_series,
    load_current_classification_actual_rate,
    load_current_classification_at_threshold,
    load_current_classification_metrics,
    load_current_classification_pr_curve_and_confusion,
    load_data_dictionary,
    load_data_quality_summary,
    load_data_reference_date,
    load_failure_trend,
    load_family_feature_importance,
    load_family_pr_curve_and_confusion,
    load_family_recurrence_intervals,
    load_part_failure_actual_rate,
    load_part_failure_at_threshold,
    load_part_failure_metrics,
    load_part_failure_pr_curve_and_confusion,
    load_part_failure_selected_model,
    load_priority_table,
    load_screen1_kpis,
    load_screen1_machine_status,
    load_screen1_power_by_machine,
    load_screen5_kpis,
    load_source_info,
    load_table_page,
    period_start,
)
from .audit_service import (  # noqa: E402
    change_admin_password, create_audit_tables, delete_user_accounts,
    ensure_dashboard_admin, list_user_accounts, log_failure, record_action, record_error, sync_if_csv_changed,
)
from .dashboard_auth import (  # noqa: E402
    current_session_state, end_admin_session, extend_admin_session, install_auth,
)
from .demo_mode import (  # noqa: E402
    DEMO_BADGE_TEXT, empty_list, is_demo_mode, no_session_state, noop,
)

DEMO_MODE = is_demo_mode()
if DEMO_MODE:
    # 데모 모드: 감사·계정 함수가 DB에 접속하지 않도록 이름만 no-op으로 바꾼다.
    change_admin_password = delete_user_accounts = ensure_dashboard_admin = noop
    create_audit_tables = log_failure = record_action = record_error = noop
    sync_if_csv_changed = noop
    list_user_accounts = empty_list
    current_session_state = extend_admin_session = no_session_state
    end_admin_session = noop

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

/* 테마 버튼은 텍스트 없이 아이콘만 표시한다 — "누르면 될 상태"를 보여주는
   관례(라이트에서는 다크로 바꾸는 아이콘, 다크에서는 라이트로 바꾸는
   아이콘). 아이콘 라이브러리 의존성 없이 유니코드 글리프만 쓴다. */
.theme-light #theme-btn::before { content: "☾"; }
.theme-dark  #theme-btn::before { content: "☀"; }

/* dcc.Tabs는 컨테이너를 100% 폭으로 잡고 탭 5개를 균등 분배(flex:1)한다.
   그대로 두면 "⑤ 보고서 요약"처럼 긴 라벨이 잘린다. 바깥 컨테이너는
   내용 폭으로, 각 탭은 자기 글자 폭으로 되돌린다. */
#screen-tabs-parent, #screen-tabs { width:max-content !important; }
#screen-tabs { flex-wrap:nowrap !important; }
#screen-tabs .tab { flex:0 0 auto !important; width:auto !important; }

/* 현재 화면에 적용되지 않는 필터 컨트롤 */
#period-btn-group button:disabled { opacity:0.4; cursor:not-allowed; }

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
]

# ------------------------------------------------------------
# 필터바 상호작용 — 확인된 실제 값(2026-09-27 기준
# synthetic_industrial_machine_data.csv 고유값).
# ------------------------------------------------------------
PLANT_OPTIONS = ["CHN-02", "DHR-03", "PUN-01"]
MACHINE_TYPE_OPTIONS = ["Belt Conveyor", "CNC Lathe", "EOT Crane",
                        "Hydraulic Press", "Screw Compressor"]
MACHINE_OPTIONS = load_asset_list()
# (라벨, 데이터 최신일을 포함한 일수). None = 전체 기간.
PERIOD_PRESETS = [("최근 30일", 30), ("최근 90일", 90), ("최근 1년", 365), ("전체", None)]

DEFAULT_FILTERS = {
    "plant": None, "machine_type": None, "machine": None, "period_index": 3,
}


def _period_index(filters):
    pi = (filters or DEFAULT_FILTERS).get("period_index", DEFAULT_FILTERS["period_index"])
    return pi if isinstance(pi, int) and 0 <= pi < len(PERIOD_PRESETS) else DEFAULT_FILTERS["period_index"]


def _filter_scope(filters):
    """filter-store 값 → (대상 기계 태그 목록, 기간 시작일 또는 None)."""
    f = filters or DEFAULT_FILTERS
    assets = filter_assets(f.get("plant"), f.get("machine_type"), f.get("machine"))
    return assets, period_start(PERIOD_PRESETS[_period_index(f)][1])


def _focus_asset(filters, machine_only=False):
    """② 기계 상세·③ 부품군 진단이 보여 줄 기계 — 필터의 기계, 전체면 점검 우선순위 1위.
    machine_only=True면 공장·기계 종류 필터를 보지 않는다(③은 그 필터를 적용하지 않는다)."""
    f = filters or DEFAULT_FILTERS
    if machine_only:
        f = {"machine": f.get("machine")}
    assets, _ = _filter_scope(f)
    return load_priority_table("grade", "desc", assets=assets or None)[0]["asset_tag"]

# ------------------------------------------------------------
# 보고서 대상(독자)별 산출물 구성
#
# 인증·역할 관리 시스템이 없으므로 "로그인 사용자의 직책"을 알 방법이 없다.
# 그래서 역할을 추론하지 않고 내보내기 시점에 헤더에서 직접 고르게 한다.
# 인증이 붙으면 export-dd의 기본값을 세션 역할로 채우면 된다.
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
    "task": ["현재 고장 표시 분류", "부품군 진단", "부품 고장 탐지"],
    "threshold": ["12", "13", "14"],
    "dataset": ["원자료", "기계·일 집계"],
}
DEFAULT_SEG = {"task": 0, "threshold": 0, "dataset": 0}

# ① 화면의 "점검 우선순위" 표 정렬 축. KPI "위험 기준선 초과 비율" 타일을 클릭하면
# "grade"(등급가중 고장점수, 기본) → "threshold"(기준선 초과)로 바뀐다.
DEFAULT_PRIO_SORT = {"sort_by": "grade", "direction": "desc"}

# ①에만 있는 타일이라 패턴 매칭 id를 쓴다 — 이유는 tile() 호출부 주석 참고.
KPI_FAILRATE_ID = {"type": "kpi-drill", "index": "failrate"}


# ============================================================
# 공용 프리미티브
# ============================================================

def note(text, style=None):
    """작은 보조 설명(micro-11)."""
    s = dict(MICRO_11)
    if style:
        s.update(style)
    return html.Span(text, style=s)


def slot(label, w, h, sub=None):
    """아직 채워지지 않은 영역 — 점선 테두리 + 라벨."""
    width_style = {"width": f"{w}px"} if isinstance(w, (int, float)) else {"flexGrow": "1", "minWidth": "0"}
    if h < 34:
        return html.Div(
            note(label),
            style={**width_style, "height": f"{h}px", "flexShrink": "0", "boxSizing": "border-box",
                   "border": f"1px dashed {CTRL}", "background": SUNK, "display": "flex",
                   "alignItems": "center", "justifyContent": "center", "padding": "0 6px",
                   "overflow": "hidden"},
        )
    children = [html.Span(label, style={**LABEL_12, "textAlign": "center"})]
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
            html.Div([html.Span(label, style=LABEL_12)],
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
    """필터바 드롭다운. 값은 filter-store(storage_type="local")가 원본이고,
    첫 렌더에 write_filters가 저장된 값을 드롭다운에 되돌려 놓는다."""
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
    """기간 버튼 4개 — 데이터 최신일을 포함한 최근 N일 또는 전체 기간."""
    buttons = [
        html.Button(label, id={"type": "period-btn", "index": i}, n_clicks=0,
                    style=period_btn_style(i == DEFAULT_FILTERS["period_index"]))
        for i, (label, _days) in enumerate(PERIOD_PRESETS)
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
        [html.Span(title, style={"margin": "0", "fontSize": "14px", "lineHeight": "20px",
                                  "fontWeight": "600", "color": INK}),
         html.Span(reason, style={"fontSize": "13px", "lineHeight": "18px", "color": INK2, "textAlign": "center"})],
        style={"width": f"{w}px", "height": f"{h}px", "boxSizing": "border-box",
               "border": f"1px dashed {CTRL}", "borderRadius": "4px", "display": "flex",
               "flexDirection": "column", "alignItems": "center", "justifyContent": "center",
               "gap": "6px", "padding": "16px"},
    )


# ============================================================
# 공통 컴포넌트: AppHeader / FilterBar
# ============================================================

def app_header():
    """좌: 시스템명·기준일 / 중앙: 화면 탭 4개(dcc.Tabs) / 우: 내보내기·테마 토글."""
    tab_style = {"height": "56px", "boxSizing": "border-box", "padding": "0 11px", "display": "flex",
                 "alignItems": "center", "whiteSpace": "nowrap", "flexShrink": "0",
                 "fontSize": "14px", "lineHeight": "20px", "fontWeight": "500",
                 "color": INK2, "border": "none", "borderBottom": "2px solid transparent", "background": "none"}
    tab_selected_style = {**tab_style, "fontWeight": "600", "color": INK, "borderBottom": f"2px solid {INK}"}

    left = html.Div(
        [html.Span("산업 기계 센서 기반 고장 위험 예측",
                    style={"fontSize": "16px", "fontWeight": "600", "color": INK, "whiteSpace": "nowrap"}),
         html.Div([html.Span("데이터 기준일", style=LABEL_12),
                   html.Span(load_data_reference_date(), style=NUM_12)],
                  style={"display": "flex", "alignItems": "center", "gap": "6px"})],
        style={"display": "flex", "alignItems": "center", "gap": "12px", "minWidth": "0"},
    )
    center = dcc.Tabs(
        id="screen-tabs", value="1",
        children=[dcc.Tab(label=label, value=tid, style=tab_style, selected_style=tab_selected_style)
                  for tid, label in SCREENS],
        style={"height": "56px"},
    )
    export_options = [
        {"label": f"{fmt_label} · {aud['label']}", "value": f"{fmt_value}:{aud_key}"}
        for fmt_label, fmt_value in [("PDF", "pdf"), ("Excel", "xlsx")]
        for aud_key, aud in REPORT_AUDIENCES.items()
    ]
    right = html.Div(
        [dcc.Dropdown(
             id="export-dd",
             options=export_options,
             value=f"pdf:{DEFAULT_AUDIENCE}", clearable=False,
             style={"width": "220px", "fontFamily": "inherit", "fontSize": "13px"},
         ),
         html.Button("내보내기", id="export-run-btn", n_clicks=0, style=btn_style(w=72)),
         html.Button("", id="theme-btn", n_clicks=0, title="테마 전환",
                     style=btn_style(w=32, extra={"padding": "0", "textAlign": "center", "fontSize": "16px"})),
         html.Span(DEMO_BADGE_TEXT, id="demo-badge",
                   style={"display": "inline-block" if DEMO_MODE else "none", "padding": "4px 10px",
                          "border": f"1px solid {HAIR}", "borderRadius": "12px", "fontSize": "12px",
                          "fontWeight": "600", "color": INK, "whiteSpace": "nowrap"}),
         html.Details([
             html.Summary(html.Span("관", id="profile-display"), id="profile-menu-toggle",
                          title="마이 프로필",
                          style={"listStyle": "none", "width": "34px", "height": "34px",
                                 "borderRadius": "50%", "background": INK, "color": CARD,
                                 "display": "grid", "placeItems": "center", "cursor": "pointer",
                                 "fontSize": "13px", "fontWeight": "700"}),
             html.Div([
                 html.Div(["아이디 · ", html.Span(id="profile-login-id")],
                          style={"fontSize": "12px", "marginBottom": "6px"}),
                 html.Div([
                     html.Span(["남은 시간 · ", html.Span("10:00", id="session-remaining")]),
                     html.Button(html.Img(src=SESSION_REFRESH_ICON, alt="", style={"width": "19px", "height": "19px", "display": "block"}),
                             id="session-extend-btn", n_clicks=0,
                             style={"border": "0", "background": "transparent", "color": "#111",
                                    "cursor": "pointer", "padding": "0", "marginLeft": "20px"},
                             title="로그인 시간 10분으로 초기화",
                             **{"aria-label": "로그인 시간 10분으로 초기화"}),
                 ], style={"fontSize": "12px", "marginBottom": "8px", "display": "flex",
                           "alignItems": "center"}),
                 html.Button("비밀번호 변경", id="password-open-btn", n_clicks=0,
                             style={"display": "block", "width": "100%", "padding": "8px", "marginBottom": "5px"}),
                 html.Button("계정 관리", id="account-manage-btn", n_clicks=0,
                             style={"display": "none", "width": "100%", "padding": "8px", "marginBottom": "5px"}),
                 html.Button("로그아웃", id="logout-btn", n_clicks=0,
                             style={"display": "block", "width": "100%", "padding": "8px"}),
             ], style={"position": "absolute", "right": "0", "top": "40px", "width": "210px",
                       "background": CARD, "color": INK, "border": f"1px solid {HAIR}",
                       "boxShadow": "0 8px 24px #0003", "padding": "10px", "zIndex": 100}),
         ], style={"position": "relative", "flexShrink": "0", "display": "none" if DEMO_MODE else "block"}),
         dcc.Download(id="report-download"),
         dcc.Download(id="table-download")],
        style={"display": "flex", "alignItems": "center", "justifyContent": "flex-end", "gap": "8px",
               # overflow:hidden을 여기 두면 "내보내기" 드롭다운 팝업까지
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
    """공장 / 기계 종류 / 기계 / 기간 / 초기화 · 우측 현재 필터 상태와 적용 범위 안내.
    dcc.Store(id="filter-store")가 화면 전환과 무관하게 값을 들고 있어 화면을
    옮겨도 선택이 유지된다."""
    return html.Div(
        [filter_dropdown("plant-dd", "공장", PLANT_OPTIONS, 140),
         filter_dropdown("machine-type-dd", "기계 종류", MACHINE_TYPE_OPTIONS, 140),
         filter_dropdown("machine-dd", "기계", MACHINE_OPTIONS, 180),
         period_toggle(),
         html.Button("초기화", id="reset-btn", n_clicks=0,
                     style={"height": "32px", "boxSizing": "border-box", "padding": "0 12px",
                            "border": f"1px solid {HAIR}", "borderRadius": "2px", "background": "transparent",
                            "fontFamily": "inherit", "fontSize": "13px", "fontWeight": "500", "color": INK2,
                            "cursor": "pointer", "whiteSpace": "nowrap", "flexShrink": "0"}),
         html.Div(style={"flexGrow": "1"}),
         html.Span(id="filter-scope-note", style={**MICRO_11, "whiteSpace": "nowrap", "flexShrink": "0"}),
         html.Div([html.Span("현재 필터", style={**LABEL_12, "whiteSpace": "nowrap"}),
                   html.Span(id="filter-echo", style={**NUM_12, "whiteSpace": "nowrap"})],
                  style={"display": "flex", "alignItems": "center", "gap": "6px", "flexShrink": "0"}),
         html.Span("", id="action-echo",
                   style={**MICRO_11, "width": "208px", "textAlign": "right", "flexShrink": "0",
                          "whiteSpace": "nowrap", "overflow": "hidden", "textOverflow": "ellipsis"})],
        style={"width": f"{CANVAS_W}px", "height": f"{FILTERBAR_H}px", "boxSizing": "border-box",
               "padding": f"0 {MARGIN}px", "background": CARD, "borderBottom": f"1px solid {HAIR}",
               "display": "flex", "alignItems": "center", "gap": "10px", "fontFamily": SANS, "color": INK,
               # overflow:hidden을 쓰면 자식(드롭다운 팝업 메뉴)까지 clip돼서
               # 목록이 잘리거나 다른 카드 밑에 깔린 것처럼 보인다. overflow는
               # 넣지 않고 줄바꿈만 막는다.
               "flexWrap": "nowrap",
               # app_header()와 같은 이유 — 설치된 Dash 버전이 옛 방식(z-index
               # 없는 position:absolute) 드롭다운을 쓸 경우를 대비해 필터바
               # 자체를 main보다 항상 위에 오도록 고정한다.
               "position": "relative", "zIndex": 20},
    )


# ============================================================
# 화면 ① 현황 — 행 96 / 460 / 340
# ============================================================

def kpi_value_tile(label, value_text, w=300, h=96, tid=None, scope=None):
    """KPI 값 타일. scope는 라벨 오른쪽의 집계 범위 안내(예: "기준일", "선택 기간")."""
    style = {"width": f"{w}px", "height": f"{h}px", "flexShrink": "0", "boxSizing": "border-box",
             "background": CARD, "outline": f"1px solid {CTRL if tid else HAIR}", "outlineOffset": "-1px",
             "borderRadius": "4px", "padding": "16px", "display": "flex",
             "flexDirection": "column", "gap": "8px",
             "cursor": "pointer" if tid else "default"}
    kwargs = {"id": tid, "n_clicks": 0} if tid else {}
    return html.Div(
        [html.Div([html.Span(label, style=LABEL_12)] + ([note(scope)] if scope else []),
                   style={"height": "16px", "display": "flex", "justifyContent": "space-between", "gap": "8px"}),
         html.Div(html.Span(value_text, style={"fontFamily": MONO, "fontSize": "22px",
                                                 "fontWeight": "600", "color": INK}),
                  style={"height": "32px", "display": "flex", "alignItems": "center"})],
        style=style, **kwargs,
    )


def _spark_figure(values, theme):
    """화면 ① "기계 상태" 타일의 미니 스파크라인(축·격자·여백 없는 라인)."""
    colors = FIGURE_COLORS.get(theme, FIGURE_COLORS["light"])
    fig = go.Figure(go.Scatter(y=values, mode="lines", hoverinfo="skip",
                                line=dict(color=colors["ink2"], width=1.4)))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(
        height=32, margin=dict(l=0, r=0, t=2, b=2), showlegend=False,
        paper_bgcolor=colors["card"], plot_bgcolor=colors["card"],
    )
    return fig


def _heatmap_figure(heat, theme):
    """화면 ① "고장 표시 히트맵" — 자산 × 월 그레인, 셀 = 그 달 고장 표시된
    부품-일 행 수. period는 이미 오름차순 CSV 순서라 그대로 pivot한다."""
    colors = FIGURE_COLORS.get(theme, FIGURE_COLORS["light"])
    pivot = heat.pivot(index="asset_tag", columns="period", values="failed_part_count")
    fig = go.Figure(go.Heatmap(
        z=pivot.to_numpy(), x=pivot.columns.tolist(), y=pivot.index.tolist(),
        colorscale=[[0, colors["card"]], [1, colors["ink"]]],
        hovertemplate="%{y} · %{x}<br>%{z}건<extra></extra>",
        colorbar=dict(title=dict(text="건수", font=dict(size=10)), tickfont=dict(size=10)),
    ))
    fig.update_xaxes(tickfont=dict(size=10), color=colors["muted"])
    fig.update_yaxes(tickfont=dict(size=11), color=colors["muted"], autorange="reversed")
    fig.update_layout(
        height=300, margin=dict(l=8, r=8, t=8, b=8),
        paper_bgcolor=colors["card"], plot_bgcolor=colors["card"],
        font=dict(size=11, color=colors["muted"]),
    )
    return fig


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


def _family_pr_figure(pr_cm, theme):
    """화면 ③ "부품군 진단" 드릴다운 — 선택 자산·부품군의 PR곡선."""
    colors = FIGURE_COLORS.get(theme, FIGURE_COLORS["light"])
    fig = go.Figure(go.Scatter(
        x=pr_cm["recall_curve"], y=pr_cm["precision_curve"], mode="lines",
        line=dict(color=colors["ink"], width=1.6),
    ))
    fig.update_yaxes(title="정밀도", range=[0, 1.02], gridcolor=colors["hair"],
                      tickfont=dict(size=11), color=colors["muted"])
    fig.update_xaxes(title="재현율", range=[0, 1.02], showgrid=False,
                      tickfont=dict(size=11), color=colors["muted"])
    fig.update_layout(
        height=392, margin=dict(l=48, r=16, t=8, b=40),
        paper_bgcolor=colors["card"], plot_bgcolor=colors["card"],
        font=dict(size=11, color=colors["muted"]), showlegend=False,
    )
    return fig


def _family_recur_figure(intervals, theme):
    """화면 ③ "부품군 진단" 드릴다운 — 선택 자산·부품군의 재발 간격(일) 분포."""
    colors = FIGURE_COLORS.get(theme, FIGURE_COLORS["light"])
    fig = go.Figure(go.Histogram(x=intervals, marker=dict(color=colors["ink"])))
    fig.update_yaxes(title="빈도", gridcolor=colors["hair"], tickfont=dict(size=11), color=colors["muted"])
    fig.update_xaxes(title="간격(일)", showgrid=False, tickfont=dict(size=11), color=colors["muted"])
    fig.update_layout(
        height=184, margin=dict(l=40, r=8, t=4, b=32), bargap=0.08,
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


def screen_1(seg_state=None, audience=DEFAULT_AUDIENCE, prio_sort=None, assets=None, start=None):
    # 절충안: 6타일 중 "위험 기준선 초과 기계 (대)" 한 자리만 위험 기준선 초과 비율(%)로
    # 바꾼다. 그 지표는 원래 절대 건수(분자)만 보여줘서 "전체 대비 얼마나
    # 심각한가"를 암산해야 했는데, 비율로 바꾸면 그 계산이 필요 없어진다.
    # 나머지 5개(관측 기계·고장 표시 기계·일·전력·온도·부품 금액)는 그대로 —
    # 전부 원자료 수준 지표라 "지금 뭐가 몇 대냐"에 즉답하는 역할을 유지한다.
    #
    # 정의(확정): 위험 기준선 초과 비율(%) = 위험 기준선 초과 기계 수 ÷ 전체 관측 기계 수 × 100
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
    kpis = load_screen1_kpis(assets, start)
    # "최고 베어링 온도"·"부품 출고 금액(누적)"은 삭제한다 — 남은 4개가 같은
    # 458px 폭(4×458 + 3×16 = 1880)으로 행 전체를 균등 분배한다.
    # 네 번째 값은 (label, value, 클릭 id, 집계 범위 안내).
    kpi_specs = [
        ("관측 기계 (대)", f"{kpis['observed_machines']:,}", None, None),
        ("고장 표시 기계·일", f"{kpis['failure_machine_days']:,}", None, "선택 기간"),
        ("위험 기준선 초과 비율 (%)", f"{kpis['failure_rate_pct']:.1f}", KPI_FAILRATE_ID, "기준일"),
        ("평균 소비 전력 (kW)", f"{kpis['avg_power_kw']:,.2f}", None, "기준일"),
    ]
    row_a = row(ROW_KPI, [kpi_value_tile(label, value_text, w=458, tid=tid, scope=scope)
                          for label, value_text, tid, scope in kpi_specs])

    prio_cols = [("순위", 48, "right"), ("대상", 200, "left"), ("종류", 110, "left"), ("공장", 100, "left"),
                 ("등급가중 고장점수", 150, "right"), ("기준선 초과", 110, "center"),
                 ("최근 고장 표시일", 150, "left"), ("당일 고장 표시 부품 수", 190, "right")]
    sort_by = prio_sort.get("sort_by", "grade")
    direction = prio_sort.get("direction", "desc")
    sort_idx = 5 if sort_by == "threshold" else 4
    sort_hint = ("정렬: 기준선 초과" if sort_by == "threshold"
                 else "정렬: 등급가중 고장점수 (기본)") + (" · 오름차순" if direction == "asc" else " · 내림차순")
    priority_records = load_priority_table(sort_by, direction, assets=assets)
    dir_btn = html.Button("▲ 오름차순" if direction == "asc" else "▼ 내림차순",
                           id={"type": "prio-dir-btn", "index": "screen1"}, n_clicks=0, style=btn_style(w=96))
    prio = card("점검 우선순위", 1090, ROW_MAIN,
                priority_table(prio_cols, priority_records, 32, head_h=32, sort_col=sort_idx),
                right=html.Div([note(f"기준일 스냅샷 · 기간 미적용 · {sort_hint}"),
                                 dir_btn],
                                style={"display": "flex", "alignItems": "center", "gap": "8px"}))

    def machine_tile(status_row):
        grade = status_row["current_grade"]
        emphasize = grade in ("위험", "경계")
        badge = html.Span(
            grade,
            style={"height": "18px", "boxSizing": "border-box", "padding": "0 6px",
                   "border": f"1px solid {CTRL if emphasize else HAIR}", "borderRadius": "2px",
                   "fontSize": "11px", "lineHeight": "16px",
                   "fontWeight": "600" if emphasize else "500",
                   "color": INK if emphasize else MUTED, "display": "inline-flex",
                   "alignItems": "center", "whiteSpace": "nowrap"},
        )
        return html.Div(
            [html.Div([html.Span(status_row["asset_tag"], style={"fontFamily": MONO, "fontSize": "13px",
                                                                   "color": INK, "whiteSpace": "nowrap"}),
                       html.Span(status_row["machine_type"], style={**MICRO_11, "whiteSpace": "nowrap",
                                                                     "overflow": "hidden", "textOverflow": "ellipsis"})],
                      style={"width": "96px", "flexShrink": "0", "display": "flex",
                             "flexDirection": "column", "gap": "4px", "overflow": "hidden"}),
             dcc.Graph(id={"type": "spark-chart", "index": status_row["asset_tag"]},
                       figure=_spark_figure(status_row["sparkline"], "light"),
                       config={"displayModeBar": False, "responsive": True},
                       style={"flexGrow": "1", "minWidth": "0", "height": "32px"}),
             html.Div([html.Span(f"{status_row['risk_score']:,.0f}",
                                  style={"fontFamily": MONO, "fontSize": "16px", "fontWeight": "600", "color": INK}),
                       badge],
                      style={"width": "88px", "flexShrink": "0", "display": "flex",
                             "flexDirection": "column", "gap": "4px"})],
            style={"width": "367px", "height": "72px", "flexShrink": "0", "boxSizing": "border-box",
                   "outline": f"1px solid {HAIR}", "outlineOffset": "-1px", "borderRadius": "2px",
                   "background": CARD, "padding": "10px 12px", "display": "flex", "alignItems": "center",
                   "gap": "8px"},
        )
    machine_status_rows = load_screen1_machine_status(assets)
    tiles_grid = html.Div([machine_tile(r) for r in machine_status_rows],
                           style={"display": "grid", "gridTemplateColumns": "repeat(2, minmax(0, 1fr))",
                                  "gridTemplateRows": "repeat(5, 72px)", "columnGap": "8px", "rowGap": "8px",
                                  "width": "742px", "height": "392px"})
    status = card("기계 상태", 774, ROW_MAIN, tiles_grid,
                  right=note("스파크라인 = 등급가중 고장점수 최근 30일 · 기간 미적용"))
    row_b = row(ROW_MAIN, [prio, status])

    heat_body = html.Div(
        dcc.Graph(id={"type": "heatmap-chart", "index": "screen1"},
                  figure=_heatmap_figure(load_asset_failure_heatmap(assets, start), "light"),
                  config={"displayModeBar": False, "responsive": True},
                  style={"flex": "1 1 auto", "minHeight": "0", "minWidth": "0"}),
        style={"height": "100%", "display": "flex", "flexDirection": "column"},
    )
    heat = card("고장 표시 히트맵", 1248, ROW_SUB, heat_body,
                right=note("셀 = 그 달 고장 표시된 부품-일 행 수 합계 · 자산 × 월"))

    power_rows = load_screen1_power_by_machine(assets, start)
    max_power = max((r["avg_power_kw"] for r in power_rows), default=1.0) or 1.0
    prows = html.Div(
        [html.Div([html.Span(r["asset_tag"], style={"fontFamily": MONO, "fontSize": "12px", "color": INK,
                                                      "width": "88px", "flexShrink": "0"}),
                   html.Div(bar(r["avg_power_kw"] / max_power * 100, 6), style={"flexGrow": "1"}),
                   html.Span(f"{r['avg_power_kw']:,.2f}",
                             style={**NUM_12, "width": "56px", "flexShrink": "0", "textAlign": "right"})],
                  style={"height": "25px", "display": "flex", "alignItems": "center", "gap": "8px"})
         for r in power_rows],
    )
    power_body = html.Div([prows], style={"display": "flex", "flexDirection": "column"})
    power = card("기계별 평균 소비 전력 (kW)", 616, ROW_SUB, power_body, right=note(f"{len(power_rows)}개 · 내림차순 · 선택 기간 평균"))
    row_c = row(ROW_SUB, [heat, power])

    return html.Div([row_a, row_b, row_c],
                     style={"display": "flex", "flexDirection": "column", "gap": f"{GUTTER}px"})


# ============================================================
# 화면 ② 기계 상세 — 행 88 / 520 / 288
# ============================================================

def screen_2(seg_state=None, audience=DEFAULT_AUDIENCE, asset_tag=None, start=None):
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
    detail = load_asset_detail_kpis(asset_tag, start)

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
         metric("평균 소비 전력 (kW)", 140, f"{detail['avg_power_kw']:,.2f}")],
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
        + [html.Div(style={"height": f"{SMULT_AXIS_H}px", "flexShrink": "0"})],
        style={"width": "160px", "flexShrink": "0", "display": "flex", "flexDirection": "column",
               "gap": f"{SMULT_GAP}px"},
    )
    sm_body = hstack(
        [sensor_labels,
         dcc.Graph(id={"type": "smult-chart", "index": "screen2"},
                   figure=_smult_figure(load_asset_sensor_series(asset_tag, start), "light"),
                   config={"displayModeBar": False, "responsive": True},
                   style={"flex": "1 1 auto", "minHeight": "0", "minWidth": "0"})],
        8, style={"height": f"{SMULT_BODY_H}px"},
    )
    smult = card("센서 8종 스몰 멀티플", 1248, 520, sm_body,
                 right=note("x축 공유 · 밴드 = 위험 기준선 초과일"))

    # "이상 점수 추이"·"군집 위치"·"고장 직전 센서 변화"·"동종 기계 대비" 4개
    # 카드를 삭제한다. K-Means 결과(군집·이상 점수)는 화면④ 데이터 조회의
    # 그레인 토글로 옮겨 원자료 조회로는 남길 계획이었으나, 그 산출물
    # (dataVerification/outputs/kmeans_assignments.csv)이 리포지토리 어디에도
    # 없다 — 대시보드는 모델을 재실행하지 않는다는 공통 제약과 충돌해 새로
    # 만들지 않았다(별도 오프라인 클러스터링 스크립트가 먼저 필요).
    # "부품 출고 이력"만 남아 오른쪽 칸(616 × 520)을 그대로 채운다.
    parts_history = load_asset_parts_history(asset_tag, start)
    if parts_history.empty:
        parts_body = empty_state("기간 내 부품 출고 없음", "선택한 기간에 이 기계의 부품 출고 금액이 0이다", 584, 452)
    else:
        parts_body = dcc.Graph(id={"type": "parts-chart", "index": "screen2"},
                               figure=_parts_figure(parts_history, "light"),
                               config={"displayModeBar": False, "responsive": True},
                               style={"flex": "1 1 auto", "minHeight": "0", "minWidth": "0"})
    parts = card("부품 출고 이력", 616, 520, parts_body)
    row_b = row(520, [smult, parts])

    return html.Div([strip, row_b],
                     style={"display": "flex", "flexDirection": "column", "gap": f"{GUTTER}px"})


def _model_comparison_table(rows, selected_key, width=742):
    """화면 ③ "모델 비교" 표 — 과제 0(현재 고장 표시 분류)·2(부품 고장 탐지)가
    공유하는 렌더러. 열 순서는 모델 → AP → 정밀도 → 재현율 → F1 → 선택 여부로
    고정한다(Accuracy는 상단 KPI 카드에 이미 있고 양성률이 낮아 부풀려지므로
    이 표에는 넣지 않는다). ``rows``는 {"key","label","average_precision",
    "precision","recall","f1"}를 가진 dict 목록이며, 결과가 있는 모델만 담아야
    한다 — 빈 행을 채우지 않는다.
    """
    cols = [("모델", 182, "left"), ("AP", 93, "right"), ("정밀도", 93, "right"),
            ("재현율", 93, "right"), ("F1", 93, "right"), ("선택", 72, "center")]

    def cell(content, align="left", highlight=False):
        return html.Td(content, style={"padding": "0 8px", "textAlign": align, "boxSizing": "border-box",
                                        "background": SUNK if highlight else "none",
                                        "borderBottom": f"1px solid {HAIR}"})

    body_rows = []
    for r in rows:
        is_selected = r["key"] == selected_key
        body_rows.append(html.Tr([
            cell(html.Span(r["label"], style={"fontSize": "13px", "color": INK}), highlight=is_selected),
            cell(f"{r['average_precision']:.3f}", align="right", highlight=is_selected),
            cell(f"{r['precision']:.3f}", align="right", highlight=is_selected),
            cell(f"{r['recall']:.3f}", align="right", highlight=is_selected),
            cell(f"{r['f1']:.3f}", align="right", highlight=is_selected),
            cell("선택" if is_selected else "—", align="center", highlight=is_selected),
        ], style={"height": "32px"}))

    thead = html.Tr(
        [html.Th(l, style={"width": f"{w}px", "padding": "0 8px", "textAlign": a, "boxSizing": "border-box",
                            **LABEL_12, "whiteSpace": "nowrap", "borderBottom": f"1px solid {CTRL}"})
         for l, w, a in cols],
        style={"height": "32px"},
    )
    return html.Table([html.Thead(thead), html.Tbody(body_rows)],
                       style={"width": f"{width}px", "tableLayout": "fixed", "borderCollapse": "collapse"})


# 화면 ③ "주요 영향 변수" 막대 색 구분 — 새 색상 토큰이 아니라 이 카드
# 전용으로 도입한 예외다(사용자 명시 지시). 파생 변수(_lag1/_median3/...)는
# 전부 "<기본 센서명>_<변환>" 형태라 접두사만 봐도 분류된다 — 파생이 늘어나도
# 매핑을 새로 추가할 필요가 없다.
FEATURE_COLOR_HEALTH = "#0f766e"    # 건강 센서 — 청록
FEATURE_COLOR_OPERATING = "#c2660d"  # 운전 조건 — 주황
_HEALTH_SENSOR_PREFIXES = (
    "temp_bearing_degC", "temp_motor_degC", "vibration_h_mms", "vibration_v_mms", "oil_pressure_bar",
)
_OPERATING_PREFIXES = ("load_pct", "shaft_rpm", "power_consumption_kw")
_OPERATING_EXACT = {"day_of_week", "is_weekend"}


def _feature_color_category(feature: str) -> str:
    """건강 센서(청록) / 운전 조건(주황) / 장비·부품·날짜(회색, 그 외 전부)."""
    if feature.startswith(_HEALTH_SENSOR_PREFIXES):
        return "health"
    if feature.startswith(_OPERATING_PREFIXES) or feature in _OPERATING_EXACT:
        return "operating"
    return "other"


_FEATURE_CATEGORY_COLOR = {"health": FEATURE_COLOR_HEALTH, "operating": FEATURE_COLOR_OPERATING, "other": MUTED}


def _interpretation_summary(actual_rate, ap, precision, recall):
    """화면 ③ 성능 카드 바로 아래 "모델 해석 요약" 두 줄 — 현재 선택된
    과제·모델의 metrics에서 그때그때 계산한다(하드코딩 없음)."""
    multiplier = ap / actual_rate if actual_rate > 0 else float("inf")
    found = round(recall * 100)
    hit = round(precision * 100)
    line1 = (f"실제 고장률 {actual_rate * 100:.1f}% · AP {ap * 100:.1f}% · "
             f"무작위 기준 대비 {multiplier:.1f}배")
    line2 = f"고장 100건 중 약 {found}건을 찾지만, 경고 100건 중 실제 고장은 약 {hit}건"
    return html.Div(
        [html.Div(line1, style={"fontSize": "13px", "lineHeight": "20px", "fontWeight": "600", "color": INK}),
         html.Div(line2, style={"fontSize": "12px", "lineHeight": "18px", "color": INK2})],
        style={"width": "1880px", "flexShrink": "0"},
    )


def _judgment_verdict(multiplier):
    if multiplier < 1.2:
        return "사용 불가 — 무작위 대비 개선 없음"
    if multiplier < 2.0:
        return "탐색적 사용 가능 — 점검 후보를 좁히는 보조 수단"
    return "제한적 운영 검토 가능"


def _threshold_metrics_body(metrics, threshold):
    """화면 ③ "판정 임계값 조정" — 슬라이더가 움직일 때마다 다시 그리는 본문.
    ``metrics``는 load_*_at_threshold()가 돌려준 dict다."""
    tp = metrics["predicted_alerts"] - metrics["false_alarms"]
    actual_positive = tp + metrics["missed_failures"]
    sentence = (f"임계값 {threshold:.2f} 적용 시 {metrics['total_rows']:,}건 중 "
                f"{metrics['predicted_alerts']:,}건을 점검 대상으로 선정하며, "
                f"실제 고장 {actual_positive:,}건 중 {tp:,}건을 탐지합니다.")
    rows = [
        ("정밀도", f"{metrics['precision']:.3f}"), ("재현율", f"{metrics['recall']:.3f}"),
        ("F1", f"{metrics['f1']:.3f}"), ("예상 경고 건수", f"{metrics['predicted_alerts']:,}건"),
        ("오경보 건수", f"{metrics['false_alarms']:,}건"), ("놓친 고장 건수", f"{metrics['missed_failures']:,}건"),
    ]
    return html.Div(
        [html.Div([html.Span(k, style={**LABEL_12, "width": "110px", "flexShrink": "0"}),
                   html.Span(v, style={"fontSize": "13px", "fontWeight": "600", "color": INK})],
                  style={"display": "flex", "alignItems": "center", "gap": "8px", "minHeight": "18px"})
         for k, v in rows]
        + [html.Div(sentence, style={"fontSize": "12px", "lineHeight": "17px", "color": INK2, "marginTop": "6px"})],
        style={"display": "flex", "flexDirection": "column", "gap": "2px"},
    )


def _operational_judgment_body(actual_rate, ap, precision, recall, confusion):
    """화면 ③ "운영 판단" 카드 — 재발 간격(회고적 통계) 대신 현재 모델의
    실사용 가능성을 요약한다. 과제 0·2에서만 쓴다(과제 1은 실제 재발 간격
    히스토그램이 이미 있어 대체하지 않는다)."""
    multiplier = ap / actual_rate if actual_rate > 0 else float("inf")
    total = sum(confusion.values())
    predicted_alert_rate = (confusion["tp"] + confusion["fp"]) / total if total else 0.0
    rows = [
        ("실제 고장률", f"{actual_rate * 100:.1f}%"),
        ("예측 경고율", f"{predicted_alert_rate * 100:.1f}%"),
        ("AP ÷ 실제 고장률", f"{multiplier:.1f}배"),
        ("경고 100건당 실제 고장", f"{round(precision * 100)}건"),
        ("실제 고장 100건당 탐지", f"{round(recall * 100)}건"),
        ("현재 모델 판정", _judgment_verdict(multiplier)),
    ]
    return html.Div(
        [html.Div([html.Span(k, style={**LABEL_12, "width": "152px", "flexShrink": "0"}),
                   html.Span(v, style={"fontSize": "12px", "lineHeight": "15px", "color": INK,
                                        "fontWeight": "600" if k == "현재 모델 판정" else "500"})],
                  style={"display": "flex", "alignItems": "center", "gap": "8px", "minHeight": "18px"})
         for k, v in rows]
        + [html.Div("자동 부품 교체 판단에는 사용할 수 없습니다.",
                    style={**MICRO_11, "marginTop": "4px"})],
        style={"display": "flex", "flexDirection": "column", "gap": "3px"},
    )


# ============================================================
# 화면 ③ 모델·예측 — 툴바 32 / 행 96 / 460 / 272 (마지막 행이 남는 높이 흡수)
# ============================================================

def screen_3(seg_state=None, audience=DEFAULT_AUDIENCE, asset_tag=None, family=None):
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
                   seg("threshold", "위험 기준선 (등급가중 고장점수)", SEG_GROUPS["threshold"], sel=thr_sel)],
                  style={"height": "32px", "display": "flex", "alignItems": "center",
                         "justifyContent": "space-between", "gap": "16px"}),
         warn_line],
        style={"width": "1880px", "height": "52px", "display": "flex", "flexDirection": "column",
               "gap": "4px", "flexShrink": "0"},
    )

    # 과제에 따라 행 A가 갈린다 (원래 스펙의 '③ 행A 변형'):
    #   과제 0(현재 고장 표시 분류)은 실제 값(hist_gradient_boosting 기준) 타일 6개.
    #   과제 1(부품군 진단)은 KPI 타일 행 자체를 만들지 않는다 — mcomp 카드
    #   자리에 부품군별 표가 대신 들어간다.
    #   과제 2(부품 고장 탐지)는 아직 자리표시자 그대로(타일 4개 × 458).
    kpi_labels = ["정확도", "정밀도", "재현율", "F1", "ROC-AUC", "평균정밀도(AP)"]
    part_failure_kpi_labels = ["평균정밀도(AP)", "정밀도", "재현율", "F1"]
    part_failure_kpi_cols = ["average_precision", "precision", "recall", "f1"]
    if task_sel == 2:
        part_failure_metrics = load_part_failure_metrics()
        selected_model = load_part_failure_selected_model()
        selected_metrics = part_failure_metrics[selected_model]
        row_a = row(ROW_KPI, [
            kpi_value_tile(label, f"{selected_metrics[col]:.3f}", w=458)
            for label, col in zip(part_failure_kpi_labels, part_failure_kpi_cols)
        ])
    elif task_sel == 0:
        current_metrics = load_current_classification_metrics()["hist_gradient_boosting"]
        row_a = row(ROW_KPI, [
            kpi_value_tile(label, f"{current_metrics[col]:.3f}")
            for label, col in zip(kpi_labels, CLASSIFICATION_METRIC_COLUMNS)
        ])
    else:
        row_a = html.Div()

    if task_sel == 1:
        # 부품군 진단 — ...
        family_rows = load_asset_family_diagnosis(asset_tag)
        valid_families = [fr["part_family"] for fr in family_rows]
        family = family if family in valid_families else valid_families[0]
        pr_cm = load_family_pr_curve_and_confusion(asset_tag, family)
        interp_row = None
    elif task_sel == 0:
        pr_cm = load_current_classification_pr_curve_and_confusion("hist_gradient_boosting")
        actual_rate = load_current_classification_actual_rate("hist_gradient_boosting")
        interp_row = _interpretation_summary(actual_rate, current_metrics["average_precision"],
                                              current_metrics["precision"], current_metrics["recall"])
    else:
        pr_cm = load_part_failure_pr_curve_and_confusion()
        actual_rate = load_part_failure_actual_rate(selected_model)
        interp_row = _interpretation_summary(actual_rate, selected_metrics["average_precision"],
                                              selected_metrics["precision"], selected_metrics["recall"])

    if task_sel == 1:
        # 부품군 진단 — mcomp 카드 자리에 선택 자산의 부품군 9종별 표를 넣는다
        # (기존 '모델 비교' 표와는 열 구성 자체가 달라 별도로 만든다).
        # average_precision 내림차순(load_asset_family_diagnosis가 이미 정렬)
        # 기준 1행이 최초 진입 기본 선택 부품군이다.
        family_cols = [
            ("부품군", 130, "left"), ("모델", 150, "left"), ("표본 수", 64, "right"),
            ("양성률(%)", 80, "right"), ("정밀도", 82, "right"), ("재현율", 82, "right"),
            ("AP", 72, "right"), ("ROC-AUC", 82, "right"),
        ]
        def family_cell(content, align="left", highlight=False):
            # boxSizing 없이는 padding이 지정한 width 위에 더해져 8열이 카드
            # 폭(742px)을 넘기고 overflow:hidden에 잘린다(기존 '모델 비교'
            # 표에도 있던 문제이나, 그 표는 이번 범위 밖이라 손대지 않는다).
            return html.Td(content, style={"width": "auto", "padding": "0 8px", "textAlign": align,
                                            "boxSizing": "border-box", "background": SUNK if highlight else "none",
                                            "borderBottom": f"1px solid {HAIR}"})
        family_body_rows = [
            html.Tr([
                family_cell(html.Span(fr["part_family"], style={"fontSize": "13px", "color": INK}),
                            highlight=fr["part_family"] == family),
                family_cell(html.Span(fr["model"], style={"fontSize": "13px", "color": INK}),
                            highlight=fr["part_family"] == family),
                family_cell(str(fr["support"]), align="right", highlight=fr["part_family"] == family),
                family_cell(f"{fr['positive_rate'] * 100:.1f}%", align="right", highlight=fr["part_family"] == family),
                family_cell(f"{fr['precision']:.3f}", align="right", highlight=fr["part_family"] == family),
                family_cell(f"{fr['recall']:.3f}", align="right", highlight=fr["part_family"] == family),
                family_cell(f"{fr['average_precision']:.3f}", align="right", highlight=fr["part_family"] == family),
                family_cell(f"{fr['roc_auc']:.3f}", align="right", highlight=fr["part_family"] == family),
            ], id={"type": "family-row", "index": fr["part_family"]}, n_clicks=0,
               style={"height": "32px", "cursor": "pointer"})
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
        mcomp = card("부품군 진단", 774, ROW_MAIN, mcomp_body, right=note("행 = 부품군 · 열 = 지표 · 행 클릭 시 아래 카드 갱신"))
    else:
        # 모델 비교 표 — 과제 0(현재 고장 표시 분류)·2(부품 고장 탐지) 공용.
        # 열 순서는 모델 → AP → 정밀도 → 재현율 → F1 → 선택 여부로 고정하고
        # Accuracy는 넣지 않는다(상단 KPI 카드에 이미 있고 양성률이 낮아
        # 부풀려진다). 결과가 있는 모델만 행으로 넣는다.
        if task_sel == 0:
            metrics = load_current_classification_metrics()
            comparison_rows = [
                {"key": "prior", "label": "prior (기준)", **{
                    col: metrics["prior"][col] for col in ("average_precision", "precision", "recall", "f1")
                }},
                {"key": "hist_gradient_boosting", "label": "hist_gradient_boosting", **{
                    col: metrics["hist_gradient_boosting"][col]
                    for col in ("average_precision", "precision", "recall", "f1")
                }},
            ]
            selected_key = "hist_gradient_boosting"
        else:
            metrics = load_part_failure_metrics()
            comparison_rows = [{"key": m, "label": m, **vals} for m, vals in metrics.items()]
            selected_key = load_part_failure_selected_model()
        mcomp_body = html.Div([_model_comparison_table(comparison_rows, selected_key)])
        mcomp = card("모델 비교", 774, ROW_MAIN, mcomp_body,
                     right=note(f"행 = 모델 · 선택 = AP 최댓값 ({selected_key})"))

    pr_chart_id_type = {0: "classification-pr-chart", 1: "family-pr-chart", 2: "partfail-pr-chart"}[task_sel]
    pr_body = html.Div(
        dcc.Graph(id={"type": pr_chart_id_type, "index": "screen3"},
                  figure=_family_pr_figure(pr_cm, "light"),
                  config={"displayModeBar": False, "responsive": True},
                  style={"flex": "1 1 auto", "minHeight": "0", "minWidth": "0"}),
        style={"width": "584px", "height": "392px", "display": "flex", "flexDirection": "column"})
    prc_note = f"모델: {pr_cm['model']}" if "model" in pr_cm else f"{asset_tag} · {family}"
    prc = card("PR 곡선", 616, ROW_MAIN, pr_body, right=note(prc_note))

    def cm_cell(value):
        return html.Div(html.Span(str(value), style={"fontFamily": MONO, "fontSize": "24px",
                                                       "fontWeight": "600", "color": INK}),
                         style={"width": "169px", "height": "160px", "boxSizing": "border-box",
                                "display": "flex", "alignItems": "center", "justifyContent": "center",
                                "border": f"1px dashed {CTRL}", "background": SUNK})
    c = pr_cm["confusion"]
    cm_body = html.Div([
        hstack([html.Div(style={"width": "72px"}),
                html.Div("예측 고장 표시 있음", style={"width": "169px", "textAlign": "center", **LABEL_12}),
                html.Div("예측 고장 표시 없음", style={"width": "169px", "textAlign": "center", **LABEL_12})],
               8, {"height": "24px", "alignItems": "center"}),
        hstack([html.Div("실제 있음", style={"width": "72px", "display": "flex", "alignItems": "center", **LABEL_12}),
                cm_cell(c["tp"]), cm_cell(c["fn"])], 8, {"height": "160px"}),
        hstack([html.Div("실제 없음", style={"width": "72px", "display": "flex", "alignItems": "center", **LABEL_12}),
                cm_cell(c["fp"]), cm_cell(c["tn"])], 8, {"height": "160px"}),
        html.Div([html.Span("판정 임계값", style={**LABEL_12, "whiteSpace": "nowrap"}),
                  html.Span(f"{pr_cm['cutoff']:.3f}", style=NUM_12)],
                 style={"height": "32px", "display": "flex", "alignItems": "center", "gap": "8px"}),
    ])
    cmx = card("혼동행렬", 458, ROW_MAIN, cm_body)
    row_b = row(ROW_MAIN, [mcomp, prc, cmx])

    if task_sel == 1:
        fi_rows = load_family_feature_importance(family)
        max_importance = max(r["importance_mean"] for r in fi_rows)
        fi_list = html.Div(
            [html.Div([
                html.Div(r["feature"], style={"width": "220px", "flexShrink": "0", "fontSize": "12px",
                                               "color": INK, "whiteSpace": "nowrap", "overflow": "hidden",
                                               "textOverflow": "ellipsis"}),
                html.Div(style={"flexGrow": "1", "height": "7px", "borderRadius": "2px",
                                "width": f"{r['importance_mean'] / max_importance * 100}%",
                                "background": _FEATURE_CATEGORY_COLOR[_feature_color_category(r["feature"])]}),
                html.Div(f"{r['importance_mean']:.3f}",
                         style={**NUM_12, "width": "56px", "textAlign": "right", "flexShrink": "0"}),
             ], style={"height": "12px", "display": "flex", "alignItems": "center", "gap": "8px", "flexShrink": "0"})
             for r in fi_rows],
            style={"display": "flex", "flexDirection": "column", "gap": "3px"},
        )
        legend_dot = lambda color: html.Span(style={"width": "7px", "height": "7px", "borderRadius": "2px",
                                                      "background": color, "display": "inline-block"})
        fi_footer = html.Div(
            [html.Div([legend_dot(FEATURE_COLOR_HEALTH), html.Span("건강 센서", style=MICRO_11),
                       legend_dot(FEATURE_COLOR_OPERATING), html.Span("운전 조건", style=MICRO_11),
                       legend_dot(MUTED), html.Span("장비·부품·날짜", style=MICRO_11)],
                      style={"display": "flex", "alignItems": "center", "gap": "4px", "flexShrink": "0"}),
             note("중요도는 고장 원인이 아니라 고장과 함께 변화한 운전 상태를 뜻한다.",
                  {"whiteSpace": "normal", "textAlign": "right"})],
            style={"display": "flex", "alignItems": "center", "justifyContent": "space-between",
                   "gap": "12px", "marginTop": "8px", "paddingTop": "6px", "borderTop": f"1px solid {HAIR}"},
        )
        feat_body = html.Div([fi_list, fi_footer], style={"display": "flex", "flexDirection": "column"})
        feat = card("주요 영향 변수 상위 10", 774, 252, feat_body, right=note("10행 · 부품군 자체 속성(자산 무관)"))
    else:
        # prior·로지스틱회귀/랜덤포레스트 모두 변수중요도를 사전 계산해 두지
        # 않았다 — 그럴듯한 값을 새로 만들지 않고 데이터 없음을 그대로 밝힌다.
        feat_body = empty_state(
            "데이터 없음", "이 과제의 모델은 변수중요도를 사전 계산해 두지 않음", 742, 184,
        )
        feat = card("주요 영향 변수", 774, 252, feat_body, right=note("데이터 없음"))

    if task_sel == 0:
        default_threshold = pr_cm["cutoff"]
        threshold_basis = "검증 구간 기준"
        thr_metrics = load_current_classification_at_threshold("hist_gradient_boosting", default_threshold)
    elif task_sel == 2:
        default_threshold = pr_cm["cutoff"]
        # pf_within_7d는 임계값을 검증 구간에서 최적화하지 않고 0.5로 고정해
        # 둔 채로 학습됐다 — "검증 구간 기준"이라고 표시하면 사실이 아니므로
        # 있는 그대로 밝힌다.
        threshold_basis = "고정값 — 검증 구간 최적화 아님"
        thr_metrics = load_part_failure_at_threshold(selected_model, default_threshold)
    else:
        default_threshold = None
        threshold_basis = None

    slider = html.Div(
        [html.Label("판정 임계값", htmlFor="thr-slider", style={**LABEL_12, "whiteSpace": "nowrap"}),
         dcc.Slider(id={"type": "thr-slider", "index": "screen3"}, min=0, max=100, marks=None,
                    value=round(default_threshold * 100) if default_threshold is not None else 50,
                    tooltip={"placement": "bottom"}),
         note("설명용 가상 실험" + (f" · 기본값 {threshold_basis}" if threshold_basis else ""),
              {"whiteSpace": "nowrap"})],
        style={"display": "flex", "alignItems": "center", "gap": "8px", "width": "420px"},
    )
    if task_sel == 1:
        thr_body = empty_state("이 과제에는 해당 없음", "부품군 진단 과제에서는 표시하지 않음", 584, 184)
    else:
        thr_body = html.Div(_threshold_metrics_body(thr_metrics, default_threshold),
                             id={"type": "thr-metrics", "index": "screen3"})
    thr = card("판정 임계값 조정", 616, 252, thr_body, right=slider)

    # 과제 1(부품군 진단)은 실제 재발 간격 히스토그램이 이미 있어 그대로 둔다.
    # 과제 0·2는 원래 "해당 없음"이던 자리를 "운영 판단" 카드로 바꾼다 —
    # AP 향상배수 기준 규칙으로 지금 이 모델을 실사용에 써도 되는지 요약한다.
    if task_sel == 1:
        intervals = load_family_recurrence_intervals(asset_tag, family)
        if intervals:
            lead_body = html.Div(
                dcc.Graph(id={"type": "family-recur-chart", "index": "screen3"},
                          figure=_family_recur_figure(intervals, "light"),
                          config={"displayModeBar": False, "responsive": True},
                          style={"flex": "1 1 auto", "minHeight": "0", "minWidth": "0"}),
                style={"width": "426px", "height": "184px", "display": "flex", "flexDirection": "column"})
        else:
            lead_body = empty_state("이 과제에는 해당 없음",
                                     "재발 이력이 2회 미만이라 간격을 계산할 수 없음", 426, 184)
        lead = card("재발 간격", 458, 252, lead_body,
                    right=note("과거 재발 간격 — 예측이 아닌 회고적 통계"))
    else:
        lead_body = _operational_judgment_body(actual_rate,
                                                (current_metrics if task_sel == 0 else selected_metrics)["average_precision"],
                                                (current_metrics if task_sel == 0 else selected_metrics)["precision"],
                                                (current_metrics if task_sel == 0 else selected_metrics)["recall"],
                                                pr_cm["confusion"])
        lead = card("운영 판단", 458, 252, lead_body, right=note("AP 향상배수 기준 자동 판정"))
    row_c = row(252, [feat, thr, lead])

    interp_children = [interp_row] if interp_row is not None else []
    return html.Div([toolbar, row_a, *interp_children, row_b, row_c],
                     style={"display": "flex", "flexDirection": "column", "gap": f"{GUTTER}px"})


# ============================================================
# 화면 ④ 데이터 — 툴바 32 / 524 / 340
# ============================================================

# SEG_GROUPS["dataset"] 인덱스 → dashboard_data의 dataset 키.
DATASET_KEY_BY_SEG_INDEX = {0: "raw", 1: "daily"}


def _dtable_footer_text(total_rows, page, page_size=16):
    """"전체 행 수 · 표시 범위" 푸터 문구 — screen_4 초기 렌더와 정렬/페이지
    콜백이 공유한다."""
    if total_rows == 0:
        return "0행"
    start = page * page_size + 1
    end = min((page + 1) * page_size, total_rows)
    return f"{total_rows:,}행 중 {start:,}–{end:,}행 표시"


def screen_4(seg_state=None, audience=DEFAULT_AUDIENCE, assets=None, start=None):
    seg_state = seg_state or DEFAULT_SEG
    dataset_index = seg_state.get("dataset", 0)
    # 예전에 저장된 선택(없어진 데이터셋 인덱스)은 원자료로 되돌린다.
    if dataset_index not in DATASET_KEY_BY_SEG_INDEX:
        dataset_index = 0
    dataset_key = DATASET_KEY_BY_SEG_INDEX[dataset_index]

    page0 = load_table_page(dataset_key, None, "asc", 0, assets=assets, start=start)
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
    csv_btn = html.Button("CSV 내보내기", id={"type": "csv-export-btn", "index": dataset_key},
                          n_clicks=0, style=btn_style())

    toolbar = html.Div(
        [seg("dataset", "데이터셋", SEG_GROUPS["dataset"], sel=dataset_index),
         html.Div([csv_btn], style={"display": "flex", "alignItems": "center", "gap": "12px"})],
        style={"width": "1880px", "height": "32px", "flexShrink": "0", "display": "flex",
               "alignItems": "center", "justifyContent": "space-between"},
    )

    dtable = card(f"데이터 조회 · {SEG_GROUPS['dataset'][dataset_index]}", 1880, 524, dtable_body,
                  right=note("공장·기계 종류·기계·기간 필터 적용"))

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
                 right=note("데이터셋 전체 기준 · 필터 미적용"))

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
    qual = card("품질 요약", 932, 162, hstack(quality_tiles, 16), right=note("데이터셋 전체 기준 · 필터 미적용"))

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


# screen_5()는 탭에서는 빠지지만 함수 자체는 지우지 않는다 — 헤더의 내보내기
# 드롭다운(REPORT_AUDIENCES/build_report_pdf/xlsx)이 화면⑤ 없이도 정상
# 동작하는지는 확인했지만, screen_5()가 그 보고서 구성의 참조 구현이라
# 남겨 둔다(요청: "함수 자체는 삭제하지 마라").
SCREEN_BUILDERS = {"1": screen_1, "2": screen_2, "3": screen_3, "4": screen_4}


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
            ("기간", PERIOD_PRESETS[_period_index(filters)][0]),
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
app.title = "설비 모니터링 대시보드"
app.index_string = INDEX_STRING
if not DEMO_MODE:
    install_auth(app.server)
ADMIN_INPUT_STYLE = {"display": "block", "width": "100%", "boxSizing": "border-box",
                     "marginBottom": "8px", "padding": "7px", "fontSize": "13px"}

# 별도 /assets 요청을 하지 않는 내장 SVG입니다. 로그인 보호 과정에서 아이콘 파일이
# 401로 막혀 깨진 이미지로 보이는 문제를 피하기 위해 화면 코드에 직접 포함합니다.
SESSION_REFRESH_ICON = (
    "data:image/svg+xml;base64,"
    "PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCA2NCA2NCIgd2lkdGg9"
    "IjY0IiBoZWlnaHQ9IjY0IiBmaWxsPSJub25lIiBzdHJva2U9IiMxMTExMTEiIHN0cm9rZS13aWR0aD0iOSIgc3Ryb2tl"
    "LWxpbmVjYXA9ImJ1dHQiIHN0cm9rZS1saW5lam9pbj0ibWl0ZXIiPjxkZWZzPjxtYXJrZXIgaWQ9ImFycm93LWhlYWQi"
    "IG1hcmtlcldpZHRoPSIxNiIgbWFya2VySGVpZ2h0PSIxNiIgcmVmWD0iMTMiIHJlZlk9IjgiIG9yaWVudD0iYXV0byIg"
    "bWFya2VyVW5pdHM9InVzZXJTcGFjZU9uVXNlIj48cGF0aCBkPSJNMCAwIDE2IDggMCAxNloiIGZpbGw9IiMxMTExMTEi"
    "IHN0cm9rZT0ibm9uZSIvPjwvbWFya2VyPjwvZGVmcz48cGF0aCBkPSJNNTIgMjVDNDcgMTMgMzQgOSAyNCAxNCAxNiAx"
    "OCAxMSAyNSAxMSAzNCIgbWFya2VyLWVuZD0idXJsKCNhcnJvdy1oZWFkKSIvPjxwYXRoIGQ9Ik0xMiA0MEMxNyA1MiAz"
    "MCA1NiA0MCA1MSA0OCA0NyA1MyA0MCA1MyAzMSIgbWFya2VyLWVuZD0idXJsKCNhcnJvdy1oZWFkKSIvPjwvc3ZnPg=="
)

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
        dcc.Store(id="audit-sync-status"),
        dcc.Store(id="audit-audience-event"),
        # 서버가 발급한 만료 시각만 담는다. 비밀번호·키는 브라우저에 두지 않는다.
        dcc.Store(id="session-state"),
        dcc.Store(id="session-prompt-state", data={"shown": False}),
        dcc.Store(id="account-delete-mode", data=False),
        dcc.Store(id="account-delete-refresh", data=0),
        dcc.Interval(id="audit-csv-interval", interval=60_000, n_intervals=0),
        dcc.Interval(id="session-timer", interval=1_000, n_intervals=0, disabled=DEMO_MODE),
        dcc.Location(id="auth-redirect", refresh=True),
        # 이전 콜백의 State 식별자는 유지하되 사용자 입력은 제거한다.
        # 실제 actor_id는 audit_service가 검증된 Flask 세션에서 읽는다.
        dcc.Input(id="actor-id-input", type="hidden", value=""),
        # 로그아웃 확인 창과 같은 형식의 중앙 비밀번호 변경 팝업입니다.
        html.Div(id="password-panel", children=[
            html.Div([
                html.H3("비밀번호 변경", style={"marginTop": 0, "marginBottom": "18px", "textAlign": "left"}),
                html.P("현재 비밀번호를 확인한 뒤 새 비밀번호로 바꿉니다.",
                       style={"marginBottom": "18px", "textAlign": "center"}),
                dcc.Input(id="password-current", type="password", placeholder="현재 비밀번호", style=ADMIN_INPUT_STYLE),
                dcc.Input(id="password-new", type="password", placeholder="새 비밀번호 (4자 이상)", style=ADMIN_INPUT_STYLE),
                dcc.Input(id="password-confirm", type="password", placeholder="새 비밀번호 다시 입력", style=ADMIN_INPUT_STYLE),
                html.Div([
                    html.Button("변경", id="password-save-btn", n_clicks=0,
                                style={"minWidth": "120px", "padding": "11px 18px", "border": "1px solid #2563eb",
                                       "borderRadius": "6px", "background": "#2563eb", "color": "#fff",
                                       "fontSize": "15px", "fontWeight": "700", "cursor": "pointer"}),
                    html.Button("닫기", id="password-close-btn", n_clicks=0,
                                style={"minWidth": "120px", "padding": "11px 18px", "border": "1px solid #b8c0ca",
                                       "borderRadius": "6px", "background": "#fff", "color": "#1f2937",
                                       "fontSize": "15px", "fontWeight": "700", "cursor": "pointer"}),
                ], style={"display": "flex", "justifyContent": "center", "gap": "10px", "marginTop": "18px"}),
                html.Div(id="password-result", role="status",
                         style={"marginTop": "12px", "minHeight": "18px", "textAlign": "center", "color": "#b42318"}),
            ], style={"width": "360px", "background": "#fff", "color": "#222", "padding": "22px",
                      "borderRadius": "10px", "boxShadow": "0 12px 40px #0005"}),
        ], style={"display": "none", "position": "fixed", "inset": 0, "zIndex": 300,
                  "background": "#0006", "alignItems": "center", "justifyContent": "center"}),
        html.Div(id="session-expiry-modal", children=[
            html.Div([
                html.H3("로그인 시간 연장", style={"marginTop": 0, "marginBottom": "18px", "textAlign": "left"}),
                html.P("로그인 시간이 곧 만료됩니다. 계속 사용하시겠습니까?",
                       style={"marginBottom": "26px", "textAlign": "center"}),
                html.Div([
                    html.Button("예 (10초)", id="session-modal-extend-btn", n_clicks=0,
                                style={"minWidth": "128px", "padding": "11px 18px", "border": "1px solid #2563eb",
                                       "borderRadius": "6px", "background": "#2563eb", "color": "#fff",
                                       "fontSize": "15px", "fontWeight": "700", "cursor": "pointer"}),
                    html.Button("아니오", id="session-modal-logout-btn", n_clicks=0,
                                style={"minWidth": "110px", "padding": "11px 18px", "border": "1px solid #b8c0ca",
                                       "borderRadius": "6px", "background": "#fff", "color": "#1f2937",
                                       "fontSize": "15px", "fontWeight": "700", "cursor": "pointer"}),
                ], style={"display": "flex", "justifyContent": "center", "gap": "10px"}),
            ], style={"width": "360px", "background": "#fff", "color": "#222", "padding": "22px",
                      "borderRadius": "10px", "boxShadow": "0 12px 40px #0005", "textAlign": "center"}),
        ], style={"display": "none", "position": "fixed", "inset": 0, "zIndex": 300,
                  "background": "#0006", "alignItems": "center", "justifyContent": "center"}),
        html.Div(id="logout-confirm-modal", children=[
            html.Div([
                html.H3("로그아웃", style={"marginTop": 0, "marginBottom": "18px", "textAlign": "left"}),
                html.P("로그아웃하시겠습니까?", style={"marginBottom": "26px", "textAlign": "center"}),
                html.Div([
                    html.Button("예", id="logout-confirm-yes-btn", n_clicks=0,
                                style={"minWidth": "120px", "padding": "11px 18px", "border": "1px solid #2563eb",
                                       "borderRadius": "6px", "background": "#2563eb", "color": "#fff",
                                       "fontSize": "15px", "fontWeight": "700", "cursor": "pointer"}),
                    html.Button("아니오", id="logout-confirm-no-btn", n_clicks=0,
                                style={"minWidth": "120px", "padding": "11px 18px", "border": "1px solid #b8c0ca",
                                       "borderRadius": "6px", "background": "#fff", "color": "#1f2937",
                                       "fontSize": "15px", "fontWeight": "700", "cursor": "pointer"}),
                ], style={"display": "flex", "justifyContent": "center", "gap": "10px"}),
            ], style={"width": "330px", "background": "#fff", "color": "#222", "padding": "22px",
                      "borderRadius": "10px", "boxShadow": "0 12px 40px #0005", "textAlign": "center"}),
        ], style={"display": "none", "position": "fixed", "inset": 0, "zIndex": 310,
                  "background": "#0006", "alignItems": "center", "justifyContent": "center"}),
        html.Div(id="password-success-modal", children=[
            html.Div([
                html.H3("비밀번호 변경 완료", style={"marginTop": 0, "marginBottom": "18px", "textAlign": "left"}),
                html.P("비밀번호가 변경되었습니다. 다시 로그인해 주세요.",
                       style={"marginBottom": "26px", "textAlign": "center"}),
                html.Button("확인", id="password-success-confirm-btn", n_clicks=0,
                            style={"minWidth": "120px", "padding": "11px 18px", "border": "1px solid #2563eb",
                                   "borderRadius": "6px", "background": "#2563eb", "color": "#fff",
                                   "fontSize": "15px", "fontWeight": "700", "cursor": "pointer"}),
            ], style={"width": "360px", "background": "#fff", "color": "#222", "padding": "22px",
                      "borderRadius": "10px", "boxShadow": "0 12px 40px #0005", "textAlign": "center"}),
        ], style={"display": "none", "position": "fixed", "inset": 0, "zIndex": 320,
                  "background": "#0006", "alignItems": "center", "justifyContent": "center"}),
        # ADMIN 역할에서만 열 수 있는 일반 계정 목록입니다. 이름은 서버에서만 복호화합니다.
        html.Div(id="account-management-modal", children=[
            html.Div([
                html.Div([
                    html.H3("계정 관리", style={"margin": 0, "textAlign": "left"}),
                    html.Button("계정 삭제", id="account-delete-mode-btn", n_clicks=0,
                                style={"padding": "7px 11px", "border": "1px solid #b42318",
                                       "borderRadius": "6px", "background": "#fff", "color": "#b42318",
                                       "fontWeight": "700", "cursor": "pointer"}),
                ], style={"display": "flex", "alignItems": "center", "justifyContent": "space-between",
                          "marginBottom": "18px"}),
                html.Div(id="account-list-body"),
                html.Div(id="account-delete-header", children=[
                    html.Span("아이디", style={"fontWeight": "700"}),
                    html.Span("이름", style={"fontWeight": "700"}),
                    html.Span("접속 상태", style={"fontWeight": "700", "textAlign": "right"}),
                    html.Span("선택", style={"fontWeight": "700", "textAlign": "right"}),
                ], style={"display": "none", "gridTemplateColumns": "1fr 1fr 90px 38px",
                          "gap": "8px", "padding": "8px 4px", "borderBottom": "1px solid #d7dde3"}),
                dcc.Checklist(id="account-delete-selection", options=[], value=[],
                              style={"display": "none"},
                              inputStyle={"marginLeft": "8px", "cursor": "pointer"},
                              labelStyle={"display": "flex", "flexDirection": "row-reverse", "width": "100%",
                                          "alignItems": "center", "padding": "10px 4px",
                                          "borderBottom": "1px solid #eef0f2", "cursor": "pointer"}),
                html.Div(id="account-delete-feedback", role="status",
                         style={"minHeight": "18px", "fontSize": "13px", "color": "#b42318",
                                "textAlign": "center", "marginTop": "12px"}),
                html.Div([
                    html.Button("닫기", id="account-management-close-btn", n_clicks=0,
                                style={"minWidth": "120px", "padding": "11px 18px", "border": "1px solid #b8c0ca",
                                       "borderRadius": "6px", "background": "#fff", "color": "#1f2937",
                                       "fontSize": "15px", "fontWeight": "700", "cursor": "pointer"}),
                    html.Button("확인", id="account-delete-confirm-btn", n_clicks=0,
                                style={"display": "none", "minWidth": "120px", "padding": "11px 18px",
                                       "border": "1px solid #2563eb", "borderRadius": "6px",
                                       "background": "#2563eb", "color": "#fff", "fontSize": "15px",
                                       "fontWeight": "700", "cursor": "pointer", "marginLeft": "auto"}),
                ], style={"display": "flex", "justifyContent": "center", "marginTop": "20px"}),
            ], style={"width": "520px", "maxWidth": "calc(100vw - 32px)", "background": "#fff", "color": "#222",
                      "padding": "22px", "borderRadius": "10px", "boxShadow": "0 12px 40px #0005"}),
        ], style={"display": "none", "position": "fixed", "inset": 0, "zIndex": 330,
                  "background": "#0006", "alignItems": "center", "justifyContent": "center"}),
        html.Div(id="account-delete-confirm-modal", children=[
            html.Div([
                html.H3("계정 삭제", style={"marginTop": 0, "marginBottom": "18px", "textAlign": "left"}),
                html.P(id="account-delete-confirm-message",
                       style={"marginBottom": "26px", "textAlign": "center"}),
                html.Div([
                    html.Button("예", id="account-delete-yes-btn", n_clicks=0,
                                style={"minWidth": "120px", "padding": "11px 18px", "border": "1px solid #2563eb",
                                       "borderRadius": "6px", "background": "#2563eb", "color": "#fff",
                                       "fontSize": "15px", "fontWeight": "700", "cursor": "pointer"}),
                    html.Button("아니오", id="account-delete-no-btn", n_clicks=0,
                                style={"minWidth": "120px", "padding": "11px 18px", "border": "1px solid #b8c0ca",
                                       "borderRadius": "6px", "background": "#fff", "color": "#1f2937",
                                       "fontSize": "15px", "fontWeight": "700", "cursor": "pointer"}),
                ], style={"display": "flex", "justifyContent": "center", "gap": "10px"}),
            ], style={"width": "380px", "maxWidth": "calc(100vw - 32px)", "background": "#fff",
                      "color": "#222", "padding": "22px", "borderRadius": "10px",
                      "boxShadow": "0 12px 40px #0005"}),
        ], style={"display": "none", "position": "fixed", "inset": 0, "zIndex": 340,
                  "background": "#0006", "alignItems": "center", "justifyContent": "center"}),
        # 일반 계정이 PDF·Excel을 누르면 파일 생성 없이 이 안내만 보여 준다.
        html.Div(id="export-access-modal", children=[
            html.Div([
                html.H3("접근 권한", style={"marginTop": 0, "marginBottom": "18px", "textAlign": "left"}),
                html.P("접근 권한이 필요합니다.",
                       style={"marginBottom": "26px", "textAlign": "center"}),
                html.Div([
                    html.Button("확인", id="export-access-modal-close", n_clicks=0,
                                style={"minWidth": "120px", "padding": "11px 18px", "border": "1px solid #2563eb",
                                       "borderRadius": "6px", "background": "#2563eb", "color": "#fff",
                                       "fontSize": "15px", "fontWeight": "700", "cursor": "pointer"}),
                ], style={"display": "flex", "justifyContent": "center"}),
            ], style={"width": "330px", "background": "#fff", "color": "#222", "padding": "22px",
                      "borderRadius": "10px", "boxShadow": "0 12px 40px #0005", "textAlign": "center"}),
        ], style={"display": "none", "position": "fixed", "inset": 0, "zIndex": 340,
                  "background": "#0006", "alignItems": "center", "justifyContent": "center"}),
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
    Input("prio-sort-store", "data"),
    Input("filter-store", "data"),
    Input("selected-family-store", "data"),
    State("actor-id-input", "value"),
)
def render_screen(active, seg_state, prio_sort, filters, selected_family, actor_id):
    if ctx.triggered_id == "screen-tabs":
        record_action("ACT_TAB_OPEN", "화면 이동", actor_id=actor_id,
                      target_type="tab", target_id=active)
    assets, start = _filter_scope(filters)
    if active in ("1", "2", "4") and not assets:
        return empty_state("조건에 맞는 기계 없음",
                           "선택한 공장·기계 종류·기계 조합에 해당하는 기계가 없다", 1880, 400)
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
    except Exception as exc:
        log_failure("ACT_SCREEN_RENDER", "화면 조회", exc, actor_id=actor_id,
                    source="wireframe_app.render_screen", target_id=active)
        return html.Div("화면을 불러오지 못했습니다. 오류 로그를 확인해 주세요.")


# ②의 "‹ 이전 기계"/"다음 기계 ›" — 공장·기계 종류 필터 안에서 알파벳순으로
# 순환하고, 고른 기계를 필터의 기계 값으로 쓴다(write_filters가 이어서 저장).
# nav_btn()이 공용 ghost-btn과 다른 id 타입을 쓰므로 echo_action과 겹치지 않는다.
@app.callback(
    Output("machine-dd", "value", allow_duplicate=True),
    Input({"type": "machine-nav-btn", "index": ALL}, "n_clicks"),
    State("filter-store", "data"),
    State("actor-id-input", "value"),
    prevent_initial_call=True,
)
def cycle_selected_asset(_clicks, filters, actor_id):
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
    chosen = assets[idx]
    record_action("ACT_ASSET_NAVIGATE", "설비 상세 이동", actor_id=actor_id,
                  target_type="asset", target_id=chosen)
    return chosen


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
    Input({"type": "thr-slider", "index": ALL}, "value"),
    State("seg-store", "data"),
    prevent_initial_call=True,
)
def recompute_threshold_metrics(slider_values, seg_state):
    if not slider_values or slider_values[0] is None:
        return no_update
    task_sel = (seg_state or DEFAULT_SEG).get("task", 0)
    threshold = slider_values[0] / 100
    if task_sel == 0:
        metrics = load_current_classification_at_threshold("hist_gradient_boosting", threshold)
    elif task_sel == 2:
        metrics = load_part_failure_at_threshold(load_part_failure_selected_model(), threshold)
    else:
        return no_update
    return [_threshold_metrics_body(metrics, threshold)]


# ------------------------------------------------------------
# 테마 전환 — 루트 className만 갈아끼우면 CSS 변수가 전부 따라 바뀐다
# ------------------------------------------------------------
@app.callback(
    Output("theme-store", "data"),
    Input("theme-btn", "n_clicks"),
    State("theme-store", "data"),
    State("actor-id-input", "value"),
    prevent_initial_call=True,
)
def toggle_theme(_n, current, actor_id):
    new_theme = "light" if (current or "light") == "dark" else "dark"
    record_action("ACT_THEME_CHANGE", "테마 전환", actor_id=actor_id,
                  target_type="theme", target_id=new_theme)
    return new_theme


@app.callback(
    Output("root", "className"),
    Input("theme-store", "data"),
)
def apply_theme(theme):
    theme = theme if theme in ("light", "dark") else "light"
    return f"theme-{theme}"


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
    return [_heatmap_figure(load_asset_failure_heatmap(assets, start), theme or "light")]


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
    return [_family_pr_figure(pr_cm, theme or "light")]


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


# ③ "현재 고장 표시 분류"/"부품 고장 탐지" 과제의 PR곡선도 같은 방식으로
# 재색칠한다. 이 둘은 선택 자산·부품군과 무관한 전체 테스트 구간 지표라
# family-pr-chart와 달리 State가 필요 없다 — heatmap-chart와 같은 패턴.
@app.callback(
    Output({"type": "classification-pr-chart", "index": ALL}, "figure"),
    Input("theme-store", "data"),
    Input("screen-tabs", "value"),
)
def recolor_classification_pr_chart(theme, _active_tab):
    pr_cm = load_current_classification_pr_curve_and_confusion("hist_gradient_boosting")
    return [_family_pr_figure(pr_cm, theme or "light")]


@app.callback(
    Output({"type": "partfail-pr-chart", "index": ALL}, "figure"),
    Input("theme-store", "data"),
    Input("screen-tabs", "value"),
)
def recolor_partfail_pr_chart(theme, _active_tab):
    pr_cm = load_part_failure_pr_curve_and_confusion()
    return [_family_pr_figure(pr_cm, theme or "light")]


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
    State("actor-id-input", "value"),
)
def write_filters(plant, machine_type, machine, _period_clicks, _reset_clicks, current, actor_id):
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
    record_action("ACT_FILTER_CHANGE", "조회 조건 변경", actor_id=actor_id,
                  target_type="filter", target_id=str(trigger)[:100], detail=new)
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
    Input("screen-tabs", "value"),
    Input("seg-store", "data"),
)
def apply_filter_scope(active, seg_state):
    period_count = len(PERIOD_PRESETS)
    if active != "3":
        return False, False, False, [False] * period_count, ""
    family_task = (seg_state or DEFAULT_SEG).get("task", 0) == 1
    note_text = ("모델 지표는 고정 평가 구간 전체 결과라 공장·기계 종류·기간 필터 미적용"
                 + ("" if family_task else " · 기계 필터는 부품군 진단 과제에만 적용"))
    return True, True, not family_task, [True] * period_count, note_text


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
    pi = _period_index(data)
    styles = [period_btn_style(i == pi) for i in range(len(PERIOD_PRESETS))]
    label, days = PERIOD_PRESETS[pi]
    start = period_start(days)
    period_text = label if start is None else f"{label} {start:%Y-%m-%d}~{load_data_reference_date()}"
    echo = (f"공장={data.get('plant') or '전체'} · 종류={data.get('machine_type') or '전체'} · "
            f"기계={data.get('machine') or '전체'} · 기간={period_text}")
    return styles, echo


# ------------------------------------------------------------
# 세그먼티드 컨트롤 — ③ 과제 / ③ 위험 기준선 / ④ 데이터셋
# 선택이 바뀌면 seg-store가 갱신되고, render_screen이 화면을 다시 그린다.
# ------------------------------------------------------------
@app.callback(
    Output("seg-store", "data"),
    Input({"type": "seg-btn", "group": ALL, "index": ALL}, "n_clicks"),
    State("seg-store", "data"),
    State("actor-id-input", "value"),
    prevent_initial_call=True,
)
def write_seg(_clicks, current, actor_id):
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
    record_action("ACT_SEGMENT_CHANGE", "분석 선택", actor_id=actor_id,
                  target_type=str(tid["group"]), target_id=str(tid["index"]))
    return new


# ------------------------------------------------------------
# 기능 미구현 버튼 — 클릭은 되고, 무엇이 없는지 그대로 알려준다
# ------------------------------------------------------------
@app.callback(
    Output("action-echo", "children"),
    Input({"type": "ghost-btn", "index": ALL}, "n_clicks"),
    State("actor-id-input", "value"),
    prevent_initial_call=True,
)
def echo_action(_clicks, actor_id):
    trig = ctx.triggered[0] if ctx.triggered else None
    if not trig or not trig.get("value"):
        return no_update
    tid = ctx.triggered_id
    label = tid["index"] if isinstance(tid, dict) else str(tid)
    record_action("ACT_UNAVAILABLE", "미구현 기능 클릭", actor_id=actor_id,
                  target_type="button", target_id=str(label), status="BLOCKED",
                  block_reason="아직 구현되지 않은 기능")
    return f"'{label}' 클릭됨 — 동작 미구현 ({NO_DATA_MARK})"


# ------------------------------------------------------------
# ① KPI "위험 기준선 초과 비율" 드릴다운 — 클릭할 때마다 "점검 우선순위" 표 정렬을
# 등급가중 고장점수 ↔ 기준선 초과 사이로 토글한다. 화면을 새로 만들지 않고
# 이미 있는 표를 재사용해 원인(어느 공장·기계·부품)까지 이어지게 한다.
# ------------------------------------------------------------
@app.callback(
    Output("prio-sort-store", "data"),
    Output("action-echo", "children", allow_duplicate=True),
    Input({"type": "kpi-drill", "index": ALL}, "n_clicks"),
    State("prio-sort-store", "data"),
    State("actor-id-input", "value"),
    prevent_initial_call=True,
)
def drill_failrate_to_priority(n_clicks_list, current, actor_id):
    # ALL 패턴이라 리스트로 온다 — ①이 화면에 없을 땐 빈 리스트, 있을 땐 [n].
    # 화면 재렌더로 타일이 다시 만들어질 때의 n_clicks=0도 함께 무시한다.
    if not n_clicks_list or not any(n_clicks_list):
        return no_update, no_update
    current = current or DEFAULT_PRIO_SORT
    new_sort_by = "grade" if current.get("sort_by") == "threshold" else "threshold"
    msg = ("'위험 기준선 초과 비율' 클릭 → 점검 우선순위를 기준선 초과 기준으로 정렬"
           if new_sort_by == "threshold" else
           "'위험 기준선 초과 비율' 다시 클릭 → 점검 우선순위를 등급가중 고장점수 기준으로 복귀")
    new_state = {"sort_by": new_sort_by, "direction": current.get("direction", "desc")}
    record_action("ACT_PRIORITY_SORT", "점검 우선순위 정렬", actor_id=actor_id,
                  target_type="priority", target_id=f"{new_sort_by}:{new_state['direction']}")
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
    Output("export-access-modal", "style"),
    Input("export-run-btn", "n_clicks"),
    Input("export-access-modal-close", "n_clicks"),
    State("export-dd", "value"),
    State("filter-store", "data"),
    State("seg-store", "data"),
    State("actor-id-input", "value"),
    prevent_initial_call=True,
)
def export_report(_run, _close, export_value, filters, seg_state, actor_id):
    if ctx.triggered_id == "export-access-modal-close":
        return no_update, no_update, {"display": "none"}

    fmt, audience = (export_value or f"pdf:{DEFAULT_AUDIENCE}").split(":")
    audit_format = "excel" if fmt == "xlsx" else fmt
    # 버튼은 보이지만, 파일을 만드는 직전에 서버 세션의 역할을 다시 확인한다.
    # 일반 사용자가 요청을 직접 바꾸더라도 PDF·Excel은 받을 수 없다.
    if not DEMO_MODE and str(session.get("role", "UNKNOWN")).upper() != "ADMIN":
        record_action("ACT_EXPORT_ACCESS_BLOCKED", "파일 내보내기 접근 차단", actor_id=actor_id,
                      target_type="export", target_id=audit_format, status="BLOCKED",
                      block_reason="관리자 역할이 필요합니다.")
        return no_update, "파일 내보내기는 관리자 계정만 사용할 수 있습니다.", {
            "display": "flex", "position": "fixed", "inset": 0, "zIndex": 340,
            "background": "#0006", "alignItems": "center", "justifyContent": "center",
        }
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
        log_failure("ACT_REPORT_EXPORT", "보고서 내보내기", exc, actor_id=actor_id,
                    source="wireframe_app.export_report", target_id=ctx.triggered_id)
        return no_update, f"내보내기 실패 — {type(exc).__name__}", {"display": "none"}
    record_action("ACT_REPORT_EXPORT", "보고서 내보내기", actor_id=actor_id,
                  target_type="export", target_id=audit_format,
                  detail={"audience": audience, "format": audit_format})
    return (dcc.send_bytes(lambda b: b.write(payload), name, type=mime),
            f"{aud['label']} 보고서 내보냄 — {name} (섹션 {len(aud['sections'])}개 · 값은 {NO_DATA_MARK})",
            {"display": "none"})


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
    State("actor-id-input", "value"),
    prevent_initial_call=True,
)
def update_dtable_page(page_current, sort_by, filters, actor_id):
    dataset_key = ctx.triggered_id["index"]
    sort_col = sort_by[0]["column_id"] if sort_by else None
    sort_dir = sort_by[0]["direction"] if sort_by else "asc"
    triggered_prop = ctx.triggered[0]["prop_id"].rsplit(".", 1)[-1] if ctx.triggered else None
    page = 0 if triggered_prop == "sort_by" else (page_current or 0)
    try:
        assets, start = _filter_scope(filters)
        result = load_table_page(dataset_key, sort_col, sort_dir, page, assets=assets, start=start)
    except Exception as exc:
        log_failure("ACT_DATA_QUERY", "데이터 표 조회", exc, actor_id=actor_id,
                    source="wireframe_app.update_dtable_page", target_id=dataset_key)
        raise
    if triggered_prop in ("sort_by", "page_current") and ctx.triggered[0].get("value") is not None:
        record_action("ACT_DATA_QUERY", "데이터 표 조회", actor_id=actor_id,
                      target_type="dataset", target_id=dataset_key,
                      detail={"page": page, "sort_column": sort_col, "sort_direction": sort_dir})
    footer_text = _dtable_footer_text(result["total_rows"], result["page"], 16)
    return result["data"], result["page_count"], result["page"], footer_text


@app.callback(
    Output("table-download", "data"),
    Input({"type": "csv-export-btn", "index": ALL}, "n_clicks"),
    State("filter-store", "data"),
    State("actor-id-input", "value"),
    prevent_initial_call=True,
)
def export_dtable_csv(_clicks, filters, actor_id):
    # 화면이 다시 그려지면 버튼이 n_clicks=0으로 재생성되며 이 콜백이 한 번
    # 더 불린다 — write_seg와 동일하게 값이 0인 호출은 무시한다.
    trig = ctx.triggered[0] if ctx.triggered else None
    if not trig or not trig.get("value"):
        return no_update
    dataset_key = ctx.triggered_id["index"]
    if dataset_key not in ("raw", "daily"):
        record_action("ACT_CSV_EXPORT", "CSV 다운로드", actor_id=actor_id,
                      target_type="dataset", target_id=dataset_key, status="BLOCKED",
                      block_reason="지원하지 않는 데이터셋")
        return no_update
    try:
        assets, start = _filter_scope(filters)
        csv_bytes, filename = export_table_csv(dataset_key, assets=assets, start=start)
    except Exception as exc:
        log_failure("ACT_CSV_EXPORT", "CSV 다운로드", exc, actor_id=actor_id,
                    source="wireframe_app.export_dtable_csv", target_id=dataset_key)
        return no_update
    record_action("ACT_CSV_EXPORT", "CSV 다운로드", actor_id=actor_id,
                  target_type="dataset", target_id=dataset_key, detail={"filename": filename})
    return dcc.send_bytes(lambda buf: buf.write(csv_bytes), filename, type="text/csv")


@app.callback(Output("password-panel", "style"), Input("password-open-btn", "n_clicks"),
              Input("password-close-btn", "n_clicks"), State("password-panel", "style"),
              prevent_initial_call=True)
def toggle_password_panel(_open, _close, style):
    updated = dict(style or {})
    updated["display"] = "none" if ctx.triggered_id == "password-close-btn" else "flex"
    return updated


@app.callback(Output("profile-display", "children"), Output("profile-login-id", "children"),
              Output("session-state", "data"), Output("account-manage-btn", "style"),
              Input("screen-tabs", "value"))
def display_profile(_active_tab):
    login_id = str(session.get("admin_id", "관리자"))
    role = str(session.get("role", "UNKNOWN")).upper()
    account_button = {"display": "block", "width": "100%", "padding": "8px", "marginBottom": "5px"}
    if role != "ADMIN":
        account_button["display"] = "none"
    return login_id[:2].upper(), login_id, current_session_state(), account_button


@app.callback(Output("account-management-modal", "style"), Output("account-list-body", "children"),
              Input("account-manage-btn", "n_clicks"), Input("account-management-close-btn", "n_clicks"),
              Input("account-delete-refresh", "data"),
              prevent_initial_call=True)
def toggle_account_management(open_clicks, close_clicks, refresh):
    """일반 계정 목록은 ADMIN 역할의 서버 세션에서만 복호화해 보여 준다."""
    if not (open_clicks or close_clicks or refresh):
        return no_update, no_update
    if ctx.triggered_id == "account-management-close-btn":
        return {"display": "none"}, no_update
    if str(session.get("role", "UNKNOWN")).upper() != "ADMIN":
        return {"display": "none"}, no_update
    rows = list_user_accounts()
    record_action("ACT_ACCOUNT_LIST_VIEW", "일반 계정 목록 조회", target_type="account", target_id="USER")
    if not rows:
        return ({"display": "flex", "position": "fixed", "inset": 0, "zIndex": 330,
                 "background": "#0006", "alignItems": "center", "justifyContent": "center"},
                html.P("생성된 일반 계정이 없습니다.", style={"textAlign": "center", "margin": "12px 0"}))
    header = html.Div([
        html.Span("아이디", style={"fontWeight": "700"}),
        html.Span("이름", style={"fontWeight": "700"}),
        html.Span("접속 상태", style={"fontWeight": "700", "textAlign": "right"}),
    ], style={"display": "grid", "gridTemplateColumns": "1fr 1fr 90px", "gap": "12px",
              "padding": "8px 4px", "borderBottom": "1px solid #d7dde3"})
    items = [header]
    for row in rows:
        online_badge = html.Span(
            "● 접속 중" if row["is_online"] else "○ 미접속",
            style={"color": "#15803d" if row["is_online"] else "#6b7280", "fontSize": "12px",
                   "fontWeight": "700", "textAlign": "right",
                   "textShadow": "0 0 7px #86efac" if row["is_online"] else "none"},
        )
        items.append(html.Div([
            html.Span(row["login_id"]), html.Span(row["name"]), online_badge,
        ], style={"display": "grid", "gridTemplateColumns": "1fr 1fr 90px", "gap": "12px",
                  "padding": "10px 4px", "borderBottom": "1px solid #eef0f2"}))
    return ({"display": "flex", "position": "fixed", "inset": 0, "zIndex": 330,
             "background": "#0006", "alignItems": "center", "justifyContent": "center"}, items)


@app.callback(Output("account-delete-selection", "options"),
              Input("account-manage-btn", "n_clicks"), Input("account-delete-refresh", "data"),
              prevent_initial_call=True)
def load_account_delete_options(open_clicks, refresh):
    """선택 행 전체를 눌러 체크할 수 있게 일반 계정만 표시한다."""
    if not (open_clicks or refresh) or str(session.get("role", "UNKNOWN")).upper() != "ADMIN":
        return []
    options = []
    for row in list_user_accounts():
        online = bool(row["is_online"])
        label = html.Div([
            html.Span(row["login_id"]),
            html.Span(row["name"]),
            html.Span("● 접속 중" if online else "○ 미접속",
                      style={"color": "#15803d" if online else "#6b7280", "fontSize": "12px",
                             "fontWeight": "700", "textAlign": "right"}),
        ], style={"display": "grid", "gridTemplateColumns": "1fr 1fr 90px", "gap": "8px",
                  "alignItems": "center", "width": "100%"})
        options.append({"label": label, "value": row["login_id"]})
    return options


@app.callback(Output("account-delete-mode", "data"), Output("account-delete-selection", "value"),
              Input("account-manage-btn", "n_clicks"),
              Input("account-management-close-btn", "n_clicks"),
              Input("account-delete-mode-btn", "n_clicks"),
              Input("account-delete-refresh", "data"),
              State("account-delete-mode", "data"), prevent_initial_call=True)
def toggle_account_delete_mode(_open, _close, _mode_click, _refresh, current_mode):
    if ctx.triggered_id == "account-delete-mode-btn":
        if str(session.get("role", "UNKNOWN")).upper() != "ADMIN":
            return False, []
        return not bool(current_mode), []
    return False, []


@app.callback(Output("account-list-body", "style"), Output("account-delete-header", "style"),
              Output("account-delete-selection", "style"),
              Output("account-delete-confirm-btn", "style"),
              Output("account-delete-mode-btn", "children"),
              Input("account-delete-mode", "data"))
def show_account_delete_mode(enabled):
    confirm_style = {"display": "inline-block" if enabled else "none", "minWidth": "120px",
                     "padding": "11px 18px", "border": "1px solid #2563eb", "borderRadius": "6px",
                     "background": "#2563eb", "color": "#fff", "fontSize": "15px", "fontWeight": "700",
                     "cursor": "pointer", "marginLeft": "auto"}
    header_style = {"display": "grid" if enabled else "none",
                    "gridTemplateColumns": "1fr 1fr 90px 38px", "gap": "8px",
                    "padding": "8px 4px", "borderBottom": "1px solid #d7dde3"}
    return ({"display": "none" if enabled else "block"}, header_style,
            {"display": "block" if enabled else "none"}, confirm_style,
            "선택 취소" if enabled else "계정 삭제")


@app.callback(Output("account-delete-confirm-modal", "style"),
              Output("account-delete-confirm-message", "children"),
              Output("account-delete-refresh", "data"),
              Output("account-delete-feedback", "children"),
              Input("account-delete-confirm-btn", "n_clicks"),
              Input("account-delete-no-btn", "n_clicks"),
              Input("account-delete-yes-btn", "n_clicks"),
              Input("account-manage-btn", "n_clicks"),
              Input("account-management-close-btn", "n_clicks"),
              State("account-delete-selection", "value"),
              State("account-delete-refresh", "data"), prevent_initial_call=True)
def confirm_account_delete(_confirm, _no, _yes, _open, _close, selected, refresh):
    triggered = ctx.triggered_id
    hidden = {"display": "none"}
    if triggered in {"account-manage-btn", "account-management-close-btn", "account-delete-no-btn"}:
        session.pop("pending_account_delete", None)
        return hidden, "", no_update, ""
    if str(session.get("role", "UNKNOWN")).upper() != "ADMIN":
        session.pop("pending_account_delete", None)
        record_action("ACT_ACCOUNT_DELETE_BLOCKED", "일반 계정 삭제 차단", target_type="account",
                      status="BLOCKED", block_reason="관리자 권한 필요")
        return hidden, "", no_update, "접근 권한이 필요합니다."
    selected = list(dict.fromkeys(selected or []))
    if triggered == "account-delete-confirm-btn":
        if not selected:
            return hidden, "", no_update, "삭제할 계정을 선택해 주세요."
        session["pending_account_delete"] = selected
        return ({"display": "flex", "position": "fixed", "inset": 0, "zIndex": 340,
                 "background": "#0006", "alignItems": "center", "justifyContent": "center"},
                f"선택한 일반 계정 {len(selected)}개를 정말로 삭제하시겠습니까?",
                no_update, "")
    if triggered != "account-delete-yes-btn":
        return no_update, no_update, no_update, no_update
    pending = session.pop("pending_account_delete", None)
    if not pending:
        return hidden, "", no_update, "삭제 확인을 다시 진행해 주세요."
    try:
        deleted = delete_user_accounts(pending, actor_id=str(session.get("admin_id", "")))
    except (PermissionError, ValueError) as exc:
        record_action("ACT_ACCOUNT_DELETE_BLOCKED", "일반 계정 삭제 차단", target_type="account",
                      status="BLOCKED", block_reason=type(exc).__name__)
        return hidden, "", no_update, "계정 목록이 변경되었습니다. 다시 선택해 주세요."
    except Exception as exc:
        log_failure("ACT_ACCOUNT_DELETE", "일반 계정 삭제", exc,
                    actor_id=str(session.get("admin_id", "")),
                    source="wireframe_app.confirm_account_delete")
        return hidden, "", no_update, "계정 삭제 중 오류가 발생했습니다."
    return hidden, "", int(refresh or 0) + 1, f"일반 계정 {deleted}개를 삭제했습니다."


# 브라우저는 초 단위 카운트다운만 표시한다. 실제 로그인 허용·차단은
# dashboard_auth.py의 Flask before_request가 서버 시각으로 다시 판단한다.
app.clientside_callback(
    """function(_tick, sessionState, promptState) {
        const hidden = {display: 'none'};
        if (!sessionState || !sessionState.expires_at_ms) {
            return [window.dash_clientside.no_update, hidden,
                    window.dash_clientside.no_update, window.dash_clientside.no_update,
                    window.dash_clientside.no_update];
        }
        const remaining = Math.max(0, Math.ceil((sessionState.expires_at_ms - Date.now()) / 1000));
        const minutes = String(Math.floor(remaining / 60)).padStart(2, '0');
        const seconds = String(remaining % 60).padStart(2, '0');
        const label = minutes + ':' + seconds;
        if (remaining === 0) {
            window.location.assign('/login');
            return [label, hidden, {shown: true}, '/login', '예 (0초)'];
        }
        const shown = Boolean(promptState && promptState.shown);
        if (remaining <= 10) {
            const modalStyle = shown ? window.dash_clientside.no_update :
                {display: 'flex', position: 'fixed', inset: 0, zIndex: 300,
                 background: '#0006', alignItems: 'center', justifyContent: 'center'};
            return [label, modalStyle, {shown: true}, window.dash_clientside.no_update,
                    '예 (' + remaining + '초)'];
        }
        return [label, window.dash_clientside.no_update,
                window.dash_clientside.no_update, window.dash_clientside.no_update,
                window.dash_clientside.no_update];
    }""",
    Output("session-remaining", "children"),
    Output("session-expiry-modal", "style"),
    Output("session-prompt-state", "data"),
    Output("auth-redirect", "href", allow_duplicate=True),
    Output("session-modal-extend-btn", "children"),
    Input("session-timer", "n_intervals"),
    State("session-state", "data"),
    State("session-prompt-state", "data"),
    prevent_initial_call=True,
)


@app.callback(Output("session-state", "data", allow_duplicate=True),
              Output("session-prompt-state", "data", allow_duplicate=True),
              Output("session-expiry-modal", "style", allow_duplicate=True),
              Input("session-extend-btn", "n_clicks"),
              Input("session-modal-extend-btn", "n_clicks"),
              prevent_initial_call=True)
def extend_login_session(profile_clicks, modal_clicks):
    """프로필·확인 창 어느 쪽에서 눌러도 같은 서버 세션을 10분 연장한다."""
    if not (profile_clicks or modal_clicks):
        return no_update, no_update, no_update
    state = extend_admin_session()
    if state is None:
        return no_update, no_update, no_update
    return state, {"shown": False}, {"display": "none"}


@app.callback(Output("logout-confirm-modal", "style"),
              Input("logout-btn", "n_clicks"), Input("logout-confirm-no-btn", "n_clicks"),
              prevent_initial_call=True)
def toggle_logout_confirmation(logout_clicks, cancel_clicks):
    """로그아웃을 누른 즉시 세션을 지우지 않고 먼저 확인을 받는다."""
    if not (logout_clicks or cancel_clicks):
        return no_update
    if ctx.triggered_id == "logout-confirm-no-btn":
        return {"display": "none"}
    return {"display": "flex", "position": "fixed", "inset": 0, "zIndex": 310,
            "background": "#0006", "alignItems": "center", "justifyContent": "center"}


@app.callback(Output("audit-audience-event", "data"),
              Input("export-dd", "value"), State("actor-id-input", "value"),
              prevent_initial_call=True)
def log_audience_change(export_value, actor_id):
    fmt, audience = (export_value or f"pdf:{DEFAULT_AUDIENCE}").split(":")
    record_action("ACT_EXPORT_OPTION_CHANGE", "내보내기 옵션 선택", actor_id=actor_id,
                  target_type="export_option", target_id=f"{fmt}:{audience}",
                  detail={"format": fmt, "audience": audience})
    return {"format": fmt, "audience": audience}


@app.callback(Output("audit-sync-status", "data"), Input("audit-csv-interval", "n_intervals"))
def sync_failure_log(_ticks):
    try:
        return sync_if_csv_changed()
    except Exception as exc:
        record_error("ERR_FAILURE_SYNC", "관측 고장 저장", exc, source="wireframe_app.sync_failure_log")
        return {"error": type(exc).__name__}


def _log_unhandled_exception(sender, exception, **_kwargs):
    record_error("ERR_DASH_CALLBACK", "대시보드 요청", exception,
                 source="wireframe_app.flask_request")


got_request_exception.connect(_log_unhandled_exception, app.server, weak=False)


@app.callback(Output("password-result", "children"), Output("password-current", "value"),
              Output("password-new", "value"), Output("password-confirm", "value"),
              Output("password-success-modal", "style"),
              Input("password-save-btn", "n_clicks"),
              State("password-current", "value"), State("password-new", "value"),
              State("password-confirm", "value"), prevent_initial_call=True)
def update_admin_password(n_clicks, current, new, confirm):
    if not n_clicks:
        return no_update, no_update, no_update, no_update, no_update
    admin_id = session.get("admin_id")
    if new != confirm:
        record_action("ACT_PASSWORD_CHANGE", "관리자 비밀번호 변경", status="BLOCKED",
                      block_reason="새 비밀번호 확인 불일치")
        return "새 비밀번호가 일치하지 않습니다.", "", "", "", no_update
    try:
        changed = change_admin_password(admin_id, current or "", new or "")
        if not changed:
            record_action("ACT_PASSWORD_CHANGE", "관리자 비밀번호 변경", status="BLOCKED",
                          block_reason="현재 비밀번호 불일치")
            return "현재 비밀번호가 올바르지 않습니다.", "", "", "", no_update
        return "", "", "", "", {"display": "flex", "position": "fixed", "inset": 0, "zIndex": 320,
                                      "background": "#0006", "alignItems": "center", "justifyContent": "center"}
    except ValueError as exc:
        record_action("ACT_PASSWORD_CHANGE", "관리자 비밀번호 변경", status="BLOCKED",
                      block_reason=type(exc).__name__)
        return str(exc), "", "", "", no_update
    except Exception as exc:
        log_failure("ACT_PASSWORD_CHANGE", "관리자 비밀번호 변경", exc, actor_id=admin_id,
                    source="wireframe_app.update_admin_password")
        return f"비밀번호 변경 실패: {type(exc).__name__}", "", "", "", no_update


@app.callback(Output("auth-redirect", "href", allow_duplicate=True),
              Input("logout-confirm-yes-btn", "n_clicks"),
              Input("session-modal-logout-btn", "n_clicks"),
              Input("password-success-confirm-btn", "n_clicks"), prevent_initial_call=True)
def logout_admin(confirm_clicks, session_decline_clicks, password_confirm_clicks):
    if not (confirm_clicks or session_decline_clicks or password_confirm_clicks):
        return no_update
    reason_by_button = {
        "logout-confirm-yes-btn": "manual",
        "session-modal-logout-btn": "session_extend_declined",
        "password-success-confirm-btn": "password_changed",
    }
    reason = reason_by_button[ctx.triggered_id]
    end_admin_session(reason)
    return "/login"


if __name__ == "__main__":
    if DEMO_MODE:
        print("[DEMO] 데모 모드 — DB·로그인 없이 읽기 전용으로 실행합니다.")
    else:
        try:
            create_audit_tables()
            created = ensure_dashboard_admin()
            print(f"[DB] 기본 관리자 계정: {'새로 준비됨' if created else '기존 계정 사용'}")
            result = sync_if_csv_changed()
            print(f"[DB] 관측 고장 로그: {result['date']} / 새로 저장 {result['created']}건")
        except Exception as exc:
            record_error("ERR_DB_STARTUP", "로그 DB 준비", exc, source="wireframe_app.startup")
            print(f"[DB] 연결 또는 초기 적재 실패: {type(exc).__name__} — instance/audit_fallback.log 확인")
    # 최신 Dash(2.17+)는 app.run, 이전 버전은 app.run_server를 쓴다.
    # 이전 실행본이 8050 포트에 남아 오래된 시간 기록을 만들 수 있어 새 포트를 사용한다.
    # 같은 사내·가정 네트워크의 다른 기기도 접속할 수 있도록 모든 네트워크 인터페이스에서 받는다.
    # 외부 인터넷 공개는 별도의 방화벽·공유기·HTTPS 설정이 필요하므로 여기서 자동으로 열지 않는다.
    app.run(host=os.environ.get("DASHBOARD_HOST", "0.0.0.0"), port=int(os.environ.get("DASHBOARD_PORT", "8052")), debug=False)
