"""대시보드 공통 — 글자 클래스·상태 배지·레이아웃 토큰, 필터·보고서·세그먼트 상태 상수, 그리드·카드·표 도구."""

from dash import dcc, html

from ..dashboard_data import (
    filter_assets, load_asset_list, load_data_reference_date, load_data_start_date, load_priority_table,
    parse_date, period_start,
)

# ============================================================
# 디자인 토큰 — assets/00-tokens.css(design/tokens.json에서 생성)가 정본이다.
#
# 색·폰트·테두리·그림자는 assets/의 클래스(01-type.css·02-bundle.css·03-app.css)로만
# 입히고, 인라인 style에는 치수·배치만 둔다(DESIGN.md §6). 테마는 <html data-theme>를
# clientside callback으로 바꿔 전환한다(§1.3). Plotly는 CSS 변수를 읽지 못하므로
# src/theme.py가 같은 tokens.json으로 등록한 plantfloor_{light,dark} 템플릿과 C()를 쓴다(§8).
# ============================================================

INDEX_STRING = """<!DOCTYPE html>
<html lang="ko" data-theme="light">
  <head>
    {%metas%}<title>{%title%}</title>{%favicon%}{%css%}
  </head>
  <!-- Dash 기본 템플릿은 설정·스크립트를 <footer>로 감싸지만, 그러면 page_footer()와 함께
       contentinfo 랜드마크가 둘이 된다(axe landmark-no-duplicate-contentinfo). 내용이 아니라
       스크립트 자리이므로 div로 둔다. -->
  <body>{%app_entry%}<div>{%config%}{%scripts%}{%renderer%}</div></body>
</html>
"""

# 타이포 클래스(01-type.css)와 잉크 보조 클래스(02-bundle.css) 조합
TITLE_14 = "title-14"
LABEL_12 = "label-12 pf-secondary"
NOTE_12 = "label-12 pf-muted"      # 카드 머리·필터바의 보조 설명
MICRO_11 = "micro-11 pf-muted"     # 배지·단위 전용
CODE_12 = "code-12"                # 식별자(asset_tag·part_no·plant_code)
NUM_13 = "num-13"                  # 열로 쌓이는 숫자
BODY_13 = "body-13"

# 기계 등급(한글 라벨) → 상태 배지 변형(.pf-badge--*)
GRADE_BADGE = {"정상": "good", "주의": "warning", "경계": "serious", "위험": "critical"}


def _figure_template(theme):
    return f"plantfloor_{theme if theme in ('light', 'dark') else 'light'}"


def _theme(theme):
    return theme if theme in ("light", "dark") else "light"


def status_badge(label):
    """상태 배지 — 8px 점 + 글자. 상태색은 글자 라벨 없이 쓰지 않는다(DESIGN.md §2.1)."""
    variant = GRADE_BADGE.get(label, "neutral")
    return html.Span([html.Span(className="pf-badge__dot"), label],
                     className=f"pf-badge pf-badge--{variant} {MICRO_11.split()[0]}")


# 레이아웃 토큰
CANVAS_H = 1080  # 폭은 1280~1920px 유동(12열 그리드), 높이는 행 높이 토큰 합으로 고정
HEADER_H, FILTERBAR_H = 56, 56
FOOTER_H = 20  # 화면 하단 오른쪽 안내 한 줄
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
    # 기본 기간은 최근 30일 — 기준일 직전 한 달의 상태를 먼저 본다(사용자 결정 2026-10-03).
    # period_index = 눌린 기간 버튼. 시작일·종료일 칸을 직접 고치면 None이 되고 start·end가 정본이 된다.
    "plant": None, "machine_type": None, "machine": None, "period_index": 0,
    "start": None, "end": None,
}


def _period_index(filters):
    """눌린 기간 버튼의 인덱스. 날짜를 직접 지정한 상태면 None(눌린 버튼이 없다)."""
    pi = (filters or DEFAULT_FILTERS).get("period_index")
    return pi if isinstance(pi, int) and 0 <= pi < len(PERIOD_PRESETS) else None


def _period_range(filters):
    """filter-store 값 → (시작일, 종료일) Timestamp. None은 그쪽 끝이 열려 있다는 뜻이다."""
    f = filters or DEFAULT_FILTERS
    pi = _period_index(f)
    if pi is not None:
        return period_start(PERIOD_PRESETS[pi][1]), None
    return parse_date(f.get("start")), parse_date(f.get("end"))


def _period_dates(filters):
    """시작일·종료일 칸에 표시할 (시작, 종료) 문자열 — 기간 버튼이 눌려 있으면 그 버튼이 뜻하는 구간."""
    f = filters or DEFAULT_FILTERS
    pi = _period_index(f)
    if pi is None:
        return f.get("start"), f.get("end")
    start = period_start(PERIOD_PRESETS[pi][1])
    return (load_data_start_date() if start is None else f"{start:%Y-%m-%d}"), load_data_reference_date()


def _filter_scope(filters):
    """filter-store 값 → (대상 기계 태그 목록, 기간 시작일, 기간 종료일)."""
    f = filters or DEFAULT_FILTERS
    assets = filter_assets(f.get("plant"), f.get("machine_type"), f.get("machine"))
    start, end = _period_range(f)
    return assets, start, end


def _focus_asset(filters, machine_only=False):
    """② 기계 상세·③ 부품군 진단이 보여 줄 기계 — 필터의 기계, 전체면 점검 우선순위 1위.
    machine_only=True면 공장·기계 종류 필터를 보지 않는다(③은 그 필터를 적용하지 않는다)."""
    f = filters or DEFAULT_FILTERS
    if machine_only:
        f = {"machine": f.get("machine")}
    assets, _, _ = _filter_scope(f)
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
            ("① 현황", "KPI 4종", ["고위험일 (기계·일)", "고위험일 비율 (%)", "기준일 고위험 기계 (대)",
                                   "고장 표시 기계·일"]),
            ("① 현황", "점검 우선순위 (기계 행)", ["순위", "대상", "종류", "공장", "등급가중 고장점수",
                                                  "기준선 초과", "최근 30일 고위험일 수", "당일 고장 표시 부품 수"]),
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
            ("③ 모델·예측", "모델 비교", ["모델", "AP", "AP ÷ 양성 비율 (배)", "ROC-AUC",
                                          "정밀도", "재현율", "경보율 (%)"]),
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
    # 순서 = SCREEN3_TASK_KEYS. 기본(0)은 주 과제인 기계 고위험일 판별이다.
    "task": ["기계 고위험일 판별", "부품 7일 내 고장 표시", "부품 당일 고장 표시 판별", "부품군 진단"],
    "threshold": ["12", "13", "14"],
    "dataset": ["원자료", "기계·일 집계"],
    # ① 고장 표시 히트맵 가로축 단위. 순서 = HEATMAP_UNITS.
    "heat_unit": ["일", "주", "월", "년"],
}
# SEG_GROUPS["heat_unit"] 인덱스 → dashboard_data.HEATMAP_FREQ의 키.
HEATMAP_UNITS = ["day", "week", "month", "year"]
# 히트맵 기본 단위는 일 — 기본 기간(최근 30일)과 짝을 이룬다.
DEFAULT_SEG = {"task": 0, "threshold": 0, "dataset": 0, "heat_unit": 0}


# ============================================================
# 공용 프리미티브
# ============================================================

def note(text, style=None):
    """카드 머리·필터바의 보조 설명."""
    return html.Span(text, className=NOTE_12, style=style)


def slot(label, w, h, sub=None):
    """아직 채워지지 않은 영역 — 라벨만 있는 자리."""
    width_style = {"width": f"{w}px"} if isinstance(w, (int, float)) else {"flexGrow": "1", "minWidth": "0"}
    children = [html.Span(label, className=LABEL_12, style={"textAlign": "center"})]
    if sub and h >= 34:
        children.append(html.Span(sub, className=MICRO_11, style={"textAlign": "center"}))
    return html.Div(
        children, className="pf-placeholder",
        style={**width_style, "height": f"{h}px", "flexShrink": "0", "display": "flex",
               "flexDirection": "column", "alignItems": "center", "justifyContent": "center",
               "gap": "2px", "padding": "0 6px", "overflow": "hidden"},
    )


LAYOUT_COL = 142  # --layout-col: 1920 캔버스에서 12열 그리드 한 칸의 폭


def _span(w):
    """1920 캔버스 기준 폭(px)을 12열 그리드 칸 수로 바꾼다 — 칸 142 + 거터 16(레이아웃 토큰).
    예: 1880 → 12, 1090 → 7, 774 → 5, 458 → 3. 실제 폭은 화면 폭을 따른다."""
    return max(1, min(12, round((w + GUTTER) / (LAYOUT_COL + GUTTER))))


def card(title, w, h, body, right=None):
    """카드(.pf-card) — hairline 테두리 + radius-md, 그림자 없음. 제목 줄 36px 고정.
    w는 1920 캔버스 기준 폭이며 12열 그리드 칸 수로 바뀐다(_span)."""
    return html.Section(
        [
            html.Div(
                [html.H2(title, className=f"pf-card__title {TITLE_14}", style={"whiteSpace": "nowrap"}),
                 html.Div(right, className="pf-card__actions")],
                className="pf-card__head",
            ),
            html.Div(body, className="pf-card__body",
                     style={"display": "flex", "flexDirection": "column", "minWidth": "0"}),
        ],
        className="pf-card",
        style={"gridColumn": f"span {_span(w)}", "height": f"{h}px", "minWidth": "0"},
    )


def row(height, children, gap=16):
    """12열 그리드 행 — 칸 폭은 minmax(0, 1fr)이라 1280~1920px 화면 폭을 따라 줄고 는다."""
    return html.Div(children, style={"display": "grid", "gridTemplateColumns": "repeat(12, minmax(0, 1fr))",
                                      "gap": f"{gap}px", "height": f"{height}px"})


def col(width, height, children, gap=16):
    return html.Div(children, style={"gridColumn": f"span {_span(width)}", "height": f"{height}px",
                                      "minWidth": "0", "display": "flex",
                                      "flexDirection": "column", "gap": f"{gap}px"})


def hstack(children, gap=8, style=None):
    s = {"display": "flex", "gap": f"{gap}px"}
    if style:
        s.update(style)
    return html.Div(children, style=s)


def tile(label, w=300, h=96, sub="값 · value-28", tid=None):
    """값이 아직 없는 KPI 자리. tid를 주면 클릭 가능한 타일이 된다 — html.Div도
    n_clicks를 받으므로 버튼으로 바꾸지 않아도 된다."""
    kwargs = {"id": tid, "n_clicks": 0} if tid else {}
    return html.Div(
        [html.Span(label, className=f"pf-kpi__label {LABEL_12}"), slot(sub, 168, 32)],
        className="pf-card pf-kpi" + (" pf-kpi--action" if tid else ""),
        style={"gridColumn": f"span {_span(w)}", "height": f"{h}px", "minWidth": "0"}, **kwargs,
    )


def bar(pct=45, h=8):
    """더미 숫자가 아니라 '여기 값이 온다'는 자리 표시일 뿐 — 길이에 의미 없음."""
    return html.Div(className="pf-placeholder", style={"width": f"{pct}%", "height": f"{h}px"})


def btn(label, w=None, bid=None):
    """기능이 아직 없는 버튼은 {"type": "ghost-btn"} 패턴을 달아, 눌렀을 때 하단
    action-echo에 '무엇이 눌렸고 무엇이 아직 없는지'를 그대로 표시한다."""
    return html.Button(
        label, id={"type": "ghost-btn", "index": bid or label}, n_clicks=0,
        className="pf-btn label-12", style={"width": f"{w}px"} if w else None,
    )


def seg(group, label_text, options, sel=0, disabled=False, help_text=None):
    """세그먼티드 컨트롤(.pf-seg) — 실제로 선택이 바뀐다. group은 seg-store의 키다.
    ③ 과제 / ③ 위험 기준선 / ④ 데이터셋 / ① 히트맵 단위가 각각 하나의 group.
    help_text를 주면 라벨 옆에 물음표 설명을 단다."""
    buttons = [
        html.Button(o, id={"type": "seg-btn", "group": group, "index": i}, n_clicks=0, disabled=disabled,
                    className="pf-seg__item label-12", **{"aria-pressed": "true" if i == sel else "false"})
        for i, o in enumerate(options)
    ]
    label = [html.Span(label_text, className=LABEL_12, style={"whiteSpace": "nowrap"})]
    if help_text:
        label.append(help_icon(help_text))
    return html.Div(
        label + [html.Div(buttons, className="pf-seg")],
        style={"display": "flex", "alignItems": "center", "gap": "8px"},
    )


def filter_dropdown(dd_id, label, options, w):
    """필터바 드롭다운. 값은 filter-store(storage_type="local")가 원본이고,
    첫 렌더에 write_filters가 저장된 값을 드롭다운에 되돌려 놓는다."""
    return html.Div(
        [html.Span(label, className=f"pf-field__label {LABEL_12}"),
         dcc.Dropdown(
             id=dd_id, options=[{"label": o, "value": o} for o in options],
             value=None, placeholder="전체", clearable=True,
             style={"width": f"{w}px"},
         )],
        className="pf-field", style={"gap": "6px", "flexShrink": "0"},
    )


def period_toggle():
    """기간 버튼 4개 — 데이터 최신일을 포함한 최근 N일 또는 전체 기간."""
    buttons = [
        html.Button(label, id={"type": "period-btn", "index": i}, n_clicks=0,
                    className="pf-seg__item label-12",
                    **{"aria-pressed": "true" if i == DEFAULT_FILTERS["period_index"] else "false"})
        for i, (label, _days) in enumerate(PERIOD_PRESETS)
    ]
    return html.Div(
        [html.Span("기간", className=LABEL_12, style={"whiteSpace": "nowrap"}),
         html.Div(buttons, id="period-btn-group", className="pf-seg")],
        style={"display": "flex", "alignItems": "center", "gap": "8px", "flexShrink": "0"},
    )


def date_range_inputs():
    """기간을 직접 지정하는 시작일·종료일 칸. 값을 고치면 기간 버튼 선택이 풀리고,
    버튼을 누르면 그 버튼이 뜻하는 구간이 이 칸에 채워진다(write_filters)."""
    data_start, data_end = load_data_start_date(), load_data_reference_date()

    # dcc.Input은 aria-* 속성을 받지 않는다 — 접근 이름은 assets/04-a11y.js가 붙인다.
    def box(input_id):
        return dcc.Input(id=input_id, type="date", debounce=True, className="pf-control pf-date",
                         min=data_start, max=data_end)

    return html.Div(
        [box("start-date"),
         html.Span("~", className=f"{LABEL_12} pf-muted"),
         box("end-date")],
        style={"display": "flex", "alignItems": "center", "gap": "6px", "flexShrink": "0"},
    )


def legend(items):
    """카드 머리의 작은 범례(.pf-legend) — items는 (표식 종류, 글자) 목록.
    표식 종류: line(series-1 선) · dash(점선 경계) · critical(status-critical 점) · band(고위험 띠).
    상태색 표식은 항상 글자와 같이 둔다(DESIGN §2.1)."""
    shape = {"critical": " pf-legend__key--dot"}
    return html.Div(
        [html.Span([html.Span(className=f"pf-legend__key{shape.get(kind, '')} pf-key--{kind}"), text],
                   className="pf-legend__item")
         for kind, text in items],
        className="pf-legend label-12",
    )


def help_icon(text):
    """라벨 옆 물음표 — 마우스를 올리면 설명(title)이 뜬다. 보조기술은 aria-label로 같은 글을 읽는다.
    KPI 카드는 .pf-card가 overflow:hidden이라 직접 그린 말풍선이 잘린다 — 브라우저 기본 title을 쓴다."""
    return html.Span("?", className="pf-help micro-11", title=text, role="img", **{"aria-label": text})


# ── 표 정렬 ─────────────────────────────────────────────────────────
# 모든 표는 열 제목 자체가 정렬 버튼이다. 같은 열을 누를 때마다 오름차순 → 내림차순 →
# 기본 순서로 돈다. 상태는 table-sort-store = {표 id: {"col": 열 id, "direction": "asc"|"desc"}}
# 한 곳에 두고, 화살표(▲ 오름차순 · ▼ 내림차순)는 지금 정렬 중인 열에만 붙는다.
SORT_NEXT = {None: "asc", "asc": "desc", "desc": None}
ARIA_SORT = {"asc": "ascending", "desc": "descending"}
_SORT_ACTION = {None: "오름차순으로 정렬", "asc": "내림차순으로 정렬", "desc": "기본 순서로 되돌림"}


def table_sort(store, table_id):
    """table-sort-store에서 표 하나의 (열 id, 방향). 기본 순서면 (None, None)."""
    state = (store or {}).get(table_id) or {}
    direction = state.get("direction")
    return (state.get("col"), direction) if direction in ARIA_SORT else (None, None)


def next_sort(store, table_id, col_id):
    """같은 열을 다시 누르면 다음 상태로, 다른 열을 누르면 그 열의 오름차순부터 시작한다."""
    col, direction = table_sort(store, table_id)
    new = SORT_NEXT[direction if col == col_id else None]
    updated = dict(store or {})
    updated[table_id] = {"col": col_id, "direction": new} if new else {}
    return updated


def sort_header(label, table_id, col_id, store, align="left"):
    """정렬 가능한 열 제목(<th> 안의 버튼). 정렬 중인 열은 제목을 진하게, 방향은 ▲▼로 표시한다."""
    col, direction = table_sort(store, table_id)
    active = direction if col == col_id else None
    action = _SORT_ACTION[active]
    return html.Th(
        html.Button(
            [html.Span(label),
             # 화살표 자리는 비어 있어도 폭을 잡아 둔다 — 정렬할 때 열 폭이 흔들리지 않게.
             html.Span({"asc": "▲", "desc": "▼"}.get(active, ""), className="pf-th-sort__arrow",
                       **{"aria-hidden": "true"})],
            id={"type": "sort-th", "table": table_id, "col": col_id}, n_clicks=0, type="button",
            className="pf-th-sort" + (" pf-th-sort--active" if active else ""),
            title=f"{label} · 누르면 {action}",
            style={"justifyContent": {"right": "flex-end", "center": "center"}.get(align, "flex-start")},
            **{"aria-label": f"{label}, {action}"},
        ),
        className="label-12 pf-th--sortable" + (" pf-th--num" if align == "right" else ""),
        style={"textAlign": align},
        **{"aria-sort": ARIA_SORT.get(active, "none")},
    )


def sort_rows(rows, key, direction):
    """기본 순서면 그대로 두고, 아니면 key 값으로 정렬한다. 값이 없는(None) 행은 방향과
    상관없이 맨 아래에 둔다 — 빈 값이 맨 위로 올라와 1등처럼 보이지 않게."""
    if direction not in ARIA_SORT:
        return list(rows)
    present = [r for r in rows if key(r) is not None]
    missing = [r for r in rows if key(r) is None]
    return sorted(present, key=key, reverse=direction == "desc") + missing


def sortable(table_id, table):
    """정렬 콜백이 표만 다시 그릴 수 있도록 표를 id 달린 칸에 담는다(화면 전체를 다시 그리면
    다른 카드의 상태 — 슬라이더 위치·표 페이지·히트맵 확대 — 가 초기화된다)."""
    return html.Div(table, id={"type": "sort-table", "table": table_id},
                    style={"height": "100%", "minHeight": "0", "display": "flex", "flexDirection": "column"})


def table_placeholder(cols, nrows, row_h, head_h=32, first_idx=False, sort_col=None, width=None, cell_h=6):
    """cols: [(라벨, 폭px, 정렬)] — 셀 내용은 실제 값이 아니라 자리 막대."""
    thead = html.Tr(
        [html.Th(f"{l}{' ▼' if i == sort_col else ''}", className=LABEL_12,
                 style={"width": f"{w}px", "textAlign": a})
         for i, (l, w, a) in enumerate(cols)],
        style={"height": f"{head_h}px"},
    )
    body_rows = []
    for r in range(nrows):
        cells = []
        for i, (l, w, a) in enumerate(cols):
            if first_idx and i == 0:
                cell = html.Span(str(r + 1), className=f"{NUM_13} pf-muted")
            else:
                just = "flex-end" if a == "right" else ("center" if a == "center" else "flex-start")
                pct = 45 if a != "left" else 70
                cell = html.Div(bar(pct, cell_h), style={"display": "flex", "justifyContent": just})
            cells.append(html.Td(cell, style={"textAlign": a}))
        body_rows.append(html.Tr(cells, style={"height": f"{row_h}px"}))
    tw = width or sum(c[1] for c in cols)
    return html.Table(
        [html.Thead(thead), html.Tbody(body_rows)], className="pf-table",
        style={"width": f"{tw}px", "tableLayout": "fixed", "flexShrink": "0"},
    )


def table_scroll(table, label):
    """카드 안 표 스크롤 영역 — 키보드로도 스크롤할 수 있게 초점을 받고 이름을 가진다."""
    return html.Div(table, className="pf-table__scroll", tabIndex="0", role="region",
                    **{"aria-label": label})


def empty_state(title, reason, h=None):
    """확정되지 않은 값은 0/—이 아니라 이유가 적힌 빈 상태(.pf-empty)로 표시한다.
    폭은 부모를 채우고, h를 주지 않으면 높이도 부모를 채운다."""
    return html.Div(
        [html.Span(title, className=f"pf-empty__title {TITLE_14}"),
         html.Span(reason, className=BODY_13)],
        className="pf-empty" + (" pf-empty--inline" if h is not None and h < 160 else ""),
        style={"width": "100%", "height": f"{h}px" if h is not None else "100%"},
    )
