"""화면 ① 현황 — KPI 타일, 점검 우선순위 표, 기계 상태·히트맵·전력."""

from dash import dcc, html

from ..dashboard_data import (
    load_asset_failure_heatmap, load_month_coverage, load_priority_table, load_screen1_kpis,
    load_screen1_machine_status, load_screen1_power_by_machine,
)
from .base import (
    _span, BODY_13, card, CODE_12, DEFAULT_AUDIENCE, DEFAULT_POWER_SORT, DEFAULT_PRIO_SORT,
    direction_buttons, GUTTER, help_icon, LABEL_12, MICRO_11, note, NUM_13, row, ROW_KPI, ROW_MAIN,
    ROW_SUB, status_badge, table_scroll,
)
from .figures import _heatmap_figure, _spark_figure


# ============================================================
# 화면 ① 현황 — 행 96 / 460 / 340
# ============================================================

def kpi_value_tile(label, value_text, w=300, h=96, scope=None, help_text=None, sub=None):
    """KPI 값 타일(.pf-kpi). 라벨 오른쪽에는 둘 중 하나가 온다 — help_text는 물음표에 달리는
    설명(①), scope는 값이 어디 것인지 적는 짧은 글(③의 모델 이름).
    sub는 값 옆의 보조 비교(예: 직전 같은 기간 값)."""
    tail = [help_icon(help_text)] if help_text else ([note(scope)] if scope else [])
    return html.Div(
        [html.Div([html.Span(label, className=f"pf-kpi__label {LABEL_12}")] + tail,
                  style={"display": "flex", "justifyContent": "space-between",
                         "alignItems": "flex-start", "gap": "8px"}),
         html.Div([html.Span(value_text, className="pf-kpi__value value-28")]
                  + ([html.Span(sub, className="pf-kpi__sub label-12")] if sub else []),
                  className="pf-kpi__row")],
        className="pf-card pf-kpi",
        style={"gridColumn": f"span {_span(w)}", "height": f"{h}px", "minWidth": "0"},
    )


# (열 라벨, 정렬, 열 id). 열 id는 record의 키이자 정렬 축 이름이다.
PRIO_COLS = [
    ("순위", "right", "rank"),
    ("대상", "left", "asset_tag"),
    ("종류", "left", "machine_type"),
    ("공장", "left", "plant_code"),
    ("등급가중 고장점수", "right", "failure_points"),
    ("기준선 초과", "center", "threshold_exceeded"),
    ("최근 30일 고위험일 수", "right", "high_risk_days_30d"),
    ("당일 고장 표시 부품 수", "right", "failed_part_count"),
]
# 열 id → load_priority_table()의 정렬 축. 순위는 기본 우선순위(등급가중 고장점수) 순이고,
# 기준선 초과는 "초과한 기계 먼저, 동률이면 고장점수" 축을 쓴다.
PRIO_SORT_AXIS = {"rank": "grade", "failure_points": "grade", "threshold_exceeded": "threshold"}
ARIA_SORT = {"asc": "ascending", "desc": "descending"}


def priority_table(records, sort_by, direction):
    """점검 우선순위 표(.pf-table) — load_priority_table()이 만든 자산별 값을 채운다.
    숫자 열은 num-13 우측 정렬, 식별자(asset_tag·plant_code)는 code-12.
    열 머리마다 ▲▼ 버튼이 있어 그 열 기준으로 오름차/내림차순 정렬한다.
    열 폭은 내용에 맞추고(자동 배치), 카드보다 넓으면 카드 안에서만 가로 스크롤한다."""
    align_items = {"right": "flex-end", "center": "center"}
    head_cells = []
    for label, align, field in PRIO_COLS:
        active = direction if field == sort_by else None
        head_cells.append(html.Th(
            html.Div([html.Span(label), direction_buttons("prio-sort-btn", label, active, field)],
                     style={"display": "flex", "alignItems": "center", "gap": "4px",
                            "justifyContent": align_items.get(align, "flex-start")}),
            className="label-12 pf-th--sortable" + (" pf-th--num" if align == "right" else ""),
            style={"textAlign": align},
            **{"aria-sort": ARIA_SORT[active] if active else "none"},
        ))
    thead = html.Tr(head_cells)
    num_fields = {"rank", "failure_points", "high_risk_days_30d", "failed_part_count"}
    code_fields = {"asset_tag", "plant_code"}
    body_rows = []
    for record in records:
        cells = []
        for _label, a, field in PRIO_COLS:
            value = record[field]
            if field == "failure_points":
                text = f"{value:,.0f}"
            elif field == "threshold_exceeded":
                text = "예" if value else "아니오"
            else:
                text = str(value)
            if field in num_fields:
                cls = f"{NUM_13} pf-td--num" + (" pf-muted" if field == "rank" else "")
            elif field in code_fields:
                cls = CODE_12
            else:
                cls = BODY_13
            cells.append(html.Td(text, className=cls, style={"textAlign": a}))
        body_rows.append(html.Tr(cells))
    return table_scroll(html.Table([html.Thead(thead), html.Tbody(body_rows)], className="pf-table"),
                        "점검 우선순위 표")


# KPI 라벨 옆 물음표에 다는 설명. 값이 무엇을 세는지와 집계 범위를 한 문장으로 적는다
# ("선택 기간"·"기준일" 같은 라벨을 카드에 따로 두는 대신 여기에 넣는다).
KPI_HELP = {
    "high_risk_days": "선택 기간에 고장점수가 12 이상이던 (기계×날짜) 건수. "
                      "고장점수는 그날 고장 표시된 부품의 중요도 합(A=4·B=2·C=1)이다.",
    "high_risk_rate_pct": "선택 기간의 전체 (기계×날짜) 중 고위험일이 차지한 비율. "
                          "옆의 작은 값은 바로 앞 같은 길이 기간의 같은 비율이다.",
    "latest_high_risk_machines": "선택 기간의 마지막 날 하루만 보고 센 고위험 기계 수. "
                                 "기간 합계가 아니라 그날 하루의 스냅샷이다.",
    "failure_machine_days": "선택 기간에 부품이 하나라도 고장 표시된 (기계×날짜) 건수. "
                            "고장점수와 달리 부품 중요도를 따지지 않는다.",
}


def screen_1(seg_state=None, audience=DEFAULT_AUDIENCE, prio_sort=None, assets=None, start=None, end=None,
             power_sort=DEFAULT_POWER_SORT):
    # KPI 4개는 판단에 쓰는 값만 둔다(관측 기계 수·평균 소비 전력은 뺐다).
    #   고위험일(기계·일)·고위험일 비율(%) — 선택 기간. 비율 옆에 직전 같은 길이 기간 값을 둔다.
    #   기준일 고위험 기계(대) — 하루치 스냅샷. 비율(1대 = 10.0%)로 쓰면 정밀해 보이기만 해서 건수로 둔다.
    #   고장 표시 기계·일 — 선택 기간.
    # 고위험 = 고장점수 12 이상 고정(dashboard_data.HIGH_RISK_THRESHOLD). ③의 12/13/14 토글은
    # 모델 비교 결과만 바꾸고 이 값에는 영향을 주지 않는다.
    prio_sort = prio_sort or DEFAULT_PRIO_SORT
    kpis = load_screen1_kpis(assets, start, end)
    # "최고 베어링 온도"·"부품 출고 금액(누적)"은 삭제한다 — 남은 4개가 같은
    # 458px 폭(4×458 + 3×16 = 1880)으로 행 전체를 균등 분배한다.
    # 집계 범위("선택 기간"·"기준일")는 카드 라벨 대신 물음표 설명으로 내렸다.
    # 고위험일 = 고장점수 12 이상인 기계·일. 하루치 스냅샷 대신 선택 기간 값을 앞에 둔다.
    prev = kpis["prev_high_risk_rate_pct"]
    kpi_specs = [
        ("고위험일 (기계·일)", f"{kpis['high_risk_days']:,}", "high_risk_days", None),
        ("고위험일 비율 (%)", f"{kpis['high_risk_rate_pct']:.1f}", "high_risk_rate_pct",
         f"직전 같은 기간 {prev:.1f}" if prev is not None else None),
        ("기준일 고위험 기계 (대)", f"{kpis['latest_high_risk_machines']:,}", "latest_high_risk_machines", None),
        ("고장 표시 기계·일", f"{kpis['failure_machine_days']:,}", "failure_machine_days", None),
    ]
    row_a = row(ROW_KPI, [kpi_value_tile(label, value_text, w=458, help_text=KPI_HELP[key], sub=sub)
                          for label, value_text, key, sub in kpi_specs])

    sort_by = prio_sort.get("sort_by", DEFAULT_PRIO_SORT["sort_by"])
    direction = prio_sort.get("direction", DEFAULT_PRIO_SORT["direction"])
    priority_records = load_priority_table(PRIO_SORT_AXIS.get(sort_by, sort_by), direction, assets=assets)
    prio = card("점검 우선순위", 1090, ROW_MAIN,
                priority_table(priority_records, sort_by, direction),
                right=note("기준일 스냅샷 · 기간 미적용 · 열 머리 ▲▼로 정렬"))

    def machine_tile(status_row):
        return html.Div(
            [html.Div([html.Span(status_row["asset_tag"], className=CODE_12, style={"whiteSpace": "nowrap"}),
                       html.Span(status_row["machine_type"], className=MICRO_11,
                                 style={"whiteSpace": "nowrap", "overflow": "hidden", "textOverflow": "ellipsis"})],
                      style={"width": "84px", "flexShrink": "0", "display": "flex",
                             "flexDirection": "column", "gap": "4px", "overflow": "hidden"}),
             dcc.Graph(id={"type": "spark-chart", "index": status_row["asset_tag"]},
                       figure=_spark_figure(status_row["sparkline"], "light"),
                       config={"displayModeBar": False, "responsive": True},
                       style={"flexGrow": "1", "minWidth": "0", "height": "32px"}),
             html.Div([html.Span(f"{status_row['risk_score']:,.0f}", className="value-20"),
                       status_badge(status_row["current_grade"])],
                      style={"width": "64px", "flexShrink": "0", "display": "flex",
                             "flexDirection": "column", "alignItems": "flex-start", "gap": "2px"})],
            className="pf-machinetile",
            style={"minWidth": "0", "minHeight": "0", "padding": "8px 12px",
                   "display": "flex", "alignItems": "center", "gap": "8px"},
        )
    machine_status_rows = load_screen1_machine_status(assets)
    tiles_grid = html.Div([machine_tile(r) for r in machine_status_rows],
                           style={"display": "grid", "gridTemplateColumns": "repeat(2, minmax(0, 1fr))",
                                  "gridTemplateRows": "repeat(5, minmax(0, 1fr))", "columnGap": "8px",
                                  "rowGap": "8px", "height": "100%"})
    status = card("기계 상태", 774, ROW_MAIN, tiles_grid,
                  right=note("등급(고장점수) 정상 0 · 주의 1~5 · 경계 6~11 · 위험 12 이상 · "
                             "스파크라인 최근 30일 · 기간 미적용"))
    row_b = row(ROW_MAIN, [prio, status])

    heat_body = html.Div(
        dcc.Graph(id={"type": "heatmap-chart", "index": "screen1"},
                  figure=_heatmap_figure(load_asset_failure_heatmap(assets, start, end),
                                       load_month_coverage(start, end), "light"),
                  config={"displayModeBar": False, "responsive": True},
                  style={"flex": "1 1 auto", "minHeight": "0", "minWidth": "0"}),
        style={"height": "100%", "display": "flex", "flexDirection": "column"},
    )
    heat = card("고장 표시 히트맵", 1248, ROW_SUB, heat_body,
                right=note("셀 = 그 달 고장 표시된 부품-일 행 수 합계 · 부분 = 관측 일수가 그 달보다 적은 달"))

    power_rows = load_screen1_power_by_machine(assets, start, end, power_sort)
    max_power = max((r["avg_power_kw"] for r in power_rows), default=1.0) or 1.0
    # 단일 계열 막대는 series-1 한 색이다 — 값에 따라 색을 바꾸지 않는다(.pf-risk).
    prows = html.Div(
        [html.Div([html.Span(r["asset_tag"], className=CODE_12, style={"width": "88px", "flexShrink": "0"}),
                   html.Span(html.Span(className="pf-risk__fill",
                                       style={"width": f"{r['avg_power_kw'] / max_power * 100:.1f}%"}),
                             className="pf-risk__track"),
                   html.Span(f"{r['avg_power_kw']:,.2f}", className=f"pf-risk__value {NUM_13}",
                             style={"width": "56px", "flexShrink": "0"})],
                  className="pf-risk", style={"height": "25px"})
         for r in power_rows],
    )
    power_body = html.Div([prows], style={"display": "flex", "flexDirection": "column"})
    power = card("기계별 평균 소비 전력 (kW)", 616, ROW_SUB, power_body,
                 right=html.Div([note(f"{len(power_rows)}개 · 선택 기간 평균"),
                                 direction_buttons("power-sort-btn", "평균 소비 전력", power_sort, "screen1")],
                                style={"display": "flex", "alignItems": "center", "gap": "6px"}))
    row_c = row(ROW_SUB, [heat, power])

    return html.Div([row_a, row_b, row_c],
                     style={"display": "flex", "flexDirection": "column", "gap": f"{GUTTER}px"})
