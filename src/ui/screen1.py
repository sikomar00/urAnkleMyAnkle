"""화면 ① 현황 — KPI 타일, 점검 우선순위 표, 기계 상태·히트맵·전력."""

from dash import dcc, html

from ..dashboard_data import (
    load_asset_failure_heatmap, load_asset_list, load_period_coverage, load_priority_table, load_screen1_kpis,
    load_screen1_machine_status, load_screen1_power_by_machine,
)
from .base import (
    _span, BODY_13, card, CODE_12, DEFAULT_AUDIENCE, DEFAULT_SEG, GUTTER, help_icon, HEATMAP_UNITS,
    LABEL_12, MICRO_11, note, NUM_13, row, ROW_KPI, ROW_MAIN, ROW_SUB, seg, SEG_GROUPS, sort_header,
    sort_rows, sortable, status_badge, table_scroll, table_sort,
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


def rate_change(now, prev):
    """고위험일 비율의 직전 같은 기간 대비 변화 — 추세 글리프(▲▼, DESIGN §6) + 증가·감소 글자.
    고위험일 비율은 오를수록 나쁘므로 증가는 status-critical-text, 감소는 status-good-text로 쓴다
    (색만으로 뜻을 나르지 않도록 "증가·감소" 글자를 함께 둔다). 차이는 화면에 보이는 소수 첫째 자리 값끼리 뺀다."""
    diff = round(round(now, 1) - round(prev, 1), 1)
    if diff == 0:
        return f"직전 같은 기간과 같음 ({prev:.1f})"
    worse = diff > 0
    return [html.Span(f"{'▲' if worse else '▼'} {abs(diff):.1f}%p {'증가' if worse else '감소'}",
                      className=f"pf-delta pf-delta--{'worse' if worse else 'better'}"),
            f" · 직전 같은 기간 {prev:.1f}"]


# (열 라벨, 정렬, 열 id). 열 id는 record의 키이자 load_priority_table()의 정렬 열 이름이다.
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


def priority_table(records, store):
    """점검 우선순위 표(.pf-table) — load_priority_table()이 만든 자산별 값을 채운다.
    숫자 열은 num-13 우측 정렬, 식별자(asset_tag·plant_code)는 code-12. 열 제목을 누르면
    그 열로 오름차순 → 내림차순 → 기본(우선순위) 순서로 돈다. 기준선을 넘은 기계의 행은
    status-critical-tint로 칠한다(DESIGN §6 — 표의 행 강조는 이 한 종류뿐이다).
    열 폭은 내용에 맞추고(자동 배치), 카드보다 넓으면 카드 안에서만 가로 스크롤한다."""
    thead = html.Tr([sort_header(label, "priority", field, store, align) for label, align, field in PRIO_COLS])
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
            if field == "threshold_exceeded" and value:
                cls += " pf-strong"
            cells.append(html.Td(text, className=cls, style={"textAlign": a}))
        body_rows.append(html.Tr(cells, className="pf-tr--flagged" if record["threshold_exceeded"] else None))
    return table_scroll(html.Table([html.Thead(thead), html.Tbody(body_rows)], className="pf-table"),
                        "점검 우선순위 표")


def priority_block(store, assets):
    """정렬 콜백과 screen_1()이 함께 쓰는 "점검 우선순위" 표."""
    col, direction = table_sort(store, "priority")
    return sortable("priority", priority_table(load_priority_table(col, direction, assets=assets), store))


# "기계별 평균 소비 전력" 표 — (열 라벨, 정렬, 열 id). 기본 순서는 평균 소비 전력 내림차순.
POWER_COLS = [("기계", "left", "asset_tag"), ("평균 소비 전력", "left", "avg_power_kw")]


def power_block(store, assets, start, end):
    """기계별 평균 소비 전력 — 막대가 든 압축 표. 단일 계열 막대는 series-1 한 색이고
    값에 따라 색을 바꾸지 않는다(.pf-risk). 열 제목을 누르면 다른 표와 같은 순서로 정렬된다."""
    col, direction = table_sort(store, "power")
    power_rows = load_screen1_power_by_machine(assets, start, end)
    max_power = max((r["avg_power_kw"] for r in power_rows), default=1.0) or 1.0
    if col:
        power_rows = sort_rows(power_rows, lambda r, c=col: r[c], direction)
    body = [
        html.Tr([
            html.Td(r["asset_tag"], className=CODE_12, style={"width": "88px"}),
            html.Td(html.Div(
                [html.Span(html.Span(className="pf-risk__fill",
                                     style={"width": f"{r['avg_power_kw'] / max_power * 100:.1f}%"}),
                           className="pf-risk__track"),
                 html.Span(f"{r['avg_power_kw']:,.2f}", className=f"pf-risk__value {NUM_13}",
                           style={"width": "56px", "flexShrink": "0"})],
                className="pf-risk")),
        ])
        for r in power_rows
    ]
    thead = html.Tr([sort_header(label, "power", field, store, align) for label, align, field in POWER_COLS])
    return sortable("power", table_scroll(
        html.Table([html.Thead(thead), html.Tbody(body)], className="pf-table pf-table--compact"),
        "기계별 평균 소비 전력 표"))


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


def screen_1(seg_state=None, audience=DEFAULT_AUDIENCE, table_sort=None, assets=None, start=None, end=None):
    # KPI 4개는 판단에 쓰는 값만 둔다(관측 기계 수·평균 소비 전력은 뺐다).
    #   고위험일(기계·일)·고위험일 비율(%) — 선택 기간. 비율 옆에 직전 같은 길이 기간 값을 둔다.
    #   기준일 고위험 기계(대) — 하루치 스냅샷. 비율(1대 = 10.0%)로 쓰면 정밀해 보이기만 해서 건수로 둔다.
    #   고장 표시 기계·일 — 선택 기간.
    # 고위험 = 고장점수 12 이상 고정(dashboard_data.HIGH_RISK_THRESHOLD). ③의 12/13/14 토글은
    # 모델 비교 결과만 바꾸고 이 값에는 영향을 주지 않는다.
    kpis = load_screen1_kpis(assets, start, end)
    # "최고 베어링 온도"·"부품 출고 금액(누적)"은 삭제한다 — 남은 4개가 같은
    # 458px 폭(4×458 + 3×16 = 1880)으로 행 전체를 균등 분배한다.
    # 집계 범위("선택 기간"·"기준일")는 카드 라벨 대신 물음표 설명으로 내렸다.
    # 고위험일 = 고장점수 12 이상인 기계·일. 하루치 스냅샷 대신 선택 기간 값을 앞에 둔다.
    prev = kpis["prev_high_risk_rate_pct"]
    kpi_specs = [
        ("고위험일 (기계·일)", f"{kpis['high_risk_days']:,}", "high_risk_days", None),
        ("고위험일 비율 (%)", f"{kpis['high_risk_rate_pct']:.1f}", "high_risk_rate_pct",
         rate_change(kpis["high_risk_rate_pct"], prev) if prev is not None else None),
        ("기준일 고위험 기계 (대)", f"{kpis['latest_high_risk_machines']:,}", "latest_high_risk_machines", None),
        ("고장 표시 기계·일", f"{kpis['failure_machine_days']:,}", "failure_machine_days", None),
    ]
    row_a = row(ROW_KPI, [kpi_value_tile(label, value_text, w=458, help_text=KPI_HELP[key], sub=sub)
                          for label, value_text, key, sub in kpi_specs])

    prio = card("점검 우선순위", 1090, ROW_MAIN, priority_block(table_sort, assets),
                right=html.Div(
                    [note("기준일 스냅샷 · 기간 미적용"),
                     help_icon("순위 = 등급가중 고장점수가 높은 순서의 점검 우선순위다. 열 제목을 누르면 그 열로 "
                               "오름차순 → 내림차순 → 기본 순서로 바뀌고, 다른 열로 정렬해도 순위 번호는 그대로다. "
                               "붉은 행은 고장점수가 위험 기준선(12) 이상인 기계다.")],
                    style={"display": "flex", "alignItems": "center", "gap": "8px"}))

    def machine_tile(status_row):
        # 기계 종류는 자르지 않는다(DESIGN §8) — 가장 긴 "Screw Compressor"가 한 줄에 들어가는 폭이고,
        # 좁은 화면에서는 어절 단위로 줄을 바꾼다.
        return html.Div(
            [html.Div([html.Span(status_row["asset_tag"], className=CODE_12, style={"whiteSpace": "nowrap"}),
                       html.Span(status_row["machine_type"], className=f"{MICRO_11} pf-sentence")],
                      style={"width": "104px", "flexShrink": "0", "display": "flex",
                             "flexDirection": "column", "gap": "4px"}),
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

    unit_index = (seg_state or DEFAULT_SEG).get("heat_unit", DEFAULT_SEG["heat_unit"])
    unit = HEATMAP_UNITS[unit_index if 0 <= unit_index < len(HEATMAP_UNITS) else DEFAULT_SEG["heat_unit"]]
    heat_body = html.Div(
        dcc.Graph(id={"type": "heatmap-chart", "index": "screen1"},
                  figure=_heatmap_figure(load_asset_failure_heatmap(assets, start, end, unit),
                                         load_period_coverage(start, end, unit), "light"),
                  config={"displayModeBar": False, "responsive": True},
                  style={"flex": "1 1 auto", "minHeight": "0", "minWidth": "0"}),
        style={"height": "100%", "display": "flex", "flexDirection": "column"},
    )
    heat = card(
        "고장 표시 히트맵", 1248, ROW_SUB, heat_body,
        right=html.Div(
            [help_icon("칸 하나 = 그 기계가 그 기간에 고장 표시된 (부품 × 날짜) 건수. "
                       "1월 5일에 부품 3개, 1월 9일에 부품 2개가 표시됐으면 5다. "
                       "날짜 끝의 '부분'은 그 구간의 일부 날짜만 데이터에 있다는 뜻이다 — "
                       "건수가 적은 것이 덜 위험했다는 뜻이 아니다."),
             seg("heat_unit", "단위", SEG_GROUPS["heat_unit"], sel=unit_index),
             html.Button("초기 배율", id={"type": "heat-reset-btn", "index": "screen1"}, n_clicks=0,
                         className="pf-btn label-12", title="확대·이동한 히트맵을 처음 배율로 되돌린다",
                         style={"flexShrink": "0"})],
            style={"display": "flex", "alignItems": "center", "gap": "8px"}))

    power = card("기계별 평균 소비 전력 (kW)", 616, ROW_SUB, power_block(table_sort, assets, start, end),
                 right=note(f"{len(assets if assets is not None else load_asset_list())}대 · 선택 기간 평균"))
    row_c = row(ROW_SUB, [heat, power])

    return html.Div([row_a, row_b, row_c],
                     style={"display": "flex", "flexDirection": "column", "gap": f"{GUTTER}px"})
