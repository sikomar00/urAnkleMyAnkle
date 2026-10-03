"""화면 ① 현황 — KPI 타일, 점검 우선순위 표, 기계 상태·히트맵·전력."""

from dash import dcc, html

from ..dashboard_data import (
    load_asset_failure_heatmap, load_month_coverage, load_priority_table, load_screen1_kpis,
    load_screen1_machine_status, load_screen1_power_by_machine,
)
from .base import (
    _span, BODY_13, card, CODE_12, DEFAULT_AUDIENCE, DEFAULT_PRIO_SORT, GUTTER, KPI_FAILRATE_ID, LABEL_12,
    MICRO_11, note, NUM_13, row, ROW_KPI, ROW_MAIN, ROW_SUB, status_badge, table_scroll,
)
from .figures import _heatmap_figure, _spark_figure


# ============================================================
# 화면 ① 현황 — 행 96 / 460 / 340
# ============================================================

def kpi_value_tile(label, value_text, w=300, h=96, tid=None, scope=None, sub=None):
    """KPI 값 타일(.pf-kpi). scope는 라벨 오른쪽의 집계 범위 안내(예: "기준일", "선택 기간"),
    sub는 값 옆의 보조 비교(예: 직전 같은 기간 값)."""
    kwargs = {"id": tid, "n_clicks": 0} if tid else {}
    return html.Div(
        [html.Div([html.Span(label, className=f"pf-kpi__label {LABEL_12}")] + ([note(scope)] if scope else []),
                  style={"display": "flex", "justifyContent": "space-between", "gap": "8px"}),
         html.Div([html.Span(value_text, className="pf-kpi__value value-28")]
                  + ([html.Span(sub, className="pf-kpi__sub label-12")] if sub else []),
                  className="pf-kpi__row")],
        className="pf-card pf-kpi" + (" pf-kpi--action" if tid else ""),
        style={"gridColumn": f"span {_span(w)}", "height": f"{h}px", "minWidth": "0"}, **kwargs,
    )


def priority_table(cols, records, sort_col=None, direction="desc"):
    """점검 우선순위 표(.pf-table) — load_priority_table()이 만든 자산별 값을 채운다.
    숫자 열은 num-13 우측 정렬, 식별자(asset_tag·plant_code)는 code-12.
    열 폭은 내용에 맞추고(자동 배치), 카드보다 넓으면 카드 안에서만 가로 스크롤한다."""
    arrow = " ▲" if direction == "asc" else " ▼"
    thead = html.Tr(
        [html.Th(f"{l}{arrow if i == sort_col else ''}",
                 className="label-12" + (" pf-th--num" if a == "right" else ""),
                 style={"textAlign": a})
         for i, (l, a) in enumerate(cols)],
    )
    field_order = ["rank", "asset_tag", "machine_type", "plant_code", "failure_points",
                   "threshold_exceeded", "high_risk_days_30d", "failed_part_count"]
    num_fields = {"rank", "failure_points", "high_risk_days_30d", "failed_part_count"}
    code_fields = {"asset_tag", "plant_code"}
    body_rows = []
    for record in records:
        cells = []
        for (l, a), field in zip(cols, field_order):
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


def screen_1(seg_state=None, audience=DEFAULT_AUDIENCE, prio_sort=None, assets=None, start=None):
    # KPI 4개는 판단에 쓰는 값만 둔다(관측 기계 수·평균 소비 전력은 뺐다).
    #   고위험일(기계·일)·고위험일 비율(%) — 선택 기간. 비율 옆에 직전 같은 길이 기간 값을 둔다.
    #   기준일 고위험 기계(대) — 하루치 스냅샷. 비율(1대 = 10.0%)로 쓰면 정밀해 보이기만 해서 건수로 둔다.
    #   고장 표시 기계·일 — 선택 기간.
    # 고위험 = 고장점수 12 이상 고정(dashboard_data.HIGH_RISK_THRESHOLD). ③의 12/13/14 토글은
    # 모델 비교 결과만 바꾸고 이 값에는 영향을 주지 않는다.
    # "기준일 고위험 기계" 타일을 클릭하면 아래 "점검 우선순위" 표가 기준선 초과 기준으로
    # 다시 정렬된다 — 새 카드 없이 이미 있는 표에서 어느 기계인지 이어 보게 한다.
    prio_sort = prio_sort or DEFAULT_PRIO_SORT
    # kpi-failrate-tile은 화면 ①에만 존재하고 ②~⑤로 넘어가면 DOM에서 사라진다.
    # 일반 문자열 id로 Input을 걸면 그 화면들에서 "ID not found in layout"
    # 콘솔 경고가 뜬다 — 패턴 매칭 id({"type":...})를 쓰면 Dash가 "지금 이
    # id를 가진 컴포넌트가 0개일 수 있다"를 정상 상태로 취급해 경고가 안 뜬다.
    kpis = load_screen1_kpis(assets, start)
    # "최고 베어링 온도"·"부품 출고 금액(누적)"은 삭제한다 — 남은 4개가 같은
    # 458px 폭(4×458 + 3×16 = 1880)으로 행 전체를 균등 분배한다.
    # 네 번째 값은 (label, value, 클릭 id, 집계 범위 안내).
    # 고위험일 = 고장점수 12 이상인 기계·일. 하루치 스냅샷 대신 선택 기간 값을 앞에 둔다.
    prev = kpis["prev_high_risk_rate_pct"]
    kpi_specs = [
        ("고위험일 (기계·일)", f"{kpis['high_risk_days']:,}", None, "선택 기간", None),
        ("고위험일 비율 (%)", f"{kpis['high_risk_rate_pct']:.1f}", None, "선택 기간",
         f"직전 같은 기간 {prev:.1f}" if prev is not None else None),
        ("기준일 고위험 기계 (대)", f"{kpis['latest_high_risk_machines']:,}", KPI_FAILRATE_ID, "기준일", None),
        ("고장 표시 기계·일", f"{kpis['failure_machine_days']:,}", None, "선택 기간", None),
    ]
    row_a = row(ROW_KPI, [kpi_value_tile(label, value_text, w=458, tid=tid, scope=scope, sub=sub)
                          for label, value_text, tid, scope, sub in kpi_specs])

    prio_cols = [("순위", "right"), ("대상", "left"), ("종류", "left"), ("공장", "left"),
                 ("등급가중 고장점수", "right"), ("기준선 초과", "center"),
                 ("최근 30일 고위험일 수", "right"), ("당일 고장 표시 부품 수", "right")]
    sort_by = prio_sort.get("sort_by", "grade")
    direction = prio_sort.get("direction", "desc")
    sort_idx = 5 if sort_by == "threshold" else 4
    sort_hint = ("정렬: 기준선 초과" if sort_by == "threshold"
                 else "정렬: 등급가중 고장점수 (기본)") + (" · 오름차순" if direction == "asc" else " · 내림차순")
    priority_records = load_priority_table(sort_by, direction, assets=assets)
    dir_btn = html.Button("▲ 오름차순" if direction == "asc" else "▼ 내림차순",
                           id={"type": "prio-dir-btn", "index": "screen1"}, n_clicks=0,
                           className="pf-btn label-12", style={"width": "96px"})
    prio = card("점검 우선순위", 1090, ROW_MAIN,
                priority_table(prio_cols, priority_records, sort_col=sort_idx, direction=direction),
                right=html.Div([note(f"기준일 스냅샷 · 기간 미적용 · {sort_hint}"),
                                 dir_btn],
                                style={"display": "flex", "alignItems": "center", "gap": "8px"}))

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
                  figure=_heatmap_figure(load_asset_failure_heatmap(assets, start), load_month_coverage(start), "light"),
                  config={"displayModeBar": False, "responsive": True},
                  style={"flex": "1 1 auto", "minHeight": "0", "minWidth": "0"}),
        style={"height": "100%", "display": "flex", "flexDirection": "column"},
    )
    heat = card("고장 표시 히트맵", 1248, ROW_SUB, heat_body,
                right=note("셀 = 그 달 고장 표시된 부품-일 행 수 합계 · 부분 = 관측 일수가 그 달보다 적은 달"))

    power_rows = load_screen1_power_by_machine(assets, start)
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
    power = card("기계별 평균 소비 전력 (kW)", 616, ROW_SUB, power_body, right=note(f"{len(power_rows)}개 · 내림차순 · 선택 기간 평균"))
    row_c = row(ROW_SUB, [heat, power])

    return html.Div([row_a, row_b, row_c],
                     style={"display": "flex", "flexDirection": "column", "gap": f"{GUTTER}px"})
