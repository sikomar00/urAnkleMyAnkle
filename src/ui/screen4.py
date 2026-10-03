"""화면 ④ 데이터 — 데이터 조회 표, 품질 요약, 데이터 사전. 화면 ⑤(보고서 요약 참조 구현) 포함."""

from dash import dash_table, dcc, html

from ..dashboard_data import (
    load_data_dictionary, load_data_quality_summary, load_failure_trend, load_screen5_kpis,
    load_source_info, load_table_page,
)
from .base import (
    card, CODE_12, col, DEFAULT_AUDIENCE, DEFAULT_SEG, empty_state, GUTTER, hstack, LABEL_12, note, NUM_13,
    REPORT_AUDIENCES, row, ROW_KPI, ROW_MAIN, ROW_SUB, seg, SEG_GROUPS, slot, table_scroll, tile,
)
from .figures import _trend_figure
from .screen1 import kpi_value_tile


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


def screen_4(seg_state=None, audience=DEFAULT_AUDIENCE, assets=None, start=None, end=None):
    seg_state = seg_state or DEFAULT_SEG
    dataset_index = seg_state.get("dataset", 0)
    # 예전에 저장된 선택(없어진 데이터셋 인덱스)은 원자료로 되돌린다.
    if dataset_index not in DATASET_KEY_BY_SEG_INDEX:
        dataset_index = 0
    dataset_key = DATASET_KEY_BY_SEG_INDEX[dataset_index]

    page0 = load_table_page(dataset_key, None, "asc", 0, assets=assets, start=start, end=end)
    columns_prop = [{"name": c["label"], "id": c["id"]} for c in page0["columns"]]
    right_align_ids = [c["id"] for c in page0["columns"] if c["align"] == "right"]
    # 숫자 열(우측 정렬)과 식별자 열은 고정폭 글꼴 — 색·글꼴은 03-app.css(.pf-dtable)와
    # 아래 css 규칙이 토큰으로 입힌다. style_* 에는 치수·정렬만 둔다.
    mono_ids = right_align_ids + [c["id"] for c in page0["columns"]
                                  if c["id"] in ("transaction_date", "asset_tag", "plant_code", "part_no")]
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
        style_table={"overflowX": "auto", "width": "100%"},
        style_header={"height": "32px", "minHeight": "32px", "maxHeight": "32px"},
        style_cell={"padding": "0 8px", "height": "24px", "minHeight": "24px", "maxHeight": "24px",
                    "textAlign": "left"},
        style_cell_conditional=[
            {"if": {"column_id": cid}, "textAlign": "right"}
            for cid in right_align_ids
        ],
        # 내장 페이저 padding·여백을 줄여 카드 높이(524px) 예산에 맞춘다
        # (정정 #2 — 행 높이·page_size는 그대로 두고 페이저만 압축).
        css=[
            {"selector": ".previous-next-container", "rule": "padding:2px 0; margin:0;"},
            {"selector": ".previous-next-container button", "rule": "padding:2px 4px; margin:0 2px;"},
            {"selector": ".page-number, .current-page-container",
             "rule": "font-family:var(--font-mono); font-size:11px; color:var(--ink-secondary); margin:0 2px;"},
            # 진단으로 특정한 진짜 원인: dash_table 기본 번들 스타일시트의
            # ".dash-spreadsheet-inner tr { height:30px; min-height:30px; }"가
            # style_cell의 24px 지정을 무시하고 행 높이를 30px로 고정한다
            # (.dash-cell-value/-container 문제가 아님). 같은 선택자로
            # 이 표에만 재정의한다.
            {"selector": ".dash-spreadsheet-inner tr",
             "rule": "height:24px; min-height:24px;"},
        ] + [{"selector": f'td[data-dash-column="{cid}"]',
              "rule": "font-family:var(--font-mono); font-variant-numeric:tabular-nums;"} for cid in mono_ids],
    )
    footer = html.Div(
        html.Span(_dtable_footer_text(page0["total_rows"], 0, 16),
                  id={"type": "dtable-footer", "index": dataset_key}, className=LABEL_12),
        style={"height": "20px", "display": "flex", "alignItems": "center"},
    )
    dtable_body = html.Div([table, footer], className="pf-dtable")
    csv_btn = html.Button("CSV 내보내기", id={"type": "csv-export-btn", "index": dataset_key},
                          n_clicks=0, className="pf-btn label-12")

    toolbar = html.Div(
        [seg("dataset", "데이터셋", SEG_GROUPS["dataset"], sel=dataset_index),
         html.Div([csv_btn], style={"display": "flex", "alignItems": "center", "gap": "12px"})],
        style={"height": "32px", "flexShrink": "0", "display": "flex",
               "alignItems": "center", "justifyContent": "space-between"},
    )

    dtable = card(f"데이터 조회 · {SEG_GROUPS['dataset'][dataset_index]}", 1880, 524, dtable_body,
                  right=note("공장·기계 종류·기계·기간 필터 적용"))

    dict_cols = [("컬럼명", "left"), ("타입", "left"), ("단위", "left"),
                 ("결측률 (%)", "right"), ("설명", "left")]

    def dict_table(records):
        """데이터 사전 표(.pf-table) — 글자를 자르지 않는다. 열 폭은 내용에 맞추고 설명만
        줄바꿈하며, 22행이 카드 높이를 넘으면 카드 안에서 세로로 스크롤한다(헤더 고정)."""
        thead = html.Tr(
            [html.Th(l, className="label-12" + (" pf-th--num" if a == "right" else ""),
                     style={"height": "24px", "padding": "0 8px", "textAlign": a})
             for l, a in dict_cols],
        )
        body_rows = []
        for record in records:
            missing_pct = record["missing_pct"]
            missing_text = "—" if missing_pct is None else f"{missing_pct:.2f}"
            values = [record["column"], record["dtype_label"], record["unit"],
                      missing_text, record["description"]]
            cells = []
            for (l, a), text in zip(dict_cols, values):
                cls = {"컬럼명": CODE_12, "결측률 (%)": f"{NUM_13} pf-td--num"}.get(l, "label-12")
                cell_style = {"height": "24px", "padding": "0 8px", "textAlign": a}
                if l == "설명":
                    cell_style["whiteSpace"] = "normal"
                cells.append(html.Td(text, className=cls, style=cell_style))
            body_rows.append(html.Tr(cells))
        return table_scroll(html.Table([html.Thead(thead), html.Tbody(body_rows)], className="pf-table"),
                            "데이터 사전 표")

    ddict = card("데이터 사전 (22열)", 932, ROW_SUB, dict_table(load_data_dictionary()),
                 right=note("데이터셋 전체 기준 · 필터 미적용"))

    def info_box(text, h=94):
        """품질·출처 카드의 문장 칸 — 같은 줄의 칸들이 카드 폭을 나눠 가진다. screen_4 전용."""
        return html.Div(
            html.Span(text, className="label-12", style={"whiteSpace": "pre-line", "wordBreak": "keep-all"}),
            style={"flex": "1 1 0", "minWidth": "0", "height": f"{h}px", "overflow": "hidden"},
        )

    q = load_data_quality_summary()
    period_tile = html.Div(
        [html.Span(f"기간: {q['period_days']:,}일", className="label-12"),
         html.Span(f"{q['period_start']} ~ {q['period_end']}", className=CODE_12, style={"whiteSpace": "nowrap"})],
        style={"flex": "1 1 0", "minWidth": "0", "height": "94px", "overflow": "hidden",
               "display": "flex", "flexDirection": "column"},
    )
    quality_tiles = [
        info_box(f"중복: 복합키(날짜·기계·부품) 중복 {q['composite_key_duplicates']:,}건 · "
                 f"완전 중복 {q['full_duplicates']:,}건"),
        info_box(f"wo_type: '작업 없음' 범주 {q['wo_type_blank_count']:,}행 ({q['wo_type_blank_pct']:.2f}%) · "
                 f"결측 아님"),
        period_tile,
        info_box(f"센서값 반복: 기계×날짜 {q['group_count']:,}개 그룹, 그룹당 부품 행 {q['rows_per_group']:,}개에 "
                 f"센서값 동일 반복"),
    ]
    qual = card("품질 요약", 932, 162, hstack(quality_tiles, 16), right=note("데이터셋 전체 기준 · 필터 미적용"))

    s = load_source_info()
    source_lines = "\n".join([s["source_name"], f"{s['row_count']:,}행 × {s['col_count']}열",
                               s["access_date_note"]])
    src = card("출처 · 라이선스 · 합성 데이터 한계", 932, 162,
               hstack([info_box(source_lines), info_box(s["license"]), info_box(s["limitations"])], 16))
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
    caveat_top = empty_state("모델 최종 평가 전", "평가 구간·지표는 평가 완료 후 기재", 78)
    caveat_bot = html.Div(
        html.Span("합성 데이터 · 교육용 — 이 보고서의 모든 수치는 실제 설비 이력이 아니다 "
                  "(헤더 배지와 동일 문구를 보고서 산출물에도 유지)",
                  className=LABEL_12, style={"textAlign": "center"}),
        className="pf-placeholder",
        style={"height": "78px", "display": "flex", "alignItems": "center",
               "justifyContent": "center", "padding": "8px 16px"},
    )
    caveat_card = card("데이터·모델 신뢰도 고지", 932, ROW_SUB,
                       html.Div([caveat_top, caveat_bot],
                                style={"display": "flex", "flexDirection": "column", "gap": "8px"}))
    row_c = row(ROW_SUB, [summary_card, caveat_card])

    return html.Div([row_a, row_b, row_c],
                     style={"display": "flex", "flexDirection": "column", "gap": f"{GUTTER}px"})
