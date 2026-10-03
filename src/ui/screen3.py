"""화면 ③ 모델·예측 — 과제별 모델·기준선 비교, 혼동행렬, 임계값, 영향 변수, 부품군 진단."""

from dash import dcc, html

from ..dashboard_data import (
    load_asset_family_diagnosis, load_asset_list, load_comparison_at_cutoff, load_comparison_config,
    load_comparison_feature_importance, load_comparison_pr_curve, load_family_feature_importance,
    load_family_pr_curve_and_confusion, load_family_recurrence_intervals, load_focus_model,
    load_model_comparison,
)
from .base import (
    BODY_13, card, CODE_12, DEFAULT_AUDIENCE, DEFAULT_SEG, empty_state, GUTTER, hstack, LABEL_12, note,
    NOTE_12, NUM_13, row, ROW_KPI, ROW_MAIN, seg, SEG_GROUPS, table_scroll,
)
from .figures import _family_recur_figure, _pr_figure
from .screen1 import kpi_value_tile


# 화면 ③ 과제 ①~③의 비교 결과 키(dashboard_data.COMPARISON_TASKS)와 표시 정보.
# SEG_GROUPS["task"]와 같은 순서다. ④ 부품군 진단은 비교표 대신 부품군별 표를 쓴다.
SCREEN3_TASK_KEYS = ["machine_risk", "part_within_7d", "part_current", "family"]
SCREEN3_FAMILY_INDEX = 3
SCREEN3_TASK_INFO = {
    # (단위, 무엇을 하는가, 혼동행렬 '있음'의 뜻)
    "machine_risk": ("기계·일", "당일 센서로 같은 날 판별", "고위험일(고장점수 {thr} 이상)"),
    "part_within_7d": ("기계·부품·일", "당일까지의 정보로 7일 내 고장 표시 확률", "7일 내 고장 표시"),
    "part_current": ("기계·부품·일", "당일 센서로 같은 날 판별 · 부품 식별 정보 없이 학습", "고장 표시"),
}
CUTOFF_POLICY_TEXT = {
    "validation_f1": "검증 구간 F1 최대",
    "fixed_0.5": "0.5 고정(원 실험이 검증 구간을 쓰지 않음)",
}

# 화면 ③ "주요 영향 변수" 막대 범주 — 범주색은 series 토큰을 고정 순서로 쓴다
# (건강 센서 = series-1, 운전 조건 = series-2, 그 외 = chart-muted; 03-app.css
# .pf-fi--*). 파생 변수(_lag1/_median3/...)는 전부 "<기본 센서명>_<변환>" 형태라
# 접두사만 봐도 분류된다 — 파생이 늘어나도 매핑을 새로 추가할 필요가 없다.
_HEALTH_SENSOR_PREFIXES = (
    "temp_bearing_degC", "temp_motor_degC", "vibration_h_mms", "vibration_v_mms", "oil_pressure_bar",
)
_OPERATING_PREFIXES = ("load_pct", "shaft_rpm", "power_consumption_kw")
_OPERATING_EXACT = {"day_of_week", "is_weekend"}


def _feature_color_category(feature: str) -> str:
    """건강 센서 / 운전 조건 / 장비·부품·이력·날짜(그 외 전부)."""
    if feature.startswith(_HEALTH_SENSOR_PREFIXES):
        return "health"
    if feature.startswith(_OPERATING_PREFIXES) or feature in _OPERATING_EXACT:
        return "operating"
    return "other"


def _screen3_threshold(task_key, seg_state):
    """12/13/14 토글은 기계 고위험일 판별에만 적용된다. 다른 과제는 None."""
    if task_key != "machine_risk":
        return None
    index = (seg_state or DEFAULT_SEG).get("threshold", 0)
    return int(SEG_GROUPS["threshold"][index if 0 <= index < len(SEG_GROUPS["threshold"]) else 0])


def _screen3_task_key(seg_state):
    index = (seg_state or DEFAULT_SEG).get("task", 0)
    return SCREEN3_TASK_KEYS[index if 0 <= index < len(SCREEN3_TASK_KEYS) else 0]


def _selection_text(row):
    labels = []
    if bool(row["production"]):
        labels.append("운영(팀 결정)")
    if bool(row["selected_by_validation"]):
        labels.append("검증 AP 최고")
    return " · ".join(labels) or "—"


def _comparison_table(rows, focus_model):
    """화면 ③ 모델 비교표 — 같은 분할·평가 구간의 기준 A·B와 모델들.
    정확도는 KPI가 아니라 이 표에만 두고, 옆에 '항상 없음 예측 정확도'를 붙인다."""
    cols = [("모델", "left"), ("AP", "right"), ("AP ÷ 양성 비율 (배)", "right"), ("ROC-AUC", "right"),
            ("정밀도", "right"), ("재현율", "right"), ("경보율 (%)", "right"), ("상위 10% 정밀도", "right"),
            ("정확도", "right"), ("항상 없음 예측 정확도", "right"), ("선택", "left")]

    def num(text):
        return html.Td(text, className=f"{NUM_13} pf-td--num")

    body_rows = [
        html.Tr([
            html.Td(r["model_label"], className=BODY_13),
            num(f"{r['average_precision']:.3f}"), num(f"{r['ap_lift']:.2f}"), num(f"{r['roc_auc']:.3f}"),
            num(f"{r['precision']:.3f}"), num(f"{r['recall']:.3f}"), num(f"{r['alert_rate'] * 100:.1f}"),
            num(f"{r['top10_precision']:.3f}"), num(f"{r['accuracy']:.3f}"),
            num(f"{r['always_negative_accuracy']:.3f}"),
            html.Td(_selection_text(r), className=BODY_13),
        ], **{"aria-selected": "true" if r["model"] == focus_model else "false"})
        for r in rows
    ]
    thead = html.Tr([html.Th(l, className="label-12" + (" pf-th--num" if a == "right" else ""),
                             style={"textAlign": a}) for l, a in cols])
    return table_scroll(html.Table([html.Thead(thead), html.Tbody(body_rows)], className="pf-table"), "모델 비교 표")


def _confusion_body(confusion, positive_name):
    """혼동행렬 — 칸마다 건수와 행 비율(실제 있음·없음 각각을 100%로 본 비율)."""
    def cell(count, row_total):
        share = count / row_total * 100 if row_total else 0.0
        return html.Div([html.Span(f"{count:,}", className="value-20"),
                         html.Span(f"행의 {share:.1f}%", className=f"{LABEL_12}")],
                        className="pf-cm-cell",
                        style={"flex": "1 1 0", "minWidth": "0", "display": "flex", "flexDirection": "column",
                               "alignItems": "center", "justifyContent": "center", "gap": "4px"})

    positives = confusion["tp"] + confusion["fn"]
    negatives = confusion["fp"] + confusion["tn"]

    def body_row(label, left, right, total):
        return hstack([html.Div(label, className=LABEL_12, style={"width": "64px", "display": "flex",
                                                                  "alignItems": "center"}),
                       cell(left, total), cell(right, total)], 8, {"flex": "1 1 0", "minHeight": "0"})

    return html.Div([
        hstack([html.Div(style={"width": "64px"}),
                html.Div("판정 있음", className=LABEL_12, style={"flex": "1 1 0", "textAlign": "center"}),
                html.Div("판정 없음", className=LABEL_12, style={"flex": "1 1 0", "textAlign": "center"})],
               8, {"minHeight": "24px", "alignItems": "center"}),
        body_row("실제 있음", confusion["tp"], confusion["fn"], positives),
        body_row("실제 없음", confusion["fp"], confusion["tn"], negatives),
        html.Div(f"있음 = {positive_name}", className=NOTE_12),
    ], style={"display": "flex", "flexDirection": "column", "gap": "8px", "height": "100%"})


def _threshold_body(metrics, positive_name):
    """판정 임계값 카드 본문 — 슬라이더 값(저장된 Test 점수에 다시 적용)으로 다시 그린다."""
    caught = metrics["tp"]
    actual = metrics["tp"] + metrics["fn"]
    rows = [("경보율 (%)", f"{metrics['alert_rate'] * 100:.1f}"), ("경보 건수", f"{metrics['alerts']:,}"),
            ("정밀도", f"{metrics['precision']:.3f}"), ("재현율", f"{metrics['recall']:.3f}"),
            ("오경보 건수", f"{metrics['fp']:,}"), ("놓친 건수", f"{metrics['fn']:,}")]
    sentence = (f"임계값 {metrics['cutoff']:.2f}이면 {metrics['rows']:,}건 중 {metrics['alerts']:,}건에 "
                f"경보를 내고, 실제 {positive_name} {actual:,}건 중 {caught:,}건을 잡는다.")
    return html.Div(
        [html.Div([html.Span(k, className=LABEL_12, style={"width": "96px", "flexShrink": "0"}),
                   html.Span(v, className=f"{NUM_13} pf-strong")],
                  style={"display": "flex", "alignItems": "center", "gap": "8px", "minHeight": "18px"})
         for k, v in rows]
        + [html.Div(sentence, className=LABEL_12, style={"marginTop": "6px"})],
        style={"display": "flex", "flexDirection": "column", "gap": "2px"},
    )


def _importance_body(fi_rows, note_text):
    """영향 변수 막대 목록 + 범주 범례. 음수(섞어도 AP가 오른 변수)는 막대 길이 0."""
    max_importance = max([r["importance_mean"] for r in fi_rows] + [1e-12])

    def legend_item(category, label):
        return html.Span([html.Span(className=f"pf-legend__key pf-legend__key--dot pf-fi--{category}"), label],
                         className="pf-legend__item")

    fi_list = html.Div(
        [html.Div([
            html.Div(r["feature"], className=CODE_12, title=r["feature"],
                     style={"width": "150px", "flexShrink": "0", "whiteSpace": "nowrap",
                            "overflow": "hidden", "textOverflow": "ellipsis"}),
            html.Div(html.Div(className=f"pf-fi-bar pf-fi--{_feature_color_category(r['feature'])}",
                              style={"width": f"{max(r['importance_mean'], 0) / max_importance * 100:.1f}%"}),
                     style={"flexGrow": "1", "minWidth": "0"}),
            html.Div(f"{r['importance_mean']:.3f}", className=f"{NUM_13} pf-muted",
                     style={"width": "48px", "textAlign": "right", "flexShrink": "0"}),
         ], style={"height": "16px", "display": "flex", "alignItems": "center", "gap": "8px", "flexShrink": "0"})
         for r in fi_rows],
        style={"display": "flex", "flexDirection": "column", "gap": "4px"},
    )
    footer = html.Div(
        [html.Div([legend_item("health", "건강 센서"), legend_item("operating", "운전 조건"),
                   legend_item("other", "그 외")], className="pf-legend label-12"),
         note(note_text, {"whiteSpace": "normal"})],
        className="pf-divider-top",
        style={"display": "flex", "flexDirection": "column", "gap": "4px", "marginTop": "8px", "paddingTop": "6px"},
    )
    return html.Div([fi_list, footer], style={"display": "flex", "flexDirection": "column"})


def _screen3_toolbar(seg_state, task_key, threshold, meta_text):
    thr_sel = seg_state.get("threshold", 0)
    if task_key == "machine_risk":
        # 위험 기준선은 표의 축이 아니라 '라벨 정의'다 — 바꾸면 정답이 바뀌므로 그 기준으로
        # 학습한 결과로 표 전체가 바뀐다. 필터로 착각하지 않도록 컨트롤 바로 아래에 상시 둔다.
        changed = thr_sel != DEFAULT_SEG["threshold"]
        warn_text = ("위험 기준선은 필터가 아니라 학습 라벨 정의다 — 바꾸면 그 기준으로 학습한 결과로 아래 전체가 바뀐다"
                     + (f" · 현재 {threshold}점 기준" if changed else ""))
    else:
        changed = False
        warn_text = "위험 기준선(12/13/14)은 기계 고위험일 판별에만 적용된다"
    return html.Div(
        [html.Div([seg("task", "과제", SEG_GROUPS["task"], sel=seg_state.get("task", 0)),
                   seg("threshold", "위험 기준선 (등급가중 고장점수)", SEG_GROUPS["threshold"], sel=thr_sel,
                       disabled=task_key != "machine_risk")],
                  style={"height": "32px", "display": "flex", "alignItems": "center",
                         "justifyContent": "space-between", "gap": "16px"}),
         html.Div([html.Span(meta_text, className=LABEL_12,
                             style={"whiteSpace": "nowrap", "overflow": "hidden", "textOverflow": "ellipsis",
                                    "minWidth": "0"}),
                   html.Span(warn_text, className="pf-inline-note label-12" + (" pf-inline-note--emphasis" if changed else ""),
                             style={"whiteSpace": "nowrap", "flexShrink": "0"})],
                  style={"display": "flex", "alignItems": "center", "justifyContent": "space-between",
                         "gap": "16px", "height": "20px"})],
        style={"height": "60px", "display": "flex", "flexDirection": "column", "gap": "8px", "flexShrink": "0"},
    )


# ============================================================
# 화면 ③ 모델·예측 — 과제 ①~③: 툴바 60 / KPI 96 / 비교표 292 / 432
#                    과제 ④: 툴바 60 / 460 / 252
# ============================================================

def screen_3(seg_state=None, audience=DEFAULT_AUDIENCE, asset_tag=None, family=None):
    seg_state = seg_state or DEFAULT_SEG
    task_key = _screen3_task_key(seg_state)
    if task_key == "family":
        return _family_screen(seg_state, asset_tag, family)
    threshold = _screen3_threshold(task_key, seg_state)
    grain, what, positive_template = SCREEN3_TASK_INFO[task_key]
    positive_name = positive_template.format(thr=threshold)
    config = load_comparison_config(task_key)
    rows = load_model_comparison(task_key, threshold)
    focus = load_focus_model(task_key, threshold)
    focus_row = next(r for r in rows if r["model"] == focus)

    meta_text = (f"{grain} · {what} · Test {config['test_period']} · {focus_row['test_rows']:,}행 · "
                 f"양성 비율 {focus_row['positive_rate'] * 100:.1f}%")
    toolbar = _screen3_toolbar(seg_state, task_key, threshold, meta_text)

    focus_scope = focus_row["model_label"] + (" · 운영" if bool(focus_row["production"]) else "")
    kpi_specs = [("AP", f"{focus_row['average_precision']:.3f}"),
                 ("AP ÷ 양성 비율 (배)", f"{focus_row['ap_lift']:.2f}"),
                 ("경보율 (%)", f"{focus_row['alert_rate'] * 100:.1f}"),
                 ("재현율", f"{focus_row['recall']:.3f}")]
    row_a = row(ROW_KPI, [kpi_value_tile(label, value, w=458, scope=focus_scope) for label, value in kpi_specs])

    policy = CUTOFF_POLICY_TEXT.get(focus_row["cutoff_policy"], focus_row["cutoff_policy"])
    table_note = f"같은 Test 구간 · 판정 기준 {policy}"
    if any(bool(r["production"]) for r in rows):
        table_note += " · 운영 모델 RF는 팀 결정 — 근거 TODO(사용자 확인)"
    comparison = card("모델 비교", 1880, 292, _comparison_table(rows, focus), right=note(table_note))
    row_b = row(292, [comparison])

    pr = load_comparison_pr_curve(task_key, focus, threshold)
    pr_body = html.Div(
        dcc.Graph(id={"type": "comparison-pr-chart", "index": "screen3"},
                  figure=_pr_figure(pr, "light", height=360),
                  config={"displayModeBar": False, "responsive": True},
                  style={"flex": "1 1 auto", "minHeight": "0", "minWidth": "0"}),
        style={"flex": "1 1 auto", "minHeight": "0", "display": "flex", "flexDirection": "column"})
    prc = card("PR 곡선", 458, 432, pr_body, right=note(focus_row["model_label"]))

    at_cutoff = load_comparison_at_cutoff(task_key, focus, focus_row["cutoff"], threshold)
    cmx = card("혼동행렬", 458, 432,
               html.Div(_confusion_body(at_cutoff, positive_name), id={"type": "cm-body", "index": "screen3"},
                        style={"height": "100%"}),
               right=note("행 비율(%)"))

    slider = html.Div(
        [html.Label("판정 임계값", htmlFor="thr-slider", className=LABEL_12, style={"whiteSpace": "nowrap"}),
         html.Div(dcc.Slider(id={"type": "thr-slider", "index": "screen3"}, min=0, max=100, step=1, marks=None,
                             value=round(focus_row["cutoff"] * 100), tooltip={"placement": "bottom"}),
                  style={"flex": "1 1 auto", "minWidth": "0"})],
        style={"display": "flex", "alignItems": "center", "gap": "8px"},
    )
    thr_body = html.Div(
        [slider,
         note(f"저장된 Test 점수에 임계값만 다시 적용하는 가상 실험 · 기본값 {focus_row['cutoff']:.2f}({policy})",
              {"whiteSpace": "normal"}),
         html.Div(_threshold_body(at_cutoff, positive_name), id={"type": "thr-metrics", "index": "screen3"})],
        style={"display": "flex", "flexDirection": "column", "gap": "8px"},
    )
    thr = card("판정 임계값", 458, 432, thr_body)

    fi_rows = load_comparison_feature_importance(task_key, threshold)
    if fi_rows:
        fi_note = f"{MODEL_LABEL_SHORT.get(fi_rows[0]['model'], fi_rows[0]['model'])} · 검증 구간 permutation importance(AP 감소량)"
        if "rf_reproduction_max_abs_diff" in config:
            fi_note += (f" · 같은 설정으로 다시 학습한 RF 기준(저장된 예측과 점수 최대 차이 "
                        f"{config['rf_reproduction_max_abs_diff']:.2f})")
        feat = card("주요 영향 변수 상위 10", 458, 432,
                    _importance_body(fi_rows, "중요도는 원인이 아니라 판별에 기여한 정도다 · " + fi_note))
    else:
        feat = card("주요 영향 변수", 458, 432,
                    empty_state("데이터 없음", "이 과제의 영향 변수 결과 파일(feature_importance.csv)이 없다"))
    row_c = row(432, [prc, cmx, thr, feat])

    return html.Div([toolbar, row_a, row_b, row_c],
                     style={"display": "flex", "flexDirection": "column", "gap": f"{GUTTER}px"})


MODEL_LABEL_SHORT = {"logistic_regression": "로지스틱 회귀", "random_forest": "랜덤 포레스트",
                     "hist_gradient_boosting": "HGB"}


def _family_screen(seg_state, asset_tag, family):
    """화면 ③ ④ 부품군 진단 — 선택 기계의 부품군 9종 표·PR 곡선·혼동행렬·영향 변수·재발 간격."""
    assets = load_asset_list()
    asset_tag = asset_tag if asset_tag in assets else assets[0]
    family_rows = load_asset_family_diagnosis(asset_tag)
    valid_families = [fr["part_family"] for fr in family_rows]
    family = family if family in valid_families else valid_families[0]
    pr_cm = load_family_pr_curve_and_confusion(asset_tag, family)
    c = pr_cm["confusion"]
    pr_cm = {**pr_cm, "positive_rate": (c["tp"] + c["fn"]) / max(1, sum(c.values()))}
    toolbar = _screen3_toolbar(seg_state, "family", None,
                               f"기계·부품군·일 · 당일 센서로 같은 날 진단 · 선택 기계 {asset_tag}")

    # average_precision 내림차순(load_asset_family_diagnosis가 이미 정렬) 1행이 최초 진입 기본 선택 부품군이다.
    family_cols = [
        ("부품군", "left"), ("모델", "left"), ("표본 수", "right"),
        ("양성률 (%)", "right"), ("정밀도", "right"), ("재현율", "right"),
        ("AP", "right"), ("ROC-AUC", "right"),
    ]

    def family_num(text):
        return html.Td(text, className=f"{NUM_13} pf-td--num")
    family_body_rows = [
        html.Tr([
            html.Td(fr["part_family"], className=BODY_13),
            html.Td(fr["model"], className=BODY_13),
            family_num(str(fr["support"])),
            family_num(f"{fr['positive_rate'] * 100:.1f}"),
            family_num(f"{fr['precision']:.3f}"),
            family_num(f"{fr['recall']:.3f}"),
            family_num(f"{fr['average_precision']:.3f}"),
            family_num(f"{fr['roc_auc']:.3f}"),
        ], id={"type": "family-row", "index": fr["part_family"]}, n_clicks=0,
           style={"cursor": "pointer"},
           **{"aria-selected": "true" if fr["part_family"] == family else "false"})
        for fr in family_rows
    ]
    family_thead = html.Tr([html.Th(l, className="label-12" + (" pf-th--num" if a == "right" else ""),
                                    style={"textAlign": a})
                            for l, a in family_cols])
    mcomp_body = table_scroll(html.Table([html.Thead(family_thead), html.Tbody(family_body_rows)],
                                         className="pf-table"), "부품군 진단 표")
    mcomp = card("부품군 진단", 774, ROW_MAIN, mcomp_body, right=note("행 = 부품군 · 행 클릭 시 오른쪽·아래 카드 갱신"))

    pr_body = html.Div(
        dcc.Graph(id={"type": "family-pr-chart", "index": "screen3"},
                  figure=_pr_figure(pr_cm, "light"),
                  config={"displayModeBar": False, "responsive": True},
                  style={"flex": "1 1 auto", "minHeight": "0", "minWidth": "0"}),
        style={"flex": "1 1 auto", "minHeight": "0", "display": "flex", "flexDirection": "column"})
    prc = card("PR 곡선", 616, ROW_MAIN, pr_body, right=note(f"{asset_tag} · {family}"))
    cmx = card("혼동행렬", 458, ROW_MAIN, _confusion_body(c, f"{family} 고장 표시"),
               right=note(f"판정 임계값 {pr_cm['cutoff']:.3f} · 행 비율(%)"))
    row_b = row(ROW_MAIN, [mcomp, prc, cmx])

    fi_rows = load_family_feature_importance(family)
    feat = card("주요 영향 변수 상위 10", 1090, 252,
                _importance_body(fi_rows, "중요도는 고장 원인이 아니라 고장과 함께 변화한 운전 상태를 뜻한다 · 부품군 자체 속성(기계 무관)"))
    intervals = load_family_recurrence_intervals(asset_tag, family)
    if intervals:
        lead_body = html.Div(
            dcc.Graph(id={"type": "family-recur-chart", "index": "screen3"},
                      figure=_family_recur_figure(intervals, "light"),
                      config={"displayModeBar": False, "responsive": True},
                      style={"flex": "1 1 auto", "minHeight": "0", "minWidth": "0"}),
            style={"flex": "1 1 auto", "minHeight": "0", "display": "flex", "flexDirection": "column"})
    else:
        lead_body = empty_state("이 과제에는 해당 없음", "재발 이력이 2회 미만이라 간격을 계산할 수 없음")
    lead = card("재발 간격", 774, 252, lead_body, right=note("과거 재발 간격 — 예측이 아닌 회고적 통계"))
    row_c = row(252, [feat, lead])

    return html.Div([toolbar, row_b, row_c],
                     style={"display": "flex", "flexDirection": "column", "gap": f"{GUTTER}px"})
