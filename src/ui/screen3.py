"""화면 ③ 모델·예측 — 과제별 모델·기준선 비교와 성능 해석, 혼동행렬, 판정 임계값, 영향 변수,
부품군 진단과 운영 판단."""

from dash import dcc, html

from ..dashboard_data import (
    load_asset_family_diagnosis, load_asset_list, load_comparison_at_cutoff, load_comparison_config,
    load_comparison_feature_importance, load_comparison_pr_curve, load_family_feature_importance,
    load_family_pr_curve_and_confusion, load_focus_model, load_model_comparison,
)
from .base import (
    BODY_13, card, DEFAULT_AUDIENCE, DEFAULT_SEG, empty_state, GUTTER, help_icon, LABEL_12,
    legend, note, NOTE_12, NUM_13, row, ROW_KPI, ROW_MAIN, seg, SEG_GROUPS, sort_header, sort_rows, sortable,
    status_badge, table_scroll, table_sort,
)
from .figures import _pr_figure
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
# 성능 해석 문장에 쓰는 짧은 이름 — "있음"의 뜻을 문장 안에서 부를 때.
SCREEN3_SHORT_NAME = {"machine_risk": "고위험일", "part_within_7d": "7일 내 고장 표시",
                      "part_current": "당일 고장 표시"}
CUTOFF_POLICY_TEXT = {
    "validation_f1": "검증 구간 F1 최대",
    "fixed_0.5": "0.5 고정(원 실험이 검증 구간을 쓰지 않음)",
}
MODEL_LABEL_SHORT = {"logistic_regression": "로지스틱 회귀", "random_forest": "랜덤 포레스트",
                     "hist_gradient_boosting": "HGB"}

THRESHOLD_HELP = ("위험 기준선은 고위험일을 정하는 학습 라벨의 기준이다. 그날 고장 표시된 부품의 중요도 합"
                  "(A=4·B=2·C=1)이 이 값 이상이면 고위험일로 보고, 기준마다 따로 학습한 모델의 결과를 보여 준다. "
                  "필터가 아니라 정답의 정의를 바꾸는 것이라 아래 지표가 모두 바뀐다. "
                  "기계 고위험일 판별 과제에만 적용된다.")

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


# 영향 변수 이름을 사람이 읽는 한국어로 바꾼다. 센서 파생 변수는 "<센서>_<수식어…>_<변환>"
# 문법이라 센서 이름 + 수식어 + 변환을 · 로 이어 붙인다. 원래 이름은 마우스를 올리면 보인다.
_SENSOR_KO = {
    "temp_bearing_degC": "베어링 온도", "temp_motor_degC": "모터 온도",
    "vibration_h_mms": "수평 진동", "vibration_v_mms": "수직 진동", "oil_pressure_bar": "오일 압력",
    "load_pct": "부하율", "shaft_rpm": "회전 속도", "power_consumption_kw": "소비 전력",
}
_FEATURE_KO = {
    "machine_type": "기계 종류", "plant_code": "공장", "asset_tag": "기계", "part_no": "부품",
    "part_family": "부품군", "criticality": "부품 중요도", "uom": "수량 단위", "unit_cost_inr": "부품 단가",
    "day_of_week": "요일", "is_weekend": "주말 여부", "month_sin": "월(계절 sin)", "month_cos": "월(계절 cos)",
    "breakdown_days_prev_7d": "최근 7일 고장 표시 일수", "breakdown_days_prev_30d": "최근 30일 고장 표시 일수",
    "days_since_last_observed_breakdown": "마지막 고장 표시 후 경과일",
    "has_prior_observed_breakdown": "과거 고장 표시 여부",
    "affected_count_7d": "최근 7일 부품군 고장 표시 일수", "affected_count_30d": "최근 30일 부품군 고장 표시 일수",
    "affected_lag1": "전날 부품군 고장 표시", "days_since_last_affected": "마지막 부품군 고장 표시 후 경과일",
    "severe_count_30d": "최근 30일 심각 고장 일수", "severe_lag1": "전날 심각 고장",
}
_FEATURE_TRANSFORM_KO = {
    "diff1": "전일 대비 변화", "diff7": "7일 전 대비 변화",
    "lag1": "1일 전 값", "lag3": "3일 전 값", "lag7": "7일 전 값",
    "mean7": "7일 평균", "median3": "3일 중앙값", "median7": "7일 중앙값",
    "std7": "7일 표준편차", "mad7": "7일 중앙절대편차", "slope7": "7일 기울기",
    "vs_mean7": "7일 평균 대비 차이", "vs_mean30": "30일 평균 대비 차이",
    "anomaly7": "7일 평균 대비 이탈(표준편차 배수)", "expected": "기대값",
}
_FEATURE_MODIFIER_KO = (("residual", "기대값 대비 잔차"), ("robust_z", "강건 z"))


def feature_label(name: str) -> str:
    """영향 변수 원래 이름 → 한국어 이름. 모르는 이름은 그대로 돌려준다."""
    if name in _FEATURE_KO:
        return _FEATURE_KO[name]
    for sensor, sensor_ko in _SENSOR_KO.items():
        if name == sensor:
            return sensor_ko
        if not name.startswith(sensor + "_"):
            continue
        rest, parts = name[len(sensor) + 1:], [sensor_ko]
        for token, token_ko in _FEATURE_MODIFIER_KO:
            if rest == token or rest.startswith(token + "_"):
                parts.append(token_ko)
                rest = rest[len(token):].lstrip("_")
        if rest:
            parts.append(_FEATURE_TRANSFORM_KO.get(rest, rest))
        return " · ".join(parts)
    return name


def _topic(word: str) -> str:
    """주제 조사 은/는 — 마지막 글자가 한글이고 받침이 있으면 은."""
    last = word[-1] if word else ""
    has_final = "가" <= last <= "힣" and (ord(last) - ord("가")) % 28 != 0
    return f"{word}{'은' if has_final else '는'}"


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


# ── 모델 비교 표 ─────────────────────────────────────────────────────
# (열 라벨, 정렬, 열 id, 값 → 글자). 기본 순서는 기준 A → 기준 B → 모델(비교 파일 순서).
MODEL_COLS = [
    ("모델", "left", "model_label", str),
    ("AP", "right", "average_precision", "{:.3f}".format),
    ("AP ÷ 양성 비율 (배)", "right", "ap_lift", "{:.2f}".format),
    ("ROC-AUC", "right", "roc_auc", "{:.3f}".format),
    ("정밀도", "right", "precision", "{:.3f}".format),
    ("재현율", "right", "recall", "{:.3f}".format),
    ("경보율 (%)", "right", "alert_rate", lambda v: f"{v * 100:.1f}"),
    ("상위 10% 정밀도", "right", "top10_precision", "{:.3f}".format),
    ("정확도", "right", "accuracy", "{:.3f}".format),
    ("항상 없음 예측 정확도", "right", "always_negative_accuracy", "{:.3f}".format),
    ("선택", "left", "selection", str),
]
# 열 최고값을 굵게 하는 열 — 판정 기준과 무관한 순위 지표만. 정밀도·재현율·정확도는 판정 기준에
# 따라 기준 A(전부 경보)가 1등이 되는 등 "최고"가 좋은 모델을 뜻하지 않는다.
MODEL_BEST_COLUMNS = {"average_precision", "ap_lift", "roc_auc", "top10_precision"}
MODEL_TABLE_HELP = ("굵은 숫자는 그 열의 최고값이다(AP·배수·ROC-AUC·상위 10% 정밀도 — 판정 기준과 무관한 순위 지표만). "
                    "파란 행은 위 성능 카드가 보여 주는 모델이다. 열 제목을 누르면 오름차순 → 내림차순 → 기본 순서로 바뀐다.")


def model_table(rows, focus_model, store):
    """화면 ③ 모델 비교표 — 같은 분할·평가 구간의 기준 A·B와 모델들.
    정확도는 KPI가 아니라 이 표에만 두고, 옆에 '항상 없음 예측 정확도'를 붙인다."""
    rows = [{**r, "selection": _selection_text(r)} for r in rows]
    best = {key: max(r[key] for r in rows) for key in MODEL_BEST_COLUMNS}
    col, direction = table_sort(store, "model")
    if col:
        rows = sort_rows(rows, lambda r, c=col: r[c], direction)

    def cell(r, align, key, fmt):
        if align == "left":
            return html.Td(fmt(r[key]), className=BODY_13)
        is_best = key in MODEL_BEST_COLUMNS and r[key] == best[key]
        return html.Td(fmt(r[key]), className=f"{NUM_13} pf-td--num" + (" pf-td--best" if is_best else ""))

    body_rows = [html.Tr([cell(r, align, key, fmt) for _label, align, key, fmt in MODEL_COLS],
                         **{"aria-selected": "true" if r["model"] == focus_model else "false"})
                 for r in rows]
    thead = html.Tr([sort_header(label, "model", key, store, align) for label, align, key, _ in MODEL_COLS])
    return table_scroll(html.Table([html.Thead(thead), html.Tbody(body_rows)], className="pf-table"), "모델 비교 표")


def model_table_block(store, seg_state):
    """정렬 콜백과 screen_3()이 함께 쓰는 모델 비교표."""
    task_key = _screen3_task_key(seg_state)
    threshold = _screen3_threshold(task_key, seg_state)
    return sortable("model", model_table(load_model_comparison(task_key, threshold),
                                         load_focus_model(task_key, threshold), store))


def _performance_sentence(rows, focus_row, task_key):
    """KPI 바로 아래 해석 문장 — 같은 비교 파일의 숫자로만 만든다(손으로 적은 숫자 없음).
    숫자 바로 뒤에는 조사를 붙이지 않는다(을/를·로/으로가 숫자의 끝 발음에 따라 달라진다)."""
    by_model = {r["model"]: r for r in rows}
    name, short = focus_row["model_label"], SCREEN3_SHORT_NAME[task_key]
    rate, ap = focus_row["positive_rate"], focus_row["average_precision"]
    first = (f"{name}의 AP는 {ap:.3f}, 무작위로 골랐을 때 기대되는 AP(양성 비율 {rate:.3f})의 "
             f"{focus_row['ap_lift']:.2f}배입니다.")
    second = (f"판정 기준 {focus_row['cutoff']:.2f}에서 전체의 {focus_row['alert_rate']:.1%}에 경보를 내 "
              f"{short}의 {focus_row['recall']:.1%}를 잡고, 경보 100건 중 실제 {_topic(short)} "
              f"{focus_row['precision'] * 100:.0f}건입니다.")
    sentences = [first, second]
    baseline_b = by_model.get("baseline_b")
    if baseline_b is not None:
        b_name = baseline_b["model_label"].split("·", 1)[-1].strip()
        gap = ap - baseline_b["average_precision"]
        sentences.append(
            f"기준 B({b_name})의 AP는 {baseline_b['average_precision']:.3f}, "
            + (f"이 모델이 {gap:.3f} 높습니다." if gap > 0 else
               f"이 모델보다 {-gap:.3f} 높습니다 — 과거 비율만으로도 이 모델보다 잘 맞힙니다."))
    best = next((r for r in rows if bool(r["selected_by_validation"])), None)
    if best is not None and best["model"] != focus_row["model"]:
        sentences.append(f"검증 AP가 가장 높았던 {best['model_label']}의 시험 AP는 {best['average_precision']:.3f}입니다.")
    return " ".join(sentences)


# ── 혼동행렬 ─────────────────────────────────────────────────────────
# 칸마다 결과 이름·건수·행 비율. 맞게 판정한 칸은 status-good, 놓친 고장은 status-critical,
# 오경보는 status-warning으로 칠하고 칸 이름 글자를 함께 둔다(상태색은 혼자 다니지 않는다).
_CM_CELLS = {"tp": ("맞게 잡음", "good"), "fn": ("놓침", "critical"),
             "fp": ("오경보", "warning"), "tn": ("정상 통과", "good")}


def _confusion_body(confusion, positive_name):
    """혼동행렬 — 행은 실제, 열은 모델 판정. 비율은 같은 행(실제 있음·없음 각각)을 100%로 본 값."""
    positives = confusion["tp"] + confusion["fn"]
    negatives = confusion["fp"] + confusion["tn"]

    def cell(key, row_total):
        count = confusion[key]
        name, variant = _CM_CELLS[key]
        share = count / row_total * 100 if row_total else 0.0
        return html.Div([html.Span(name, className="pf-cm-cell__name label-12"),
                         html.Span(f"{count:,}", className="value-20"),
                         html.Span(f"{share:.1f}%", className="num-13 pf-secondary")],
                        className=f"pf-cm-cell pf-cm-cell--{variant}")

    def total(text, value):
        return html.Div([html.Span(text, className=LABEL_12), html.Span(f"{value:,}", className=NUM_13)],
                        className="pf-cm-total")

    return html.Div([
        html.Div(""), html.Div("판정 있음", className=f"{LABEL_12} pf-cm-head"),
        html.Div("판정 없음", className=f"{LABEL_12} pf-cm-head"), html.Div(""),
        html.Div("실제 있음", className=f"{LABEL_12} pf-cm-side"), cell("tp", positives), cell("fn", positives),
        total("합계", positives),
        html.Div("실제 없음", className=f"{LABEL_12} pf-cm-side"), cell("fp", negatives), cell("tn", negatives),
        total("합계", negatives),
        html.Div(f"있음 = {positive_name} · 비율 = 같은 행(실제) 기준", className="label-12 pf-muted pf-cm-foot"),
    ], className="pf-cm")


# ── 판정 임계값 ───────────────────────────────────────────────────────
def _metric_tile(label, value, key_variant=None):
    """카드 안 작은 값 칸 — 라벨(필요하면 상태 점) + 값. 혼동행렬과 같은 뜻의 칸은 같은 색 점을 단다."""
    head = [html.Span(className=f"pf-legend__key pf-legend__key--dot pf-key--{key_variant}")] if key_variant else []
    return html.Div([html.Span(head + [label], className=f"{LABEL_12} pf-legend__item"),
                     html.Span(value, className="value-20 pf-num")], className="pf-metric")


def _threshold_body(metrics, positive_name):
    """판정 임계값 카드 본문 — 슬라이더 값(저장된 Test 점수에 다시 적용)으로 다시 그린다."""
    caught = metrics["tp"]
    actual = metrics["tp"] + metrics["fn"]
    tiles = [
        _metric_tile("경보율 (%)", f"{metrics['alert_rate'] * 100:.1f}"),
        _metric_tile("정밀도", f"{metrics['precision']:.3f}"),
        _metric_tile("재현율", f"{metrics['recall']:.3f}"),
        _metric_tile("경보 건수", f"{metrics['alerts']:,}"),
        _metric_tile("오경보 건수", f"{metrics['fp']:,}", "warning"),
        _metric_tile("놓친 건수", f"{metrics['fn']:,}", "critical"),
    ]
    return html.Div([
        html.Div([html.Span("현재 임계값", className=LABEL_12),
                  html.Span(f"{metrics['cutoff']:.2f}", className="value-20 pf-num")],
                 style={"display": "flex", "alignItems": "baseline", "gap": "8px"}),
        html.Div(tiles, className="pf-metric-grid"),
        html.Div(f"임계값 {metrics['cutoff']:.2f} 기준으로 {metrics['rows']:,}건 중 {metrics['alerts']:,}건에 경보를 내고, "
                 f"실제 {positive_name} {actual:,}건 중 {caught:,}건을 잡습니다.", className=f"{BODY_13} pf-sentence"),
    ], style={"display": "flex", "flexDirection": "column", "gap": "10px"})


# ── 주요 영향 변수 ────────────────────────────────────────────────────
def _importance_body(fi_rows):
    """영향 변수 순위 목록 — 한 줄에 순위·한국어 이름·값, 그 아래 막대. 음수(섞어도 AP가 오른 변수)는
    막대 길이 0. 원래 변수 이름은 마우스를 올리면 보인다."""
    max_importance = max([r["importance_mean"] for r in fi_rows] + [1e-12])
    items = [
        html.Div([
            html.Div([html.Span(f"{rank}", className="num-13 pf-muted pf-fi__rank"),
                      html.Span(feature_label(r["feature"]), className=BODY_13, title=r["feature"]),
                      html.Span(f"{r['importance_mean']:.3f}", className="num-13 pf-secondary pf-fi__value")],
                     className="pf-fi__head"),
            html.Div(html.Div(className=f"pf-fi-bar pf-fi--{_feature_color_category(r['feature'])}",
                              style={"width": f"{max(r['importance_mean'], 0) / max_importance * 100:.1f}%"}),
                     className="pf-fi__track"),
        ], className="pf-fi")
        for rank, r in enumerate(fi_rows, start=1)
    ]
    footer = legend([("health", "건강 센서"), ("operating", "운전 조건"), ("other", "그 외")])
    return html.Div([html.Div(items, className="pf-fi-list"), html.Div(footer, className="pf-divider-top pf-fi-foot")],
                    style={"display": "flex", "flexDirection": "column", "height": "100%"})


def _fi_card_right(help_text):
    """영향 변수 카드 머리 — "원인이 아니다"는 한 줄은 늘 보이게 두고 자세한 설명은 물음표에 둔다."""
    return html.Div([note("원인이 아니라 판별 기여도"), help_icon(help_text)],
                    style={"display": "flex", "alignItems": "center", "gap": "8px"})


# ── 운영 판단 ─────────────────────────────────────────────────────────
# 판정 규칙은 이 대시보드가 정한 기준이다(현장 검증을 거친 기준이 아니다) — 카드 물음표에 그대로 적는다.
VERDICT_MIN_LIFT, VERDICT_ASSIST_PRECISION, VERDICT_OPERATE = 1.2, 0.5, 0.8
VERDICT_RULES = (f"판정 규칙 — AP ÷ 실제 고장률이 {VERDICT_MIN_LIFT} 미만이면 사용 불가, "
                 f"경고 100건당 실제 고장이 {VERDICT_ASSIST_PRECISION * 100:.0f}건 미만이면 탐색적 사용 가능, "
                 f"{VERDICT_ASSIST_PRECISION * 100:.0f}건 이상이면 점검 보조 사용 가능, "
                 f"{VERDICT_OPERATE * 100:.0f}건 이상이면서 실제 고장 100건당 탐지가 {VERDICT_OPERATE * 100:.0f}건 이상이면 "
                 "운영 검토 가능. 이 규칙은 대시보드가 정한 기준이며 현장 검증을 거친 기준이 아니다.")


def operational_verdict(lift, precision, recall):
    """(판정, 배지 변형, 설명 문장, 판정 근거). 판정 순서는 VERDICT_RULES와 같다."""
    per_alert, per_failure = f"{precision * 100:.1f}건", f"{recall * 100:.1f}건"
    assist, operate = f"{VERDICT_ASSIST_PRECISION * 100:.0f}건", f"{VERDICT_OPERATE * 100:.0f}건"
    if lift < VERDICT_MIN_LIFT:
        return ("사용 불가", "critical",
                "무작위로 고른 것과 거의 같아 판단 근거로 쓸 수 없습니다. 이 부품군은 기존 점검 주기를 따릅니다.",
                f"AP ÷ 실제 고장률 {lift:.2f}배 — 기준 {VERDICT_MIN_LIFT}배 미만")
    if precision >= VERDICT_OPERATE and recall >= VERDICT_OPERATE:
        return ("운영 검토 가능", "good",
                "경고 대부분이 실제 고장이고 고장도 대부분 잡습니다. 현장 검증을 거쳐 운영 판단에 쓸 수 있는지 검토합니다.",
                f"경고 100건당 실제 고장 {per_alert} · 실제 고장 100건당 탐지 {per_failure} — 둘 다 {operate} 이상")
    if precision >= VERDICT_ASSIST_PRECISION:
        detail = ("고장 누락도 적지만 오경보가 남아 있어" if recall >= VERDICT_OPERATE
                  else "놓치는 고장이 있어")
        return ("점검 보조 사용 가능", "warning",
                f"경고의 절반 이상이 실제 고장이지만 {detail} 자동 부품 교체 판단에는 사용할 수 없습니다. "
                "점검 우선순위를 정하는 보조 근거로 사용합니다.",
                f"경고 100건당 실제 고장 {per_alert} — {assist} 이상, 운영 검토 기준(둘 다 {operate} 이상)에는 못 미침")
    if recall >= 0.7:
        detail = "고장 누락은 적지만 오경보가 많아"
    elif recall >= 0.4:
        detail = "고장의 일부를 놓치고 오경보도 많아"
    else:
        detail = "고장을 많이 놓치고 오경보도 많아"
    return ("탐색적 사용 가능", "serious",
            f"{detail} 자동 부품 교체 판단에는 사용할 수 없습니다. 점검 후보를 좁히는 보조 수단으로만 사용합니다.",
            f"AP ÷ 실제 고장률 {lift:.2f}배({VERDICT_MIN_LIFT}배 이상)이지만 경고 100건당 실제 고장 {per_alert} — "
            f"점검 보조 기준 {assist} 미만")


def _operation_body(confusion, average_precision):
    """운영 판단 카드 본문 — 혼동행렬(같은 판정 임계값)과 AP로 계산한다."""
    total = sum(confusion.values())
    positives = confusion["tp"] + confusion["fn"]
    alerts = confusion["tp"] + confusion["fp"]
    rate = positives / total if total else 0.0
    precision = confusion["tp"] / alerts if alerts else 0.0
    recall = confusion["tp"] / positives if positives else 0.0
    lift = average_precision / rate if rate else 0.0
    verdict, variant, sentence, reason = operational_verdict(lift, precision, recall)
    tiles = [
        _metric_tile("실제 고장률 (%)", f"{rate * 100:.1f}"),
        _metric_tile("예측 경고율 (%)", f"{alerts / total * 100 if total else 0:.1f}"),
        _metric_tile("AP ÷ 실제 고장률 (배)", f"{lift:.2f}"),
        _metric_tile("경고 100건당 실제 고장 건수", f"{precision * 100:.1f}"),
        _metric_tile("실제 고장 100건당 탐지 건수", f"{recall * 100:.1f}"),
    ]
    return html.Div([
        html.Div(tiles, className="pf-metric-grid"),
        html.Div([html.Div([html.Span("현재 모델 판정", className=LABEL_12), status_badge(verdict, variant)],
                           style={"display": "flex", "alignItems": "center", "gap": "12px"}),
                  html.P(sentence, className="body-14 pf-sentence", style={"margin": "0"}),
                  html.Div(f"판정 근거 · {reason}", className=f"{NOTE_12} pf-sentence")],
                 className="pf-verdict"),
    ], style={"display": "flex", "flexDirection": "column", "gap": "16px"})


# ── 도구줄 ────────────────────────────────────────────────────────────
def _machine_picker(asset_tag):
    """부품군 진단의 기계 선택 — 화면 ③은 필터바를 숨기므로 여기서 고른다. 고른 기계는 전역 필터의
    기계로 쓰여 다른 화면에도 이어진다(wireframe_app.pick_family_asset)."""
    return html.Div(
        [html.Span("기계", className=f"pf-field__label {LABEL_12}"),
         dcc.Dropdown(id={"type": "family-asset-dd", "index": "screen3"},
                      options=[{"label": a, "value": a} for a in load_asset_list()],
                      value=asset_tag, clearable=False, searchable=False, style={"width": "130px"})],
        className="pf-field", style={"gap": "6px", "flexShrink": "0"},
    )


def _screen3_toolbar(seg_state, task_key, meta_text, machine_picker=None):
    left = [seg("task", "과제", SEG_GROUPS["task"], sel=seg_state.get("task", 0))]
    if machine_picker is not None:
        left.append(machine_picker)
    return html.Div(
        [html.Div([html.Div(left, style={"display": "flex", "alignItems": "center", "gap": "24px"}),
                   seg("threshold", "위험 기준선 (등급가중 고장점수)", SEG_GROUPS["threshold"],
                       sel=seg_state.get("threshold", 0), disabled=task_key != "machine_risk",
                       help_text=THRESHOLD_HELP)],
                  style={"height": "32px", "display": "flex", "alignItems": "center",
                         "justifyContent": "space-between", "gap": "16px"}),
         html.Div(meta_text, className=LABEL_12,
                  style={"height": "20px", "display": "flex", "alignItems": "center", "whiteSpace": "nowrap",
                         "overflow": "hidden", "textOverflow": "ellipsis", "minWidth": "0"})],
        style={"height": "60px", "display": "flex", "flexDirection": "column", "gap": "8px", "flexShrink": "0"},
    )


# ============================================================
# 화면 ③ 모델·예측 — 필터바를 숨겨 본문 예산이 984px이다.
#   과제 ①~③: 도구줄 60 / KPI 96 / 해석 56 / 비교표 276 / 432
#   과제 ④:   도구줄 60 / 460 / 432
# ============================================================

def screen_3(seg_state=None, audience=DEFAULT_AUDIENCE, asset_tag=None, family=None, table_sort=None):
    seg_state = seg_state or DEFAULT_SEG
    task_key = _screen3_task_key(seg_state)
    if task_key == "family":
        return _family_screen(seg_state, asset_tag, family, table_sort)
    threshold = _screen3_threshold(task_key, seg_state)
    grain, what, positive_template = SCREEN3_TASK_INFO[task_key]
    positive_name = positive_template.format(thr=threshold)
    config = load_comparison_config(task_key)
    rows = load_model_comparison(task_key, threshold)
    focus = load_focus_model(task_key, threshold)
    focus_row = next(r for r in rows if r["model"] == focus)

    meta_text = (f"{grain} · {what} · Test {config['test_period']} · {focus_row['test_rows']:,}행 · "
                 f"양성 비율 {focus_row['positive_rate'] * 100:.1f}%")
    toolbar = _screen3_toolbar(seg_state, task_key, meta_text)

    focus_scope = focus_row["model_label"] + (" · 운영" if bool(focus_row["production"]) else "")
    kpi_specs = [("AP", f"{focus_row['average_precision']:.3f}"),
                 ("AP ÷ 양성 비율 (배)", f"{focus_row['ap_lift']:.2f}"),
                 ("경보율 (%)", f"{focus_row['alert_rate'] * 100:.1f}"),
                 ("재현율", f"{focus_row['recall']:.3f}")]
    row_a = row(ROW_KPI, [kpi_value_tile(label, value, w=458, scope=focus_scope) for label, value in kpi_specs])
    insight = html.Section(
        [html.Span("성능 해석", className="label-12 pf-strong pf-insight__label"),
         html.P(_performance_sentence(rows, focus_row, task_key), className=f"{BODY_13} pf-insight__text pf-sentence")],
        className="pf-insight", style={"height": "56px"}, **{"aria-label": "모델 성능 해석"})

    policy = CUTOFF_POLICY_TEXT.get(focus_row["cutoff_policy"], focus_row["cutoff_policy"])
    table_note = f"같은 Test 구간 · 판정 기준 {policy}"
    if any(bool(r["production"]) for r in rows):
        table_note += " · 운영 모델 RF는 팀 결정 — 근거 TODO(사용자 확인)"
    comparison = card("모델 비교", 1880, 276, model_table_block(table_sort, seg_state),
                      right=html.Div([note(table_note), help_icon(MODEL_TABLE_HELP)],
                                     style={"display": "flex", "alignItems": "center", "gap": "8px"}))
    row_b = row(276, [comparison])

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
               right=note(focus_row["model_label"]))

    slider = html.Div(
        [html.Label("임계값 조정 (0 ~ 1)", htmlFor="thr-slider", className=LABEL_12),
         dcc.Slider(id={"type": "thr-slider", "index": "screen3"}, min=0, max=100, step=1, marks=None,
                    value=round(focus_row["cutoff"] * 100))],
        style={"display": "flex", "flexDirection": "column", "gap": "4px"},
    )
    thr_body = html.Div(
        [slider,
         html.Div(_threshold_body(at_cutoff, positive_name), id={"type": "thr-metrics", "index": "screen3"})],
        style={"display": "flex", "flexDirection": "column", "gap": "10px"},
    )
    thr = card("판정 임계값", 458, 432, thr_body,
               right=html.Div([note("가상 실험 · 재학습 없음"),
                               help_icon(f"저장된 Test 점수에 임계값만 다시 적용하는 가상 실험이다. 모델을 다시 학습하지 "
                                         f"않는다. 기본값({focus_row['cutoff']:.2f})은 {policy} 기준으로 정한 값이다. "
                                         "놓친 건수·오경보 건수의 점 색은 혼동행렬 칸의 색과 같다.")],
                              style={"display": "flex", "alignItems": "center", "gap": "8px"}))

    fi_rows = load_comparison_feature_importance(task_key, threshold)
    if fi_rows:
        fi_note = (f"중요도는 원인이 아니라 판별에 기여한 정도다. "
                   f"{MODEL_LABEL_SHORT.get(fi_rows[0]['model'], fi_rows[0]['model'])} · "
                   "검증 구간 permutation importance(그 변수를 섞었을 때 AP가 줄어든 양)")
        if "rf_reproduction_max_abs_diff" in config:
            fi_note += (f" · 같은 설정으로 다시 학습한 RF 기준(저장된 예측과 점수 최대 차이 "
                        f"{config['rf_reproduction_max_abs_diff']:.2f})")
        feat = card(f"주요 영향 변수 상위 {len(fi_rows)}", 458, 432, _importance_body(fi_rows),
                    right=_fi_card_right(fi_note))
    else:
        feat = card("주요 영향 변수", 458, 432,
                    empty_state("데이터 없음", "이 과제의 영향 변수 결과 파일(feature_importance.csv)이 없다"))
    row_c = row(432, [prc, cmx, thr, feat])

    return html.Div([toolbar, row_a, insight, row_b, row_c],
                     style={"display": "flex", "flexDirection": "column", "gap": f"{GUTTER}px"})


# ── 부품군 진단 ───────────────────────────────────────────────────────
FAMILY_COLS = [
    ("부품군", "left", "part_family", str),
    ("모델", "left", "model_label", str),
    ("표본 수", "right", "support", "{:,}".format),
    ("양성률 (%)", "right", "positive_rate", lambda v: f"{v * 100:.1f}"),
    ("정밀도", "right", "precision", "{:.3f}".format),
    ("재현율", "right", "recall", "{:.3f}".format),
    ("AP", "right", "average_precision", "{:.3f}".format),
    ("ROC-AUC", "right", "roc_auc", "{:.3f}".format),
]


def family_table(family_rows, family, store):
    """부품군 9종 표 — 행을 누르면 그 부품군으로 오른쪽·아래 카드가 바뀐다.
    기본 순서는 AP 내림차순(load_asset_family_diagnosis가 이미 정렬)."""
    rows = [{**fr, "model_label": MODEL_LABEL_SHORT.get(fr["model"], fr["model"])} for fr in family_rows]
    col, direction = table_sort(store, "family")
    if col:
        rows = sort_rows(rows, lambda r, c=col: r[c], direction)
    body_rows = [
        html.Tr([html.Td(fmt(r[key]), className=(f"{NUM_13} pf-td--num" if align == "right" else BODY_13))
                 for _label, align, key, fmt in FAMILY_COLS],
                id={"type": "family-row", "index": r["part_family"]}, n_clicks=0,
                style={"cursor": "pointer"},
                **{"aria-selected": "true" if r["part_family"] == family else "false"})
        for r in rows
    ]
    thead = html.Tr([sort_header(label, "family", key, store, align) for label, align, key, _ in FAMILY_COLS])
    return table_scroll(html.Table([html.Thead(thead), html.Tbody(body_rows)], className="pf-table"),
                        "부품군 진단 표")


def _family_selection(asset_tag, family):
    """(기계, 부품군 행들, 선택 부품군) — 알 수 없는 기계·부품군은 첫 값(AP 최고 부품군)으로 바꾼다."""
    assets = load_asset_list()
    asset_tag = asset_tag if asset_tag in assets else assets[0]
    family_rows = load_asset_family_diagnosis(asset_tag)
    valid = [fr["part_family"] for fr in family_rows]
    return asset_tag, family_rows, family if family in valid else valid[0]


def family_table_block(store, asset_tag, family):
    """정렬 콜백과 _family_screen()이 함께 쓰는 부품군 진단 표."""
    asset_tag, family_rows, family = _family_selection(asset_tag, family)
    return sortable("family", family_table(family_rows, family, store))


def _family_screen(seg_state, asset_tag, family, table_sort):
    """화면 ③ ④ 부품군 진단 — 선택 기계의 부품군 9종 표·PR 곡선·혼동행렬·영향 변수·운영 판단."""
    asset_tag, family_rows, family = _family_selection(asset_tag, family)
    selected = next(fr for fr in family_rows if fr["part_family"] == family)
    pr_cm = load_family_pr_curve_and_confusion(asset_tag, family)
    c = pr_cm["confusion"]
    pr_cm = {**pr_cm, "positive_rate": (c["tp"] + c["fn"]) / max(1, sum(c.values()))}
    toolbar = _screen3_toolbar(seg_state, "family", "기계·부품군·일 · 당일 센서로 같은 날 진단 · 기계를 고르면 다른 화면에도 이어진다",
                               machine_picker=_machine_picker(asset_tag))

    mcomp = card("부품군 진단", 774, ROW_MAIN, family_table_block(table_sort, asset_tag, family),
                 right=note("행 = 부품군 · 행을 누르면 오른쪽·아래 카드가 바뀐다"))
    pr_body = html.Div(
        dcc.Graph(id={"type": "family-pr-chart", "index": "screen3"},
                  figure=_pr_figure(pr_cm, "light"),
                  config={"displayModeBar": False, "responsive": True},
                  style={"flex": "1 1 auto", "minHeight": "0", "minWidth": "0"}),
        style={"flex": "1 1 auto", "minHeight": "0", "display": "flex", "flexDirection": "column"})
    prc = card("PR 곡선", 616, ROW_MAIN, pr_body, right=note(f"{asset_tag} · {family}"))
    cmx = card("혼동행렬", 458, ROW_MAIN, _confusion_body(c, f"{family} 고장 표시"),
               right=note(f"판정 임계값 {pr_cm['cutoff']:.3f}"))
    row_b = row(ROW_MAIN, [mcomp, prc, cmx])

    fi_rows = load_family_feature_importance(family)
    feat = card(f"주요 영향 변수 상위 {len(fi_rows)}", 1090, 432, _importance_body(fi_rows),
                right=_fi_card_right("중요도는 고장 원인이 아니라 고장과 함께 변화한 운전 상태를 뜻한다. "
                                     "부품군 자체의 속성이라 기계와 무관하게 같다."))
    operation = card("운영 판단", 774, 432, _operation_body(c, selected["average_precision"]),
                     right=html.Div([note(f"{asset_tag} · {family} · 판정 임계값 {pr_cm['cutoff']:.3f}"),
                                     help_icon(VERDICT_RULES)],
                                    style={"display": "flex", "alignItems": "center", "gap": "8px"}))
    row_c = row(432, [feat, operation])

    return html.Div([toolbar, row_b, row_c],
                     style={"display": "flex", "flexDirection": "column", "gap": f"{GUTTER}px"})
