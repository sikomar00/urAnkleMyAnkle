"""화면 ② 기계 상세 — 정보 띠, 센서 소형 다중 그래프, 부품 출고 금액."""

from dash import dcc, html

from ..dashboard_data import (
    load_asset_detail_kpis, load_asset_list, load_asset_parts_history, load_asset_sensor_series,
    load_sensor_outlier_bounds, OUTLIER_BASELINE_END, OUTLIER_Z,
)
from .base import (
    BODY_13, card, CODE_12, DEFAULT_AUDIENCE, empty_state, GUTTER, help_icon, hstack, LABEL_12, legend,
    note, NUM_13, row, status_badge,
)
from .figures import (
    _parts_figure, _smult_figure, SCREEN2_ROW_H, sensor_outside_mask, SMULT_AXIS_H, SMULT_BODY_H, SMULT_GAP,
    SMULT_PANEL_H, SMULT_SENSORS,
)


SENSOR_HELP = (f"센서마다 이 기계의 학습 구간({OUTLIER_BASELINE_END:%Y-%m-%d} 이전) 정상일(고장 표시가 없는 날) "
               f"값으로 정상 범위를 정한다. 점선은 중앙값 ± {OUTLIER_Z} × 1.4826 × MAD(강건 z {OUTLIER_Z})이고, "
               "빨간 점은 그 범위를 벗어난 날이다. 붉은 세로 띠는 고장점수 12 이상인 고위험일이다. "
               "설비 이상 실험(asset_anomaly_features)의 강건 z 점수와 같은 기준이다.")


# ============================================================
# 화면 ② 기계 상세 — 정보 띠 88 / 824
# ============================================================

def screen_2(seg_state=None, audience=DEFAULT_AUDIENCE, asset_tag=None, start=None, end=None):
    def value_box(content, w, h=26, cls=BODY_13):
        """strip 전용 값 칸 — screen_2() 안에서만 쓰인다."""
        return html.Div(
            html.Span(content, className=cls,
                      style={"whiteSpace": "nowrap", "overflow": "hidden", "textOverflow": "ellipsis"}),
            style={"width": f"{w}px", "height": f"{h}px",
                   "display": "flex", "alignItems": "center", "overflow": "hidden"},
        )

    def metric(label, w, value, cls=NUM_13):
        return html.Div([html.Span(label, className=LABEL_12, style={"whiteSpace": "nowrap"}),
                          value_box(value, w, cls=cls)],
                         style={"width": f"{w}px", "display": "flex", "flexDirection": "column", "gap": "4px"})

    def nav_btn(label, index):
        """공용 btn()은 id 타입이 항상 "ghost-btn"이라 전역 echo_action이
        같이 반응해 '미구현' 문구를 띄운다. 여기서는 실제 기계 전환 콜백만
        반응하도록 다른 id 타입을 쓴다."""
        return html.Button(label, id={"type": "machine-nav-btn", "index": index},
                            n_clicks=0, className="pf-btn label-12")

    assets = load_asset_list()
    asset_tag = asset_tag if asset_tag in assets else assets[0]
    detail = load_asset_detail_kpis(asset_tag, start, end)

    strip = html.Section(
        [nav_btn("‹ 이전 기계", "prev"),
         html.Div([value_box(detail["asset_tag"], 220, cls=f"{CODE_12} title-14"),
                   value_box(f"{detail['machine_type']} · {detail['plant_code']}", 220, h=22,
                             cls=f"{BODY_13} pf-secondary")],
                  style={"display": "flex", "flexDirection": "column", "gap": "4px"}),
         nav_btn("다음 기계 ›", "next"),
         html.Div(style={"flexGrow": "1"}),
         metric("현재 등급", 120, status_badge(detail["current_grade"]), cls=""),
         metric("위험도", 120, f"{detail['risk_score']:,.0f}"),
         metric("최근 고장 표시일", 140, detail["last_failure_date"] or "—", cls=CODE_12),
         metric("고장 표시 일수 (일)", 140, f"{detail['failure_days_count']:,}"),
         metric("평균 소비 전력 (kW)", 140, f"{detail['avg_power_kw']:,.2f}")],
        className="pf-card",
        style={"height": "88px", "flexDirection": "row",
               "alignItems": "center", "gap": "16px"},
    )

    # 라벨 열은 HTML로 두고 오른쪽 차트만 Plotly로 그린다. 패널 높이·간격이
    # _smult_figure()의 subplot 치수(SMULT_*)와 같아야 1:1로 정렬된다.
    series = load_asset_sensor_series(asset_tag, start, end)
    bounds = load_sensor_outlier_bounds(asset_tag)

    def sensor_label(label, column):
        outside = int(sensor_outside_mask(series, column, bounds[column]).sum())
        count = (html.Span([html.Span(className="pf-legend__key pf-legend__key--dot pf-key--critical"),
                            f"경계 밖 {outside}일"], className=f"{LABEL_12} pf-legend__item")
                 if outside else html.Span("경계 밖 없음", className=f"{LABEL_12} pf-muted"))
        return html.Div([html.Span(label, className=f"{LABEL_12} pf-strong"), count],
                        style={"height": f"{SMULT_PANEL_H}px", "flexShrink": "0", "display": "flex",
                               "flexDirection": "column", "justifyContent": "center", "gap": "4px"})

    sensor_labels = html.Div(
        [sensor_label(label, column) for label, column in SMULT_SENSORS]
        + [html.Div(style={"height": f"{SMULT_AXIS_H}px", "flexShrink": "0"})],
        style={"width": "160px", "flexShrink": "0", "display": "flex", "flexDirection": "column",
               "gap": f"{SMULT_GAP}px"},
    )
    sm_body = hstack(
        [sensor_labels,
         dcc.Graph(id={"type": "smult-chart", "index": "screen2"},
                   figure=_smult_figure(series, "light", bounds),
                   config={"displayModeBar": False, "responsive": True},
                   style={"flex": "1 1 auto", "minHeight": "0", "minWidth": "0"})],
        8, style={"height": f"{SMULT_BODY_H}px"},
    )
    smult = card("센서 8종 종합 지표", 1248, SCREEN2_ROW_H, sm_body,
                 right=html.Div([legend([("line", "측정값"), ("dash", "정상 범위 경계"),
                                         ("critical", "경계 밖"), ("band", "고위험일")]),
                                 help_icon(SENSOR_HELP)],
                                style={"display": "flex", "alignItems": "center", "gap": "12px"}))

    # "이상 점수 추이"·"군집 위치"·"고장 직전 센서 변화"·"동종 기계 대비" 4개
    # 카드를 삭제한다. K-Means 결과(군집·이상 점수)는 화면④ 데이터 조회의
    # 그레인 토글로 옮겨 원자료 조회로는 남길 계획이었으나, 그 산출물
    # (dataVerification/outputs/kmeans_assignments.csv)이 리포지토리 어디에도
    # 없다 — 대시보드는 모델을 재실행하지 않는다는 공통 제약과 충돌해 새로
    # 만들지 않았다(별도 오프라인 클러스터링 스크립트가 먼저 필요).
    # "부품 출고 이력"만 남아 오른쪽 칸(616 × 520)을 그대로 채운다.
    parts_history = load_asset_parts_history(asset_tag, start, end)
    if parts_history.empty:
        parts_body = empty_state("기간 내 부품 출고 없음", "선택한 기간에 이 기계의 부품 출고 금액이 0이다")
    else:
        parts_body = dcc.Graph(id={"type": "parts-chart", "index": "screen2"},
                               figure=_parts_figure(parts_history, "light"),
                               config={"displayModeBar": False, "responsive": True},
                               style={"flex": "1 1 auto", "minHeight": "0", "minWidth": "0"})
    parts = card("부품 출고 금액 상위 10 (INR)", 616, SCREEN2_ROW_H, parts_body,
                 right=note("선택 기간 합계"))
    row_b = row(SCREEN2_ROW_H, [smult, parts])

    return html.Div([strip, row_b],
                     style={"display": "flex", "flexDirection": "column", "gap": f"{GUTTER}px"})
