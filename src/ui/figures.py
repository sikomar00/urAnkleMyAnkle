"""대시보드 Plotly 그림 — 모두 src/theme.py의 plantfloor_{light,dark} 템플릿과 C()를 쓴다."""

from datetime import timedelta

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..theme import C
from .base import _figure_template, _theme


def _spark_figure(values, theme):
    """화면 ① "기계 상태" 타일의 미니 스파크라인 — series-1 2px, 축·격자·범례 없음."""
    fig = go.Figure(go.Scatter(y=values, mode="lines", hoverinfo="skip",
                                line=dict(color=C("series-1", _theme(theme)), width=2)))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(template=_figure_template(theme), height=32, margin=dict(l=0, r=0, t=2, b=2))
    return fig


# 히트맵 순차 파랑. 0은 색이 아니라 표면(surface-sunken)이다(DESIGN.md §2.1).
_SEQ_BLUE = ["seq-blue-100", "seq-blue-200", "seq-blue-300", "seq-blue-400",
             "seq-blue-500", "seq-blue-600", "seq-blue-700"]


def _heatmap_figure(heat, coverage, theme):
    """화면 ① "고장 표시 히트맵" — 자산 × 기간(일·주·월·년) 그레인, 셀 = 그 기간에
    고장 표시된 부품-일 행 수. period는 이미 오름차순 CSV 순서라 그대로 pivot한다.
    coverage(load_period_coverage)로 관측 일수가 달력상 일수보다 적은 칸에 '부분'을 붙인다."""
    theme = _theme(theme)
    pivot = heat.pivot(index="asset_tag", columns="period", values="failed_part_count")
    periods = pivot.columns.tolist()
    partial = [p for p in periods if coverage.get(p, (0, 0))[0] < coverage.get(p, (0, 0))[1]]
    # 칸이 많아지면(일 단위 3년 = 1,095칸) 2px 간격이 칸보다 넓어져 격자가 사라진다.
    xgap = 2 if len(periods) <= 120 else 0
    step = max(1, len(periods) // 6)
    tickvals = [p for p in periods if p in set(periods[::step]) | set(partial)]
    observed_days = np.tile([coverage.get(p, (0, 0))[0] for p in periods], (len(pivot.index), 1))
    zmax = max(int(pivot.to_numpy().max()), 1)
    first = 1 / zmax  # 값 1의 위치 — 0만 표면색, 1 이상은 seq-blue-100부터
    colorscale = ([[0, C("surface-sunken", theme)], [first * 0.999, C("surface-sunken", theme)]]
                  + [[first + (1 - first) * i / (len(_SEQ_BLUE) - 1), C(name, theme)]
                     for i, name in enumerate(_SEQ_BLUE)])
    fig = go.Figure(go.Heatmap(
        z=pivot.to_numpy(), x=periods, y=pivot.index.tolist(), customdata=observed_days,
        zmin=0, zmax=zmax, colorscale=colorscale, xgap=xgap, ygap=2,
        hovertemplate="%{y} · %{x}<br>%{z}건 · 관측 %{customdata}일<extra></extra>",
        colorbar=dict(title=dict(text="건수", font=dict(size=10)), tickfont=dict(size=10)),
    ))
    fig.update_xaxes(type="category", tickmode="array", tickvals=tickvals,
                     ticktext=[f"{p} 부분" if p in partial else p for p in tickvals],
                     tickfont=dict(size=10), showgrid=False)
    fig.update_yaxes(tickfont=dict(size=11), showgrid=False, autorange="reversed")
    fig.update_layout(template=_figure_template(theme), height=270, margin=dict(l=8, r=8, t=8, b=8),
                      hovermode="closest")
    return fig


def _trend_figure(df, theme):
    """화면 ⑤ "고장·위험 추세" 카드용 라인 차트(참조 화면 전용)."""
    theme = _theme(theme)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=df["transaction_date"], y=df["failure_days"],
                              name="고장 표시 건수", mode="lines",
                              line=dict(color=C("series-1", theme), width=2)),
                  secondary_y=False)
    fig.add_trace(go.Scatter(x=df["transaction_date"], y=df["high_risk_pct"],
                              name="위험 기준선 초과 비율 (%)", mode="lines",
                              line=dict(color=C("series-2", theme), width=2)),
                  secondary_y=True)
    fig.update_layout(template=_figure_template(theme), margin=dict(l=48, r=48, t=8, b=32),
                      legend=dict(orientation="h", y=1.14, x=0), showlegend=True)
    fig.update_yaxes(title_text="건수", secondary_y=False)
    fig.update_yaxes(title_text="%", showgrid=False, secondary_y=True)
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
# 왼쪽 라벨 열과 차트 내부 패널이 같은 치수를 써야 1:1로 정렬되므로 상수로 묶어 둔다.
# 화면 ② 아래 행 높이 = 본문 928 - 기계 정보 줄 88 - 거터 16. 카드 본문(824 - 70)에
# 패널 78×8 + 간격 12×7 + 공유 x축 40 = 748이 들어간다. 간격 한가운데에 센서 사이 구분선을 긋는다
# (screen2 — 간격이 좁으면 위 패널의 아래 경계 점선과 아래 패널의 위 경계 점선이 붙어 보인다).
SCREEN2_ROW_H = 824
SMULT_PANEL_H, SMULT_GAP, SMULT_AXIS_H = 78, 12, 40
# 패널 위아래 여백(값 범위의 비율) — 경계 점선과 경계 밖 점이 패널 가장자리에 붙거나 잘리지 않게 한다.
SMULT_Y_PAD = 0.12
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
                 fillcolor=color, line_width=0, layer="below")
            for a, b in _true_runs(flags)]


def sensor_outside_mask(df, column, bound):
    """센서 값이 정상 범위(bound["lower"]~bound["upper"]) 밖인 날."""
    return (df[column] < bound["lower"]) | (df[column] > bound["upper"])


def _smult_figure(df, theme, bounds=None):
    """화면 ② "센서 8종 종합 지표" — 모든 패널 series-1 단색(축·격자·범례 없음,
    맨 아래 공유 x축만) + 위험 기준선 초과일 세로 밴드(chart-band-critical).
    밴드는 yref="paper"라 8개 패널과 그 사이 간격까지 하나로 관통한다.
    bounds(load_sensor_outlier_bounds)를 주면 센서마다 정상 범위 경계를 점선(chart-muted)으로 긋고
    범위 밖 날을 status-critical 점으로 찍는다(DESIGN §8 — 점선은 임계선, 점 둘레 surface-card 2px 링).
    경계선은 trace로 그려 y축 범위가 경계까지 넓어지게 한다(shape는 자동 범위에 들어가지 않는다)."""
    theme = _theme(theme)
    fig = make_subplots(rows=len(SMULT_SENSORS), cols=1, shared_xaxes=True,
                        vertical_spacing=SMULT_GAP / SMULT_PLOT_H)
    dates = df["transaction_date"]
    for r, (label, column) in enumerate(SMULT_SENSORS, start=1):
        fig.add_trace(go.Scatter(x=dates, y=df[column], mode="lines",
                                  line=dict(color=C("series-1", theme), width=1.5),
                                  showlegend=False, hoverinfo="skip"),
                      row=r, col=1)
        if not bounds or df.empty:
            continue
        bound = bounds[column]
        low = min(df[column].min(), bound["lower"])
        high = max(df[column].max(), bound["upper"])
        pad = (high - low) * SMULT_Y_PAD or 1
        fig.update_yaxes(range=[low - pad, high + pad], row=r, col=1)
        for level in (bound["lower"], bound["upper"]):
            fig.add_trace(go.Scatter(x=[dates.iloc[0], dates.iloc[-1]], y=[level, level], mode="lines",
                                      line=dict(color=C("chart-muted", theme), width=1, dash="dash"),
                                      showlegend=False, hoverinfo="skip"),
                          row=r, col=1)
        outside = df[sensor_outside_mask(df, column, bound)]
        if not outside.empty:
            fig.add_trace(go.Scatter(
                x=outside["transaction_date"], y=outside[column], mode="markers", showlegend=False,
                marker=dict(color=C("status-critical", theme), size=7,
                            line=dict(color=C("surface-card", theme), width=2)),
                hovertemplate=(f"%{{x|%Y-%m-%d}} · {label} %{{y:.2f}}<br>"
                               f"정상 범위 {bound['lower']:.2f} ~ {bound['upper']:.2f}<extra></extra>")),
                row=r, col=1)
    # 밴드는 shapes로 한 번에 넘긴다 — 구간마다 add_vrect()를 호출하면 호출마다
    # figure 전체를 다시 검증해 2분이 넘게 걸린다(일괄 할당은 30ms 수준).
    fig.update_layout(
        template=_figure_template(theme), height=SMULT_BODY_H, margin=dict(l=0, r=0, t=0, b=SMULT_AXIS_H),
        shapes=_band_shapes(df["transaction_date"], df["is_high_risk_day"], C("chart-band-critical", theme)),
        hovermode="closest",
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_xaxes(visible=True, showgrid=False, tickformat="%Y-%m-%d", row=len(SMULT_SENSORS), col=1)
    return fig


def _parts_figure(df, theme):
    """화면 ② "부품 출고 이력" — 부품별 출고 금액 상위 10개 가로 막대(series-1 단일 계열).
    금액 내림차순이 위로 오도록 autorange="reversed"."""
    theme = _theme(theme)
    labels = [f"{no}<br>{desc}" for no, desc in zip(df["part_no"], df["part_description"])]
    fig = go.Figure(go.Bar(
        x=df["total_issue_value_inr"], y=labels, orientation="h", width=0.4,
        marker_color=C("series-1", theme),
        text=[f"{v:,.0f}" for v in df["total_issue_value_inr"]],
        textposition="outside", textfont=dict(size=11),
        cliponaxis=False,
        customdata=df["part_description"],
        hovertemplate="%{customdata}<br>%{text} INR<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed", showgrid=False, tickfont=dict(size=11))
    fig.update_xaxes(visible=False, range=[0, df["total_issue_value_inr"].max() * 1.35])
    fig.update_layout(template=_figure_template(theme), height=SCREEN2_ROW_H - 76,
                      margin=dict(l=0, r=0, t=0, b=0), hovermode="closest")
    return fig


def _pr_figure(pr, theme, height=392):
    """화면 ③ PR 곡선(series-1 단일 계열) + Test 양성 비율 수평선.
    양성 비율선은 아무 정보 없이 무작위로 경보를 낼 때의 정밀도다 — 곡선이 이 선 위에 있어야 한다."""
    theme = _theme(theme)
    fig = go.Figure(go.Scatter(
        x=pr["recall_curve"], y=pr["precision_curve"], mode="lines",
        line=dict(color=C("series-1", theme), width=2),
        hovertemplate="재현율 %{x:.2f}<br>정밀도 %{y:.2f}<extra></extra>",
    ))
    rate = pr.get("positive_rate")
    if rate is not None:
        fig.add_hline(y=rate, line=dict(color=C("chart-muted", theme), width=1, dash="dash"),
                      annotation_text=f"양성 비율 {rate * 100:.1f}%", annotation_position="top right",
                      annotation_font_size=10)
    fig.update_yaxes(title="정밀도", range=[0, 1.02], tickfont=dict(size=11))
    fig.update_xaxes(title="재현율", range=[0, 1.02], tickfont=dict(size=11))
    fig.update_layout(template=_figure_template(theme), height=height, margin=dict(l=48, r=16, t=8, b=40),
                      hovermode="closest")
    return fig
