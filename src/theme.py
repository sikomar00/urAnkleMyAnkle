"""Plotly 시각 양식 — DESIGN.md 8절.

Plotly는 CSS 변수를 읽지 못하므로 색을 Python 쪽에 이중 관리해야 한다. 그 이중
관리가 어긋나는 것이 이 시스템에서 두 번째로 흔한 회귀이므로, 값을 손으로 적지 않고
design/tokens.json 을 그대로 읽는다 — assets/00-tokens.css 와 같은 정본이다.

import 시 pio.templates 에 plantfloor_light / plantfloor_dark 를 등록한다. 모든
figure 가 이 템플릿을 쓰게 하고, 차트마다 layout 을 따로 쓰지 않는다.
테마 전환은 콜백에서 fig.update_layout(template=f"plantfloor_{theme}") 로 갈아끼운다.
"""
import json
from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio

_TOKENS_PATH = Path(__file__).resolve().parents[1] / "design/tokens.json"

_TOK = {t["name"]: t["value"] for t in
        json.loads(_TOKENS_PATH.read_text(encoding="utf-8"))["color"]["tokens"]}


def C(name: str, theme: str = "light") -> str:
    return _TOK[name][theme]


def build_template(theme: str) -> go.layout.Template:
    return go.layout.Template(layout=dict(
        paper_bgcolor=C("surface-card", theme),
        plot_bgcolor=C("surface-card", theme),
        colorway=[C(f"series-{i}", theme) for i in range(1, 9)],
        font=dict(family="Pretendard, system-ui, 'Malgun Gothic', 'Apple SD Gothic Neo', sans-serif",
                  size=11, color=C("ink-muted", theme)),
        xaxis=dict(showgrid=False, zerolinecolor=C("chart-axis", theme),
                   linecolor=C("chart-axis", theme), ticks="outside",
                   tickcolor=C("chart-muted", theme)),
        yaxis=dict(showgrid=True, gridcolor=C("chart-grid", theme), gridwidth=1,
                   zerolinecolor=C("chart-axis", theme)),
        margin=dict(l=44, r=16, t=8, b=32),
        hovermode="x unified",
        showlegend=False,
    ))


for _t in ("light", "dark"):
    pio.templates[f"plantfloor_{_t}"] = build_template(_t)
