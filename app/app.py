"""
Dash 앱 진입점. 화면 5개(app/pages/*)를 탭으로 묶어서 실행한다.

실행 (반드시 레포 루트에서):
    python -m app.app

⚠️ `python app/app.py` 로 실행하면 sys.path[0]이 app/ 폴더가 되어
   `from src...` import가 전부 실패합니다. -m 옵션을 쓰면 현재
   작업 디렉터리가 import 경로에 들어갑니다.
"""
from dash import Dash, Input, Output, dcc, html

from app.pages import ai4i, data_lookup, energy_status, forecast, report_summary

# 탭 value -> (표시 이름, 해당 page 모듈)
TABS = {
    "tab-energy": ("에너지 현황", energy_status),
    "tab-forecast": ("예측", forecast),
    "tab-ai4i": ("AI4I 예지보전", ai4i),
    "tab-data": ("데이터 조회", data_lookup),
    "tab-report": ("보고서 요약", report_summary),
}

app = Dash(__name__, suppress_callback_exceptions=True)
server = app.server

app.layout = html.Div([
    html.H1("예지보전·에너지 통합 대시보드"),
    dcc.Tabs(
        id="main-tabs",
        value="tab-energy",
        children=[dcc.Tab(label=name, value=key) for key, (name, _) in TABS.items()],
    ),
    html.Div(id="tab-content"),
])


@app.callback(Output("tab-content", "children"), Input("main-tabs", "value"))
def render_tab(value):
    """선택된 탭의 page 모듈에서 layout을 꺼내 화면에 꽂는다."""
    _, module = TABS[value]
    return module.layout


# 각 page가 register_callbacks(app)을 정의했으면 등록.
# 아직 없는 page는 건너뛰므로, 구현 순서에 상관없이 앱이 뜬다.
for _name, _module in TABS.values():
    _register = getattr(_module, "register_callbacks", None)
    if callable(_register):
        _register(app)


if __name__ == "__main__":
    app.run(debug=True)
