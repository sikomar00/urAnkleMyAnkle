"""
Dash 앱 진입점. 화면 5개(app/pages/*)를 탭으로 묶어서 실행한다.

실행:
    python app/app.py
"""
import dash
from dash import Dash, dcc, html

# TODO: app/pages/의 각 모듈에서 layout, register_callbacks를 import

app = Dash(__name__)
server = app.server

app.layout = html.Div([
    html.H1("예지보전·에너지 통합 대시보드"),
    dcc.Tabs(id="main-tabs", value="tab-energy", children=[
        dcc.Tab(label="에너지 현황", value="tab-energy"),
        dcc.Tab(label="예측", value="tab-forecast"),
        dcc.Tab(label="AI4I 예지보전", value="tab-ai4i"),
        dcc.Tab(label="데이터 조회", value="tab-data"),
        dcc.Tab(label="보고서 요약", value="tab-report"),
    ]),
    html.Div(id="tab-content"),
])

# TODO: main-tabs value에 따라 tab-content를 각 page의 layout으로
# 교체하는 콜백 작성

if __name__ == "__main__":
    app.run(debug=True)
