"""
화면: AI4I 예지보전
요건: 설비유형 필터, 고장 확률 분류 결과, 혼동행렬 또는 지표

TODO:
- outputs/ai4i_metrics.csv, outputs/ai4i_scored.csv 읽어서 구현
  (split == "test" 행만 사용해서 지표 계산할 것 — 학습 데이터
  성능을 보여주면 안 됨)
"""
from dash import html

layout = html.Div([
    html.H2("AI4I 예지보전 (TODO)"),
])
