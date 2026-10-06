"""
화면: 데이터 조회
요건: CSV 업로드·검증, DataTable, 필터, 결과 다운로드

TODO:
- dcc.Upload로 CSV 업로드 받기
- src.validate.validate_ai4i_csv로 검증
- dash_table.DataTable로 표시 (필터 기능 켜기)
- dcc.Download + dcc.send_string으로 결과 다운로드
  (한글 깨짐 방지를 위해 "\ufeff" + csv 문자열로 BOM 붙이기)
"""
from dash import html

layout = html.Div([
    html.H2("데이터 조회 (TODO)"),
])
