[app/pages] 폴더 설명
---------------------------------
용도: Dash 화면(탭/페이지) 5개를 파일별로 분리해둔 곳입니다.

파일 ↔ 요건 화면 매칭:
- energy_status.py  : 에너지 현황 (공장·기간 필터, KPI, 전력추이)
- forecast.py        : 예측 (실제값·예측값, MAE·RMSE, 기간 조회)
- ai4i.py             : AI4I 예지보전 (설비유형 필터, 고장확률,
                        혼동행렬/지표)
- data_lookup.py      : 데이터 조회 (CSV 업로드·검증, DataTable,
                        필터, 다운로드) — src/validate.py 사용
- report_summary.py   : 보고서 요약 (핵심 결론, 한계, 개선제안)

각 파일은 해당 화면의 레이아웃(layout)과 콜백(callback) 함수를
담습니다. app.py에서 이 파일들을 불러와 하나의 앱으로 합칩니다.
