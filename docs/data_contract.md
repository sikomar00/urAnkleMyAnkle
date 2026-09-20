# 데이터 계약 (outputs/ 폴더 파일 스키마)

**이 문서가 outputs/ 폴더의 실제 csv 파일 구조와 항상 일치해야
합니다.** 컬럼을 바꾸려면 이 문서를 먼저 수정하고 팀에 공유한
다음 코드를 바꾸세요.

> ⚠️ Dash는 `outputs/` 만 읽습니다. `data/processed/` 는 .gitignore로
> 제외되므로, 화면에 필요한 결과는 반드시 `outputs/` 에 저장하세요.

---

## 에너지 트랙

### outputs/energy_daily.csv — 일 단위 집계 (에너지 현황 화면)

| 컬럼 | 타입 | 설명 |
|---|---|---|
| date | date | 일자 |
| region | string | 지역/공장 구분 (필터용) |
| usage_kwh | float | 일 총 사용량 |
| peak_kw | float | 일 최대부하 |
| load_factor | float | 부하율 = 평균부하 / 최대부하 |
| avg_temp | float | 평균기온 (기상 병합 결과) |

### outputs/energy_forecast.csv — 예측 결과

| 컬럼 | 타입 | 설명 |
|---|---|---|
| ds | date | 날짜 |
| y_actual | float | 실제값 (test 기간만, train 기간은 결측) |
| yhat | float | 예측값 |
| yhat_lower | float | 예측구간 하한 (baseline/linreg는 결측 가능) |
| yhat_upper | float | 예측구간 상한 |
| model | string | `baseline` / `linreg` / `prophet` |

### outputs/energy_scores.csv — 모델 성능 비교

| 컬럼 | 타입 | 설명 |
|---|---|---|
| model | string | `baseline` / `linreg` / `prophet` |
| MAE | float | |
| RMSE | float | |
| R2 | float | |
| n_test | int | 평가에 쓴 행 수 |
| horizon_days | int | 예측 지평(일). baseline과 동일해야 공정한 비교 |

> **요건 대응**: "회귀(기준모델 대비 MAE/RMSE/R²)"는 `linreg` vs
> `baseline` 비교, "Prophet 시계열 예측"은 `prophet` 행입니다.
> 셋 다 있어야 요건을 채웁니다.

---

## 변경 이력

| 날짜 | 변경 내용 | 변경자 |
|---|---|---|
| 2026-09-18 | 최초 작성 | |
| 2026-09-18 | threshold·클래스비율 컬럼 추가, energy_daily / kmeans_selection / cluster_profile / classification_report 신설, linreg 추가, 업로드 형식 절 추가 | |
