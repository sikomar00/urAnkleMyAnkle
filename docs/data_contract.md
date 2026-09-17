# 데이터 계약 (outputs/ 폴더 파일 스키마)

**이 문서가 outputs/ 폴더의 실제 csv 파일 구조와 항상 일치해야
합니다.** 컬럼을 바꾸려면 이 문서를 먼저 수정하고 팀에 공유한
다음 코드를 바꾸세요.

## outputs/energy_forecast.csv (에너지 예측 결과)

| 컬럼 | 타입 | 설명 |
|---|---|---|
| ds | date | 날짜 |
| y_actual | float | 실제값 (test 기간만, train 기간은 결측 가능) |
| yhat | float | 예측값 |
| yhat_lower | float | 예측구간 하한 |
| yhat_upper | float | 예측구간 상한 |
| model | string | 모델 이름 (baseline / prophet 등) |

## outputs/energy_scores.csv (에너지 모델 성능 비교)

| 컬럼 | 타입 | 설명 |
|---|---|---|
| model | string | 모델 이름 |
| MAE | float | |
| RMSE | float | |
| R2 | float | |

## outputs/ai4i_metrics.csv (AI4I 분류 모델 성능 비교)

| 컬럼 | 타입 | 설명 |
|---|---|---|
| model | string | baseline / tree / rf |
| precision | float | test 세트 기준 |
| recall | float | test 세트 기준 |
| f1 | float | test 세트 기준 |
| tn | int | 혼동행렬 True Negative |
| fp | int | 혼동행렬 False Positive |
| fn | int | 혼동행렬 False Negative |
| tp | int | 혼동행렬 True Positive |

## outputs/ai4i_scored.csv (개별 데이터 단위 결과)

| 컬럼 | 타입 | 설명 |
|---|---|---|
| udi | int | 식별자 (표시용, 모델 입력 아님) |
| type | string | L/M/H |
| air_k | float | Air temperature [K] |
| process_k | float | Process temperature [K] |
| rpm | float | Rotational speed |
| torque | float | Torque [Nm] |
| tool_wear | float | Tool wear [min] |
| failure | int | 실제 Machine failure (0/1) |
| split | string | train / test |
| proba | float | RF 모델의 고장 확률 예측값 |
| cluster | int | K-Means 군집 번호 |
| anomaly | int | 거리 기반 이상치 여부 (0/1) |

## 변경 이력

| 날짜 | 변경 내용 | 변경자 |
|---|---|---|
| | 최초 작성 | |
