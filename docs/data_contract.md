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

## AI4I 트랙

### outputs/ai4i_metrics.csv — 분류 성능 비교 + 임계값 변화표

| 컬럼 | 타입 | 설명 |
|---|---|---|
| model | string | `baseline` / `tree` / `rf` |
| threshold | float | 판정 임계값 (0.5, 0.3, 0.2 …) |
| precision | float | test 세트 기준 |
| recall | float | test 세트 기준 |
| f1 | float | test 세트 기준 |
| tn / fp / fn / tp | int | 혼동행렬 |
| n_train | int | 학습 행 수 |
| n_test | int | 평가 행 수 |
| pos_rate_train | float | 학습셋 고장 비율 |
| pos_rate_test | float | 평가셋 고장 비율 (stratify 검증용) |

> `model` + `threshold` 조합이 행의 식별자입니다. 임계값을 바꿔
> 여러 행을 넣을 때 `model` 만으로는 중복이 되어 Dash 필터가 꼬입니다.

### outputs/ai4i_scored.csv — 행 단위 결과

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

### outputs/ai4i_kmeans_selection.csv — k 선택 근거

| 컬럼 | 타입 | 설명 |
|---|---|---|
| k | int | 군집 개수 |
| inertia | float | 관성(SSE). 엘보우 판단용 |
| silhouette | float | 실루엣 계수 |

> **발표 대비**: "k를 왜 이 값으로 정했나"는 비지도 파트에서
> 가장 높은 확률로 들어오는 질문입니다. 이 파일이 답변 근거입니다.

### outputs/ai4i_cluster_profile.csv — 군집별 특징

| 컬럼 | 타입 | 설명 |
|---|---|---|
| cluster | int | 군집 번호 |
| n | int | 소속 행 수 |
| failure_rate | float | 군집 내 실제 고장 비율 |
| dist_mean | float | 중심까지 평균 거리 |
| air_k_mean / process_k_mean / rpm_mean / torque_mean / tool_wear_mean | float | 군집 중심 해석용 |

> **요건 대응**: "정상/이상 군집의 특징과 한계 해석" 서술의 근거표.

### outputs/ai4i_classification_report.txt

sklearn `classification_report` 출력을 그대로 저장. 요건 명시 항목.

---

## 업로드 CSV 형식 (데이터 조회 화면)

`src/validate.py` 의 `REQUIRED_COLUMNS` 는 **개명 후** 컬럼명
(`air_k`, `process_k` …)을 기준으로 합니다. AI4I 원본 CSV의 컬럼명은
`Air temperature [K]` 형태라서 그대로 올리면 검증에 실패합니다.

**팀 결정 (TODO: 택 1 하고 나머지 줄 삭제)**
- [ ] **A. 관대하게** — `validate_ai4i_csv` 가 원본 컬럼명도 받아서
      내부에서 rename. 심사자가 원본 파일을 그냥 올려도 동작.
- [ ] **B. 엄격하게** — `outputs/sample_upload.csv` 를 제공하고
      화면에 다운로드 버튼 + "이 형식으로 올리세요" 안내.

컬럼 매핑표는 `src/features.py` 의 `COLUMN_RENAME_MAP` 에 있습니다.

---

## 변경 이력

| 날짜 | 변경 내용 | 변경자 |
|---|---|---|
| 2026-09-18 | 최초 작성 | |
| 2026-09-18 | threshold·클래스비율 컬럼 추가, energy_daily / kmeans_selection / cluster_profile / classification_report 신설, linreg 추가, 업로드 형식 절 추가 | |
