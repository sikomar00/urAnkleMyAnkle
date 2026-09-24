# 산업 장비 점수·4단계 위험도 모델 설계

작성일: 2026-09-24

## 1. 목적

현재 장비 당일 모델은 장비·날짜별 `failure_points`를 12, 13, 14 등의 기준으로
각각 이진화해 별도 분류 모델을 학습한다. 이 방식은 0점과 11점을 같은 정상으로,
12점과 24점을 같은 위험으로 취급해 점수의 심각도 정보를 잃는다.

이번 변경의 목적은 장비·날짜 단위 센서와 장비·날짜 단위 정답을 일치시키면서 다음
두 과제를 병렬로 평가하는 것이다.

1. 센서와 과거 정보로 `failure_points` 자체를 예측하는 회귀 과제
2. `failure_points`를 네 단계로 변환해 상태를 분류하는 다중분류 과제

기존 `target_ge_12`, `target_ge_13`, `target_ge_14` 이진 모델은 기준선 비교를 위해
삭제하거나 변경하지 않는다.

## 2. 중요한 해석 제한

원본 데이터에는 실제 기계 정지, 정지시간 또는 생산 손실 여부가 없다. 따라서
`failure_points`와 네 단계 상태는 부품별 `breakdown_flag`를 집계한 대리 지표이며,
실제 기계 고장을 확정하는 값으로 표현하지 않는다.

상태 이름은 다음 영문 값을 저장하고 사용자 문서에서는 한국어로 함께 설명한다.

| 저장값 | 한국어 표시 | 의미 |
|---|---|---|
| `normal` | 정상 | 고장 표시 부품이 없는 0점 |
| `caution` | 주의 | 1점 이상 5점 이하 |
| `risk` | 위험 | 6점 이상 11점 이하 |
| `high_risk` | 고위험 | 12점 이상 |

`high_risk`를 실제 기계 고장 또는 생산 정지로 부르지 않는다.

## 3. 등급 경계

초기 실험의 공통 경계는 다음과 같다.

\[
L(S)=
\begin{cases}
\mathrm{normal},&S=0,\\
\mathrm{caution},&1\le S\le5,\\
\mathrm{risk},&6\le S\le11,\\
\mathrm{high\_risk},&S\ge12.
\end{cases}
\]

이 경계는 다음 이유로 선택한다.

1. 0점은 고장 표시 부품이 없는 유일하게 명확한 정상 상태다.
2. 1~5점은 소수의 낮은 중요도 고장 또는 A등급 한 개 고장을 포함한다.
3. 6~11점은 기존 12점 고위험 기준에 접근하는 중간 상태다.
4. 12점 이상은 기존 이진 모델과 직접 비교할 수 있다.
5. 전체 데이터에서 각 단계가 학습 가능한 수의 표본을 가진다.

등급 경계는 첫 실험에서 고정한다. 기계별 운영 임계값은 이 변경의 범위에 포함하지
않고 후속 단계에서 생산 영향과 오경보 비용을 근거로 설계한다.

## 4. 데이터 단위와 Feature

한 행은 `transaction_date + machine_type + asset_tag`로 식별되는 장비·날짜다.
기존 `build_asset_daily()`가 만든 장비 일별 데이터를 재사용한다.

회귀와 4단계 분류는 동일한 Feature를 사용한다.

- `machine_type`
- `asset_tag`
- 센서 8개
- `day_of_week`
- `month_sin`
- `month_cos`

센서 8개는 다음과 같다.

- `temp_bearing_degC`
- `temp_motor_degC`
- `vibration_h_mms`
- `vibration_v_mms`
- `oil_pressure_bar`
- `load_pct`
- `shaft_rpm`
- `power_consumption_kw`

당일 1차 실험에서는 미래 정보, `breakdown_flag`, `failure_points`, 등급 Target을
Feature로 사용하지 않는다. 기존 날짜 분할을 그대로 사용한다.

- 학습: `label_end_date < 2024-01-01`
- 검증: `transaction_date >= 2024-01-01`이고 `label_end_date < 2024-07-01`
- 테스트: `transaction_date >= 2024-07-01`

## 5. 준비 과제

`src/asset_features.py`에 다음 공개 함수를 추가한다.

### `add_asset_severity(frame)`

`failure_points`를 검증하고 `severity_level`과 순서형 숫자
`severity_code`를 추가한다.

| `severity_level` | `severity_code` |
|---|---:|
| `normal` | 0 |
| `caution` | 1 |
| `risk` | 2 |
| `high_risk` | 3 |

음수, 결측 또는 숫자가 아닌 `failure_points`가 있으면 실패한다.

### `prepare_asset_score_current(frame)`

`failure_points`를 회귀 Target으로 갖는 장비 당일 과제를 준비한다.

### `prepare_asset_severity_current(frame)`

`severity_level`을 다중분류 Target으로 갖는 장비 당일 과제를 준비한다.

기존 분류용 `PreparedTask` 계약을 무리하게 확장하지 않고 회귀용
`PreparedRegressionTask`를 `src/industrial_data.py`에 별도로 둔다.

## 6. 모델

### 6.1 점수 회귀

기준 모델과 학습 모델은 다음과 같다.

1. 학습 구간 평균을 예측하는 `DummyRegressor`
2. `RandomForestRegressor`
3. `HistGradientBoostingRegressor`

검증 구간 MAE가 가장 낮은 학습 모델을 선택한다. Target은 0 이상의 점수이므로
저장 전 예측값은 0 이상으로 제한한다. 학습 과정에서 테스트 구간을 모델 선택에
사용하지 않는다.

### 6.2 4단계 분류

기존 전처리 계약과 다음 모델을 재사용한다.

1. `DummyClassifier(strategy="prior")`
2. `LogisticRegression(class_weight="balanced")`
3. `RandomForestClassifier(class_weight="balanced")`
4. `HistGradientBoostingClassifier`

검증 구간 Macro F1이 가장 높은 학습 모델을 선택한다. 다중분류에서는 이진 확률
cutoff를 적용하지 않고 가장 높은 클래스 확률을 예측 등급으로 사용한다.

## 7. 평가 지표

### 7.1 회귀 지표

- MAE
- RMSE
- \(R^2\)
- Spearman 순위상관

추가로 실제 점수와 예측점수를 같은 네 단계로 변환해 다음을 계산한다.

실제 점수는 정수 경계로 변환한다. 회귀 예측값은 연속값이므로 0 이상으로 제한한 뒤
가장 가까운 정수 점수에 대응하도록 다음 경계를 사용한다.

| 회귀 예측값 | 예측 등급 |
|---:|---|
| \(0\le\hat S<0.5\) | `normal` |
| \(0.5\le\hat S<5.5\) | `caution` |
| \(5.5\le\hat S<11.5\) | `risk` |
| \(11.5\le\hat S\) | `high_risk` |

이 경계를 공통 함수로 구현하여 학습 평가, 저장 예측 및 테스트가 서로 다른 반올림
규칙을 사용하지 않도록 한다.

- Accuracy
- Macro Precision
- Macro Recall
- Macro F1
- 등급별 Precision, Recall, F1, support
- Confusion Matrix

### 7.2 4단계 분류 지표

- Accuracy
- Macro Precision
- Macro Recall
- Macro F1
- Weighted F1
- 등급별 Precision, Recall, F1, support
- Confusion Matrix

고위험 상태만 기존 이진 모델과 비교하기 위해 다음 값도 제공한다.

- `high_risk` 대 나머지의 Precision
- `high_risk` 대 나머지의 Recall
- `high_risk` 대 나머지의 F1

희귀한 `high_risk` 등급 때문에 Accuracy만으로 모델을 선택하거나 성능을 판단하지
않는다.

## 8. 실행 파일과 산출물

새 실행 파일은 다음 하나를 추가한다.

```text
src/current_asset_score_model.py
```

기본 실행 명령은 다음과 같다.

```bash
python src/current_asset_score_model.py
```

기본 산출물 폴더는 `outputs/asset_score_current`다.

```text
outputs/asset_score_current/
├── regression_metrics.csv
├── severity_metrics.csv
├── test_predictions.csv
├── run_config.json
├── feature_importance.csv
└── models/
```

`test_predictions.csv`에는 최소한 다음 값을 저장한다.

- 날짜, 기계 종류, 장비 식별자
- 실제 `failure_points`
- 예측 `failure_points`
- 실제 `severity_level`
- 예측 `severity_level`
- 각 등급의 예측 확률
- 선택된 모델 이름

## 9. 코드 경계

예상 변경 파일은 다음과 같다.

- 수정: `src/industrial_data.py`
- 수정: `src/asset_features.py`
- 추가: `src/industrial_regression.py`
- 추가: `src/current_asset_score_model.py`
- 추가 또는 수정: 관련 테스트
- 수정: 실행 및 Feature 가이드

기존 장비 이진 모델, 부품 모델, 미래 예측 모델의 동작과 기본 산출물 경로는
변경하지 않는다.

## 10. 테스트 요구사항

최소한 다음을 자동 테스트한다.

1. 점수 0, 1, 5, 6, 11, 12가 정확한 경계 등급으로 변환되는지
2. 음수·결측·비숫자 점수를 거부하는지
3. 회귀와 분류 Target이 Feature에 포함되지 않는지
4. 기존 장비 일별 점수 합산 결과가 유지되는지
5. 시간 분할 이후 세 구간에 필요한 Target이 존재하는지
6. 회귀 모델 저장 후 다시 불러온 예측이 동일한지
7. 다중분류 모델 저장 후 클래스 순서와 확률 열이 일치하는지
8. 빈 구간 또는 단일 등급 구간을 명확히 건너뛰는지
9. 기존 전체 테스트가 그대로 통과하는지

## 11. 성공 조건

첫 구현은 성능 향상을 미리 보장하지 않는다. 다음 조건을 만족하면 재설계 실험이
완료된 것으로 본다.

1. 기존 이진 모델과 동일한 시간 분할로 공정하게 비교한다.
2. 회귀와 4단계 분류 결과를 기계 종류별로 모두 확인할 수 있다.
3. Dummy 기준과 비교해 학습 모델의 개선 여부를 명시한다.
4. `high_risk` 등급의 Precision·Recall·F1을 별도로 제공한다.
5. 성능이 낮더라도 결과를 숨기거나 임계값으로 양성률을 강제하지 않는다.

Family 기반 Target, 미래 7일 점수 예측, 기계별 운영 임계값 최적화, 실제 정지 데이터
결합은 이번 구현이 검증된 뒤 진행하는 후속 범위다.
