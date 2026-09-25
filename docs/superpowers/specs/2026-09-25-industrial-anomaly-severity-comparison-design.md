# 산업 장비 센서 이상 특징·고위험 기준 비교 설계

작성일: 2026-09-25

## 1. 목적

현재 장비·날짜 단위 4단계 분류는 센서 원값과 달력 정보만 사용한다. 센서 원값은
기계 종류와 개별 장비의 평상시 운전 수준 차이를 직접 표현하지 못하고, 현재 공통
등급은 고위험을 `failure_points >= 12`로만 정의한다.

이번 변경은 다음 두 질문을 같은 시간 분할에서 검증한다.

1. 개별 장비와 기계 종류의 정상 상태에서 벗어난 정도, 최근 센서 변화와 여러
   센서의 동시 이상을 Feature로 추가하면 4단계 분류 성능이 개선되는가?
2. 고위험 시작점을 12점과 13점으로 정의했을 때 등급 분포와 예측 성능은 어떻게
   달라지는가?

최종 Target은 두 기준 모두 `normal`, `caution`, `risk`, `high_risk` 네 등급이다.
별도의 이진 분류 모델로 목표를 바꾸지 않는다.

## 2. 해석 제한

원본 데이터의 `breakdown_flag`는 부품별 고장 표시이며, 실제 기계 정지, 정지시간,
생산 손실 또는 정비 수행 결과가 아니다. 장비 일별 `failure_points`는 부품별 표시와
중요도 가중치를 합산한 대리 점수다.

따라서 다음 용어를 구분한다.

| 이름 | 정의 | 해석 |
|---|---|---|
| `part_breakdown_rate` | 부품 행 중 `breakdown_flag == 1` 비율 | 부품 고장 표시율 |
| `asset_issue_day_rate` | 장비·날짜 중 `failure_points >= 1` 비율 | 문제가 하나 이상 표시된 장비일 비율 |
| `asset_high_risk_day_rate` | 장비·날짜 중 고위험 기준 이상 비율 | 고위험 점수 발생률 |

위 값을 실제 기계 고장률 또는 정지율이라고 부르지 않는다.

## 3. 공통 4단계 기준과 비교 기준

기계 종류마다 점수 경계를 완화하지 않는다. 모든 기계는 같은 부품 수와 중요도
가중치 구조를 사용하므로 같은 점수는 같은 절대 등급으로 해석한다. 기계별 차이는
경계를 변경하는 대신 발생률과 성능을 별도로 보고한다.

### 3.1 고위험 12점 기준

\[
L_{12}(S)=
\begin{cases}
\mathrm{normal},&S=0,\\
\mathrm{caution},&1\le S\le5,\\
\mathrm{risk},&6\le S\le11,\\
\mathrm{high\_risk},&S\ge12.
\end{cases}
\]

### 3.2 고위험 13점 기준

\[
L_{13}(S)=
\begin{cases}
\mathrm{normal},&S=0,\\
\mathrm{caution},&1\le S\le5,\\
\mathrm{risk},&6\le S\le12,\\
\mathrm{high\_risk},&S\ge13.
\end{cases}
\]

두 기준의 차이는 12점 장비일의 정답이 `high_risk`인지 `risk`인지다. 테스트
구간에는 12점 장비일이 70건 있다. `high_risk` 비율은 12점 기준 221/1,850
(11.95%), 13점 기준 151/1,850(8.16%)다.

두 기준은 서로 다른 Target이므로 Macro F1 숫자만 비교해 어느 기준이 더 옳다고
단정하지 않는다. 12점 기준은 더 민감한 정책이고, 13점 기준은 더 높은 점수에
집중하는 정책으로 함께 보고한다.

## 4. 데이터 단위와 분할

한 행은 `transaction_date + machine_type + asset_tag`로 식별되는 장비·날짜다.
부품 20개 행의 센서값이 같은지 검증한 뒤 장비·날짜 한 행으로 집계하는 기존
`build_asset_daily()` 계약을 유지한다.

시간 분할은 현재 실험과 동일하게 고정한다.

- 학습: `label_end_date < 2024-01-01`
- 검증: `transaction_date >= 2024-01-01`이고 `label_end_date < 2024-07-01`
- 테스트: `transaction_date >= 2024-07-01`

모든 기준값, 결측 보정, 인코딩과 모델 선택은 학습·검증까지만 사용한다. 테스트는
최종 평가에만 사용한다.

## 5. Feature 실험군

센서 8개는 다음과 같다.

- `temp_bearing_degC`
- `temp_motor_degC`
- `vibration_h_mms`
- `vibration_v_mms`
- `oil_pressure_bar`
- `load_pct`
- `shaft_rpm`
- `power_consumption_kw`

두 고위험 기준 각각에 대해 다음 네 Feature 구성을 평가한다.

| 실험군 | Feature | 검증 목적 |
|---|---|---|
| A | 기존 센서 원값, 기계·장비 식별자, 달력 | 현재 기준선 재현 |
| B | A + 장비별·기계 종류별 Robust Z-score | 정상 범위 대비 편차 효과 |
| C | A + 센서 과거 변화량·3일/7일 추세 | 시간 변화 효과 |
| D | A + B + C + 동시 이상 요약 | 전체 결합 효과 |

총 실험 조합은 `A12`, `B12`, `C12`, `D12`, `A13`, `B13`, `C13`, `D13`이다.

## 6. 정상 기준과 Robust Z-score

센서별 정상 기준은 학습 구간 중 `failure_points == 0`인 행으로만 적합한다. 검증과
테스트 행은 기준 계산에 포함하지 않는다.

Robust Z-score는 다음처럼 정의한다.

\[
z_{\mathrm{robust}}
=\frac{x-\operatorname{median}(x_{\mathrm{normal}})}
{1.4826\operatorname{MAD}(x_{\mathrm{normal}})}.
\]

센서별 기준 선택 순서는 다음과 같다.

1. 정상 표본이 30건 이상이고 MAD가 0보다 큰 `asset_tag` 기준
2. 같은 조건을 만족하는 `machine_type` 기준
3. 전체 학습 정상행 기준
4. 전체 정상행의 MAD도 0이면 표준편차 사용
5. 표준편차도 0이면 상수 센서로 기록하고 Z-score를 0으로 둠

사용한 기준 수준, 정상 표본 수, 중앙값, MAD와 대체 사유를
`zscore_baselines.csv`에 저장한다. 같은 학습 기준을 검증·테스트 및 저장 모델의
추론 과정에서 재사용한다.

## 7. 과거 변화·추세와 동시 이상

과거 Feature는 `asset_tag`별 날짜 순서로 생성한다. 당일 탐지이므로 당일 센서
원값은 사용할 수 있지만, 과거 통계에는 미래 또는 현재 이후 값이 들어가면 안 된다.

센서별 후보는 다음과 같다.

- `lag1`, `lag3`, `lag7`
- 당일값과 `lag1`, `lag7`의 차이
- `shift(1)` 이후 최근 3일·7일 중앙값
- `shift(1)` 이후 최근 7일 MAD
- 최근 7일 선형 변화 기울기

동시 이상 요약은 다음을 포함한다.

- `max_abs_robust_z`
- `mean_abs_robust_z`
- `abs(z) >= 2`인 센서 수
- `abs(z) >= 3`인 센서 수
- 온도 센서와 진동 센서가 동시에 `abs(z) >= 2`인지 여부

초기 과거가 부족해 생기는 결측은 학습 파이프라인의 결측 보정으로 처리하고, 해당
행을 미래값으로 채우지 않는다.

## 8. 모델과 선택 규칙

두 기준과 네 Feature 실험군에서 다음 모델을 같은 조건으로 비교한다.

1. `DummyClassifier`: 학습 등급 분포만 사용하는 기준선
2. `LogisticRegression(class_weight="balanced")`
3. `RandomForestClassifier(class_weight="balanced")`
4. 가중치를 적용하지 않은 `HistGradientBoostingClassifier`
5. 학습 등급 빈도의 역수를 `sample_weight`로 적용한
   `HistGradientBoostingClassifier`

Dummy는 선택 대상이 아니다. 각 `고위험 기준 x Feature 실험군 x 평가 Scope` 안에서
학습 모델을 먼저 선택한다. 검증 Macro F1이 가장 높은 모델을 우선 선택하고, 최고
Macro F1과 0.01 이내인 모델이 둘 이상이면 `high_risk` Recall, 그다음
`high_risk` Precision이 높은 모델을 선택한다.

그다음 같은 고위험 기준과 Scope 안에서 A~D의 선택 모델을 비교해 동일한 규칙으로
대표 Feature 실험군을 정한다. 테스트 성능으로 모델이나 Feature 구성을 선택하지
않는다. 대표 모델 외에도 모든 실험 결과를 산출물에 남긴다.

현재처럼 전체 모델과 기계 종류별 모델을 모두 평가한다. 전체 모델은
`machine_type`과 `asset_tag`를 Feature로 사용하고, 기계별 모델은 해당 기계 종류
안에서 학습한다. `asset_tag`별 별도 모델은 표본 수가 작아 최종 선택 모델로 삼지
않고, 같은 전체 또는 기계별 모델의 예측을 `asset_tag`별로 나눠 평가한다.

## 9. 평가 지표와 비교 방법

각 실험 조합과 모델에 대해 다음을 저장한다.

- Accuracy
- Macro Precision, Macro Recall, Macro F1
- Weighted F1
- 등급별 Precision, Recall, F1, support
- `high_risk` 대 나머지의 Precision, Recall, F1
- Confusion Matrix

주요 비교는 다음처럼 같은 Target 안에서 수행한다.

- `A12 -> B12`: Z-score 효과
- `A12 -> C12`: 과거 변화·추세 효과
- `A12 -> D12`: 전체 결합 효과
- `A13 -> B13`, `A13 -> C13`, `A13 -> D13`: 13점 기준의 같은 비교

12점과 13점 정책 간에는 고위험 표본 수, 고위험 Recall·Precision, 12점 행의 예측,
기계별 발생률을 함께 제시한다. Target이 다르다는 사실을 결과 문서에 명시한다.

## 10. 실행 인터페이스와 산출물

기존 `current_asset_score_model.py`와 `outputs/asset_score_current`는 변경하지 않는다.
새 비교 실험은 별도 실행 파일과 출력 폴더를 사용한다.

예상 실행 명령은 다음과 같다.

```bash
python src/current_asset_severity_experiments.py \
  --high-risk-thresholds 12 13 \
  --feature-sets A B C D
```

기본 출력 구조는 다음과 같다.

```text
outputs/asset_severity_experiments/
├── experiment_metrics.csv
├── class_metrics.csv
├── confusion_matrices.csv
├── test_predictions.csv
├── machine_failure_profile.csv
├── feature_importance.csv
├── zscore_baselines.csv
├── run_config.json
├── experiment_summary.md
└── models/
```

`machine_failure_profile.csv`는 `machine_type`과 `asset_tag` 수준에서 세 발생률과
점수 평균·중앙값을 분리해 저장한다. `experiment_summary.md`는 다음을 한국어로
정리한다.

1. 12점·13점 기준의 등급 분포 차이
2. A~D Feature 조합별 검증·테스트 결과
3. 기존 모델 대비 Macro F1 및 고위험 Recall 변화
4. 기계 종류와 `asset_tag`별 발생률 및 성능
5. 12점 장비일 70건의 예측 변화
6. 합성 데이터와 대리 Target의 해석 한계

## 11. 오류 처리와 재현성

- 고위험 기준은 6보다 큰 정수만 허용한다.
- Feature 실험군 이름이 A~D가 아니면 실행을 거부한다.
- 같은 장비·날짜의 센서값 충돌은 기존처럼 오류로 처리한다.
- 정상 기준에 사용된 행 수와 대체 경로를 반드시 기록한다.
- 분할 날짜, 임계값, Feature 목록, 모델 파라미터와 난수 시드를
  `run_config.json`에 저장한다.
- 빈 검증·테스트 구간, 학습 단일 등급 또는 평가 불가능한 범위는 이유와 함께
  `skipped` 상태로 남긴다.

## 12. 테스트 요구사항

최소한 다음을 자동 테스트한다.

1. 11·12·13점이 두 기준에서 올바른 등급으로 변환되는지
2. 모든 기계에 같은 점수 경계가 적용되는지
3. Robust Z-score 기준이 학습 정상행으로만 적합되는지
4. 장비 기준 부족·MAD 0일 때 기계 종류와 전체 기준으로 대체되는지
5. 미래 센서값을 바꿔도 과거 날짜 Feature가 바뀌지 않는지
6. A~D Feature 집합이 설계한 열만 포함하는지
7. Target 관련 열이 Feature에 들어가지 않는지
8. 모델 선택의 Macro F1·고위험 Recall·Precision 우선순위가 유지되는지
9. 저장 모델을 다시 불러온 예측과 확률 열 순서가 동일한지
10. 산출물 스키마와 12점 행 비교 결과가 일관적인지
11. 기존 전체 테스트가 그대로 통과하는지

## 13. 성공 조건

구현 완료 조건은 성능 향상을 미리 가정하지 않는다.

1. 두 고위험 기준과 네 Feature 구성이 같은 분할에서 재현 가능하게 실행된다.
2. Z-score와 과거 Feature에 검증·테스트 정보 누수가 없다.
3. 전체, 기계 종류, `asset_tag` 수준의 발생률과 성능이 분리되어 제공된다.
4. 12점과 13점 기준의 차이를 실제 기계 정지율로 과장하지 않는다.
5. D가 A보다 낮더라도 결과를 숨기지 않고 원인을 결과 문서에 기록한다.
6. 기존 실행 파일과 산출물 계약이 깨지지 않고 전체 테스트가 통과한다.

실제 고장·정지 데이터 결합, 미래 7일 이상탐지, 정비 비용 기반 운영 임계값과
대시보드 연결은 이번 변경 이후의 후속 범위다.
