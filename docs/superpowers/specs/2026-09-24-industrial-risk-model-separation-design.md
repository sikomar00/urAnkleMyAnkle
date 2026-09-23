# 산업 장비 위험 모델 분리 설계

작성일: 2026-09-24  
대상 프로젝트: `urAnkleMyAnkle`

## 1. 목적

현재 `src/eda.py`에는 장비·날짜별 고장점수 기반 당일 탐지 실험이 있고,
`src/industrial_features.py`에는 부품 단위 당일 분류와 향후 7일 예측 로직이 함께 있다.
데이터 단위와 정답이 다른 과제를 한 흐름에서 다루면 모델 입력, 평가 결과 및 저장된
모델의 의미가 혼동될 수 있다.

이번 변경의 목적은 다음 네 과제를 명시적으로 분리하면서 공통 학습·평가 코드를
재사용하는 것이다.

1. 장비 단위 당일 위험 탐지
2. 장비 단위 향후 7일 위험 예측
3. 부품 단위 당일 고장 탐지
4. 부품 단위 향후 7일 고장 예측

미래 예측 시점은 날짜 \(t\)의 센서 측정이 끝난 시점으로 정의한다. 따라서
\(X_{\le t}\)를 사용하여 \(t+1\)일부터 \(t+7\)일까지의 위험을 예측한다.

## 2. 설계 원칙

1. 장비와 부품의 데이터 단위를 코드 수준에서 분리한다.
2. 당일 탐지와 미래 예측의 Target을 서로 다른 함수에서 생성한다.
3. Feature 생성에는 예측 시점 이후 정보를 사용하지 않는다.
4. 미래 Target을 관측하는 기간이 학습·검증·테스트 경계를 넘지 않게 한다.
5. 모델 선택과 확률 임계값 선택에는 검증 데이터만 사용한다.
6. 테스트 데이터는 최종 평가에 한 번만 사용한다.
7. 클래스 불균형 문제에서는 Accuracy보다 Average Precision을 우선한다.
8. 기존 실행 파일은 호환 진입점으로 유지하여 기존 명령이 갑자기 깨지지 않게 한다.

## 3. 파일 구조

구현 후의 주요 구조는 다음과 같다.

```text
src/
├── industrial_data.py
│   └── 데이터 로딩, 공통 컬럼 검사 및 날짜 분할
├── asset_features.py
│   ├── prepare_asset_current()
│   └── prepare_asset_forecast()
├── part_features.py
│   ├── prepare_part_current()
│   └── prepare_part_forecast()
├── industrial_training.py
│   └── 공통 학습, 모델 선택, 임계값 선택, 평가 및 결과 저장
├── current_asset_model.py
├── forecast_asset_model.py
├── current_part_model.py
├── forecast_part_model.py
├── current_state_model.py
│   └── current_part_model.py를 호출하는 호환 진입점
├── timeseries_model.py
│   └── forecast_part_model.py를 호출하는 호환 진입점
└── eda.py
    └── 학습 진입점이 아닌 결과 비교·시각화 용도
```

결과 폴더는 다음처럼 분리한다.

```text
outputs/
├── asset_current/
├── asset_forecast_7d/
├── part_current/
└── part_forecast_7d/
```

## 4. 공통 데이터 정의

부품 중요도 가중치는 다음과 같이 유지한다.

\[
w_i=
\begin{cases}
4,& \mathrm{criticality}=A,\\
2,& \mathrm{criticality}=B,\\
1,& \mathrm{criticality}=C.
\end{cases}
\]

장비 \(a\)의 날짜 \(t\)별 고장점수는 다음과 같다.

\[
S_{a,t}
=
\sum_{i\in\mathrm{parts}(a)}
\mathrm{breakdown\_flag}_{a,i,t}\,w_i.
\]

기존 `target_gt_12`는 실제 조건이 `>= 12`였으므로 이름과 연산자가 일치하지 않는다.
새 Target 이름은 연산 의미가 분명하도록 `target_ge_12`, `target_ge_13`,
`target_ge_14`로 정한다.

## 5. 장비 단위 당일 위험 탐지

### 5.1 데이터 단위

한 행은 다음 키로 식별한다.

```text
transaction_date + machine_type + asset_tag
```

원본 부품 행에 반복 기록된 센서값은 동일 장비·동일 날짜 안에서 같은 값인지 먼저
검사한 다음 한 행으로 집계한다. 서로 다른 값이 발견되면 임의로 평균하거나 첫 값을
고르지 않고 오류를 발생시킨다.

### 5.2 Target

각 점수 기준 \(\tau\in\{12,13,14\}\)에 대해 다음 Target을 만든다.

\[
y^{\mathrm{asset,current}}_{\tau}(t)
=
\mathbf{1}\{S_{a,t}\ge\tau\}.
\]

### 5.3 Feature

초기 Feature는 다음과 같다.

```text
machine_type
asset_tag
temp_bearing_degC
temp_motor_degC
vibration_h_mms
vibration_v_mms
oil_pressure_bar
load_pct
shaft_rpm
power_consumption_kw
day_of_week
month_sin
month_cos
```

`asset_tag`는 기존 장비의 시간대별 행동을 학습하는 데 도움을 줄 수 있지만, 학습에
없던 새 장비에 대한 일반화 성능을 보장하지 않는다. 따라서 전체, 기계 종류별 및
장비별 성능을 구분하여 저장한다.

## 6. 장비 단위 향후 7일 위험 예측

### 6.1 미래 Target

기본 예측 기간은 7일이며 CLI의 `--horizon`으로 변경할 수 있게 한다. 점수 기준
\(\tau\)에 대해 전체 미래 위험 Target은 다음과 같다.

\[
y^{\mathrm{asset,any\_risk}}_{\tau,7}(t)
=
\max_{1\le k\le7}
\mathbf{1}\{S_{a,t+k}\ge\tau\}.
\]

기본 운영 Target인 신규 위험은 오늘 정상인 장비만 예측 모집단에 남긴다.

\[
y^{\mathrm{asset,new\_risk}}_{\tau,7}(t)
=
y^{\mathrm{asset,any\_risk}}_{\tau,7}(t)
\quad\text{subject to}\quad S_{a,t}<\tau.
\]

`new_risk_7d`는 당일 탐지 이후 예방점검 대상을 찾는 기본 모델이다.
`any_risk_7d`는 현재 위험 상태의 지속까지 포함한 비교 모델이다. 두 결과는
`risk_definition` 컬럼으로 구분한다.

향후 7일 날짜가 모두 존재하지 않으면 해당 행에 음성 Target을 채우지 않고 학습
대상에서 제외한다.

### 6.2 센서 Feature

각 센서 \(x\)에 대해 날짜 \(t\)까지의 정보만 사용하여 다음 값을 만든다.

\[
x_t,\quad x_{t-1},\quad x_{t-3},\quad x_{t-7},
\]

\[
\overline{x}_{t,7}=\frac{1}{7}\sum_{j=0}^{6}x_{t-j},
\qquad
\sigma_{t,7}=\operatorname{std}(x_t,\ldots,x_{t-6}),
\]

\[
\Delta x_t=x_t-x_{t-1}.
\]

센서 Feature는 장비 단위인 `asset_tag`별로 계산한다.

### 6.3 과거 위험 Feature

고장점수 기반 이력은 예측 당일의 고장 결과를 사용하지 않고 반드시 `shift(1)` 이후
계산한다.

```text
failure_points_lag1
failure_points_mean7
failure_points_max7
risk_event_count_30d
days_since_last_risk
```

연산 순서는 `groupby -> shift(1) -> rolling`로 고정한다.

## 7. 부품 단위 당일 고장 탐지

### 7.1 데이터 단위와 Target

한 행은 다음 키로 식별한다.

```text
transaction_date + asset_tag + part_no
```

Target은 당일의 원본 고장 표시다.

\[
y^{\mathrm{part,current}}_{a,i,t}
=
\mathrm{breakdown\_flag}_{a,i,t}.
\]

### 7.2 Feature

```text
machine_type
asset_tag
part_no
criticality
plant_code
센서 8개
day_of_week
month_sin
month_cos
```

현재 구현처럼 `part_no`를 제외하면 동일 장비·날짜의 부품들은 모델에 같은 입력으로
보이지만 서로 다른 정답을 가질 수 있다. 부품을 구분하기 위해 `part_no`와
`criticality`를 사용한다. 다만 기존 부품 식별자를 기억하는 효과가 포함될 수 있으므로,
시간 분할 결과를 새 부품 일반화 성능으로 해석하지 않는다.

## 8. 부품 단위 향후 7일 고장 예측

### 8.1 Target

\[
y^{\mathrm{part,7d}}_{a,i,t}
=
\max_{1\le k\le7}
\mathrm{breakdown\_flag}_{a,i,t+k}.
\]

오늘 이미 고장인 부품은 신규 고장 예측 모집단에서 제외한다.

\[
\mathrm{breakdown\_flag}_{a,i,t}=0.
\]

### 8.2 Feature

센서 이력은 장비 단위 `asset_tag`별로 다음 값을 계산한다.

```text
sensor_current
sensor_lag1
sensor_lag3
sensor_lag7
sensor_mean7
sensor_std7
sensor_diff1
```

고장 이력은 `asset_tag + part_no`별로 계산한다.

```text
breakdown_lag1
breakdown_count_7d
breakdown_count_30d
days_since_last_breakdown
```

정적 Feature로 `machine_type`, `asset_tag`, `part_no`, `criticality`,
`plant_code`를 사용한다.

## 9. 시간 분할과 누수 방지

기본 검증 시작일은 2024-01-01, 테스트 시작일은 2024-07-01로 유지한다. 미래
Target에는 Target 관측 종료일 `label_end_date`를 기록한다.

검증 시작일을 \(V\), 테스트 시작일을 \(T\)라고 하면 다음처럼 분할한다.

\[
\mathrm{Train}
=
\{(t,y_t):\mathrm{label\_end\_date}<V\},
\]

\[
\mathrm{Validation}
=
\{(t,y_t):t\ge V,
\ \mathrm{label\_end\_date}<T\},
\]

\[
\mathrm{Test}
=
\{(t,y_t):t\ge T\}.
\]

예를 들어 테스트가 2024-07-01에 시작할 때 2024-06-25의 7일 Target은
2024-07-02까지 참조한다. 이 행은 검증 데이터에서 제외한다.

## 10. 모델 학습 및 선택

각 과제는 동일한 공통 인터페이스로 다음 모델을 비교한다.

1. `DummyClassifier`: 양성률 기준선
2. `LogisticRegression`: 해석 가능한 선형 기준모델
3. `RandomForestClassifier`: 기존 `eda.py` 결과와의 비교
4. `HistGradientBoostingClassifier`: 비선형 관계 비교

LightGBM은 초기 기본 모델에서 제외한다. 기본 평가 구조를 안정화한 후 선택적 모델로
추가할 수 있으며, 추가할 때는 macOS `libomp` 같은 네이티브 의존성을 문서화한다.

최종 모델은 검증 데이터의 Average Precision이 가장 높은 모델로 선택한다.

\[
m^*=\arg\max_m\operatorname{AP}_{\mathrm{validation}}(m).
\]

모델과 임계값 선택이 끝난 뒤 테스트 데이터에 한 번 적용한다.

## 11. 확률 임계값 정책

다음 세 정책을 지원한다.

### 11.1 F1 최대화

\[
\tau^*=\arg\max_\tau F_1(\tau).
\]

기존 실험과 비교하기 위한 기본 임계값으로 사용한다.

### 11.2 최소 Precision

\[
\max_\tau \operatorname{Recall}(\tau)
\quad\text{subject to}\quad
\operatorname{Precision}(\tau)\ge p_{\min}.
\]

기본 `p_min`은 0.30으로 하되 CLI에서 변경할 수 있게 한다.

### 11.3 상위 비율

검증 데이터의 위험도 상위 비율에 대응하는 확률을 임계값으로 선택한다. 기본 비교
비율은 10%이며 CLI에서 변경할 수 있게 한다.

세 정책의 결과를 모두 저장하고, 모델 파일의 기본 임계값은 기존 비교를 위해 F1
정책으로 둔다.

## 12. 평가 지표

주지표는 클래스 불균형에 적합한 Average Precision으로 한다. 다음 지표를 함께
저장한다.

```text
average_precision
roc_auc
accuracy
precision
recall
f1
true_negative
false_positive
false_negative
true_positive
```

점검 우선순위 평가를 위해 다음 지표도 계산한다.

```text
precision_at_5pct
recall_at_5pct
precision_at_10pct
recall_at_10pct
precision_at_20pct
recall_at_20pct
lift_at_10pct
```

\[
\operatorname{Lift@10\%}
=
\frac{\operatorname{Precision@10\%}}{\text{전체 양성률}}.
\]

## 13. 출력 계약

각 결과 폴더에는 다음 파일을 생성한다.

```text
metrics.csv
test_predictions.csv
feature_importance.csv
run_config.json
models/
```

`metrics.csv`에는 결과의 의미를 식별할 수 있도록 다음 컬럼을 포함한다.

```text
grain
mode
target
risk_definition
score_threshold
horizon
scope_kind
scope_name
model
threshold_policy
probability_cutoff
```

`test_predictions.csv`에는 원본 키, 실제 Target, 모델 점수, 적용 임계값, 예측값 및
모델 식별자를 포함한다. 저장된 모델에는 Feature 목록, Target, 데이터 단위, 모드,
위험 정의, 시간 범위, 선택된 임계값 및 모델명을 함께 기록한다.

## 14. CLI 계약

기본 실행 예시는 다음과 같다.

```bash
python src/current_asset_model.py
python src/forecast_asset_model.py --horizon 7 --risk-definition new
python src/forecast_asset_model.py --horizon 7 --risk-definition any
python src/current_part_model.py
python src/forecast_part_model.py --horizon 7
```

공통 선택 인자는 다음과 같다.

```text
--data
--output
--scope
--machine-type
--asset-tag
--validation-start
--test-start
--threshold-policy
--min-precision
--top-fraction
--random-state
```

장비 모델은 `--score-threshold`를 여러 번 지정할 수 있고, 생략하면 12, 13, 14를
모두 실행한다. 미래 모델은 `--horizon`을 지원한다.

## 15. 테스트 계약

구현은 다음 실패 테스트를 먼저 추가한 후 진행한다.

1. 장비별 `failure_points` 합산 정확성
2. `failure_points >= 12`일 때 `target_ge_12=1`인지 확인
3. 미래 Target이 날짜 \(t\)를 제외하고 \(t+1,\ldots,t+7\)만 참조하는지 확인
4. 향후 7일이 완전하지 않은 행이 제외되는지 확인
5. `new_risk_7d`에서 오늘 위험한 장비가 제외되는지 확인
6. `any_risk_7d`에서 오늘 위험한 장비가 유지되는지 확인
7. 부품 Feature에 다른 부품의 고장 이력이 섞이지 않는지 확인
8. 미래 원본 값을 변경해도 과거 시점 Feature가 변하지 않는지 확인
9. 검증 Target 관측 기간이 테스트 구간을 침범하지 않는지 확인
10. 네 실행 파일이 소규모 합성 데이터로 끝까지 실행되는지 확인

추가로 다음 검증을 수행한다.

1. `pytest` 전체 실행
2. 각 CLI의 `--help` 실행
3. 네 모델의 기본 명령 실행
4. 출력 CSV 스키마 검사
5. 모델 저장·재로딩 후 예측 일치 검사
6. 기존 호환 진입점 실행 확인

## 16. 범위와 비범위

이번 구현 범위에는 네 과제의 분리, 누수 없는 Feature/Target 생성, 공통 학습·평가,
세 임계값 정책, 결과 저장 및 테스트가 포함된다.

다음 항목은 이번 초기 구현에 포함하지 않는다.

1. 실시간 스트리밍 추론 서비스
2. 웹 대시보드 화면 변경
3. 자동 재학습 스케줄러
4. 새로운 장비·새로운 부품에 대한 별도 외삽 평가
5. SHAP 기반 상세 설명
6. LightGBM 기본 모델 편입

## 17. 알려진 한계

1. 데이터는 교육용 합성 데이터이므로 실제 설비로 일반화할 수 없다.
2. `asset_tag`와 `part_no`는 기존 대상을 기억하는 효과를 만들 수 있다.
3. `new_risk_7d`는 신규 위험 발생을, `any_risk_7d`는 위험 지속까지 포함하므로 두
   지표를 직접 같은 과제로 비교해서는 안 된다.
4. 하나의 검증 기간에서 선택한 임계값은 양성 수가 적을 때 불안정할 수 있다.
5. F1 최대 임계값은 운영 비용을 반영하지 않으므로 최소 Precision과 상위 비율 결과를
   함께 검토해야 한다.

## 18. 완료 조건

다음 조건을 모두 충족하면 구현이 완료된 것으로 판단한다.

1. 네 실행 파일이 독립적으로 실행된다.
2. 장비·부품, 당일·미래 결과가 각 출력 폴더에 분리된다.
3. `new_risk_7d`와 `any_risk_7d`가 명시적으로 구분된다.
4. 미래 Feature와 날짜 분할에 대한 누수 방지 테스트가 통과한다.
5. 기준모델과 세 학습모델의 검증·테스트 결과가 저장된다.
6. 세 임계값 정책과 Top-K 계열 지표가 출력된다.
7. 기존 `current_state_model.py`, `timeseries_model.py` 실행이 유지된다.
8. 전체 테스트와 실행 검증 결과가 문서화된다.
