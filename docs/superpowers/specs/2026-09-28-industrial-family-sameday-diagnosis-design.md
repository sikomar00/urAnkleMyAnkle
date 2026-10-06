# 산업 장비 당일 Family 이상·심각 진단 설계

작성일: 2026-09-28  
대상 프로젝트: `urAnkleMyAnkle`

## 세 줄 요약

1. 미래 예측 전에 당일 장비 센서로 9개 부품 Family의 이상 여부를 각각 진단한다.
2. `Family 구성품 하나 이상 고장`을 이상 Target으로, `2개 이상 또는 A등급 부품 고장`을 심각 Target으로 분리한다.
3. 당일값·과거 시계열 Feature·운전조건 보정 잔차를 비교하고, 시간 Feature의 개선이 확인될 때만 순차 모델을 검토한다.

## 한 페이지 요약

현재 부품별 당일·미래 예측과 장비 점수 기반 4단계 분류는 실용적인 성능을 만들지
못했다. 그러나 각 `part_no`에는 장비 10대의 약 3년치 관측치가 있으므로 부품별
분석 자체가 불가능했던 것은 아니다. 문제는 센서가 장비·날짜 단위로 공유되고,
개별 부품 고장과 센서 사이의 안정적인 관계가 약했다는 점이다.

이번 실험은 개별 부품 번호 대신 현장에서 점검 범위를 줄일 수 있는 `part_family`를
진단한다. 한 행은 `transaction_date + machine_type + asset_tag`인 장비·날짜다.
Family별로 `affected`와 `severe` 두 이진 Target을 만들며, 여러 Family가 동시에
양성일 수 있는 다중라벨 문제로 다룬다.

\[
y^{\mathrm{affected}}_{a,f,t}
=\mathbf{1}\{N^{\mathrm{failed}}_{a,f,t}\ge1\},
\]

\[
y^{\mathrm{severe}}_{a,f,t}
=\mathbf{1}\{N^{\mathrm{failed}}_{a,f,t}\ge2
\ \lor\ A^{\mathrm{failed}}_{a,f,t}=1\}.
\]

첫 구현은 LSTM을 사용하지 않는다. 현재 센서만 사용하는 A, 과거 센서와 과거
Family 상태를 추가한 B, 운전조건으로 설명되는 정상 변화를 제거한 잔차까지 넣은
C를 Logistic Regression, Random Forest, HistGradientBoosting으로 비교한다. 시간
순서 분할을 사용하고, 정상 기준과 잔차 모델은 학습 구간에서만 적합한다. 모델과
Feature 집합은 검증 Average Precision을 우선하여 선택하고 테스트는 최종 평가에만
사용한다.

주요 결과는 Family별 Precision, Recall, F1, Average Precision, ROC AUC와 양성률
대비 Lift다. Accuracy는 보조 지표로만 제공한다. 검증에서 시계열 Feature가 정적
기준선보다 Macro Average Precision을 0.02 이상 높이고 다수 Family에서 같은 개선
방향을 보일 때만 LSTM·Temporal CNN 같은 순차 모델을 다음 단계로 검토한다. 당일
Family 진단이 성립하지 않으면 미래 Family 예측으로 넘어가지 않는다.

## 1. 목적과 범위

이번 변경의 목적은 다음 두 질문을 검증하는 것이다.

1. 당일 센서와 운전조건으로 어느 부품 Family에 이상이 표시됐는지 진단할 수 있는가?
2. 현재값만 사용할 때보다 과거 센서 변화와 운전조건 보정 잔차를 사용할 때 성능이
   개선되는가?

초기 범위는 당일 진단만 포함한다. 다음 항목은 이번 구현 범위에서 제외한다.

- 내일 또는 향후 3·7일 Family 고장 예측
- 개별 `part_no` 예측
- LSTM·GRU·Temporal CNN 구현
- 실제 장비 정지·생산 손실 예측
- 기존 4단계 장비 점수 모델의 제거 또는 대체

당일 Family 진단 결과가 유효할 때만 미래 예측을 별도 설계한다.

## 2. 데이터 해석과 제한

원본 센서 8개는 부품 센서가 아니라 장비·날짜 센서다. 같은 장비·날짜의 20개
부품 행은 같은 센서값을 가진다. 각 부품은 장비 10대와 약 3년을 합쳐 10,950개의
관측치를 가지므로 부품별 시계열 분석은 가능하지만, 센서가 어느 물리적 부품에서
측정됐는지는 알 수 없다.

따라서 모델 출력은 다음처럼 해석한다.

- 허용: `Bearing Family 이상 의심`, `Electrical Family 심각 의심`
- 금지: `Bearing 전체 부품 고장 확정`, `MRO-10045 교체 확정`

`breakdown_flag`는 합성 데이터의 부품별 표시다. 실제 장비 정지, 고장 시각, 정비
확정 결과 또는 생산 손실을 의미하지 않는다. 결과 보고서에 이 제한을 명시한다.

## 3. 데이터 단위와 Family 목록

입력 한 행은 다음 키로 식별되는 장비·날짜다.

```text
transaction_date + machine_type + asset_tag
```

진단 대상 Family는 원본 데이터에 존재하는 다음 9개로 고정한다.

1. `Bearing`
2. `Seal & Gasket`
3. `Drive Belt`
4. `Filter`
5. `Electrical`
6. `Coupling`
7. `Lubrication`
8. `Sensor`
9. `Fastener`

입력 데이터에서 Family 목록이 달라지면 조용히 열을 추가하거나 삭제하지 않고
검사 결과를 오류 또는 명시적인 스키마 변경으로 처리한다.

## 4. Target 정의

장비 \(a\), Family \(f\), 날짜 \(t\)에서 다음 값을 계산한다.

- \(N^{\mathrm{parts}}_{a,f,t}\): Family 전체 부품 수
- \(N^{\mathrm{failed}}_{a,f,t}\): `breakdown_flag == 1`인 Family 부품 수
- \(A^{\mathrm{failed}}_{a,f,t}\): 고장 부품 중 `criticality == A`가 있으면 1

### 4.1 Family 이상 Target

\[
y^{\mathrm{affected}}_{a,f,t}
=\mathbf{1}\{N^{\mathrm{failed}}_{a,f,t}\ge1\}.
\]

구성품 하나라도 고장으로 표시되면 해당 Family는 `affected`다. 이 Target은
`Family 전체 고장`이 아니라 `Family 점검 필요`로 해석한다.

### 4.2 Family 심각 Target

\[
y^{\mathrm{severe}}_{a,f,t}
=\mathbf{1}\{N^{\mathrm{failed}}_{a,f,t}\ge2
\ \lor\ A^{\mathrm{failed}}_{a,f,t}=1\}.
\]

고장 부품이 두 개 이상이거나 A등급 부품 하나가 고장으로 표시되면 `severe`다.

### 4.3 Target 관계 검사

모든 행과 Family에서 다음이 성립해야 한다.

\[
y^{\mathrm{severe}}_{a,f,t}
\le y^{\mathrm{affected}}_{a,f,t}.
\]

Family 전체 부품이 A등급인 경우 두 Target이 같을 수 있다. 현재 `Bearing`은 세
부품이 모두 A등급이므로 `affected`와 `severe`가 동일하다. 구현은 이를 오류로 보지
않되 `identical_target` 상태로 보고서에 기록하고 중복 모델 학습은 생략할 수 있다.

### 4.4 모든 구성품 고장

\[
y^{\mathrm{all}}_{a,f,t}
=\mathbf{1}\{N^{\mathrm{failed}}_{a,f,t}=N^{\mathrm{parts}}_{a,f,t}\}.
\]

`all`은 발생률 통계로만 저장하고 학습 Target으로 사용하지 않는다. 현재 발생률은
Family별 약 0.17~1.76%로 너무 낮아 기본 진단 Target으로 부적합하다.

## 5. 다중라벨 구현 방식

하루에 여러 Family가 동시에 양성일 수 있으므로 단일 다중분류로 만들지 않는다.
공통 장비·날짜 Feature Frame 위에서 `Family x Target 종류`별 독립 이진 모델을
학습한다.

```text
9 Family x affected
9 Family x severe
```

Target이 동일하거나 학습·검증 구간에서 양성 표본이 부족한 조합은 이유와 함께
`identical_target` 또는 `insufficient_positive_rows`로 기록한다. 독립 이진 모델의
예측을 합쳐 장비·날짜별 다중라벨 결과를 만든다.

## 6. 시간 분할과 누수 방지

기존 날짜 경계를 유지한다.

- 학습: `transaction_date < 2024-01-01`
- 검증: `2024-01-01 <= transaction_date < 2024-07-01`
- 테스트: `transaction_date >= 2024-07-01`

당일 진단이므로 Target 종료일은 입력 날짜와 같다. 다음 규칙을 적용한다.

1. Robust 기준과 잔차 모델은 학습 구간에서만 적합한다.
2. 모델 및 Feature 집합 선택과 확률 임계값 선택은 검증 구간까지만 사용한다.
3. 테스트는 최종 성능 산출에 한 번만 사용한다.
4. 과거 통계는 `asset_tag` 또는 `asset_tag + family` 안에서 날짜순으로 계산한다.
5. 과거 Target 이력은 반드시 `shift(1)` 후 계산한다.
6. 미래값이나 같은 날의 Family Target을 Feature로 사용하지 않는다.

## 7. Feature 실험군

### 7.1 실험 A: 당일 정적 Feature

다음 현재 정보만 사용한다.

- `machine_type`, `asset_tag`
- 센서 8개 원값
- 요일과 월 주기값

센서 8개는 베어링 온도, 모터 온도, 수평·수직 진동, 오일압력, 부하, 회전수,
전력소비다.

### 7.2 실험 B: 과거 시계열 Feature

A에 다음 센서별 Feature를 추가한다.

- `lag1`, `lag3`, `lag7`
- `diff1`, `diff7`
- `median3`, `median7`
- `mad7`
- `slope7`

`median3`, `median7`, `mad7`, `slope7`은 현재일을 제외한 `shift(1)` 이후 과거값으로
계산한다. 당일 진단이므로 현재 센서값 자체는 사용할 수 있다.

Family별 과거 상태는 다음을 별도 추가한다.

- 전일 `affected`
- 과거 7일·30일 `affected` 횟수
- 마지막 `affected` 이후 경과일
- 전일 `severe`
- 과거 30일 `severe` 횟수

센서 시계열 효과와 과거 Family 상태 효과를 구분할 수 있도록 결과에 두 Feature
그룹의 중요도를 분리해 저장한다.

### 7.3 실험 C: 운전조건 보정 잔차

B에 건강상태 센서의 운전조건 보정 잔차를 추가한다. 건강상태 센서는 다음 5개다.

- `temp_bearing_degC`
- `temp_motor_degC`
- `vibration_h_mms`
- `vibration_v_mms`
- `oil_pressure_bar`

운전조건은 다음 3개다.

- `load_pct`
- `shaft_rpm`
- `power_consumption_kw`

학습 구간에서 건강상태 센서별 기대값 모델을 적합한다.

\[
\widehat{x}_{a,t}
=f(\mathrm{machine\_type},\mathrm{asset\_tag},
\mathrm{load},\mathrm{rpm},\mathrm{power},\mathrm{calendar}).
\]

잔차는 다음과 같다.

\[
r_{a,t}=x_{a,t}-\widehat{x}_{a,t}.
\]

기대값 모델은 센서별 `HistGradientBoostingRegressor`로 고정한다. 적합 데이터는
학습 구간에서 9개 Family의 `affected`가 모두 0인 완전 정상 장비일만 사용한다.
입력은 `machine_type`, `asset_tag`, 세 운전조건과 달력 Feature다. 범주형 Feature는
학습 파이프라인에서 인코딩하고 수치형 결측은 학습 중앙값으로 보정한다. 정상
장비일이 30건 미만인 장비는 별도 잔차 모델을 만들지 않고 전체 학습 정상행으로
적합한 공통 모델을 사용한다. 검증·테스트 행과 Target은 기대값 모델 적합에 사용하지
않는다.

잔차의 Robust Z-score와 `lag1`, `lag3`, `lag7`, `median7`, `mad7`, `slope7`을 만든다.
검증·테스트의 기대값은 학습 구간에서 적합한 모델만 사용해 계산한다.

## 8. 모델 비교

각 `Family x Target 종류 x Feature 실험군`에 대해 다음을 비교한다.

1. `DummyClassifier`: Family 양성률 기준선
2. `LogisticRegression(class_weight="balanced")`
3. `RandomForestClassifier(class_weight="balanced")`
4. `HistGradientBoostingClassifier`

첫 구현에서는 LSTM·GRU·Temporal CNN을 사용하지 않는다. 일반 모델에 과거
Feature를 제공했을 때 시간 정보가 실제로 성능을 높이는지 먼저 검증한다.

## 9. 모델·Feature 선택과 임계값

모델과 Feature 집합은 검증 구간의 Average Precision을 우선해 선택한다. 같은
Average Precision과 0.01 이내인 후보가 여러 개면 F1, Recall, Precision 순으로
선택한다. Dummy는 선택 대상이 아니다.

선택 모델에는 다음 임계값 정책을 각각 적용한다.

1. 검증 F1 최대 임계값
2. 검증 Precision 0.70 이상에서 Recall 최대
3. 검증 Precision 0.80 이상에서 Recall 최대
4. 위험점수 상위 10% 임계값

요구 Precision을 달성할 수 없으면 임계값을 임의로 낮추지 않고 `unavailable`로
기록한다. 테스트에서 임계값을 다시 고르지 않는다.

## 10. 평가 지표

Family별로 다음을 저장한다.

- 양성률과 support
- Accuracy
- Precision, Recall, F1
- ROC AUC
- Average Precision
- 상위 5·10·20% Precision, Recall, Lift
- TN, FP, FN, TP

다중라벨 전체 지표는 다음을 저장한다.

- Macro Precision, Recall, F1, Average Precision
- Micro Precision, Recall, F1, Average Precision
- Hamming loss
- 장비·날짜 단위 전체 라벨 일치율

Accuracy는 양성률이 낮은 Family에서 과대평가될 수 있으므로 모델 선택의 주요
지표로 사용하지 않는다.

## 11. 시계열 우선 진행 조건

순차 모델을 다음 단계로 검토하려면 B 또는 C가 A보다 다음 조건을 충족해야 한다.

1. 검증 Macro Average Precision이 0.02 이상 개선된다.
2. `affected` Target의 9개 Family 중 최소 5개에서 Average Precision이 개선된다.
3. 테스트 Macro Average Precision이 A보다 낮아지지 않는다.
4. 개선이 과거 Target 이력 하나에만 의존하지 않고 센서 시계열 또는 잔차 Feature에도
   나타난다.

조건을 충족하면 LSTM 또는 Temporal CNN을 별도 설계·구현한다. 충족하지 않으면
현재 합성 데이터에는 유효한 시간 패턴이 부족하다고 결론 내리고 순차 모델을 만들지
않는다.

## 12. 실행 인터페이스와 산출물

새 실행 파일은 기존 장비·부품 모델과 분리한다.

예상 명령은 다음과 같다.

```bash
python -m src.current_family_diagnosis \
  --feature-sets A B C \
  --targets affected severe \
  --scope overall
```

기본 출력 폴더는 다음과 같다.

```text
outputs/family_current/
├── metrics.csv
├── multilabel_metrics.csv
├── target_profile.csv
├── test_predictions.csv
├── feature_importance.csv
├── residual_baselines.csv
├── run_config.json
├── experiment_summary.md
└── models/
```

`experiment_summary.md`는 한글 세 줄 요약과 한 페이지 요약을 포함하고 다음 내용을
정리한다.

1. Family별 `affected`, `severe`, `all` 발생률
2. A·B·C 성능 비교
3. Family별 Precision·Recall·F1·Average Precision
4. Precision 0.70·0.80 정책의 사용 가능 여부
5. 기계 종류와 장비별 성능
6. 순차 모델 진행 조건 충족 여부
7. 합성 데이터와 대리 Target의 한계

## 13. 코드 구조

예상 파일 책임은 다음과 같다.

```text
src/
├── family_features.py
│   ├── 장비·날짜 집계
│   ├── affected·severe·all Target 생성
│   └── Family 과거 이력 생성
├── family_residual_features.py
│   ├── 학습 구간 기대 센서 모델 적합
│   └── 잔차·잔차 시계열 Feature 생성
├── family_diagnosis_experiments.py
│   ├── Family별 이진 모델 비교
│   ├── Feature 집합 선택
│   └── 다중라벨 결과 결합
├── family_diagnosis_report.py
│   └── 한글 요약 보고서 생성
└── current_family_diagnosis.py
    └── CLI 진입점
```

기존 `industrial_data.py`, `industrial_training.py`, `asset_anomaly_features.py`의 공통
날짜 분할, 모델 생성, 임계값과 지표 계산은 계약이 맞는 범위에서 재사용한다. 기존
모델의 동작을 바꾸기 위한 리팩터링은 하지 않는다.

## 14. 오류 처리와 재현성

- 같은 장비·날짜에서 센서값이 다르면 오류로 처리한다.
- Family 구성 부품 수가 날짜에 따라 달라지면 오류로 처리한다.
- `breakdown_flag`가 0·1이 아니면 오류로 처리한다.
- `criticality`가 A·B·C가 아니면 오류로 처리한다.
- `severe == 1`이고 `affected == 0`인 행이 있으면 오류로 처리한다.
- 학습 또는 검증 양성 표본이 부족하면 이유와 함께 해당 조합을 건너뛴다.
- 결측 과거 Feature는 미래값으로 채우지 않고 학습 파이프라인에서 보정한다.
- 난수 시드, 날짜 경계, 모델 파라미터, Feature 목록과 Target 정의를
  `run_config.json`에 저장한다.

## 15. 테스트 요구사항

최소한 다음을 자동 테스트한다.

1. Family별 구성 부품 수와 목록이 올바르게 집계되는지
2. 부품 하나 고장에서 `affected == 1`이 되는지
3. 부품 두 개 고장에서 `severe == 1`이 되는지
4. A등급 부품 하나 고장에서 `severe == 1`이 되는지
5. `severe <= affected` 관계가 모든 Target에서 유지되는지
6. 모든 구성품 고장 Target이 통계에는 포함되고 학습 Target에서는 제외되는지
7. 여러 Family가 동시에 양성인 다중라벨 행이 보존되는지
8. 과거 Family 상태가 현재 Target 계산 전에 `shift(1)` 되는지
9. 미래 센서와 미래 Target을 바꿔도 과거 날짜 Feature가 변하지 않는지
10. 잔차 기대값 모델이 학습 구간에서만 적합되는지
11. A·B·C Feature 집합에 Target 또는 사후 정보가 포함되지 않는지
12. 검증에서 선택한 모델과 임계값이 테스트에서 바뀌지 않는지
13. Precision 0.70·0.80 미달 정책이 `unavailable`로 저장되는지
14. 저장 모델을 다시 불러온 예측과 원래 예측이 일치하는지
15. 기존 전체 테스트가 그대로 통과하는지

## 16. 성공 및 중단 조건

구현 성공은 높은 예측 성능을 미리 가정하지 않는다. 다음이 모두 충족되면 구현이
완료된 것으로 본다.

1. 두 Target과 세 Feature 실험이 누수 없이 재현된다.
2. Family별·전체 지표와 발생률이 저장된다.
3. 정적 Feature 대비 시간 Feature의 개선 여부가 수치로 판정된다.
4. Precision 0.70·0.80 정책의 가능 여부가 숨김없이 보고된다.
5. 기존 테스트와 신규 테스트가 통과한다.

다음 조건이면 미래 예측으로 넘어가지 않는다.

- `affected`의 다수 Family에서 선택 모델 Average Precision이 양성률 기준선과 거의
  같음
- 시계열·잔차 Feature가 정적 Feature보다 안정적으로 개선되지 않음
- 높은 Precision 정책이 Recall을 사실상 0으로 만듦
- 성능이 특정 장비 하나 또는 과거 Target 이력에만 의존함

위 조건에 해당하면 현재 데이터에서 Family 당일 진단이 충분히 식별되지 않는다는
결론을 보고서에 남기고, 데이터 구조 또는 실제 정비 정답 확보를 다음 과제로 둔다.
