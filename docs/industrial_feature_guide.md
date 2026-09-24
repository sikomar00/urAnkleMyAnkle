# 산업 위험 모델 Feature·Target 가이드

## 1. 공통 정의

원본 한 행은 장비의 한 부품을 나타냅니다. 부품 중요도 가중치는

\[
w(A)=4,\qquad w(B)=2,\qquad w(C)=1
\]

이고, 장비 \(a\)의 날짜 \(t\) 고장점수는

\[
S_{a,t}=\sum_{i\in\mathrm{parts}(a)}
\mathrm{breakdown\_flag}_{a,i,t}\,w_i
\]

입니다. 같은 장비·날짜에 반복된 센서값이 서로 다르면 임의로 평균하지 않고 오류를
발생시킵니다.

## 2. 장비 단위 당일 탐지

- 그룹 키: `transaction_date + machine_type + asset_tag`
- Target:

\[
y^{\mathrm{asset,current}}_{\tau}(t)=\mathbf{1}\{S_{a,t}\ge\tau\},
\qquad \tau\in\{12,13,14\}.
\]

- Feature: `machine_type`, `asset_tag`, 센서 8개, 요일, 월 주기값
- 용도: 오늘 이미 위험한 장비를 즉시 찾는 탐지

`target_ge_12`처럼 이름과 실제 `>=` 연산을 일치시켰습니다.

## 3. 장비 단위 향후 7일 예측

- 그룹 키: Feature 이력은 `asset_tag`, 일별 집계는 장비·날짜
- 전체 위험 Target:

\[
y^{\mathrm{asset,any}}_{\tau,7}(t)
=\max_{1\le k\le 7}\mathbf{1}\{S_{a,t+k}\ge\tau\}.
\]

- 신규 위험 Target: 위 Target을 사용하되 \(S_{a,t}<\tau\)인 오늘 정상 장비만 포함
- 기본값: `--horizon 7 --risk-definition new`

센서 \(x\)마다 날짜 \(t\)에서 알 수 있는 다음 Feature를 사용합니다.

\[
x_t,\ x_{t-1},\ x_{t-3},\ x_{t-7},\
\overline{x}_{t,7},\ \sigma_{t,7},\ x_t-x_{t-1}.
\]

`current`, `mean7`, `std7`에는 예측 시점인 날짜 \(t\)가 포함됩니다. 반면 고장점수
이력은 당일 결과를 Feature로 유출하지 않도록 반드시 `shift(1)` 후 계산합니다.

- `failure_points_lag1`
- `failure_points_mean7`, `failure_points_max7`
- `risk_event_count_30d`
- `days_since_last_risk`

## 4. 부품 단위 당일 탐지

- 그룹 키: `transaction_date + asset_tag + part_no`
- Target:

\[
y^{\mathrm{part,current}}_{a,i,t}
=\mathrm{breakdown\_flag}_{a,i,t}.
\]

- Feature: `machine_type`, `asset_tag`, `part_no`, `criticality`, `plant_code`,
  센서 8개, 요일, 월 주기값

같은 장비의 여러 부품은 센서값이 같아도 정답이 다를 수 있으므로 `part_no`와
`criticality`를 포함합니다.

## 5. 부품 단위 향후 7일 예측

- Target:

\[
y^{\mathrm{part,7d}}_{a,i,t}
=\max_{1\le k\le7}\mathrm{breakdown\_flag}_{a,i,t+k}.
\]

- 모집단: 날짜 \(t\)에 `breakdown_flag == 0`인 부품
- 센서 이력 그룹: `asset_tag`
- 고장 이력 그룹: `asset_tag + part_no`

센서 Feature는 장비 미래 모델과 같은 현재값·lag·7일 평균/표준편차·변화량을
사용합니다. 부품 고장 이력은 `breakdown_lag1`, `breakdown_count_7d`,
`breakdown_count_30d`, `days_since_last_breakdown`이며 모두 `shift(1)` 이후 값입니다.

## 6. 시간 분할과 누수 방지

미래 Target의 마지막 관측일을 `label_end_date`에 저장합니다. 검증 시작일을 \(V\),
테스트 시작일을 \(T\)라 하면

\[
\mathrm{Train}=\{t:\mathrm{label\_end\_date}<V\},
\]

\[
\mathrm{Validation}=\{t:t\ge V,\ \mathrm{label\_end\_date}<T\},
\qquad
\mathrm{Test}=\{t:t\ge T\}.
\]

따라서 6월 말 기준일의 7일 Target이 7월 테스트 기간을 참조하면 검증에서 제외됩니다.
또한 미래 7개 달력 날짜 중 하루라도 빠지면 음성으로 간주하지 않고 그 기준일 자체를
제외합니다.

## 7. 해석할 때 주의할 점

1. 현재 CSV는 교육용 합성 데이터입니다. 실제 센서의 열화 패턴과 고장 전조가 약하거나
   무작위에 가까우면 복잡한 모델도 높은 성능을 낼 수 없습니다.
2. `asset_tag`, `part_no`는 기존 대상을 기억하는 데 도움을 주지만 새 장비·새 부품에 대한
   일반화 능력을 증명하지 않습니다.
3. 클래스가 불균형하므로 Accuracy만 보면 안 됩니다. Average Precision, Precision,
   Recall, F1, 상위 10% Recall/Lift를 함께 봐야 합니다.
4. `breakdown_flag`, 미래 Target, 정비 후에 기록되는 값은 입력 Feature로 쓰면 안 됩니다.
5. 성능을 높이기 전에 센서가 실제로 고장 이전에 변하는지, 동일 Feature에 서로 다른
   정답이 얼마나 섞이는지 확인해야 합니다.
