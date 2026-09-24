# 산업 위험 모델 코드 가이드

## 1. 전체 실행 흐름

```text
원본 CSV
  → Feature/Target 생성
  → 시간 분할
  → 세 모델 비교
  → 검증 데이터로 모델·임계값 선택
  → 테스트 평가와 결과 저장
  → eda.py로 네 과제 결과 비교
```

학습과 테스트의 호출 관계는 다음과 같습니다.

```text
*_model.py
  ├─ industrial_data.load_industrial_data()
  ├─ asset_features.prepare_*() 또는 part_features.prepare_*()
  └─ industrial_cli.run_tasks()
       └─ industrial_training.run_experiment_suite()
```

## 2. 실행 파일

1. `current_asset_model.py`
   - 목적: 장비의 당일 고장점수가 기준 이상인지 탐지
   - 기본 Target: `target_ge_12`, `target_ge_13`, `target_ge_14`
   - 실행: `python src/current_asset_model.py`
   - 선택 인자: `--score-threshold`, `--scope`, 날짜 분할, 임계값 정책

2. `forecast_asset_model.py`
   - 목적: 오늘까지의 정보로 향후 7일 장비 위험 예측
   - 기본 Target: 현재 정상 장비의 `target_new_risk_7d_ge_*`
   - 실행: `python src/forecast_asset_model.py --horizon 7 --risk-definition new`
   - 비교 실행: `--risk-definition any`

3. `current_part_model.py`
   - 목적: 부품의 당일 `breakdown_flag` 탐지
   - 실행: `python src/current_part_model.py`

4. `forecast_part_model.py`
   - 목적: 오늘 정상인 부품의 향후 7일 고장 예측
   - 실행: `python src/forecast_part_model.py --horizon 7`

5. `eda.py`
   - 목적: 네 출력의 선택 모델 테스트 지표 비교
   - 실행: `python src/eda.py`
   - 주의: 학습은 하지 않으므로 먼저 네 모델 명령을 실행해야 함

## 3. 공통 모듈

### `industrial_data.py`

- 공통 컬럼명과 센서 목록을 정의합니다.
- `load_industrial_data()`가 CSV와 필수 컬럼을 검사합니다.
- `PreparedTask`가 데이터, Feature, Target, grain/mode 메타데이터를 묶습니다.
- `split_by_date()`가 `label_end_date`를 사용해 미래 라벨의 경계 누수를 막습니다.
- `iter_scopes()`가 overall, 기계 종류별, 장비별 평가 범위를 만듭니다.

### `asset_features.py`

- `build_asset_daily()`가 부품 행을 장비·날짜 한 행으로 집계합니다.
- `prepare_asset_current()`가 점수 기준별 당일 Target을 만듭니다.
- `prepare_asset_forecast()`가 향후 기간 Target과 센서·고장점수 이력을 만듭니다.

### `part_features.py`

- `prepare_part_current()`가 부품 식별정보를 포함한 당일 과제를 만듭니다.
- `prepare_part_forecast()`가 장비 센서 이력과 부품별 고장 이력을 결합합니다.

### `industrial_training.py`

- `build_model()`이 범주형 One-Hot 인코딩과 수치 결측 대치를 포함한 파이프라인을 만듭니다.
- 비교 모델은 Logistic Regression, Random Forest, HistGradientBoosting입니다.
- `run_experiment_suite()`가 검증 Average Precision으로 모델을 선택합니다.
- `choose_thresholds()`가 F1, 최소 Precision, 상위 비율 정책을 계산합니다.
- `ranking_metrics()`가 상위 5/10/20% Precision·Recall·Lift를 계산합니다.

### `industrial_cli.py`

- 네 실행 파일의 공통 인자를 한곳에서 정의합니다.
- 생략된 점수 기준을 12/13/14로, 생략된 임계값 정책을 세 정책 전체로 정규화합니다.
- `run_tasks()`가 명령행 인자를 공통 학습기로 전달합니다.

## 4. 데이터 누수 방지

1. `label_end_date`
   - 2024-06-28의 7일 Target은 7월 데이터를 봅니다.
   - 이를 6월 검증 행으로 넣으면 테스트 기간 정보가 검증에 섞이므로 제외합니다.

2. `shift(1)`
   - 날짜 \(t\)의 고장 결과는 예측 시점에 이미 아는 센서값과 달리 정답에 해당합니다.
   - 고장 이력은 먼저 한 칸 이동한 뒤 rolling하여 \(t-1\)까지의 값만 사용합니다.

3. 미래 날짜 연속성
   - 7개 행이 있어도 중간 하루가 빠지면 정확한 7일 구간이 아닙니다.
   - 각 미래 날짜가 정확히 \(t+1,\ldots,t+7\)인지 검사하고 아니면 제외합니다.

4. 전처리 fitting
   - 인코더와 결측 대치는 학습 구간에만 fit합니다.
   - 검증에서 처음 나타난 범주는 `handle_unknown="ignore"`로 처리합니다.

## 5. 결과 파일 읽기

### `metrics.csv`

한 행은 과제·scope·모델·데이터 분할·임계값 정책 조합입니다.

- `selected_model`: 검증 AP가 가장 높아 선택된 모델인지 여부
- `split`: `validation` 또는 `test`
- `threshold_policy`: `f1`, `min_precision`, `top_fraction`
- `average_precision`: 불균형 데이터에서 순위 품질을 보는 핵심 지표
- `precision`, `recall`, `f1`: 선택 임계값을 적용한 분류 지표
- `precision_at_10pct`, `recall_at_10pct`, `lift_at_10pct`: 상위 10% 점검 전략 성능

예를 들어 Recall은 높지만 Precision이 0.10이면 고장을 많이 찾는 대신 경보 10건 중
약 1건만 실제 양성이라는 뜻입니다. 운영 점검 비용에 맞춰 정책을 골라야 합니다.

### `test_predictions.csv`

선택 모델의 테스트 행별 `risk_score`, `probability_cutoff`, `prediction`을 저장합니다.
어떤 장비/부품을 점검할지 확인할 때 사용합니다.

### `feature_importance.csv`

선택 모델의 permutation importance입니다. 값이 낮다고 인과적으로 무관하다고 단정할
수 없고, 서로 중복된 Feature가 있으면 중요도가 나뉠 수 있습니다.

### `run_config.json`과 `models/`

`run_config.json`은 모델 목록, 날짜 분할, 임계값 정책, 난수 시드를 기록합니다.
`models/`의 joblib 파일에는 선택 파이프라인과 Feature/Target 메타데이터가 들어 있습니다.

## 6. 기존 파일 호환성

- `current_state_model.py` → `current_part_model.py`, 기본 출력 `outputs/current_state`
- `timeseries_model.py` → `forecast_part_model.py`, 기본 출력 `outputs/timeseries`
- `industrial_features.py` → 기존 import가 깨지지 않게 공통 데이터 함수와 부품 준비 함수 재노출

새 코드는 네 명시적 실행 파일을 쓰는 것이 좋고, 기존 자동화가 있을 때만 호환 파일을
사용하면 됩니다.
