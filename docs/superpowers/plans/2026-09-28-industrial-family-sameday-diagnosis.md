# 산업 장비 당일 Family 이상·심각 진단 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 장비·날짜 센서로 9개 부품 Family의 당일 `affected`와 `severe`를 다중라벨로 진단하고, 당일값·과거 시계열·운전조건 보정 잔차의 성능을 같은 시간 분할에서 비교한다.

**Architecture:** 원본 부품 행을 `장비·날짜·Family` 긴 형식으로 집계하고 Family별 독립 이진 모델을 학습한 뒤 장비·날짜 다중라벨 결과로 결합한다. Feature 생성, 운전조건 잔차, 실험 실행, 보고서와 CLI를 분리하며 기존 장비·부품 모델은 변경하지 않는다. LSTM은 구현하지 않고 시계열 Feature가 정적 기준선을 개선할 때만 후속 과제로 검토한다.

**Tech Stack:** Python 3.11, pandas, NumPy, scikit-learn 1.6.1, joblib, pytest

**Spec:** `docs/superpowers/specs/2026-09-28-industrial-family-sameday-diagnosis-design.md`

## 세 줄 요약

1. `하나 이상 고장=affected`, `2개 이상 또는 A등급 고장=severe`를 9개 Family별로 만든다.
2. 정적 A, 과거 시계열 B, 운전조건 잔차 C를 Logistic·Random Forest·HistGradientBoosting으로 비교한다.
3. Family 당일 진단이 검증·테스트에서 성립할 때만 미래 예측 또는 순차 모델로 진행한다.

## 한 페이지 요약

구현은 7개 독립 작업으로 나눈다. 먼저 `family_features.py`가 원본 219,000개 부품
행을 98,550개 `장비·날짜·Family` 행으로 집계하고 `affected`, `severe`, `all_failed`
Target을 만든다. 다음으로 센서 과거값과 Family 과거 상태를 미래 누수 없이 추가해
정적 Feature A와 시계열 Feature B를 구성한다.

`family_residual_features.py`는 학습 구간의 모든 Family 정상 장비일만 사용해 건강상태
센서 5개의 기대값을 적합한다. 장비별 정상 표본이 30건 이상이면 장비별 모델을,
부족하거나 처음 보는 장비이면 전체 정상행 모델을 사용한다. 실제값과 기대값의 잔차,
잔차 Robust Z-score와 과거 잔차 통계를 추가해 Feature C를 만든다.

실험 엔진은 Family와 Target별 독립 이진 문제를 Logistic Regression, Random Forest,
HistGradientBoosting으로 학습한다. 모델과 Feature 집합은 검증 Average Precision으로
선택하고, 검증 F1 최대·Precision 0.70·Precision 0.80·상위 10%의 네 임계값 정책을
테스트에 고정 적용한다. Accuracy는 보조 지표로만 저장한다.

선택 예측은 다중라벨 행렬로 결합해 Macro·Micro Precision, Recall, F1, Average
Precision, Hamming loss와 전체 라벨 일치율을 계산한다. 최종 CLI는 결과 CSV,
저장 모델, 실행 설정과 한글 보고서를 `outputs/family_current/`에 생성한다. 전체
테스트와 실제 원본 데이터 실행까지 완료한 뒤, B·C가 A를 개선했는지와 순차 모델
진행 조건을 결과 보고서에 명시한다.

## Global Constraints

- 문서, CLI 설명, 오류 메시지와 자동 보고서는 한글로 작성한다.
- 데이터 단위는 `transaction_date + machine_type + asset_tag + part_family` 긴 형식이다.
- Family 목록은 설계 문서의 9개로 고정하고 구성 변화는 오류로 처리한다.
- `affected = failed_parts >= 1`, `severe = failed_parts >= 2 or failed_A_part`, `all_failed = failed_parts == total_parts`를 사용한다.
- `all_failed`는 발생률 통계에만 포함하고 학습 Target으로 사용하지 않는다.
- 학습은 2024-01-01 이전, 검증은 2024-01-01~2024-06-30, 테스트는 2024-07-01 이후다.
- 과거 센서와 과거 Family 상태는 달력일 기준이며 Target 이력은 반드시 `shift(1)` 후 계산한다.
- 잔차 기대값 모델과 Robust 기준은 학습 구간의 모든 Family 정상 장비일만 사용한다.
- 모델·Feature·임계값 선택은 검증 데이터까지만 사용하고 테스트는 최종 평가에만 사용한다.
- 모델 선택 1순위는 Average Precision이며 0.01 이내 후보는 F1, Recall, Precision 순으로 정한다.
- Precision 0.70·0.80을 달성하지 못하는 정책은 임계값을 낮추지 않고 `unavailable`로 기록한다.
- `breakdown_flag`, 당일 Family Target과 미래 정보는 Feature에 포함하지 않는다.
- 기존 실행 파일, 기존 모델 출력과 기존 테스트 동작을 보존한다.
- 높은 성능을 구현 완료 조건으로 두지 않으며 실패 결과도 숨기지 않는다.

## Review Focus

- 날짜별 Family 구성 부품이 누락되거나 추가되면 조용히 비율을 바꾸지 않고 Task 1에서 오류로 검증한다.
- A등급 한 개 고장과 B·C등급 두 개 고장이 모두 `severe`가 되는지 Task 1에서 검증한다.
- 날짜가 빠진 장비에서 `lag1`이 직전 관측행이 아니라 전날 달력일인지 Task 2에서 검증한다.
- 검증·테스트의 극단값을 바꿔도 잔차 모델과 기준값이 변하지 않는지 Task 3에서 검증한다.
- Family Target이 동일하거나 양성 표본이 부족할 때 중복 학습·모델 선택 오류 없이 상태를 남기는지 Task 4에서 검증한다.

---

### Task 1: Family 당일 집계와 Target 계약

**Files:**
- Create: `src/family_features.py`
- Create: `tests/test_family_features.py`

**Interfaces:**
- Consumes: `industrial_data.py`의 공통 컬럼 상수와 원본 부품 DataFrame
- Produces: `FAMILY_NAMES: tuple[str, ...]`
- Produces: `FAMILY_TARGETS = ("affected", "severe")`
- Produces: `build_family_daily(raw: pd.DataFrame) -> pd.DataFrame`
- Produces: `build_family_target_profile(family_daily: pd.DataFrame) -> pd.DataFrame`

- [ ] **Step 1: 9개 Family와 20개 부품을 갖는 최소 장비·날짜 fixture를 작성한다**

`tests/test_family_features.py`에 날짜 3일, 장비 1대, 설계 문서의 Family·부품 수·criticality를 명시하는 `raw_family_rows()` fixture를 만든다. 같은 장비·날짜의 센서 8개는 동일하게 고정한다.

- [ ] **Step 2: Target 경계와 다중 Family 양성 테스트를 작성한다**

```python
def test_build_family_daily_creates_affected_severe_and_all(raw_family_rows):
    result = build_family_daily(raw_family_rows)
    bearing = result.query("part_family == 'Bearing'").sort_values("transaction_date")
    assert bearing["affected"].tolist() == [0, 1, 1]
    assert bearing["severe"].tolist() == [0, 1, 1]
    assert bearing["all_failed"].tolist() == [0, 0, 1]
    assert (result["severe"] <= result["affected"]).all()


def test_one_a_or_two_non_a_parts_are_severe(raw_family_rows):
    result = build_family_daily(raw_family_rows)
    assert select_day(result, "Electrical", "2023-01-02")["severe"] == 1
    assert select_day(result, "Filter", "2023-01-03")["severe"] == 1


def test_multiple_families_remain_positive_on_same_asset_day(raw_family_rows):
    result = build_family_daily(raw_family_rows)
    day = result.query("transaction_date == '2023-01-03'")
    assert set(day.loc[day.affected.eq(1), "part_family"]) == {
        "Bearing", "Filter"
    }
```

- [ ] **Step 3: 입력 계약 오류 테스트를 작성한다**

`breakdown_flag`가 0·1이 아닌 경우, `criticality`가 A·B·C가 아닌 경우, 같은 날짜의 센서 충돌, 날짜별 Family 구성 부품 수 변화와 예상하지 않은 Family가 각각 `ValueError`를 발생시키는지 테스트한다.

- [ ] **Step 4: 신규 모듈 부재로 테스트가 실패하는지 확인한다**

Run: `/Users/kodohyeon/Documents/project_LS/venv/bin/python -m pytest tests/test_family_features.py -v`

Expected: `ModuleNotFoundError: No module named 'src.family_features'`

- [ ] **Step 5: Family 긴 형식 집계를 구현한다**

`build_family_daily()`는 센서 일관성과 `transaction_date + asset_tag + part_no` 유일성을 검사하고, 장비·날짜·Family별 `total_parts`, `failed_parts`, `failed_a_parts`, `affected`, `severe`, `all_failed`를 만든다. 센서·기계 정보는 검증 후 한 번만 보존하고 달력 Feature를 추가한다.

- [ ] **Step 6: Family 발생률 프로필을 구현한다**

`build_family_target_profile()`은 Family별 전체·학습·검증·테스트 행 수, `affected`, `severe`, `all_failed` 건수와 비율, 구성 부품 수를 반환한다.

- [ ] **Step 7: Family Target 테스트를 통과시킨다**

Run: `/Users/kodohyeon/Documents/project_LS/venv/bin/python -m pytest tests/test_family_features.py -v`

Expected: 모든 테스트 PASS

- [ ] **Step 8: 커밋한다**

```bash
git add src/family_features.py tests/test_family_features.py
git commit -m "feat: add same-day family targets"
```

### Task 2: 과거 센서·Family Feature와 A·B 구성

**Files:**
- Modify: `src/family_features.py`
- Modify: `tests/test_family_features.py`

**Interfaces:**
- Consumes: Task 1의 `build_family_daily()`와 `asset_anomaly_features.build_sensor_history_features()`
- Produces: `FAMILY_HISTORY_FEATURES: tuple[str, ...]`
- Produces: `build_family_history_features(family_daily: pd.DataFrame) -> pd.DataFrame`
- Produces: `FamilyDiagnosisFeatures(frame, feature_sets, residual_transformer)` dataclass
- Produces: `prepare_family_features(raw, validation_start="2024-01-01", feature_sets=("A", "B")) -> FamilyDiagnosisFeatures`

- [ ] **Step 1: 달력 lag와 미래 비누수 테스트를 작성한다**

```python
def test_family_lag_uses_previous_calendar_day(raw_family_rows):
    daily = build_family_daily(raw_family_rows)
    daily = daily.loc[~daily.transaction_date.eq("2023-01-02")]
    result = build_family_history_features(daily)
    row = select_day(result, "Bearing", "2023-01-03")
    assert pd.isna(row["affected_lag1"])
    assert pd.isna(row["load_pct_lag1"])


def test_future_target_change_does_not_change_past_history(raw_family_rows):
    before = build_family_history_features(build_family_daily(raw_family_rows))
    changed = raw_family_rows.copy()
    changed.loc[changed.transaction_date.eq("2023-01-03"), "breakdown_flag"] = 1
    after = build_family_history_features(build_family_daily(changed))
    columns = ["affected_lag1", "affected_count_7d", "severe_count_30d"]
    assert_frame_values_equal(before, after, date="2023-01-02", columns=columns)
```

- [ ] **Step 2: Feature 집합과 금지 열 테스트를 작성한다**

`prepare_family_features()`의 A에는 현재 센서·기계·장비·달력만, B에는 A와 센서 과거 Feature·Family 과거 Feature만 들어가는지 검사한다. `breakdown_flag`, `affected`, `severe`, `all_failed`, `failed_parts`, `failed_a_parts`가 Feature에 없음을 단언한다.

- [ ] **Step 3: 공개 함수 부재로 테스트가 실패하는지 확인한다**

Run: `/Users/kodohyeon/Documents/project_LS/venv/bin/python -m pytest tests/test_family_features.py -v`

Expected: `build_family_history_features` 또는 `prepare_family_features` import 실패

- [ ] **Step 4: Family 과거 상태를 달력일 기준으로 구현한다**

`asset_tag + part_family`별 달력 인덱스에서 `shift(1)` 후 `affected_lag1`, `affected_count_7d`, `affected_count_30d`, `days_since_last_affected`, `severe_lag1`, `severe_count_30d`를 계산한다.

- [ ] **Step 5: 센서 과거 Feature를 장비·날짜 기준으로 한 번 계산해 병합한다**

Family 반복 행에서 센서를 먼저 `asset_tag + transaction_date` 한 행으로 축소하고 기존 `build_sensor_history_features()`를 호출한다. 병합은 `many_to_one` 검증을 사용하며 Family Target과 센서 이력이 서로 섞이지 않게 한다.

- [ ] **Step 6: A·B Feature 준비 dataclass를 구현한다**

```python
@dataclass(frozen=True)
class FamilyDiagnosisFeatures:
    frame: pd.DataFrame
    feature_sets: dict[str, tuple[str, ...]]
    residual_transformer: object | None = None
```

`prepare_family_features()`는 요청 Feature가 A·B 외 값이면 `ValueError`를 발생시키고, 날짜·Family 정렬이 고정된 Frame을 반환한다.

- [ ] **Step 7: 과거 Feature 테스트를 통과시킨다**

Run: `/Users/kodohyeon/Documents/project_LS/venv/bin/python -m pytest tests/test_family_features.py -v`

Expected: 모든 테스트 PASS

- [ ] **Step 8: 커밋한다**

```bash
git add src/family_features.py tests/test_family_features.py
git commit -m "feat: add family history feature sets"
```

### Task 3: 운전조건 보정 잔차와 Feature C

**Files:**
- Create: `src/family_residual_features.py`
- Create: `tests/test_family_residual_features.py`
- Modify: `src/family_features.py`
- Modify: `tests/test_family_features.py`

**Interfaces:**
- Consumes: Task 2의 Family 긴 형식 Frame과 학습 날짜 경계
- Produces: `HEALTH_SENSOR_COLUMNS`, `OPERATING_CONTEXT_COLUMNS`, `RESIDUAL_FEATURES`
- Produces: `OperatingResidualTransformer.fit(train: pd.DataFrame) -> OperatingResidualTransformer`
- Produces: `OperatingResidualTransformer.transform(frame: pd.DataFrame) -> pd.DataFrame`
- Produces: `OperatingResidualTransformer.artifact_table() -> pd.DataFrame`
- Extends: `prepare_family_features(..., feature_sets=("A", "B", "C"))`

- [ ] **Step 1: 완전 정상 학습행과 장비별 모델 선택 테스트를 작성한다**

`fit()`이 같은 장비·날짜의 9개 Family 중 하나라도 `affected == 1`이면 그 날짜를 정상 적합에서 제외하는지 검사한다. 정상 장비일 30건 이상인 장비는 `selected_scope="asset_tag"`, 부족한 장비는 `selected_scope="global"`로 기록되는지 검사한다.

- [ ] **Step 2: 학습 구간 외 데이터 비누수 테스트를 작성한다**

```python
def test_validation_and_test_values_do_not_change_fitted_residual_models(family_rows):
    train = family_rows.loc[family_rows.transaction_date.lt("2024-01-01")]
    original = OperatingResidualTransformer(min_normal_rows=30).fit(train)
    changed = family_rows.copy()
    changed.loc[changed.transaction_date.ge("2024-01-01"), HEALTH_SENSOR_COLUMNS] = 9999
    refit = OperatingResidualTransformer(min_normal_rows=30).fit(
        changed.loc[changed.transaction_date.lt("2024-01-01")]
    )
    pd.testing.assert_frame_equal(original.artifact_table(), refit.artifact_table())
```

- [ ] **Step 3: 처음 보는 장비와 MAD 0 fallback 테스트를 작성한다**

처음 보는 `asset_tag`는 global 기대값 모델과 global 잔차 기준을 사용하고, 잔차 MAD가 0이면 표준편차, 표준편차도 0이면 상수 0 Z-score로 처리되는지 검사한다.

- [ ] **Step 4: 신규 모듈 부재로 테스트가 실패하는지 확인한다**

Run: `/Users/kodohyeon/Documents/project_LS/venv/bin/python -m pytest tests/test_family_residual_features.py -v`

Expected: `ModuleNotFoundError`

- [ ] **Step 5: 직렬화 가능한 잔차 변환기를 구현한다**

센서별 global `HistGradientBoostingRegressor`와 정상 표본 30건 이상인 장비별 모델을 적합한다. global 모델 입력은 `machine_type`, `asset_tag`, 운전조건 3개와 달력 Feature고, 장비별 모델 입력은 운전조건 3개와 달력 Feature다. 건강상태 센서마다 실제값·예측값·잔차를 생성한다.

- [ ] **Step 6: 잔차 Robust 기준과 잔차 이력을 구현한다**

학습 정상 잔차만으로 장비별→global MAD 기준을 만들고 잔차 Robust Z-score를 계산한다. 잔차와 잔차 Z-score의 `lag1`, `lag3`, `lag7`, `median7`, `mad7`, `slope7`은 달력일 기준으로 생성한다.

- [ ] **Step 7: Feature C를 준비 흐름에 연결한다**

`prepare_family_features()`는 전체 데이터를 먼저 날짜 분할하지 않고, `validation_start` 이전 Family Frame만 변환기 `fit()`에 전달한 뒤 전체 Frame에 `transform()`한다. C는 B와 잔차 Feature를 포함한다.

- [ ] **Step 8: 잔차 및 Feature C 테스트를 통과시킨다**

Run: `/Users/kodohyeon/Documents/project_LS/venv/bin/python -m pytest tests/test_family_residual_features.py tests/test_family_features.py -v`

Expected: 모든 테스트 PASS

- [ ] **Step 9: 커밋한다**

```bash
git add src/family_residual_features.py src/family_features.py tests/test_family_residual_features.py tests/test_family_features.py
git commit -m "feat: add operating-adjusted family residuals"
```

### Task 4: Family별 모델 비교·선택·임계값

**Files:**
- Create: `src/family_diagnosis_experiments.py`
- Create: `tests/test_family_diagnosis_experiments.py`

**Interfaces:**
- Consumes: `FamilyDiagnosisFeatures`, `industrial_training.build_model()`, `classification_metrics()`, `ranking_metrics()`
- Produces: `FamilyExperimentResult(metrics, predictions, importances, models, multilabel_input)` dataclass
- Produces: `select_family_model(validation_metrics: pd.DataFrame, tolerance=0.01) -> str`
- Produces: `select_family_feature_set(validation_metrics: pd.DataFrame, tolerance=0.01) -> str`
- Produces: `choose_family_thresholds(y_true, scores) -> list[ThresholdSelection]`
- Produces: `run_family_experiments(prepared, output_dir, max_iter=100, random_state=42) -> FamilyExperimentResult`

- [ ] **Step 1: Average Precision 우선 모델·Feature 선택 테스트를 작성한다**

Average Precision이 가장 높은 후보를 선택하고 0.01 이내 동률에서 F1→Recall→Precision 순서가 적용되는지 각각 테스트한다. 테스트 행의 지표를 섞어도 선택 결과가 변하지 않는지 검사한다.

- [ ] **Step 2: Precision 정책 테스트를 작성한다**

검증 점수에서 F1 최대, Precision 0.70, Precision 0.80, 상위 10% 임계값을 만들고, 목표 Precision을 만족하지 못할 때 해당 정책이 `unavailable`인지 검사한다.

- [ ] **Step 3: 동일 Target과 양성 부족 상태 테스트를 작성한다**

Bearing의 `affected`와 `severe`가 동일하면 하나의 모델 결과를 재사용하고 `identical_target`을 기록하는지 검사한다. 학습 또는 검증 양성이 10건 미만이거나 단일 클래스면 `insufficient_positive_rows`를 기록하고 학습하지 않는지 검사한다.

- [ ] **Step 4: 검증 임계값의 테스트 고정과 저장 모델 재현 테스트를 작성한다**

테스트 Target을 바꿔도 모델·Feature·임계값 선택이 변하지 않으며, joblib로 저장 후 다시 불러온 확률이 원래 확률과 일치하는지 검사한다.

- [ ] **Step 5: 신규 모듈 부재로 테스트가 실패하는지 확인한다**

Run: `/Users/kodohyeon/Documents/project_LS/venv/bin/python -m pytest tests/test_family_diagnosis_experiments.py -v`

Expected: `ModuleNotFoundError`

- [ ] **Step 6: 선택 함수와 네 임계값 정책을 구현한다**

선택 함수는 검증행만 허용하고 Average Precision→F1→Recall→Precision→후보 선언 순서로 결정한다. Precision 정책 이름은 `f1`, `min_precision_70`, `min_precision_80`, `top_10pct`로 고정한다.

- [ ] **Step 7: Family별 A·B·C 모델 비교를 구현한다**

각 Family와 Target에서 A·B·C별 Logistic Regression, Random Forest, HistGradientBoosting을 학습한다. 모델별 검증 지표로 대표 모델을 정하고 대표 모델끼리 Feature 집합을 선택한다. Dummy 양성률은 `prior_average_precision`으로 저장하되 선택 대상에서 제외한다.

- [ ] **Step 8: 테스트 평가·예측·중요도·모델 저장을 구현한다**

선택되지 않은 후보도 검증 지표를 남긴다. 선택 조합은 네 임계값 정책으로 테스트 지표와 예측을 저장하고 permutation importance를 계산한다. 저장 모델에는 pipeline, Family, Target, Feature 집합, Feature 목록, 임계값과 잔차 변환기를 포함한다.

- [ ] **Step 9: 실험 엔진 테스트를 통과시킨다**

Run: `/Users/kodohyeon/Documents/project_LS/venv/bin/python -m pytest tests/test_family_diagnosis_experiments.py -v`

Expected: 모든 테스트 PASS

- [ ] **Step 10: 커밋한다**

```bash
git add src/family_diagnosis_experiments.py tests/test_family_diagnosis_experiments.py
git commit -m "feat: compare same-day family diagnosis models"
```

### Task 5: 다중라벨 지표와 한글 보고서

**Files:**
- Create: `src/family_diagnosis_report.py`
- Create: `tests/test_family_diagnosis_report.py`
- Modify: `src/family_diagnosis_experiments.py`
- Modify: `tests/test_family_diagnosis_experiments.py`

**Interfaces:**
- Consumes: Task 4의 선택 예측과 Task 1의 Target profile
- Produces: `build_multilabel_metrics(predictions: pd.DataFrame) -> pd.DataFrame`
- Produces: `evaluate_sequence_gate(metrics: pd.DataFrame) -> dict[str, object]`
- Produces: `render_family_summary(metrics, multilabel_metrics, profile, sequence_gate) -> str`

- [ ] **Step 1: 다중라벨 행렬 결합 테스트를 작성한다**

장비·날짜·Target·정책별 9개 Family 실제값과 예측값을 pivot하고 Macro·Micro Precision, Recall, F1, Average Precision, Hamming loss와 전체 라벨 일치율이 수작업 기대값과 일치하는지 테스트한다.

- [ ] **Step 2: 시계열 진행 조건 테스트를 작성한다**

검증 Macro AP가 A 대비 0.02 이상, affected Family 5개 이상 개선, 테스트 Macro AP 비하락, 센서 시계열·잔차 importance가 0보다 큰 경우만 `eligible=True`인지 검사한다. 하나라도 충족하지 않으면 사유 목록을 반환하는지 검사한다.

- [ ] **Step 3: 보고서 필수 문구 테스트를 작성한다**

보고서에 세 줄 요약, 한 페이지 요약, Family별 발생률, A·B·C 비교, Precision 0.70·0.80 가용성, 기계·장비별 결과, 순차 모델 판정과 합성 데이터 제한이 포함되는지 검사한다.

- [ ] **Step 4: 신규 모듈 부재로 테스트가 실패하는지 확인한다**

Run: `/Users/kodohyeon/Documents/project_LS/venv/bin/python -m pytest tests/test_family_diagnosis_report.py -v`

Expected: `ModuleNotFoundError`

- [ ] **Step 5: 다중라벨 지표와 순차 모델 gate를 구현한다**

평가 불가능한 Family는 macro 평균에서 제외하되 제외 목록을 별도 열에 기록한다. Average Precision은 확률점수, Hamming loss와 전체 일치율은 검증에서 고정된 임계값 예측을 사용한다.

- [ ] **Step 6: 한글 보고서 렌더러를 구현한다**

성능이 낮거나 Precision 정책이 불가능하면 그대로 명시하며 장비 고장 확정이나 부품 교체 표현을 사용하지 않는다.

- [ ] **Step 7: 실험 엔진에서 결과 파일을 저장한다**

`metrics.csv`, `multilabel_metrics.csv`, `target_profile.csv`, `test_predictions.csv`, `feature_importance.csv`, `residual_baselines.csv`, `run_config.json`, `experiment_summary.md`와 `models/`를 원자적으로 쓸 수 있는 기존 패턴으로 저장한다.

- [ ] **Step 8: 보고서·실험 테스트를 통과시킨다**

Run: `/Users/kodohyeon/Documents/project_LS/venv/bin/python -m pytest tests/test_family_diagnosis_report.py tests/test_family_diagnosis_experiments.py -v`

Expected: 모든 테스트 PASS

- [ ] **Step 9: 커밋한다**

```bash
git add src/family_diagnosis_report.py src/family_diagnosis_experiments.py tests/test_family_diagnosis_report.py tests/test_family_diagnosis_experiments.py
git commit -m "feat: report multilabel family diagnosis results"
```

### Task 6: 당일 Family 진단 CLI와 사용 가이드

**Files:**
- Create: `src/current_family_diagnosis.py`
- Modify: `src/RUN_INDUSTRIAL.md`
- Modify: `docs/industrial_code_guide.md`
- Modify: `tests/test_industrial_entrypoints.py`

**Interfaces:**
- Consumes: `load_industrial_data()`, `prepare_family_features()`, `run_family_experiments()`
- Produces: `build_parser() -> argparse.ArgumentParser`
- Produces: `main() -> None`

- [ ] **Step 1: CLI 기본값과 잘못된 선택지 테스트를 작성한다**

기본 입력이 원본 CSV, 기본 출력이 `outputs/family_current`, Feature 집합이 A·B·C, Target이 affected·severe, `max_iter=100`, `random_state=42`인지 검사한다. 알 수 없는 Feature나 Target을 전달하면 argparse가 종료되는지 검사한다.

- [ ] **Step 2: 직접 실행·모듈 실행 smoke test를 작성한다**

`python src/current_family_diagnosis.py --help`와 `python -m src.current_family_diagnosis --help`가 모두 종료 코드 0이고 한글 설명을 출력하는지 `tests/test_industrial_entrypoints.py`에 추가한다.

- [ ] **Step 3: 실행 파일 부재로 테스트가 실패하는지 확인한다**

Run: `/Users/kodohyeon/Documents/project_LS/venv/bin/python -m pytest tests/test_industrial_entrypoints.py -v`

Expected: 새 실행 파일 관련 테스트 FAIL

- [ ] **Step 4: CLI를 구현한다**

`main()`은 원본 로드→A·B·C 준비→Family 실험→결과 저장만 조정한다. `--feature-sets`, `--targets`, `--output`, `--max-iter`, `--random-state`, `--validation-start`, `--test-start`를 지원한다.

- [ ] **Step 5: 실행 가이드와 코드 가이드를 한글로 갱신한다**

실행 명령, 생성 파일, Target 정의, 지표 해석, 당일 진단 성공 전 미래 예측을 하지 않는 조건을 문서화한다.

- [ ] **Step 6: CLI와 관련 테스트를 통과시킨다**

Run: `/Users/kodohyeon/Documents/project_LS/venv/bin/python -m pytest tests/test_industrial_entrypoints.py tests/test_family_features.py tests/test_family_residual_features.py tests/test_family_diagnosis_experiments.py tests/test_family_diagnosis_report.py -v`

Expected: 모든 테스트 PASS

- [ ] **Step 7: 커밋한다**

```bash
git add src/current_family_diagnosis.py src/RUN_INDUSTRIAL.md docs/industrial_code_guide.md tests/test_industrial_entrypoints.py
git commit -m "feat: add same-day family diagnosis command"
```

### Task 7: 전체 검증·실제 실행·결과 판정

**Files:**
- Create: `outputs/family_current/metrics.csv`
- Create: `outputs/family_current/multilabel_metrics.csv`
- Create: `outputs/family_current/target_profile.csv`
- Create: `outputs/family_current/test_predictions.csv`
- Create: `outputs/family_current/feature_importance.csv`
- Create: `outputs/family_current/residual_baselines.csv`
- Create: `outputs/family_current/run_config.json`
- Create: `outputs/family_current/experiment_summary.md`

**Interfaces:**
- Consumes: Task 6의 CLI와 원본 합성 데이터
- Produces: 재현 가능한 실제 결과와 다음 단계 판정

- [ ] **Step 1: 전체 테스트를 실행한다**

Run: `/Users/kodohyeon/Documents/project_LS/venv/bin/python -m pytest -q`

Expected: 기존 xfail을 제외하고 모든 테스트 PASS

- [ ] **Step 2: 전체 A·B·C 실험을 실행한다**

Run: `/Users/kodohyeon/Documents/project_LS/venv/bin/python -m src.current_family_diagnosis --feature-sets A B C --targets affected severe --output outputs/family_current --max-iter 100`

Expected: 종료 코드 0, 설계 문서의 8개 결과 파일과 `models/` 생성

- [ ] **Step 3: 결과 스키마와 행 수를 검증한다**

별도 읽기 전용 검증 명령으로 9개 Family, 두 Target 상태, A·B·C 후보, 검증·테스트 분리, 네 임계값 정책, 예측 유일 키와 `severe <= affected`가 결과 파일에서 유지되는지 검사한다. 실패하면 결과를 해석하지 않고 해당 구현 Task로 돌아간다.

- [ ] **Step 4: 결과를 기준선과 비교한다**

Family별 양성률 대비 AP Lift, 정적 A 대비 B·C의 Macro AP 변화, Precision 0.70·0.80 정책의 Recall, Family별 F1과 기계·장비 편차를 계산한다. Accuracy 단독으로 성공을 판단하지 않는다.

- [ ] **Step 5: 순차 모델·미래 예측 진행 여부를 판정한다**

설계의 네 gate 조건을 적용해 `eligible`과 실패 사유를 확인한다. gate를 통과해도 이번 Task에서 LSTM이나 미래 예측을 구현하지 않고 다음 설계 필요사항으로 기록한다.

- [ ] **Step 6: 생성된 결과 보고서를 최종 검토한다**

`outputs/family_current/experiment_summary.md`에 세 줄 요약과 한 페이지 요약, Family Target 정의, 실제 지표, 실패·성공 해석, 다음 단계 판정이 모두 포함됐는지 확인한다. 기존 미추적 `docs/industrial_failure_modeling_journey.md`는 수정하거나 커밋하지 않는다.

- [ ] **Step 7: 전체 테스트를 다시 실행한다**

Run: `/Users/kodohyeon/Documents/project_LS/venv/bin/python -m pytest -q`

Expected: 기존 xfail을 제외하고 모든 테스트 PASS

- [ ] **Step 8: 재현 가능한 결과 파일을 커밋한다**

모델 바이너리는 `outputs/family_current/models/`에 생성하되 커밋하지 않는다. CSV·JSON·MD 결과만 명시적으로 스테이징하고, 최종 보고에 재생성 명령을 남긴다.

```bash
git add \
  outputs/family_current/metrics.csv \
  outputs/family_current/multilabel_metrics.csv \
  outputs/family_current/target_profile.csv \
  outputs/family_current/test_predictions.csv \
  outputs/family_current/feature_importance.csv \
  outputs/family_current/residual_baselines.csv \
  outputs/family_current/run_config.json \
  outputs/family_current/experiment_summary.md
git commit -m "docs: record family diagnosis experiment results"
```

- [ ] **Step 9: 최종 브랜치 검토를 요청한다**

`superpowers:requesting-code-review`로 승인된 설계·계획 대비 누락, 누수, 지표 선택과 저장 산출물을 검토한다. 발견된 문제는 수정 후 관련 테스트와 전체 테스트를 다시 실행한다.

## 완료 확인표

- [ ] Family `affected`, `severe`, `all_failed` 정의가 코드·테스트·문서에서 일치한다.
- [ ] A·B·C Feature 집합에 Target 또는 미래 정보가 없다.
- [ ] 잔차 모델과 정상 기준이 학습 구간만 사용한다.
- [ ] 검증 데이터만 모델·Feature·임계값 선택에 사용한다.
- [ ] 9개 Family 다중라벨 지표가 생성된다.
- [ ] Precision 0.70·0.80 불가능 정책이 숨겨지지 않는다.
- [ ] 실제 A·B·C 결과와 순차 모델 진행 판정이 한글 보고서에 기록된다.
- [ ] 기존·신규 전체 테스트가 통과한다.
