# 부품 고장확률 기반 장비 당일 위험 모델 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 부품별 당일 고장확률을 보정해 장비의 예상 고장점수, 12점·13점 이상 확률과 4단계 상태확률을 계산하고, 장비 위험 추정과 부품 위치 특정의 성공 여부를 분리해 검증한다.

**Architecture:** 원본 부품 행을 유지한 채 A1·A2 Feature를 미래 누수 없이 만들고, 각 Feature 구성에서 Logistic Regression·Random Forest·HistGradientBoosting 중 하나를 과거 선택 구간으로 고른다. 선택 모델의 부품확률을 별도 보정 구간에서 sigmoid 보정한 뒤 가중 Poisson-binomial 분포로 장비 위험을 집계한다. 모델링, 확률 집계, 기준모델·평가, 보고서와 CLI를 분리하며 기존 당일·미래·Family 실행 흐름은 변경하지 않는다.

**Tech Stack:** Python 3.11, pandas, NumPy, SciPy, scikit-learn 1.6.1, joblib, pytest

**Spec:** `docs/superpowers/specs/2026-09-29-probabilistic-asset-risk-design.md`

## 세 줄 요약

1. A1은 당일 센서·정적정보, A2는 여기에 과거 부품 고장 이력을 추가해 부품 고장확률을 계산한다.
2. 보정확률로 예상 점수와 12점·13점 이상 확률을 만들며, 장비 위험 성능은 MAE·PR-AUC·Brier Score로 검증한다.
3. Top 3는 B0보다 유의하게 좋아질 때만 `우선 점검 후보`로 승인하고, 그렇지 않으면 `참고용 위험 순위`로 제한한다.

## 한 페이지 요약

구현은 7개 작업으로 나눈다. 먼저 원본의 `장비·날짜·부품` 계약과 같은 장비·날짜의
센서 일관성을 검사하고, 2022~2023년 학습, 2024년 1~3월 모델 선택, 4~6월 확률
보정·임계값 결정, 7월 이후 테스트의 네 구간을 만든다. A1은 장비·부품 정적정보,
센서 8개와 달력 Feature를 사용한다. A2는 현재일 정답을 포함하지 않도록 `shift(1)`한
전일 고장, 7일·30일 고장 횟수, 누적 고장률과 마지막 고장 후 경과일을 추가한다.

다음으로 보정된 부품확률과 중요도 가중치 A=4, B=2, C=1을 결합한다. 동적계획법으로
0~47점의 가중 Poisson-binomial 분포를 계산하고 예상 점수, 12점·13점 이상 확률,
정상·주의·위험·고위험 확률을 만든다. 확률 순위와 예상 점수 기여도 순위도 저장하지만,
이는 장비 위험 계산과 구분된 보조 결과다.

과거 장비×부품 고장률만 쓰는 B0를 Beta-binomial 방식으로 평활해 A1·A2와 비교한다.
부품 위치 특정은 Hit Rate@3, Recall@3, MRR과 paired bootstrap 신뢰구간을 사용한다.
보정 구간에서 A1 또는 A2가 B0와 무작위 기준을 넘고 B0 대비 Hit Rate@3 차이의
95% 신뢰구간 하한이 0보다 크며 Recall@3가 낮지 않을 때만 `approved_on_validation`을
부여한다. 같은 세 부품만 매일 반복 추천하는지도 함께 검사한다.

실험 엔진은 후보 모델을 선택 구간에서 고르고 학습+선택 구간으로 재적합한 다음,
4~6월 자료로 sigmoid 보정한다. 테스트는 모델·보정기·오경보율 5%·10% 임계값을
바꾸지 않고 한 번만 평가한다. 산출물은 부품확률, 장비 위험확률, 점수·고위험·4단계·
순위 지표, 보정표, 중요도, 실행설정, 모델과 한글 요약문이다. 실제 성능이 낮더라도
숨기지 않으며, `장비 위험은 추정 가능하지만 부품 위치는 특정 불가`라는 결론도
정상적인 실험 결과로 인정한다.

## Global Constraints

- 문서, CLI 설명, 오류 메시지와 자동 보고서는 한글로 작성한다.
- 같은 장비·날짜의 센서 8개가 20개 부품 행에 반복되는 것은 정상 데이터 구조로 취급한다.
- 원본 CSV를 변경하지 않고 새 실행 결과는 `outputs/probabilistic_asset_risk/`에만 쓴다.
- 각 장비·날짜에는 정확히 20개 부품이 있어야 하며 부품 구성·Family·중요도는 기간 중 고정되어야 한다.
- 중요도 가중치는 A=4, B=2, C=1이며 현재 데이터의 최대 장비점수는 47점이다.
- A1은 당일 센서·정적정보만, A2는 A1과 현재일 이전의 부품 고장 이력만 사용한다.
- A2 이력 계산은 반드시 `shift(1)` 후 수행하며 현재일·미래일 정답을 참조하지 않는다.
- 학습은 2024-01-01 이전, 선택은 2024-01-01~03-31, 보정은 04-01~06-30, 테스트는 07-01 이후로 고정한다.
- 모델 선택은 선택 구간, 확률 보정·알림 임계값·위치 특정 승인은 보정 구간까지만 사용한다.
- 후보 모델은 선택 구간 예상점수 MAE, 12·13점 평균 Brier Score, 부품 PR-AUC 순으로 선택하며 Top 3 지표로 고르지 않는다.
- 테스트 결과로 모델·보정기·임계값·`localization_status`를 다시 선택하지 않는다.
- 장비 위험 추정과 부품 위치 특정의 성공 여부를 반드시 별도로 결론 낸다.
- Top 3 승인 실패 시 표시명은 `참고용 위험 순위`, 상태는 `insufficient_evidence`로 고정한다.
- `breakdown_flag`, 실제 고장점수·등급, 사후 정비정보와 미래 정보는 Feature에 포함하지 않는다.
- 기존 코드·CLI·산출물 인터페이스와 기존 테스트 동작을 보존한다.
- 높은 성능은 구현 완료 조건이 아니다. 낮은 성능, 미정의 지표와 승인 실패도 그대로 저장한다.
- random state는 42로 고정하고 확률·순위·bootstrap 결과가 재현되어야 한다.

## Review Focus

- 같은 장비·날짜 20행이 네 시간 구간 중 하나에만 들어가며 행 단위 무작위 분할이 없는지 검토한다.
- A2의 누적 고장률과 최근 고장 횟수가 현재 `breakdown_flag`를 포함하지 않는지 경계 날짜로 검증한다.
- 모델 선택, sigmoid 보정, FPR 임계값과 위치 특정 승인에 테스트 데이터가 들어가지 않는지 검토한다.
- 가중 점수분포의 합이 1이고 기대값이 \(\sum_i w_i\hat p_i\)와 일치하는지 검증한다.
- `B0`, `A1`, `A2`가 동일한 장비일·평가지표·Top K 정의로 비교되는지 확인한다.
- Top 3가 좋아 보여도 보정 구간 승인조건을 통과하지 않으면 운영적인 표현으로 승격되지 않는지 확인한다.
- 상수 입력 상관계수에서 경고가 장비별로 반복되지 않고 결측값과 상태 한 건으로 기록되는지 확인한다.
- 12점과 13점 정책이 섞이지 않고 항상 \(P(S\ge13)\le P(S\ge12)\)인지 확인한다.

---

### Task 1: 원본 계약, 시간 분할과 A1·A2 Feature

**Files:**
- Create: `src/probabilistic_risk_features.py`
- Create: `tests/test_probabilistic_risk_features.py`

**Interfaces:**
- Consumes: `src.industrial_data`의 공통 컬럼 상수, `src.asset_features.CRITICALITY_WEIGHTS`, 원본 부품 DataFrame
- Produces: `A1_FEATURES`, `A2_HISTORY_FEATURES`, `PERIOD_NAMES`
- Produces: `ProbabilisticRiskFeatures(frame, feature_sets, periods, max_failure_points)` dataclass
- Produces: `prepare_probabilistic_risk_features(raw, *, selection_start="2024-01-01", calibration_start="2024-04-01", test_start="2024-07-01", expected_parts_per_asset=20) -> ProbabilisticRiskFeatures`

- [ ] **Step 1: 최소 20부품·복수 날짜 fixture를 작성한다**

`tests/test_probabilistic_risk_features.py`에 장비 2대, 날짜 40일, 장비별 20개 부품,
A 6개·B 9개·C 5개와 센서 8개를 갖는 fixture를 만든다. 같은 장비·날짜의 센서는
20행에서 동일하게 만들고 일부 날짜에만 부품 고장을 배치한다.

- [ ] **Step 2: 데이터 계약과 네 시간 구간 테스트를 작성한다**

```python
def test_prepare_keeps_each_asset_day_in_one_period(raw_part_rows):
    prepared = prepare_probabilistic_risk_features(
        raw_part_rows,
        selection_start="2023-01-11",
        calibration_start="2023-01-21",
        test_start="2023-01-31",
    )
    counts = prepared.frame.groupby(["transaction_date", "asset_tag"])["period"].nunique()
    assert counts.eq(1).all()
    assert set(prepared.periods) == {"train", "selection", "calibration", "test"}


def test_a1_and_a2_never_include_current_targets(raw_part_rows):
    prepared = prepare_probabilistic_risk_features(raw_part_rows, ...)
    forbidden = {"breakdown_flag", "failure_points", "severity_level"}
    assert forbidden.isdisjoint(prepared.feature_sets["A1"])
    assert forbidden.isdisjoint(prepared.feature_sets["A2"])
```

필수 컬럼 누락, 장비·날짜·부품 중복, 센서 충돌, 20개가 아닌 부품 수, 날짜별 부품
구성 변화, Family·중요도 변화, 0·1 이외 정답, 역전 또는 빈 날짜 구간도 각각
`ValueError`가 발생하는지 테스트한다.

- [ ] **Step 3: A2 미래 비누수와 달력일 이력을 테스트한다**

현재일 고장값만 변경해도 같은 날의 `breakdown_lag1`, `breakdown_count_7d`,
`breakdown_count_30d`, `historical_breakdown_rate`, `days_since_last_breakdown`이
변하지 않아야 한다. 날짜 하나를 제거한 fixture에서는 lag1이 직전 관측행이 아니라
전날 달력일을 의미해 결측이 되는지 검사한다. 테스트 기간의 과거 날짜 고장은 이후
테스트 날짜 Feature에는 반영되지만 같은 날에는 반영되지 않는지도 검사한다.

- [ ] **Step 4: 신규 모듈 부재로 테스트가 실패하는지 확인한다**

Run: `/Users/kodohyeon/Documents/project_LS/venv/bin/python -m pytest tests/test_probabilistic_risk_features.py -v`

Expected: `ModuleNotFoundError: No module named 'src.probabilistic_risk_features'`

- [ ] **Step 5: 엄격한 원본 검증과 실제 점수를 구현한다**

날짜를 정규화하고 `transaction_date + asset_tag + part_no` 유일성, 센서·정적정보
일관성과 20개 부품 구성을 검사한다. `criticality_weight`와 장비·날짜별
`actual_failure_points`를 추가하고 최대 가능 점수가 47인지 확인한다.

- [ ] **Step 6: A1과 A2를 구현한다**

A1에는 `machine_type`, `asset_tag`, `plant_code`, `part_no`, `part_family`,
`criticality`, 센서 8개, 요일·월 주기를 넣는다. A2 과거 이력은
`asset_tag + part_no`별 완전한 달력 인덱스에서 현재일보다 이전 값만으로 계산한다.
최초 관측일의 누적률은 결측으로 두고 모델 전처리기의 중앙값 대체에 맡긴다.

- [ ] **Step 7: Feature 테스트를 통과시킨다**

Run: `/Users/kodohyeon/Documents/project_LS/venv/bin/python -m pytest tests/test_probabilistic_risk_features.py -v`

Expected: 모든 테스트 PASS

- [ ] **Step 8: 커밋한다**

```bash
git add src/probabilistic_risk_features.py tests/test_probabilistic_risk_features.py
git commit -m "feat: add probabilistic risk feature sets"
```

### Task 2: 부품확률 순위와 가중 점수분포 집계

**Files:**
- Create: `src/probabilistic_risk_aggregation.py`
- Create: `tests/test_probabilistic_risk_aggregation.py`

**Interfaces:**
- Produces: `weighted_score_distribution(probabilities, weights, *, max_score=None) -> np.ndarray`
- Produces: `rank_part_probabilities(frame, *, probability_column="failure_probability") -> pd.DataFrame`
- Produces: `aggregate_asset_risk(ranked_parts, *, high_risk_thresholds=(12, 13)) -> pd.DataFrame`
- Produces: `severity_probabilities(score_distribution, high_risk_threshold) -> dict[str, float]`

- [ ] **Step 1: 확률 0·1과 손계산 가능한 분포 테스트를 작성한다**

```python
def test_weighted_distribution_matches_two_part_hand_calculation():
    result = weighted_score_distribution([0.25, 0.50], [4, 2])
    assert result[0] == pytest.approx(0.375)
    assert result[2] == pytest.approx(0.375)
    assert result[4] == pytest.approx(0.125)
    assert result[6] == pytest.approx(0.125)
    assert result.sum() == pytest.approx(1.0)
    assert np.dot(np.arange(len(result)), result) == pytest.approx(2.0)
```

모두 0, 모두 1, 확률 결측·무한값·범위 이탈, 양수가 아닌 정수 가중치와 분포합
오류도 테스트한다.

- [ ] **Step 2: Top K 동률 규칙과 중복 방지 테스트를 작성한다**

확률 내림차순, 가중치 내림차순, `part_no` 오름차순으로 정렬되는지 검사한다.
확률 순위와 \(w_i\hat p_i\) 기여도 순위가 별도 열에 저장되고 같은 장비일 Top 3에
중복 부품이 없는지 확인한다.

- [ ] **Step 3: 12·13점 및 4단계 확률 테스트를 작성한다**

집계 결과의 `prob_score_0`~`prob_score_47` 합, 예상 점수, 실제 점수,
`prob_ge_12`, `prob_ge_13`, 4개 상태확률 합을 검사한다. 상태확률 동률이면 더 높은
위험등급을 선택하고 항상 `prob_ge_13 <= prob_ge_12`인지 확인한다.

- [ ] **Step 4: 신규 모듈 부재로 테스트가 실패하는지 확인한다**

Run: `/Users/kodohyeon/Documents/project_LS/venv/bin/python -m pytest tests/test_probabilistic_risk_aggregation.py -v`

Expected: `ModuleNotFoundError: No module named 'src.probabilistic_risk_aggregation'`

- [ ] **Step 5: 동적계획 점수분포를 구현한다**

각 부품마다 기존 확률질량을 미고장 \((1-p_i)\)와 고장 \(p_i\) 경로로 갱신한다.
허용오차 `1e-10`에서 분포합을 검증하고 기대값과 직접 계산한
\(\sum_i w_i\hat p_i\)가 일치하지 않으면 오류를 발생시킨다.

- [ ] **Step 6: 순위와 장비 위험 집계를 구현한다**

장비·날짜·모델변형별 정확히 20행을 요구한다. 결과는 임계값 12와 13마다 한 행을
생성하고 점수분포, 상태확률, 실제·예측 등급, 확률·기여도 Top 3 문자열을 저장한다.

- [ ] **Step 7: 집계 테스트를 통과시킨다**

Run: `/Users/kodohyeon/Documents/project_LS/venv/bin/python -m pytest tests/test_probabilistic_risk_aggregation.py -v`

Expected: 모든 테스트 PASS

- [ ] **Step 8: 커밋한다**

```bash
git add src/probabilistic_risk_aggregation.py tests/test_probabilistic_risk_aggregation.py
git commit -m "feat: aggregate part probabilities into asset risk"
```

### Task 3: B0 기준모델, Top K 지표와 위치 특정 승인

**Files:**
- Create: `src/probabilistic_risk_evaluation.py`
- Create: `tests/test_probabilistic_risk_evaluation.py`

**Interfaces:**
- Produces: `SmoothedBaseRateModel(prior_strength=20.0).fit(frame).predict_proba(frame)`
- Produces: `build_ranking_metrics(ranked_parts, *, ks=(1, 3, 5)) -> pd.DataFrame`
- Produces: `random_ranking_expectations(ranked_parts, *, ks=(1, 3, 5)) -> pd.DataFrame`
- Produces: `paired_hit_rate_bootstrap(model_parts, baseline_parts, *, k=3, samples=2000, random_state=42) -> dict`
- Produces: `decide_localization_status(calibration_results) -> dict[str, object]`

- [ ] **Step 1: B0 Beta-binomial 평활 테스트를 작성한다**

전체 고장률 \(\pi\), `prior_strength=20`, \(\alpha=20\pi\),
\(\beta=20(1-\pi)\)로 장비×부품 확률을 계산하는지 손계산과 비교한다. 처음 보는
장비×부품은 전체 \(\pi\)로 fallback하고, `predict_proba()`가 전달된 현재
`breakdown_flag`를 사용하지 않는지 검사한다.

- [ ] **Step 2: 다중 고장일 Top K 지표 테스트를 작성한다**

고장이 없는 장비일은 Hit Rate·Recall·MRR 분모에서 제외하되 Precision@3의 거짓
추천 분석에는 남기는 계약을 고정한다. Hit Rate@1·3·5, Recall@1·3·5,
Precision@3, MRR, 중요도·Family별 Recall@3를 손계산 fixture와 비교한다.

- [ ] **Step 3: 무작위 기준과 장비일 paired bootstrap 테스트를 작성한다**

무작위 Top K 기대값이 각 장비일의 고장 부품 수와 20개 후보 수를 이용하는지
검사한다. bootstrap은 부품행이 아니라 장비·날짜를 재표집하고, 동일한 두 순위의
Hit@3 차이와 95% 신뢰구간이 0인지 확인한다.

- [ ] **Step 4: 위치 특정 승인·거부 테스트를 작성한다**

승인은 보정 구간에서만 다음 네 조건을 모두 요구한다.

1. Hit Rate@3가 B0와 무작위 기준보다 큼
2. B0 대비 장비일 Hit@3 차이의 paired bootstrap 95% 하한이 0보다 큼
3. Recall@3가 B0보다 낮지 않음
4. 보정 구간 전체에서 Top 3 조합이 정확히 하나로 고정되지 않음

실패 사유를 목록으로 저장하고 하나라도 실패하면 `insufficient_evidence`, 모두
통과하면 `approved_on_validation`이 되는지 테스트한다.

- [ ] **Step 5: 신규 모듈 부재로 테스트가 실패하는지 확인한다**

Run: `/Users/kodohyeon/Documents/project_LS/venv/bin/python -m pytest tests/test_probabilistic_risk_evaluation.py -v`

Expected: `ModuleNotFoundError: No module named 'src.probabilistic_risk_evaluation'`

- [ ] **Step 6: B0와 순위 평가를 구현한다**

B0는 학습+선택 구간에만 적합한다. 지표 결과에는 분모 수, 양성 부품 수, 고유
Top 3 조합 수, 가장 흔한 Top 3 조합 비율을 포함한다. 계산 불가 상황은 임의의 0이
아니라 `status`와 결측값으로 남긴다.

- [ ] **Step 7: bootstrap과 승인 판정을 구현한다**

2,000회 재표집, percentile 95% 신뢰구간과 고정 seed를 사용한다. 승인 상태는 보정
구간 자료로 한 번 결정하고 이후 테스트 지표를 전달해도 바뀌지 않는 순수 함수 계약을
테스트한다.

- [ ] **Step 8: 평가 테스트를 통과시킨다**

Run: `/Users/kodohyeon/Documents/project_LS/venv/bin/python -m pytest tests/test_probabilistic_risk_evaluation.py -v`

Expected: 모든 테스트 PASS

- [ ] **Step 9: 커밋한다**

```bash
git add src/probabilistic_risk_evaluation.py tests/test_probabilistic_risk_evaluation.py
git commit -m "feat: add localization baselines and validation gate"
```

### Task 4: 후보 모델 선택, 재적합과 sigmoid 확률 보정

**Files:**
- Create: `src/probabilistic_asset_risk.py`
- Create: `tests/test_probabilistic_asset_risk.py`

**Interfaces:**
- Consumes: Task 1 Feature, `src.industrial_training.build_model`과 `MODEL_NAMES`
- Produces: `PartModelSelection(feature_set, selected_model, selection_metrics)` dataclass
- Produces: `CalibratedPartModel(feature_set, model_name, estimator, calibrator, features)` dataclass
- Produces: `select_part_model(prepared, feature_set, *, model_names=MODEL_NAMES, max_iter=100, random_state=42) -> PartModelSelection`
- Produces: `fit_calibrated_part_model(prepared, selection, *, method="sigmoid") -> CalibratedPartModel`
- Produces: `predict_part_probabilities(model, frame, *, split) -> pd.DataFrame`

- [ ] **Step 1: 선택 우선순위 테스트를 작성한다**

선택 구간에서 예상점수 MAE 최소, 12·13점 이상 확률의 평균 Brier Score 최소,
부품행 Average Precision 최대, 고정 모델 순서 Logistic→HGB→RF 순으로 정렬되는
합성 지표표를 이용해 결정적 선택을 검사한다. Hit Rate@3가 가장 높은 후보라도 점수
MAE가 더 나쁘면 선택되지 않아야 하며, 테스트 구간 지표를 극단적으로 바꿔도 선택
결과가 변하지 않아야 한다.

- [ ] **Step 2: 학습 범위와 sigmoid 보정 비누수 테스트를 작성한다**

기반 모델 재적합은 학습+선택 구간만, `FrozenEstimator` 기반 sigmoid calibrator는
보정 구간만 보도록 spy estimator 또는 색인 기록 fixture로 확인한다. 보정 후 테스트
확률이 0~1이고 같은 입력에서 재현되는지 검사한다.

- [ ] **Step 3: 단일 클래스와 저장·재로딩 테스트를 작성한다**

선택 또는 보정 구간 정답이 단일 클래스이면 명시적 한글 `ValueError`를 내야 한다.
joblib로 기반 모델과 calibrator를 저장하고 다시 불러온 확률이 원본과 일치하는지
임시 디렉터리에서 검사한다.

- [ ] **Step 4: 신규 인터페이스 부재로 테스트가 실패하는지 확인한다**

Run: `/Users/kodohyeon/Documents/project_LS/venv/bin/python -m pytest tests/test_probabilistic_asset_risk.py -v`

Expected: `ModuleNotFoundError` 또는 공개 함수 import 실패

- [ ] **Step 5: 후보 학습과 선택을 구현한다**

A1·A2마다 세 후보를 학습 구간에 적합하고 선택 구간 부품확률을 Task 2 장비위험으로
집계해 예상점수 MAE와 12·13점 Brier Score를 계산한다. 부품행 Average Precision과
Top K 지표도 후보 진단값으로 저장하지만 Top K는 선택 순위에 사용하지 않는다.
후보별 결과를 모두 보존하고 선택된 후보만 다음 단계로 넘긴다.

- [ ] **Step 6: 재적합과 확률 보정을 구현한다**

선택된 pipeline을 학습+선택 자료로 처음부터 재적합하고 scikit-learn 1.6의
`FrozenEstimator`와 `CalibratedClassifierCV(method="sigmoid")`로 보정한다.
calibrator가 기반 estimator를 다시 적합하지 않는지 확인한다.

- [ ] **Step 7: 예측·저장 계약을 구현하고 테스트를 통과시킨다**

Run: `/Users/kodohyeon/Documents/project_LS/venv/bin/python -m pytest tests/test_probabilistic_asset_risk.py -v`

Expected: 모든 테스트 PASS

- [ ] **Step 8: 커밋한다**

```bash
git add src/probabilistic_asset_risk.py tests/test_probabilistic_asset_risk.py
git commit -m "feat: select and calibrate part probability models"
```

### Task 5: 장비 위험·점수·등급 평가와 전체 실험 오케스트레이션

**Files:**
- Modify: `src/probabilistic_risk_evaluation.py`
- Modify: `tests/test_probabilistic_risk_evaluation.py`
- Modify: `src/probabilistic_asset_risk.py`
- Modify: `tests/test_probabilistic_asset_risk.py`

**Interfaces:**
- Produces: `choose_fpr_cutoff(y_true, probabilities, *, max_fpr) -> dict`
- Produces: `build_score_metrics(asset_predictions) -> pd.DataFrame`
- Produces: `build_high_risk_metrics(asset_predictions, cutoffs) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]`
- Produces: `build_severity_metrics(asset_predictions) -> tuple[pd.DataFrame, pd.DataFrame]`
- Produces: `ProbabilisticRiskRun(...)` dataclass
- Produces: `run_probabilistic_asset_risk(raw, *, output_dir, selection_start="2024-01-01", calibration_start="2024-04-01", test_start="2024-07-01", high_risk_thresholds=(12, 13), fpr_policies=(0.05, 0.10), bootstrap_samples=2000, max_iter=100, random_state=42) -> ProbabilisticRiskRun`

- [ ] **Step 1: 점수 지표와 기준선 테스트를 작성한다**

MAE, RMSE, 실제·예상 평균, Pearson·Spearman, 기계·장비·실제 점수구간별 MAE와
예상 점수 10분위별 실제 평균을 손계산 fixture로 검증한다. 전체 학습 평균, 장비별
학습 평균, B0, A1, A2가 같은 테스트 장비일에서 비교되어야 한다.

상수 예측에서는 상관계수를 계산하지 않고 값은 결측, 상태는
`not_defined_constant_input` 한 행으로 기록하며 `ConstantInputWarning`이 반복되지
않는지 `pytest.warns`/`warnings.catch_warnings`로 검사한다.

- [ ] **Step 2: 고위험 확률과 FPR 정책 테스트를 작성한다**

12·13점별 ROC-AUC, PR-AUC, Brier Score, Log Loss와 확률 10구간 보정표를 검사한다.
보정 구간의 실제 음성 장비일에서 FPR 5%·10%를 넘지 않는 가장 낮은 cutoff를 정하고,
테스트 정답·확률을 바꿔도 cutoff가 변하지 않아야 한다. Accuracy, Precision, Recall,
F1, FPR, TN·FP·FN·TP를 저장한다.

- [ ] **Step 3: 4단계 지표 테스트를 작성한다**

12·13점 각각 Accuracy, Macro Precision·Recall·F1, Weighted F1, 등급별 지표와
4×4 혼동행렬이 올바른지 검사한다. 존재하지 않는 등급도 support 0으로 명시한다.

- [ ] **Step 4: 작은 end-to-end 실험 테스트를 작성한다**

작은 fixture에서 후보 수를 Logistic 하나, bootstrap 50회로 제한해 B0·A1·A2
부품확률과 장비위험을 생성한다. 모든 필수 산출물, 모델 파일, run config가 임시
출력 폴더에 생성되고 `localization_status`와 표시명이 승인 결과에 맞는지 검사한다.

- [ ] **Step 5: 구현 전 공개 함수 부재로 테스트가 실패하는지 확인한다**

Run: `/Users/kodohyeon/Documents/project_LS/venv/bin/python -m pytest tests/test_probabilistic_risk_evaluation.py tests/test_probabilistic_asset_risk.py -v`

Expected: 신규 평가 또는 실행 함수 import 실패

- [ ] **Step 6: 장비 평가 함수를 구현한다**

고위험 상수 발생률 기준의 Brier Score도 함께 계산한다. 지표 계산 불가 시 status를
남기고, 분류 cutoff가 없는 정책은 `unavailable`로 기록한다. 보정표 구간은 중복
경계에서도 실패하지 않도록 실제 예측값 순위 기반 10분위를 사용한다.

- [ ] **Step 7: 전체 실행과 산출물 저장을 구현한다**

실행 순서는 Feature 준비→후보 선택→재적합→보정→B0/A1/A2 부품확률→장비집계→
보정구간 정책 고정→테스트 평가→importance→저장으로 고정한다. 다음 파일을 쓴다.

```text
outputs/probabilistic_asset_risk/
├── part_probabilities.csv
├── asset_risk_predictions.csv
├── ranking_metrics.csv
├── score_metrics.csv
├── high_risk_metrics.csv
├── severity_metrics.csv
├── confusion_matrices.csv
├── calibration_table.csv
├── feature_importance.csv
├── run_config.json
└── models/
```

`run_config.json`에는 원본 SHA-256, 분할, Feature, 모든 후보와 선택 모델, 보정법,
B0 prior, cutoff, seed, 라이브러리 버전, Git commit·dirty 상태를 기록한다.
Permutation importance는 선택 구간 Average Precision으로만 계산한다.

- [ ] **Step 8: 오케스트레이션 테스트를 통과시킨다**

Run: `/Users/kodohyeon/Documents/project_LS/venv/bin/python -m pytest tests/test_probabilistic_risk_evaluation.py tests/test_probabilistic_asset_risk.py -v`

Expected: 모든 테스트 PASS

- [ ] **Step 9: 커밋한다**

```bash
git add src/probabilistic_risk_evaluation.py src/probabilistic_asset_risk.py tests/test_probabilistic_risk_evaluation.py tests/test_probabilistic_asset_risk.py
git commit -m "feat: evaluate and run probabilistic asset risk"
```

### Task 6: 한글 결과 보고서, CLI와 실행 가이드

**Files:**
- Create: `src/probabilistic_risk_report.py`
- Create: `tests/test_probabilistic_risk_report.py`
- Create: `src/current_probabilistic_asset_risk.py`
- Modify: `tests/test_industrial_entrypoints.py`
- Modify: `src/RUN_INDUSTRIAL.md`
- Modify: `src/probabilistic_asset_risk.py`

**Interfaces:**
- Produces: `render_probabilistic_risk_summary(run: ProbabilisticRiskRun) -> str`
- Produces: `src.current_probabilistic_asset_risk.build_parser()`와 `main()`
- CLI defaults: 출력 `outputs/probabilistic_asset_risk`, thresholds 12·13, FPR 0.05·0.10, bootstrap 2000

- [ ] **Step 1: 보고서 표현 경계 테스트를 작성한다**

보고서가 `세 줄 요약`, `한 페이지 요약`, `장비 위험 추정`, `부품 위치 특정`,
`한계와 다음 판단`을 모두 포함하는지 검사한다. 위치 승인 실패 fixture에서는
`우선 점검 후보`를 성능 결론으로 쓰지 않고 `참고용 위험 순위`,
`부품 특정 근거 부족`을 표시해야 한다. 장비 위험만 성공한 경우 두 결론이 섞이지
않는지도 검사한다.

- [ ] **Step 2: CLI 기본값과 잘못된 인자 테스트를 작성한다**

`--data`, `--output`, 세 날짜 경계, `--high-risk-thresholds 12 13`,
`--fpr-policies 0.05 0.10`, `--bootstrap-samples 2000`, `--max-iter 100`,
`--random-state 42`를 확인한다. 0~1 밖 FPR, 역전 날짜와 7 미만 고위험점수는
실행 전에 거부하도록 테스트한다.

- [ ] **Step 3: 신규 보고서·CLI 부재로 테스트가 실패하는지 확인한다**

Run: `/Users/kodohyeon/Documents/project_LS/venv/bin/python -m pytest tests/test_probabilistic_risk_report.py tests/test_industrial_entrypoints.py -v`

Expected: 신규 모듈 import 실패 또는 새 entrypoint 미등록 실패

- [ ] **Step 4: 보고서 생성과 `experiment_summary.md` 저장을 구현한다**

검증·테스트 표를 구분하고 MAE 개선률, 12·13점 확률 품질, 위치 특정 승인·재현 여부,
A2 이력의 추가 가치를 한글로 설명한다. 실제 지표를 근거로만 문장을 선택하고 낮은
성능을 개선으로 표현하지 않는다.

- [ ] **Step 5: CLI와 가이드를 구현한다**

스크립트 직접 실행과 module 실행을 모두 지원한다. `src/RUN_INDUSTRIAL.md`에 다음
명령, 산출물 설명, A1/A2/B0, 12·13점, Top 3 해석 주의를 추가한다.

```bash
cd /Users/kodohyeon/Documents/project_LS/urAnkleMyAnkle
source /Users/kodohyeon/Documents/project_LS/venv/bin/activate
python -m src.current_probabilistic_asset_risk
```

- [ ] **Step 6: 보고서·CLI 테스트를 통과시킨다**

Run: `/Users/kodohyeon/Documents/project_LS/venv/bin/python -m pytest tests/test_probabilistic_risk_report.py tests/test_industrial_entrypoints.py -v`

Expected: 모든 테스트 PASS

- [ ] **Step 7: 커밋한다**

```bash
git add src/probabilistic_risk_report.py src/probabilistic_asset_risk.py src/current_probabilistic_asset_risk.py src/RUN_INDUSTRIAL.md tests/test_probabilistic_risk_report.py tests/test_industrial_entrypoints.py
git commit -m "feat: add probabilistic asset risk CLI and report"
```

### Task 7: 원본 데이터 실행, 결과 검증과 최종 문서화

**Files:**
- Create: `outputs/probabilistic_asset_risk/part_probabilities.csv`
- Create: `outputs/probabilistic_asset_risk/asset_risk_predictions.csv`
- Create: `outputs/probabilistic_asset_risk/ranking_metrics.csv`
- Create: `outputs/probabilistic_asset_risk/score_metrics.csv`
- Create: `outputs/probabilistic_asset_risk/high_risk_metrics.csv`
- Create: `outputs/probabilistic_asset_risk/severity_metrics.csv`
- Create: `outputs/probabilistic_asset_risk/confusion_matrices.csv`
- Create: `outputs/probabilistic_asset_risk/calibration_table.csv`
- Create: `outputs/probabilistic_asset_risk/feature_importance.csv`
- Create: `outputs/probabilistic_asset_risk/run_config.json`
- Create: `outputs/probabilistic_asset_risk/experiment_summary.md`
- Create: `outputs/probabilistic_asset_risk/models/*.joblib`

- [ ] **Step 1: 전체 테스트를 먼저 실행한다**

Run: `/Users/kodohyeon/Documents/project_LS/venv/bin/python -m pytest -q`

Expected: 기존 150개 테스트와 신규 테스트가 모두 PASS하고 기존 xfail만 유지됨

- [ ] **Step 2: 원본 데이터로 새 CLI를 실행한다**

Run: `/Users/kodohyeon/Documents/project_LS/venv/bin/python -m src.current_probabilistic_asset_risk --data data/raw/synthetic_industrial_machine_data.csv --output outputs/probabilistic_asset_risk`

Expected: 한글 완료 메시지, traceback 없음, 필수 CSV·JSON·Markdown·joblib 생성

- [ ] **Step 3: 산출물 불변조건을 자동 검증한다**

짧은 Python 검증 명령으로 다음을 확인한다.

1. 각 모델변형·장비·날짜에 부품 20개와 중복 없는 순위 1~20이 존재함
2. 점수분포 0~47의 합이 허용오차에서 1임
3. 예상 점수가 부품별 \(w_i\hat p_i\) 합과 일치함
4. `prob_ge_13 <= prob_ge_12`임
5. 상태확률 네 개의 합이 1임
6. 테스트 cutoff와 `localization_status`가 보정 구간 결정값과 일치함
7. 원본 SHA-256와 실행 Git 정보가 `run_config.json`에 존재함
8. 저장 모델 재로딩 예측이 저장된 표본 확률과 일치함

- [ ] **Step 4: 결과를 설계 성공조건과 대조한다**

`experiment_summary.md`와 CSV에서 다음을 수치로 확인한다.

1. A1 또는 A2 점수 MAE가 장비평균·B0보다 낮은가
2. 12·13점 Brier Score가 발생률 상수 기준보다 낮은가
3. A2가 A1보다 점수 또는 보정을 개선했는가
4. Top 3가 보정 구간 승인조건을 통과했는가
5. 승인됐다면 테스트에서 B0 대비 신뢰구간 하한이 다시 0보다 큰가

성공하지 않은 항목은 재튜닝하지 않고 실패 원인과 한계를 보고서에 남긴다.

- [ ] **Step 5: diff와 작업 범위를 검토한다**

Run: `git status --short && git diff --check && git diff --stat`

Expected: 이번 구현·결과 파일만 변경되고, 기존 미추적
`docs/industrial_failure_modeling_journey.md`와 `outputs/family_current/models/`는 그대로임

- [ ] **Step 6: 최종 테스트를 다시 실행한다**

Run: `/Users/kodohyeon/Documents/project_LS/venv/bin/python -m pytest -q`

Expected: 전체 PASS, 새로운 warning 없음

- [ ] **Step 7: 생성 결과를 커밋한다**

```bash
git add outputs/probabilistic_asset_risk
git commit -m "results: add probabilistic asset risk experiment"
```

커밋 전에 모델 파일을 포함한 출력 크기를 확인하고, 저장소 정책상 허용 범위를 넘으면
사용자와 상의해 모델 binary만 제외한다. 기존 미추적 파일은 추가하지 않는다.

## 최종 검토 체크리스트

1. 설계의 A1·A2·B0, 예상 점수, 12·13점, 4단계와 Top 3 보조 실험이 모두 작업에 매핑되었다.
2. 모든 구현 작업은 실패 테스트→최소 구현→통과 확인→작은 커밋 순서다.
3. 모델 선택·보정·승인·테스트의 시간 경계가 함수와 테스트로 분리되었다.
4. 부품 특정 성능이 낮을 때도 장비 위험 결과와 섞이지 않도록 보고서 표현 테스트가 있다.
5. 수학적 집계의 핵심 불변조건과 저장·재로딩 재현성이 자동 검증된다.
6. 기존 미추적 사용자 파일은 어떤 `git add` 명령에도 포함되지 않는다.
