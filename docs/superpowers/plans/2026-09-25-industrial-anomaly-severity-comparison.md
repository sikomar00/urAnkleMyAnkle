# 산업 장비 센서 이상 특징·고위험 기준 비교 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 공통 4등급을 유지하면서 고위험 기준 12·13점과 센서 Feature A~D를 같은 시간 분할에서 비교하고, 전체·기계 종류·개별 장비 결과를 한국어로 보고한다.

**Architecture:** 기존 장비 일별 집계는 유지한다. 임계값별 등급 변환은 `asset_features.py`, 학습 정상 범위와 과거 센서 Feature는 새 `asset_anomaly_features.py`, 모델 비교·저장은 새 `asset_severity_experiments.py`, 발생률·한국어 보고서는 새 `asset_severity_report.py`가 담당한다. 새 CLI만 이 흐름을 호출하며 기존 `current_asset_score_model.py`와 `outputs/asset_score_current`는 변경하지 않는다.

**Tech Stack:** Python 3.11, pandas, NumPy, scikit-learn 1.6.1, joblib, pytest

**Spec:** `docs/superpowers/specs/2026-09-25-industrial-anomaly-severity-comparison-design.md`

## Global Constraints

- 문서, CLI 설명, 오류 메시지와 자동 요약은 한국어로 작성한다.
- Target은 `normal`, `caution`, `risk`, `high_risk` 네 등급이다.
- 모든 기계에 같은 점수 경계를 적용하고 고위험 시작점만 12와 13을 비교한다.
- 학습은 2024-01-01 이전, 검증은 2024-01-01~2024-06-30, 테스트는 2024-07-01 이후다.
- 정상 기준은 학습 구간의 `failure_points == 0` 행만 사용한다.
- 과거 Feature는 `asset_tag`별 `shift(1)` 이후 값만 사용하며 미래값으로 채우지 않는다.
- 모델은 검증 Macro F1 우선, 0.01 이내 후보는 고위험 Recall, Precision 순으로 선택한다.
- `breakdown_flag`, `failure_points`, 등급 Target과 미래 정보는 Feature에 넣지 않는다.
- 실제 기계 고장률이라는 표현 대신 부품 고장 표시율과 고위험 점수 발생률을 구분한다.
- 기존 실행 파일, 기존 출력 폴더와 기존 64개 통과 테스트의 동작을 보존한다.

## Review Focus

- 임계값에 `True`, 6 이하, 실수가 들어오면 `ValueError`가 발생하는지 Task 1에서 검증한다.
- 장비 정상 표본은 충분하지만 특정 센서 MAD만 0이면 센서별 fallback이 되는지 Task 2에서 검증한다.
- 날짜가 빠진 장비에서 `lag1`이 직전 관측행이 아니라 직전 달력 날짜인지 Task 3에서 검증한다.
- Macro F1 차이가 정확히 0.01인 후보의 선택이 결정적인지 Task 4에서 검증한다.
- 13점 기준에서 12점 행이 `risk`이고 저장 확률 순서가 고정되는지 Task 5에서 검증한다.

---

### Task 1: 임계값을 받는 공통 4등급 변환

**Files:**
- Modify: `src/asset_features.py:25-109`
- Modify: `tests/test_asset_features.py:1-110`

**Interfaces:**
- Consumes: `failure_points`, `SEVERITY_LEVELS`, `SEVERITY_CODE`
- Produces: `severity_labels(scores: Any, high_risk_threshold: int = 12) -> pd.Categorical`
- Produces: `add_asset_severity(frame: pd.DataFrame, high_risk_threshold: int = 12) -> pd.DataFrame`

- [ ] **Step 1: 12·13점 경계와 임계값 검증 테스트를 작성한다**

```python
from src.asset_features import severity_labels


@pytest.mark.parametrize(
    ("threshold", "expected"),
    [
        (12, ["risk", "high_risk", "high_risk"]),
        (13, ["risk", "risk", "high_risk"]),
    ],
)
def test_severity_labels_support_high_risk_threshold(threshold, expected):
    result = severity_labels([11, 12, 13], high_risk_threshold=threshold)
    assert result.astype("string").tolist() == expected


@pytest.mark.parametrize("bad_threshold", [True, 6, 12.5])
def test_severity_labels_reject_invalid_threshold(bad_threshold):
    with pytest.raises(ValueError, match="7 이상의 정수"):
        severity_labels([0, 1, 12], high_risk_threshold=bad_threshold)
```

- [ ] **Step 2: 테스트가 함수 부재로 실패하는지 확인한다**

Run: `/Users/kodohyeon/Documents/project_LS/venv/bin/python -m pytest tests/test_asset_features.py::test_severity_labels_support_high_risk_threshold tests/test_asset_features.py::test_severity_labels_reject_invalid_threshold -v`

Expected: `severity_labels` import 실패

- [ ] **Step 3: 임계값별 등급 변환을 구현한다**

```python
from typing import Any


def severity_labels(scores: Any, high_risk_threshold: int = 12) -> pd.Categorical:
    if (
        isinstance(high_risk_threshold, bool)
        or not isinstance(high_risk_threshold, int)
        or high_risk_threshold < 7
    ):
        raise ValueError("high_risk_threshold는 7 이상의 정수여야 합니다.")
    values = pd.to_numeric(pd.Series(scores), errors="coerce")
    if values.isna().any() or values.lt(0).any():
        raise ValueError("failure_points는 결측이 없는 0 이상의 숫자여야 합니다.")
    labels = pd.Series("high_risk", index=values.index, dtype="string")
    labels.loc[values.eq(0)] = "normal"
    labels.loc[values.between(1, 5, inclusive="both")] = "caution"
    labels.loc[
        values.between(6, high_risk_threshold - 1, inclusive="both")
    ] = "risk"
    return pd.Categorical(labels, categories=SEVERITY_LEVELS, ordered=True)
```

`add_asset_severity()`는 새 함수를 호출하고 기본값 12를 유지해 기존 호출 결과를 보존한다.

- [ ] **Step 4: 기존·신규 경계 테스트를 실행한다**

Run: `/Users/kodohyeon/Documents/project_LS/venv/bin/python -m pytest tests/test_asset_features.py -v`

Expected: 모든 테스트 PASS

- [ ] **Step 5: 커밋한다**

```bash
git add src/asset_features.py tests/test_asset_features.py
git commit -m "feat: support configurable severity thresholds"
```

### Task 2: 학습 정상행 기반 Robust Z-score

**Files:**
- Create: `src/asset_anomaly_features.py`
- Create: `tests/test_asset_anomaly_features.py`

**Interfaces:**
- Consumes: 장비 일별 DataFrame, `SENSOR_COLUMNS`
- Produces: `RobustNormalBaseline.fit(train)`, `transform(frame)`, `baseline_table()`
- Produces: `ROBUST_Z_FEATURES`

- [ ] **Step 1: 정상행 제한과 센서별 fallback 테스트를 작성한다**

```python
def test_robust_baseline_uses_asset_normal_rows_only(normal_daily):
    abnormal = normal_daily.iloc[[0]].copy()
    abnormal["failure_points"] = 20
    abnormal["temp_bearing_degC"] = 999.0
    train = pd.concat([normal_daily, abnormal], ignore_index=True)
    fitted = RobustNormalBaseline(min_normal_rows=3).fit(train)
    transformed = fitted.transform(abnormal)
    table = fitted.baseline_table().query(
        "asset_tag == 'A-1' and sensor == 'temp_bearing_degC'"
    ).iloc[0]
    expected = (999.0 - table["median"]) / table["scale"]
    assert transformed.iloc[0]["temp_bearing_degC_robust_z"] == pytest.approx(expected)


def test_zero_asset_mad_falls_back_per_sensor_to_machine(normal_daily):
    normal_daily.loc[
        normal_daily.asset_tag.eq("A-1"), "oil_pressure_bar"
    ] = 5.0
    fitted = RobustNormalBaseline(min_normal_rows=3).fit(normal_daily)
    row = fitted.baseline_table().query(
        "asset_tag == 'A-1' and sensor == 'oil_pressure_bar'"
    ).iloc[0]
    assert row["selected_scope"] == "machine_type"
    assert row["fallback_reason"] == "asset_mad_zero"


def test_robust_baseline_requires_normal_training_rows(normal_daily):
    normal_daily["failure_points"] = 1
    with pytest.raises(ValueError, match="정상행이 없습니다"):
        RobustNormalBaseline(min_normal_rows=3).fit(normal_daily)


def test_unseen_asset_uses_known_machine_baseline(normal_daily):
    fitted = RobustNormalBaseline(min_normal_rows=3).fit(normal_daily)
    unseen = normal_daily.iloc[[0]].copy()
    unseen["asset_tag"] = "A-NEW"
    transformed = fitted.transform(unseen)
    assert transformed.filter(like="_robust_z").notna().all().all()
```

`normal_daily`는 두 `asset_tag`, 센서 8개, 각각 정상행 4개를 명시적으로 만드는 fixture다.

- [ ] **Step 2: 새 모듈 부재로 실패하는지 확인한다**

Run: `/Users/kodohyeon/Documents/project_LS/venv/bin/python -m pytest tests/test_asset_anomaly_features.py -v`

Expected: `ModuleNotFoundError`

- [ ] **Step 3: 직렬화 가능한 정상 기준 클래스를 구현한다**

```python
ROBUST_Z_FEATURES = tuple(f"{column}_robust_z" for column in SENSOR_COLUMNS)
BASELINE_COLUMNS = (
    "asset_tag", "machine_type", "sensor", "normal_rows", "median", "mad",
    "scale", "selected_scope", "fallback_reason",
)


@dataclass
class RobustNormalBaseline:
    min_normal_rows: int = 30
    epsilon: float = 1e-12
    _table: pd.DataFrame = field(default_factory=pd.DataFrame, init=False)

    def fit(self, train: pd.DataFrame) -> "RobustNormalBaseline":
        if self.min_normal_rows < 1:
            raise ValueError("min_normal_rows는 1 이상이어야 합니다.")
        normal = train.loc[train["failure_points"].eq(0)].copy()
        if normal.empty:
            raise ValueError("학습 구간에 failure_points == 0인 정상행이 없습니다.")
        self._table = _build_baseline_table(
            normal, self.min_normal_rows, self.epsilon
        )
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self._table.empty:
            raise ValueError("RobustNormalBaseline.fit()을 먼저 호출해야 합니다.")
        return _apply_baseline_table(frame, self._table)

    def baseline_table(self) -> pd.DataFrame:
        if self._table.empty:
            raise ValueError("RobustNormalBaseline.fit()을 먼저 호출해야 합니다.")
        return self._table.copy()
```

`_build_baseline_table()`은 센서마다 `asset_tag -> machine_type -> global MAD -> global std -> constant` 순서로 기준을 고른다. MAD는 `median(abs(x - median(x)))`, scale은 `1.4826 * MAD`다. 마지막까지 scale이 0이면 `scale=1.0`, `selected_scope="constant"`로 기록하고 Z-score는 0으로 둔다.

- [ ] **Step 4: 테스트를 실행한다**

Run: `/Users/kodohyeon/Documents/project_LS/venv/bin/python -m pytest tests/test_asset_anomaly_features.py -v`

Expected: 네 테스트 PASS

- [ ] **Step 5: 커밋한다**

```bash
git add src/asset_anomaly_features.py tests/test_asset_anomaly_features.py
git commit -m "feat: add robust asset sensor baselines"
```

### Task 3: 과거 센서 Feature와 A~D 실험군

**Files:**
- Modify: `src/asset_anomaly_features.py`
- Modify: `tests/test_asset_anomaly_features.py`

**Interfaces:**
- Consumes: 원본 부품 DataFrame, 학습 경계, `RobustNormalBaseline`
- Produces: `AssetExperimentFeatures`, `build_sensor_history_features()`, `prepare_asset_experiment_features()`
- Produces: `FEATURE_SET_NAMES = ("A", "B", "C", "D")`

- [ ] **Step 1: 달력 lag, 미래 비누수와 Feature 집합 테스트를 작성한다**

```python
def test_history_lag_uses_previous_calendar_day(daily_sensor_rows):
    missing = daily_sensor_rows.loc[
        ~daily_sensor_rows.transaction_date.eq(pd.Timestamp("2023-01-02"))
    ]
    result = build_sensor_history_features(missing)
    january_third = result.loc[
        result.transaction_date.eq(pd.Timestamp("2023-01-03"))
    ].iloc[0]
    assert pd.isna(january_third["load_pct_lag1"])


def test_future_change_does_not_change_past_features(daily_sensor_rows):
    before = build_sensor_history_features(daily_sensor_rows)
    changed = daily_sensor_rows.copy()
    changed.loc[changed.transaction_date.eq("2023-01-10"), "load_pct"] = 999.0
    after = build_sensor_history_features(changed)
    columns = [column for column in before if column.startswith("load_pct_")]
    pd.testing.assert_series_equal(
        before.loc[before.transaction_date.eq("2023-01-05"), columns].iloc[0],
        after.loc[after.transaction_date.eq("2023-01-05"), columns].iloc[0],
    )


def test_feature_sets_are_nested_without_targets(raw_asset_history):
    prepared = prepare_asset_experiment_features(
        raw_asset_history, validation_start="2024-01-01", min_normal_rows=3
    )
    assert tuple(prepared.feature_sets) == ("A", "B", "C", "D")
    assert set(prepared.feature_sets["A"]) < set(prepared.feature_sets["B"])
    assert set(prepared.feature_sets["A"]) < set(prepared.feature_sets["C"])
    assert set(prepared.feature_sets["B"]) | set(prepared.feature_sets["C"]) < set(
        prepared.feature_sets["D"]
    )
    forbidden = {"breakdown_flag", "failure_points", "severity_level", "severity_code"}
    assert not forbidden.intersection(prepared.feature_sets["D"])
```

- [ ] **Step 2: 공개 함수 부재로 테스트가 실패하는지 확인한다**

Run: `/Users/kodohyeon/Documents/project_LS/venv/bin/python -m pytest tests/test_asset_anomaly_features.py -v`

Expected: import 또는 attribute 부재로 FAIL

- [ ] **Step 3: 과거 Feature를 과거 달력값만으로 구현한다**

```python
def _rolling_slope(values: np.ndarray) -> float:
    if np.isnan(values).any():
        return np.nan
    return float(np.polyfit(np.arange(len(values), dtype=float), values, 1)[0])


def build_sensor_history_features(daily: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for _, group in daily.groupby(ASSET_COLUMN, sort=False):
        indexed = group.sort_values(DATE_COLUMN).set_index(DATE_COLUMN)
        original_dates = indexed.index
        calendar = indexed.reindex(
            pd.date_range(original_dates.min(), original_dates.max(), freq="D")
        )
        for sensor in SENSOR_COLUMNS:
            shifted = calendar[sensor].shift(1)
            for lag in (1, 3, 7):
                calendar[f"{sensor}_lag{lag}"] = calendar[sensor].shift(lag)
            calendar[f"{sensor}_diff1"] = calendar[sensor] - calendar[f"{sensor}_lag1"]
            calendar[f"{sensor}_diff7"] = calendar[sensor] - calendar[f"{sensor}_lag7"]
            calendar[f"{sensor}_median3"] = shifted.rolling(3, min_periods=2).median()
            calendar[f"{sensor}_median7"] = shifted.rolling(7, min_periods=3).median()
            calendar[f"{sensor}_mad7"] = shifted.rolling(7, min_periods=3).apply(
                lambda x: np.median(np.abs(x - np.median(x))), raw=True
            )
            calendar[f"{sensor}_slope7"] = shifted.rolling(7, min_periods=7).apply(
                _rolling_slope, raw=True
            )
        pieces.append(calendar.loc[original_dates].reset_index())
    return pd.concat(pieces, ignore_index=True)
```

- [ ] **Step 4: A~D 준비 계약과 동시 이상 요약을 구현한다**

```python
ANOMALY_SUMMARY_FEATURES = (
    "max_abs_robust_z", "mean_abs_robust_z", "sensor_count_abs_z_ge_2",
    "sensor_count_abs_z_ge_3", "temperature_and_vibration_anomaly",
)


@dataclass(frozen=True)
class AssetExperimentFeatures:
    frame: pd.DataFrame
    feature_sets: dict[str, tuple[str, ...]]
    baselines: pd.DataFrame
    baseline_transformer: RobustNormalBaseline
```

`prepare_asset_experiment_features()`는 `build_asset_daily()`로 집계하고 `label_end_date < validation_start`인 학습행으로만 baseline을 적합한다. A는 기존 Feature, B는 A+Z-score, C는 A+과거 Feature, D는 B+C+동시 이상 요약으로 만든다.

- [ ] **Step 5: Task 2·3 테스트를 실행한다**

Run: `/Users/kodohyeon/Documents/project_LS/venv/bin/python -m pytest tests/test_asset_anomaly_features.py -v`

Expected: 모든 테스트 PASS

- [ ] **Step 6: 커밋한다**

```bash
git add src/asset_anomaly_features.py tests/test_asset_anomaly_features.py
git commit -m "feat: add asset sensor history feature sets"
```

### Task 4: 모델 후보와 결정적 선택 규칙

**Files:**
- Create: `src/asset_severity_experiments.py`
- Create: `tests/test_asset_severity_experiments.py`

**Interfaces:**
- Consumes: 기존 `build_model()`, `severity_classification_metrics()`
- Produces: `fit_experiment_model()`, `select_candidate()`, `select_feature_set()`, `EXPERIMENT_MODELS`

- [ ] **Step 1: balanced HGB와 선택 우선순위 테스트를 작성한다**

```python
def test_balanced_hist_gradient_boosting_fits_all_classes():
    frame = pd.DataFrame({"signal": range(12)})
    target = pd.Series(["normal"] * 6 + ["caution"] * 3 + ["risk"] * 2 + ["high_risk"])
    model = fit_experiment_model(
        "hist_gradient_boosting_balanced", frame, target,
        max_iter=10, random_state=42,
    )
    assert set(model.named_steps["classifier"].classes_) == set(target)


def test_select_candidate_uses_recall_at_exact_tolerance():
    metrics = pd.DataFrame([
        {"model": "random_forest", "macro_f1": 0.500,
         "high_risk_recall": 0.40, "high_risk_precision": 0.60},
        {"model": "hist_gradient_boosting", "macro_f1": 0.490,
         "high_risk_recall": 0.70, "high_risk_precision": 0.30},
    ])
    assert select_candidate(metrics, tolerance=0.01) == "hist_gradient_boosting"


def test_select_candidate_uses_precision_after_recall_tie():
    metrics = pd.DataFrame([
        {"model": "logistic_regression", "macro_f1": 0.50,
         "high_risk_recall": 0.70, "high_risk_precision": 0.20},
        {"model": "random_forest", "macro_f1": 0.50,
         "high_risk_recall": 0.70, "high_risk_precision": 0.40},
    ])
    assert select_candidate(metrics) == "random_forest"


def test_select_feature_set_uses_same_validation_rule():
    metrics = pd.DataFrame([
        {"feature_set": "A", "macro_f1": 0.510,
         "high_risk_recall": 0.30, "high_risk_precision": 0.60},
        {"feature_set": "D", "macro_f1": 0.505,
         "high_risk_recall": 0.50, "high_risk_precision": 0.40},
    ])
    assert select_feature_set(metrics) == "D"
```

- [ ] **Step 2: 새 모듈 부재로 실패하는지 확인한다**

Run: `/Users/kodohyeon/Documents/project_LS/venv/bin/python -m pytest tests/test_asset_severity_experiments.py -v`

Expected: `ModuleNotFoundError`

- [ ] **Step 3: 모델 적합과 선택을 구현한다**

```python
EXPERIMENT_MODELS = (
    "dummy_classifier", "logistic_regression", "random_forest",
    "hist_gradient_boosting", "hist_gradient_boosting_balanced",
)
LEARNED_EXPERIMENT_MODELS = EXPERIMENT_MODELS[1:]


def fit_experiment_model(model_name, frame, target, *, max_iter, random_state):
    if model_name == "dummy_classifier":
        model = Pipeline([
            ("classifier", DummyClassifier(strategy="prior", random_state=random_state))
        ])
        return model.fit(frame, target.astype(str))
    build_name = (
        "hist_gradient_boosting"
        if model_name == "hist_gradient_boosting_balanced"
        else model_name
    )
    model = build_model(build_name, frame, max_iter=max_iter, random_state=random_state)
    fit_params = {}
    if model_name == "hist_gradient_boosting_balanced":
        fit_params["classifier__sample_weight"] = compute_sample_weight(
            "balanced", target.astype(str)
        )
    return model.fit(frame, target.astype(str), **fit_params)


def _select_name(metrics, name_column, candidates, tolerance):
    eligible = metrics.loc[metrics[name_column].isin(candidates)].copy()
    if eligible.empty:
        raise ValueError("선택할 검증 지표가 없습니다.")
    best = float(eligible.macro_f1.max())
    eligible = eligible.loc[eligible.macro_f1.ge(best - tolerance - 1e-12)]
    order = {name: index for index, name in enumerate(candidates)}
    eligible["candidate_order"] = eligible[name_column].map(order)
    ranked = eligible.sort_values(
        ["high_risk_recall", "high_risk_precision", "macro_f1", "candidate_order"],
        ascending=[False, False, False, True],
    )
    return str(ranked.iloc[0][name_column])


def select_candidate(metrics: pd.DataFrame, tolerance: float = 0.01) -> str:
    return _select_name(
        metrics, "model", LEARNED_EXPERIMENT_MODELS, tolerance
    )


def select_feature_set(metrics: pd.DataFrame, tolerance: float = 0.01) -> str:
    return _select_name(metrics, "feature_set", FEATURE_SET_NAMES, tolerance)
```

- [ ] **Step 4: 테스트를 실행한다**

Run: `/Users/kodohyeon/Documents/project_LS/venv/bin/python -m pytest tests/test_asset_severity_experiments.py -v`

Expected: 네 테스트 PASS

- [ ] **Step 5: 커밋한다**

```bash
git add src/asset_severity_experiments.py tests/test_asset_severity_experiments.py
git commit -m "feat: add severity experiment model selection"
```

### Task 5: 8개 조합 실행·저장·재현

**Files:**
- Modify: `src/asset_severity_experiments.py`
- Modify: `tests/test_asset_severity_experiments.py`

**Interfaces:**
- Consumes: `AssetExperimentFeatures`, 임계값, Feature 집합, 날짜 분할
- Produces: `run_asset_severity_experiments()`, 지표·예측 CSV, JSON, joblib 모델

- [ ] **Step 1: end-to-end 산출물과 12점 경계 테스트를 작성한다**

```python
def test_runner_writes_threshold_outputs(tmp_path, experiment_raw):
    metrics = run_asset_severity_experiments(
        experiment_raw, output_dir=tmp_path,
        high_risk_thresholds=(12, 13), feature_sets=("A",),
        scope="overall", min_normal_rows=2, max_iter=10,
    )
    assert set(metrics.high_risk_threshold) == {12, 13}
    required = {
        "experiment_metrics.csv", "class_metrics.csv",
        "confusion_matrices.csv", "test_predictions.csv",
        "feature_importance.csv", "zscore_baselines.csv",
        "run_config.json", "models",
    }
    assert required <= {path.name for path in tmp_path.iterdir()}
    predictions = pd.read_csv(tmp_path / "test_predictions.csv")
    score_12 = predictions.loc[predictions.actual_failure_points.eq(12)]
    assert set(score_12.loc[score_12.high_risk_threshold.eq(12), "actual_level"]) == {"high_risk"}
    assert set(score_12.loc[score_12.high_risk_threshold.eq(13), "actual_level"]) == {"risk"}


def test_saved_model_reproduces_ordered_probabilities(tmp_path, experiment_raw):
    run_asset_severity_experiments(
        experiment_raw, output_dir=tmp_path,
        high_risk_thresholds=(13,), feature_sets=("A",),
        scope="overall", min_normal_rows=2, max_iter=10,
    )
    model_file = next((tmp_path / "models").glob("severity__13__A__overall__*.joblib"))
    payload = joblib.load(model_file)
    assert payload["class_order"] == list(SEVERITY_LEVELS)
    assert np.allclose(
        payload["pipeline"].predict_proba(payload["verification_rows"]),
        payload["verification_probabilities"],
    )


@pytest.mark.parametrize(
    ("thresholds", "feature_sets", "message"),
    [((6,), ("A",), "7 이상의 정수"), ((12,), ("E",), "A, B, C, D")],
)
def test_runner_rejects_invalid_experiment_options(
    tmp_path, experiment_raw, thresholds, feature_sets, message
):
    with pytest.raises(ValueError, match=message):
        run_asset_severity_experiments(
            experiment_raw, output_dir=tmp_path,
            high_risk_thresholds=thresholds, feature_sets=feature_sets,
            scope="overall", min_normal_rows=2, max_iter=10,
        )


def test_runner_records_single_training_class_as_skipped(
    tmp_path, experiment_raw
):
    train = pd.to_datetime(experiment_raw.transaction_date).lt("2024-01-01")
    experiment_raw.loc[train, "breakdown_flag"] = 0
    metrics = run_asset_severity_experiments(
        experiment_raw, output_dir=tmp_path,
        high_risk_thresholds=(12,), feature_sets=("A",),
        scope="overall", min_normal_rows=2, max_iter=10,
    )
    assert set(metrics.status) == {"skipped"}
    assert metrics.reason.str.contains("single train class").all()
```

`experiment_raw` fixture는 각 날짜 구간에 네 등급과 12점 행을 포함하도록 A/B/C 중요도 부품의 `breakdown_flag`를 직접 구성한다.

- [ ] **Step 2: 실행 함수 부재로 실패하는지 확인한다**

Run: `/Users/kodohyeon/Documents/project_LS/venv/bin/python -m pytest tests/test_asset_severity_experiments.py::test_runner_writes_threshold_outputs tests/test_asset_severity_experiments.py::test_saved_model_reproduces_ordered_probabilities -v`

Expected: `run_asset_severity_experiments` 부재로 FAIL

- [ ] **Step 3: 전체·기계 종류 Scope와 모델 반복을 구현한다**

```python
prepared = prepare_asset_experiment_features(
    raw, validation_start=validation_start, min_normal_rows=min_normal_rows
)
for high_risk_threshold in high_risk_thresholds:
    target = f"severity_level_ge_{high_risk_threshold}"
    prepared.frame[target] = severity_labels(
        prepared.frame["failure_points"], high_risk_threshold
    )
    for scope_kind, scope_name, subset in iter_experiment_scopes(
        prepared.frame, scope, machine_type, asset_tag
    ):
        parts = split_by_date(subset, validation_start, test_start)
        for feature_set in feature_sets:
            features = list(prepared.feature_sets[feature_set])
            for model_name in EXPERIMENT_MODELS:
                model = fit_experiment_model(
                    model_name, parts["train"][features], parts["train"][target],
                    max_iter=max_iter, random_state=random_state,
                )
```

`all`은 `overall`과 다섯 `machine_type` 모델만 만든다. 각 Feature 집합에서 검증 지표로 모델을 고르고, 같은 임계값·Scope에서 선택 모델끼리 같은 규칙으로 대표 Feature를 고른다. 테스트 성능은 선택에 사용하지 않는다.

실행 시작 시 임계값은 `severity_labels()`와 같은 규칙으로 검증하고 Feature 집합은 A~D만 허용한다. 빈 학습·검증·테스트 분할이나 학습 Target이 한 등급뿐이면 모델을 적합하지 않고 `status="skipped"`, 구체적인 `reason`을 `experiment_metrics.csv`에 남긴다.

- [ ] **Step 4: long-format 지표·혼동행렬·예측과 모델을 저장한다**

CSV 계약:

```text
experiment_metrics.csv: threshold, feature_set, scope, model, split, 선택표시, 전체 지표
class_metrics.csv: 위 식별자 + level, precision, recall, f1, support
confusion_matrices.csv: 위 식별자 + actual_level, predicted_level, count
test_predictions.csv: 날짜, 기계, 장비, 실제 점수·등급, 예측 등급, 확률 4개
feature_importance.csv: 위 식별자 + feature, importance_mean, importance_std
zscore_baselines.csv: Task 2 BASELINE_COLUMNS
```

joblib payload에는 `OrderedSeverityClassifier`, Feature 목록, 임계값, Feature 집합, class 순서, baseline transformer와 검증용 행·확률을 저장한다.

`run_config.json`에는 임계값, Feature 집합, Scope, 필터, 날짜 경계, 정상 최소 표본 수, `max_iter`, 난수 시드, 모델 목록과 sklearn 버전을 저장한다.

- [ ] **Step 5: runner와 기존 회귀 테스트를 실행한다**

Run: `/Users/kodohyeon/Documents/project_LS/venv/bin/python -m pytest tests/test_asset_severity_experiments.py tests/test_industrial_regression.py -v`

Expected: 모든 테스트 PASS

- [ ] **Step 6: 커밋한다**

```bash
git add src/asset_severity_experiments.py tests/test_asset_severity_experiments.py
git commit -m "feat: run severity threshold feature experiments"
```

### Task 6: 발생률 프로필과 한국어 결과 보고서

**Files:**
- Create: `src/asset_severity_report.py`
- Create: `tests/test_asset_severity_report.py`
- Modify: `src/asset_severity_experiments.py`

**Interfaces:**
- Consumes: 원본 부품행, 장비 일별행, 지표와 예측
- Produces: `build_failure_profile()`, `render_experiment_summary()`
- Produces: `machine_failure_profile.csv`, `experiment_summary.md`

- [ ] **Step 1: 세 발생률과 필수 보고 문구 테스트를 작성한다**

```python
def test_failure_profile_keeps_rate_definitions(raw_rows, daily_rows):
    profile = build_failure_profile(
        raw_rows, daily_rows, high_risk_thresholds=(12, 13),
        validation_start="2024-01-01", test_start="2024-07-01",
    )
    assert {
        "part_breakdown_rate", "asset_issue_day_rate",
        "asset_high_risk_day_rate", "mean_failure_points",
        "median_failure_points",
    } <= set(profile)
    assert set(profile.scope_kind) == {"machine_type", "asset_tag"}
    assert set(profile.high_risk_threshold) == {12, 13}


def test_summary_states_proxy_limit(metrics_frame, predictions_frame, profile_frame):
    text = render_experiment_summary(metrics_frame, predictions_frame, profile_frame)
    for phrase in ("실제 기계 정지율이 아닙니다", "12점 기준", "13점 기준", "12점 장비일", "Macro F1"):
        assert phrase in text
```

- [ ] **Step 2: 모듈 부재로 실패하는지 확인한다**

Run: `/Users/kodohyeon/Documents/project_LS/venv/bin/python -m pytest tests/test_asset_severity_report.py -v`

Expected: `ModuleNotFoundError`

- [ ] **Step 3: 분할·Scope별 발생률 프로필을 구현한다**

```python
PROFILE_COLUMNS = (
    "split", "scope_kind", "scope_name", "high_risk_threshold",
    "part_rows", "asset_days", "part_breakdown_rate",
    "asset_issue_day_rate", "asset_high_risk_day_rate",
    "mean_failure_points", "median_failure_points",
)
```

원본 행에서 `part_breakdown_rate`를 계산하고, 장비 일별 행에서 `asset_issue_day_rate`, `asset_high_risk_day_rate`, 점수 평균과 중앙값을 계산한다. 날짜 분할은 모델과 같고 `machine_type`, `asset_tag` 두 Scope를 모두 만든다.

- [ ] **Step 4: 한국어 Markdown 보고서를 구현한다**

고정 섹션은 다음과 같다.

```python
SUMMARY_SECTIONS = (
    "# 산업 장비 4단계 위험도 실험 결과",
    "## 1. 해석 전 주의사항",
    "## 2. 12점·13점 등급 분포",
    "## 3. Feature A~D 성능 비교",
    "## 4. 기계 종류별 결과",
    "## 5. 개별 장비별 발생률",
    "## 6. 12점 장비일 분석",
    "## 7. 다음 실험 권고",
)
```

검증에서 선택한 모델·Feature의 테스트 결과만 대표 결과로 쓰고, D가 A보다 낮으면 감소량을 그대로 기록한다.

- [ ] **Step 5: runner 저장과 테스트를 실행한다**

Run: `/Users/kodohyeon/Documents/project_LS/venv/bin/python -m pytest tests/test_asset_severity_report.py tests/test_asset_severity_experiments.py -v`

Expected: 모든 테스트 PASS

- [ ] **Step 6: 커밋한다**

```bash
git add src/asset_severity_report.py src/asset_severity_experiments.py tests/test_asset_severity_report.py tests/test_asset_severity_experiments.py
git commit -m "feat: report severity experiment results"
```

### Task 7: 전용 CLI와 한글 가이드

**Files:**
- Create: `src/current_asset_severity_experiments.py`
- Modify: `tests/test_industrial_entrypoints.py`
- Modify: `src/RUN_INDUSTRIAL.md`
- Modify: `docs/industrial_feature_guide.md`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: `run_asset_severity_experiments()`
- Produces: `python src/current_asset_severity_experiments.py`

- [ ] **Step 1: CLI 기본값과 help 테스트를 작성한다**

```python
def test_asset_severity_experiment_defaults():
    args = build_severity_experiment_parser().parse_args([])
    assert args.output == SEVERITY_EXPERIMENT_OUTPUT
    assert args.high_risk_thresholds == [12, 13]
    assert args.feature_sets == ["A", "B", "C", "D"]
    assert args.scope == "all"
    assert args.min_normal_rows == 30
```

기존 `test_entrypoint_help` 목록에 `src/current_asset_severity_experiments.py`를 추가한다.

- [ ] **Step 2: 새 CLI 부재로 실패하는지 확인한다**

Run: `/Users/kodohyeon/Documents/project_LS/venv/bin/python -m pytest tests/test_industrial_entrypoints.py -v`

Expected: 새 모듈 import 또는 help 실행 FAIL

- [ ] **Step 3: 직접 실행 가능한 CLI를 구현한다**

```python
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "asset_severity_experiments"


def build_parser(default_output: Path = DEFAULT_OUTPUT) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="장비 당일 4단계 위험도 센서 이상 Feature 비교"
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--high-risk-thresholds", nargs="+", type=int, default=[12, 13])
    parser.add_argument(
        "--feature-sets", nargs="+", choices=["A", "B", "C", "D"],
        default=["A", "B", "C", "D"],
    )
    parser.add_argument("--scope", choices=["all", "overall", "machine_type"], default="all")
    parser.add_argument("--machine-type")
    parser.add_argument("--asset-tag")
    parser.add_argument("--validation-start", default="2024-01-01")
    parser.add_argument("--test-start", default="2024-07-01")
    parser.add_argument("--min-normal-rows", type=int, default=30)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--random-state", type=int, default=42)
    return parser
```

직접 실행 import 보정은 기존 `current_asset_score_model.py` 패턴을 그대로 사용한다. `main()`은 CSV를 한 번 읽고 인자를 runner에 전달한다.

- [ ] **Step 4: 실행 가이드·Feature 가이드·gitignore를 갱신한다**

문서에 다음 실제 명령을 넣는다.

```bash
/Users/kodohyeon/Documents/project_LS/venv/bin/python src/current_asset_severity_experiments.py \
  --data dataVerification/synthetic_industrial_machine_data.csv \
  --high-risk-thresholds 12 13 --feature-sets A B C D --scope all --max-iter 100
```

Feature 가이드에는 두 등급 기준, Robust Z-score 식, A~D, 학습 정상행 제한과 세 발생률 용어를 추가한다. `.gitignore`에는 `outputs/asset_severity_experiments/`를 추가한다.

- [ ] **Step 5: 관련 테스트를 실행한다**

Run: `/Users/kodohyeon/Documents/project_LS/venv/bin/python -m pytest tests/test_industrial_entrypoints.py tests/test_asset_features.py tests/test_asset_anomaly_features.py tests/test_asset_severity_experiments.py tests/test_asset_severity_report.py -v`

Expected: 모든 테스트 PASS

- [ ] **Step 6: 커밋한다**

```bash
git add .gitignore src/current_asset_severity_experiments.py src/RUN_INDUSTRIAL.md docs/industrial_feature_guide.md tests/test_industrial_entrypoints.py
git commit -m "docs: add severity experiment command"
```

### Task 8: 전체 검증과 실제 8개 실험

**Files:**
- Verify: `tests/`
- Generate (gitignored): `outputs/asset_severity_experiments/`
- Modify only after a failing regression test identifies a defect: owner source/test from Tasks 1~7

**Interfaces:**
- Consumes: 실제 합성 CSV, 완성 CLI
- Produces: 전체 테스트 증거, 8개 조합 결과, 한국어 결과 요약

- [ ] **Step 1: 전체 테스트를 실행한다**

Run: `/Users/kodohyeon/Documents/project_LS/venv/bin/python -m pytest -q`

Expected: 기존 64개와 새 테스트 PASS, 기존 예상 실패 2개만 XFAIL

- [ ] **Step 2: 실제 8개 실험을 실행한다**

Run:

```bash
/Users/kodohyeon/Documents/project_LS/venv/bin/python src/current_asset_severity_experiments.py \
  --data dataVerification/synthetic_industrial_machine_data.csv \
  --high-risk-thresholds 12 13 --feature-sets A B C D --scope all --max-iter 100
```

Expected: exit code 0과 `outputs/asset_severity_experiments/experiment_summary.md`

- [ ] **Step 3: 산출물과 선택 행을 기계적으로 확인한다**

Run:

```bash
/Users/kodohyeon/Documents/project_LS/venv/bin/python - <<'PY'
import pandas as pd
from pathlib import Path

root = Path("outputs/asset_severity_experiments")
required = {
    "experiment_metrics.csv", "class_metrics.csv", "confusion_matrices.csv",
    "test_predictions.csv", "machine_failure_profile.csv",
    "feature_importance.csv", "zscore_baselines.csv", "run_config.json",
    "experiment_summary.md",
}
assert required <= {path.name for path in root.iterdir()}
metrics = pd.read_csv(root / "experiment_metrics.csv")
assert set(metrics.high_risk_threshold) == {12, 13}
assert set(metrics.feature_set) == {"A", "B", "C", "D"}
selected = metrics.query("split == 'test' and selected_model == True")
assert not selected.empty
predictions = pd.read_csv(root / "test_predictions.csv")
assert set(predictions.high_risk_threshold) == {12, 13}
print(selected[[
    "high_risk_threshold", "feature_set", "scope_kind", "scope_name", "model",
    "selected_feature_set", "accuracy", "macro_f1",
    "high_risk_precision", "high_risk_recall",
]].to_string(index=False))
PY
```

Expected: assertion 실패 없이 선택 결과 출력

- [ ] **Step 4: 결과의 의미를 검토한다**

1. A12 overall Macro F1이 기존 0.4851과 합리적인 범위에서 재현되는지 확인한다.
2. B·C·D가 같은 임계값의 A보다 개선 또는 악화됐는지 확인한다.
3. 12·13점 고위험 support가 각각 221·151인지 확인한다.
4. Belt Conveyor와 CNC Lathe의 고위험 Recall 0 문제가 개선됐는지 확인한다.
5. Z-score 중요도와 `day_of_week` 의존도 변화를 확인한다.
6. `asset_tag`별 고위험 점수 발생률이 보고서에 포함됐는지 확인한다.

코드 결함이면 소유 테스트를 먼저 실패하게 만든 뒤 최소 수정하고 전체 테스트와 실제 실행을 반복한다. 데이터 신호 한계면 결과를 바꾸지 않고 보고서에 기록한다.

- [ ] **Step 5: 저장소 상태를 확인한다**

Run:

```bash
git status --short --branch
git log --oneline -8
```

Expected: 소스·테스트·문서는 커밋되어 있고 출력 폴더는 gitignored 상태

- [ ] **Step 6: 실행 중 수정이 있었을 때만 마지막 커밋을 만든다**

전체 테스트 PASS를 다시 확인한 다음 관련 파일만 추가한다.

```bash
git add src tests
git commit -m "fix: finalize severity experiment execution"
```
