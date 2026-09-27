# 산업 장비 점수·4단계 위험도 모델 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 장비·날짜별 `failure_points` 회귀와 정상·주의·위험·고위험 4단계 분류를 기존 이진 모델 옆에 추가하고 동일한 시간 분할로 비교 가능한 산출물을 만든다.

**Architecture:** 기존 `build_asset_daily()`와 공통 전처리를 재사용하되 회귀 계약과 학습 실행기는 분리한다. 하나의 새 CLI가 회귀 과제와 4단계 분류 과제를 같은 scope·날짜 분할로 실행하고, 선택 모델·예측·지표를 `outputs/asset_score_current`에 저장한다. 기존 장비 이진 모델과 부품·미래 모델은 변경하지 않는다.

**Tech Stack:** Python 3.11, pandas, NumPy, scikit-learn, joblib, pytest

**Spec:** `docs/superpowers/specs/2026-09-24-industrial-asset-severity-model-design.md`

## Global Constraints

- 실제 기계 정지 Label이 없으므로 `high_risk`를 실제 고장 또는 생산 정지로 표현하지 않는다.
- 등급은 `normal=0점`, `caution=1~5점`, `risk=6~11점`, `high_risk=12점 이상`으로 고정한다.
- 회귀 예측 등급 경계는 `[0, 0.5)`, `[0.5, 5.5)`, `[5.5, 11.5)`, `[11.5, ∞)`를 사용한다.
- 회귀·분류 Feature는 `ASSET_CURRENT_FEATURES`만 사용하고 `breakdown_flag`, `failure_points`, `severity_level`, `severity_code`를 입력에 포함하지 않는다.
- 학습·검증·테스트 경계는 각각 `2024-01-01`, `2024-07-01`을 기본값으로 사용하며 전처리는 학습 구간에만 적합한다.
- 모델 선택에는 검증 구간만 사용하고 테스트 구간은 최종 평가에만 사용한다.
- 기존 `current_asset_model.py`와 기존 산출물 스키마를 변경하지 않는다.
- 테스트는 프로젝트 루트에서 `python -m pytest ...`로 실행하며 테스트 파일을 Python으로 직접 실행하지 않는다.
- LightGBM과 새 외부 의존성을 추가하지 않는다.

## Review Focus

- 회귀 예측값이 정확히 `0.5`, `5.5`, `11.5`일 때 각각 상위 등급으로 들어가야 한다. Task 2의 경계 테스트로 고정한다.
- `failure_points`가 결측·문자열·음수이면 정상 등급으로 숨기지 말고 `ValueError`가 발생해야 한다. Task 1의 입력 검증 테스트로 고정한다.
- 검증·테스트에 학습 시 없던 `asset_tag`가 나와도 One-Hot 전처리가 실패하지 않아야 한다. Task 2의 미지 범주 예측 테스트로 고정한다.
- 빈 분할 또는 학습 구간의 단일 등급은 명시적인 `skipped` 상태와 이유를 기록해야 한다. Task 3의 건너뛰기 테스트로 고정한다.
- 저장한 회귀·다중분류 모델을 다시 불러왔을 때 예측값, 클래스 순서, 확률 열 순서가 같아야 한다. Task 3의 재현성 테스트로 고정한다.

---

## 파일 구조

- `src/industrial_data.py`: 회귀 학습 데이터 계약 `PreparedRegressionTask`를 소유한다.
- `src/asset_features.py`: 점수→등급 변환과 장비 점수/등급 과제 준비를 소유한다.
- `src/industrial_training.py`: 분류와 회귀가 공유하는 전처리 생성 함수만 공개한다.
- `src/industrial_regression.py`: 회귀·4단계 분류 모델 생성, 선택, 평가, 저장을 소유한다.
- `src/current_asset_score_model.py`: 새 파이프라인의 명령행 진입점만 소유한다.
- `tests/test_asset_features.py`: 등급 경계와 준비 과제 계약을 검증한다.
- `tests/test_industrial_regression.py`: 회귀·다중분류 지표와 학습 산출물을 검증한다.
- `tests/test_industrial_entrypoints.py`: 새 CLI의 help와 기본값을 검증한다.
- `src/RUN_INDUSTRIAL.md`, `docs/industrial_code_guide.md`, `docs/industrial_feature_guide.md`: 실행·코드·Feature 설명을 갱신한다.

### Task 1: 회귀 계약과 4단계 Target 생성

**Files:**
- Modify: `src/industrial_data.py:47-85`
- Modify: `src/asset_features.py:12-116`
- Modify: `tests/test_asset_features.py:1-68`
- Test: `tests/test_industrial_data.py`

**Interfaces:**
- Consumes: 기존 `build_asset_daily(frame: pd.DataFrame) -> pd.DataFrame`, `ASSET_CURRENT_FEATURES`
- Produces: `PreparedRegressionTask`, `add_asset_severity(frame)`, `prepare_asset_score_current(frame)`, `prepare_asset_severity_current(frame)`

- [ ] **Step 1: 등급 경계와 오류 입력에 대한 실패 테스트 작성**

`tests/test_asset_features.py`의 import에 새 함수를 추가하고 다음 테스트를 작성한다.

```python
import numpy as np

from src.asset_features import (
    add_asset_severity,
    build_asset_daily,
    prepare_asset_current,
    prepare_asset_forecast,
    prepare_asset_score_current,
    prepare_asset_severity_current,
)


def test_asset_severity_uses_all_integer_boundaries():
    frame = pd.DataFrame({"failure_points": [0, 1, 5, 6, 11, 12, 30]})
    result = add_asset_severity(frame)
    assert result["severity_level"].astype("string").tolist() == [
        "normal", "caution", "caution", "risk", "risk", "high_risk",
        "high_risk",
    ]
    assert result["severity_code"].tolist() == [0, 1, 1, 2, 2, 3, 3]


@pytest.mark.parametrize("bad_value", [-1, np.nan, "bad"])
def test_asset_severity_rejects_invalid_failure_points(bad_value):
    with pytest.raises(ValueError, match="failure_points"):
        add_asset_severity(pd.DataFrame({"failure_points": [bad_value]}))
```

- [ ] **Step 2: 과제 준비 함수의 Target 누수 방지 실패 테스트 작성**

```python
def test_asset_score_and_severity_tasks_keep_targets_out_of_features():
    score_task = prepare_asset_score_current(_rows())
    severity_task = prepare_asset_severity_current(_rows())

    assert score_task.target == "failure_points"
    assert severity_task.target == "severity_level"
    assert "failure_points" not in score_task.features
    assert "severity_level" not in severity_task.features
    assert "severity_code" not in severity_task.features
    assert score_task.frame["severity_level"].astype("string").tolist() == [
        "high_risk"
    ]
```

- [ ] **Step 3: 새 테스트를 실행해 실패 확인**

Run: `python -m pytest tests/test_asset_features.py -k "severity or score_and" -v`

Expected: FAIL with import errors for `add_asset_severity`, `prepare_asset_score_current`, and `prepare_asset_severity_current`.

- [ ] **Step 4: `PreparedRegressionTask` 최소 계약 구현**

`src/industrial_data.py`의 `PreparedTask` 아래에 다음 계약을 추가한다.

```python
@dataclass(frozen=True)
class PreparedRegressionTask:
    """회귀 Feature, Target과 과제 메타데이터를 보관한다."""

    frame: pd.DataFrame
    features: tuple[str, ...]
    target: str
    grain: str
    mode: str
    risk_definition: str = "score"

    def __post_init__(self) -> None:
        if self.target in self.features:
            raise ValueError("Target 컬럼을 Feature에 포함할 수 없습니다.")
        missing = [
            column
            for column in (*self.features, self.target)
            if column not in self.frame
        ]
        if missing:
            raise ValueError(
                f"PreparedRegressionTask에 필요한 컬럼이 없습니다: {missing}"
            )
```

- [ ] **Step 5: 점수→등급 함수와 준비 함수 구현**

`src/asset_features.py`에 다음 상수와 함수를 추가한다.

```python
SEVERITY_LEVELS = ("normal", "caution", "risk", "high_risk")
SEVERITY_CODE = {name: index for index, name in enumerate(SEVERITY_LEVELS)}


def add_asset_severity(frame: pd.DataFrame) -> pd.DataFrame:
    """정수 고장점수를 정상·주의·위험·고위험으로 변환한다."""
    if "failure_points" not in frame:
        raise ValueError("failure_points 컬럼이 필요합니다.")
    result = frame.copy()
    scores = pd.to_numeric(result["failure_points"], errors="coerce")
    if scores.isna().any() or scores.lt(0).any():
        raise ValueError("failure_points는 결측이 없는 0 이상의 숫자여야 합니다.")
    labels = pd.Series("high_risk", index=result.index, dtype="string")
    labels.loc[scores.eq(0)] = "normal"
    labels.loc[scores.between(1, 5, inclusive="both")] = "caution"
    labels.loc[scores.between(6, 11, inclusive="both")] = "risk"
    result["failure_points"] = scores
    result["severity_level"] = pd.Categorical(
        labels, categories=SEVERITY_LEVELS, ordered=True
    )
    result["severity_code"] = labels.map(SEVERITY_CODE).astype(int)
    return result


def prepare_asset_score_current(frame: pd.DataFrame) -> PreparedRegressionTask:
    daily = add_asset_severity(build_asset_daily(frame))
    return PreparedRegressionTask(
        frame=daily,
        features=ASSET_CURRENT_FEATURES,
        target="failure_points",
        grain="asset",
        mode="current",
    )


def prepare_asset_severity_current(frame: pd.DataFrame) -> PreparedTask:
    daily = add_asset_severity(build_asset_daily(frame))
    return PreparedTask(
        frame=daily,
        features=ASSET_CURRENT_FEATURES,
        target="severity_level",
        grain="asset",
        mode="current",
        risk_definition="severity_4class",
    )
```

`PreparedRegressionTask`를 `src/asset_features.py` import 목록에 추가한다.

- [ ] **Step 6: 경계·계약 테스트 통과 확인**

Run: `python -m pytest tests/test_asset_features.py tests/test_industrial_data.py -v`

Expected: 모든 테스트 PASS.

- [ ] **Step 7: Task 1 커밋**

```bash
git add src/industrial_data.py src/asset_features.py tests/test_asset_features.py tests/test_industrial_data.py
git commit -m "feat: add asset severity targets"
```

### Task 2: 공통 전처리와 회귀·등급 평가 도구

**Files:**
- Modify: `src/industrial_training.py:240-303`
- Create: `src/industrial_regression.py`
- Create: `tests/test_industrial_regression.py`
- Test: `tests/test_industrial_training.py`

**Interfaces:**
- Consumes: `SEVERITY_LEVELS`, 기존 범주형/수치형 scikit-learn 전처리 규칙
- Produces: `build_preprocessor(frame)`, `build_regression_model(...)`, `predicted_severity(...)`, `score_regression_metrics(...)`, `severity_classification_metrics(...)`

- [ ] **Step 1: 공통 전처리 재사용과 미지 범주에 대한 실패 테스트 작성**

`tests/test_industrial_training.py`에 다음 테스트를 추가한다.

```python
from src.industrial_training import build_preprocessor


def test_shared_preprocessor_accepts_unseen_category():
    train = pd.DataFrame({"asset_tag": ["A-1", "A-2"], "signal": [1.0, 2.0]})
    valid = pd.DataFrame({"asset_tag": ["NEW"], "signal": [3.0]})
    preprocessor = build_preprocessor(train)
    preprocessor.fit(train)
    transformed = preprocessor.transform(valid)
    assert transformed.shape[0] == 1


@pytest.mark.parametrize(
    "leak", ["failure_points", "severity_level", "severity_code"]
)
def test_validate_features_rejects_asset_score_targets(leak):
    with pytest.raises(ValueError, match=leak):
        validate_features(["signal", leak], "severity_level")
```

- [ ] **Step 2: 예측점수 경계와 지표 실패 테스트 작성**

`tests/test_industrial_regression.py`를 만들고 다음 테스트를 작성한다.

```python
import numpy as np
import pandas as pd
import pytest

from src.industrial_regression import (
    build_regression_model,
    predicted_severity,
    score_regression_metrics,
    severity_classification_metrics,
)


def test_predicted_severity_uses_half_point_boundaries():
    result = predicted_severity([-2.0, 0.49, 0.5, 5.49, 5.5, 11.49, 11.5])
    assert result.tolist() == [
        "normal", "normal", "caution", "caution", "risk", "risk",
        "high_risk",
    ]


def test_regression_metrics_include_score_and_high_risk_results():
    metrics = score_regression_metrics(
        [0, 4, 8, 12], [0.1, 4.4, 8.2, 12.1]
    )
    assert metrics["mae"] == pytest.approx(0.2)
    assert metrics["macro_f1"] == 1.0
    assert metrics["high_risk_recall"] == 1.0


def test_severity_metrics_keep_fixed_class_order_when_class_is_absent():
    metrics = severity_classification_metrics(
        ["normal", "risk"], ["normal", "normal"]
    )
    assert metrics["confusion_matrix"] == [
        [1, 0, 0, 0],
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    assert metrics["high_risk_support"] == 0
```

- [ ] **Step 3: 새 테스트를 실행해 실패 확인**

Run: `python -m pytest tests/test_industrial_training.py::test_shared_preprocessor_accepts_unseen_category tests/test_industrial_regression.py -v`

Expected: FAIL because `build_preprocessor` and `industrial_regression` do not exist.

- [ ] **Step 4: 기존 전처리를 공개 함수로 추출**

`src/industrial_training.py`에서 `build_model()`의 전처리 부분을 다음 함수로 추출하고 `build_model()`이 이를 호출하게 한다.

```python
def build_preprocessor(frame: pd.DataFrame) -> ColumnTransformer:
    """학습 Frame의 범주형·수치형 컬럼에 맞는 전처리를 만든다."""
    categorical, numeric = _column_types(frame)
    transformers: list[tuple[str, Any, list[str]]] = []
    if categorical:
        transformers.append((
            "category",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            categorical,
        ))
    if numeric:
        transformers.append((
            "numeric",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            numeric,
        ))
    return ColumnTransformer(transformers=transformers, remainder="drop")
```

`build_model()` 안에서는 다음 한 줄을 사용한다.

```python
preprocess = build_preprocessor(frame)
```

같은 파일의 `validate_features()`에는 장비 점수·등급 Target을 입력에서 차단하는 다음
규칙을 추가한다.

```python
ASSET_SCORE_TARGETS = {"failure_points", "severity_level", "severity_code"}

# validate_features() 안에서 기존 forbidden 계산에 합친다.
forbidden = forbidden | (set(features) & ASSET_SCORE_TARGETS)
```

- [ ] **Step 5: 회귀 모델과 연속 예측 등급 변환 구현**

`src/industrial_regression.py`에 다음 기반 코드를 작성한다.

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.pipeline import Pipeline

from .asset_features import SEVERITY_LEVELS
from .industrial_training import build_model, build_preprocessor

REGRESSION_MODELS = (
    "dummy_regressor",
    "random_forest_regressor",
    "hist_gradient_boosting_regressor",
)
LEARNED_REGRESSION_MODELS = REGRESSION_MODELS[1:]


def build_regression_model(
    model_name: str,
    frame: pd.DataFrame,
    max_iter: int = 100,
    random_state: int = 42,
) -> Pipeline:
    preprocess = build_preprocessor(frame)
    if model_name == "dummy_regressor":
        regressor = DummyRegressor(strategy="mean")
    elif model_name == "random_forest_regressor":
        regressor = RandomForestRegressor(
            n_estimators=max(50, max_iter),
            random_state=random_state,
            n_jobs=-1,
        )
    elif model_name == "hist_gradient_boosting_regressor":
        regressor = HistGradientBoostingRegressor(
            max_iter=max_iter,
            early_stopping=False,
            random_state=random_state,
        )
    else:
        raise ValueError(f"지원하지 않는 회귀 모델입니다: {model_name}")
    return Pipeline([("preprocess", preprocess), ("regressor", regressor)])


def predicted_severity(scores: Any) -> np.ndarray:
    values = np.maximum(np.asarray(scores, dtype=float), 0.0)
    return np.select(
        [values < 0.5, values < 5.5, values < 11.5],
        ["normal", "caution", "risk"],
        default="high_risk",
    )
```

- [ ] **Step 6: 고정 순서 다중분류 지표 구현**

같은 파일에 다음 지표 함수를 추가한다.

```python
def severity_classification_metrics(
    y_true: Any, y_pred: Any
) -> dict[str, Any]:
    actual = np.asarray(y_true, dtype=str)
    predicted = np.asarray(y_pred, dtype=str)
    precision, recall, f1, support = precision_recall_fscore_support(
        actual,
        predicted,
        labels=list(SEVERITY_LEVELS),
        zero_division=0,
    )
    result: dict[str, Any] = {
        "accuracy": float(accuracy_score(actual, predicted)),
        "macro_precision": float(precision_score(
            actual, predicted, labels=list(SEVERITY_LEVELS),
            average="macro", zero_division=0,
        )),
        "macro_recall": float(recall_score(
            actual, predicted, labels=list(SEVERITY_LEVELS),
            average="macro", zero_division=0,
        )),
        "macro_f1": float(f1_score(
            actual, predicted, labels=list(SEVERITY_LEVELS),
            average="macro", zero_division=0,
        )),
        "weighted_f1": float(f1_score(
            actual, predicted, labels=list(SEVERITY_LEVELS),
            average="weighted", zero_division=0,
        )),
        "confusion_matrix": confusion_matrix(
            actual, predicted, labels=list(SEVERITY_LEVELS)
        ).tolist(),
    }
    for index, level in enumerate(SEVERITY_LEVELS):
        result[f"{level}_precision"] = float(precision[index])
        result[f"{level}_recall"] = float(recall[index])
        result[f"{level}_f1"] = float(f1[index])
        result[f"{level}_support"] = int(support[index])
    actual_high = actual == "high_risk"
    predicted_high = predicted == "high_risk"
    result["high_risk_precision"] = float(precision_score(
        actual_high, predicted_high, zero_division=0
    ))
    result["high_risk_recall"] = float(recall_score(
        actual_high, predicted_high, zero_division=0
    ))
    result["high_risk_binary_f1"] = float(f1_score(
        actual_high, predicted_high, zero_division=0
    ))
    return result
```

- [ ] **Step 7: 회귀 지표 구현**

```python
def score_regression_metrics(y_true: Any, y_pred: Any) -> dict[str, Any]:
    actual = np.asarray(y_true, dtype=float)
    predicted = np.maximum(np.asarray(y_pred, dtype=float), 0.0)
    severity = severity_classification_metrics(
        pd.cut(
            actual,
            bins=[-np.inf, 0, 5, 11, np.inf],
            labels=SEVERITY_LEVELS,
        ).astype("string"),
        predicted_severity(predicted),
    )
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(mean_squared_error(actual, predicted) ** 0.5),
        "r2": float(r2_score(actual, predicted)) if len(actual) >= 2 else None,
        "spearman": float(pd.Series(actual).corr(
            pd.Series(predicted), method="spearman"
        )) if len(actual) >= 2 else None,
        **severity,
    }
```

구현할 때 실제 점수의 등급 변환은 `add_asset_severity()`와 경계가 어긋나지 않도록 공통 `observed_severity(scores)` 함수로 추출한다. `pd.cut()`의 0점 포함 방식에 의존하지 말고 `0`, `1~5`, `6~11`, `12+`를 명시적으로 검사한다.

- [ ] **Step 8: 평가 도구 테스트 통과와 기존 전처리 회귀 확인**

Run: `python -m pytest tests/test_industrial_regression.py tests/test_industrial_training.py -v`

Expected: 모든 테스트 PASS. 기존 `test_logistic_regression_scales_numeric_features`도 PASS.

- [ ] **Step 9: Task 2 커밋**

```bash
git add src/industrial_training.py src/industrial_regression.py tests/test_industrial_training.py tests/test_industrial_regression.py
git commit -m "feat: add asset score training primitives"
```

### Task 3: 회귀·4단계 분류 실행기와 저장 계약

**Files:**
- Modify: `src/industrial_regression.py`
- Modify: `tests/test_industrial_regression.py`

**Interfaces:**
- Consumes: `PreparedRegressionTask`, `PreparedTask`, `split_by_date()`, `iter_scopes()`, `build_regression_model()`, 기존 `build_model()`
- Produces: `run_asset_score_suite(score_task, severity_task, *, output_dir, scope, machine_type, asset_tag, max_iter, validation_start, test_start, random_state) -> tuple[pd.DataFrame, pd.DataFrame]`

- [ ] **Step 1: 작은 3구간 학습 Fixture 작성**

`tests/test_industrial_regression.py`에 각 분할이 네 등급을 포함하는 Fixture를 추가한다.

```python
from src.asset_features import SEVERITY_LEVELS
from src.industrial_data import PreparedRegressionTask, PreparedTask


def _score_tasks(single_train_class: bool = False):
    dates = pd.to_datetime([
        "2023-12-01", "2023-12-02", "2023-12-03", "2023-12-04",
        "2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04",
        "2024-07-01", "2024-07-02", "2024-07-03", "2024-07-04",
    ])
    points = [0, 3, 8, 12] * 3
    if single_train_class:
        points[:4] = [0, 0, 0, 0]
    frame = pd.DataFrame({
        "transaction_date": dates,
        "label_end_date": dates,
        "machine_type": ["Press"] * 12,
        "asset_tag": ["A-1", "A-2"] * 6,
        "signal": np.arange(12, dtype=float),
        "failure_points": points,
    })
    frame["severity_level"] = np.select(
        [frame.failure_points.eq(0), frame.failure_points.le(5),
         frame.failure_points.le(11)],
        ["normal", "caution", "risk"],
        default="high_risk",
    )
    features = ("machine_type", "asset_tag", "signal")
    return (
        PreparedRegressionTask(
            frame, features, "failure_points", "asset", "current"
        ),
        PreparedTask(
            frame, features, "severity_level", "asset", "current",
            risk_definition="severity_4class",
        ),
    )
```

- [ ] **Step 2: 산출물과 선택 모델 실패 테스트 작성**

```python
def test_asset_score_suite_writes_all_outputs(tmp_path):
    score_task, severity_task = _score_tasks()
    regression, severity = run_asset_score_suite(
        score_task,
        severity_task,
        output_dir=tmp_path,
        scope="overall",
        max_iter=10,
    )
    assert regression.query("selected_model == True").shape[0] == 2
    assert severity.query("selected_model == True").shape[0] == 2
    assert {
        "regression_metrics.csv", "severity_metrics.csv",
        "test_predictions.csv", "feature_importance.csv", "run_config.json",
    } <= {path.name for path in tmp_path.iterdir()}
    predictions = pd.read_csv(tmp_path / "test_predictions.csv")
    assert {
        "actual_failure_points", "predicted_failure_points",
        "actual_severity_level", "regression_severity_level",
        "predicted_severity_level", "prob_normal", "prob_caution",
        "prob_risk", "prob_high_risk",
    } <= set(predictions)
```

검증과 테스트 각각 한 행의 선택 모델 지표를 저장하므로 `selected_model == True`가 두 행이어야 한다.

- [ ] **Step 3: 단일 등급 학습 구간 건너뛰기 실패 테스트 작성**

```python
def test_asset_score_suite_records_single_train_class_as_skipped(tmp_path):
    score_task, severity_task = _score_tasks(single_train_class=True)
    _, severity = run_asset_score_suite(
        score_task,
        severity_task,
        output_dir=tmp_path,
        scope="overall",
        max_iter=10,
    )
    assert set(severity["status"]) == {"skipped"}
    assert "single train class" in severity.iloc[0]["reason"]
```

- [ ] **Step 4: 저장 모델 재현성 실패 테스트 작성**

```python
def test_saved_asset_score_models_reproduce_predictions_and_class_order(tmp_path):
    score_task, severity_task = _score_tasks()
    run_asset_score_suite(
        score_task,
        severity_task,
        output_dir=tmp_path,
        scope="overall",
        max_iter=10,
    )
    regression_file = next((tmp_path / "models").glob("regression__*.joblib"))
    severity_file = next((tmp_path / "models").glob("severity__*.joblib"))
    regression_payload = joblib.load(regression_file)
    severity_payload = joblib.load(severity_file)
    rows = score_task.frame.loc[:, list(score_task.features)].head(3)
    assert np.allclose(
        regression_payload["pipeline"].predict(rows),
        regression_payload["verification_predictions"],
    )
    assert severity_payload["class_order"] == list(SEVERITY_LEVELS)
    assert np.allclose(
        severity_payload["pipeline"].predict_proba(rows),
        severity_payload["verification_probabilities"],
    )
```

- [ ] **Step 5: 실행기 테스트를 실행해 실패 확인**

Run: `python -m pytest tests/test_industrial_regression.py -k "suite or saved" -v`

Expected: FAIL because `run_asset_score_suite` does not exist.

- [ ] **Step 6: 실행기 메타데이터와 고정 출력 스키마 정의**

`src/industrial_regression.py`에 다음 상수와 헬퍼를 추가한다.

```python
REGRESSION_METRIC_COLUMNS = (
    "task_type", "scope_kind", "scope_name", "model", "selected_model",
    "split", "status", "reason", "train_rows", "valid_rows", "test_rows",
    "mae", "rmse", "r2", "spearman", "accuracy", "macro_precision",
    "macro_recall", "macro_f1", "weighted_f1", "confusion_matrix_json",
    "normal_precision", "normal_recall", "normal_f1", "normal_support",
    "caution_precision", "caution_recall", "caution_f1", "caution_support",
    "risk_precision", "risk_recall", "risk_f1", "risk_support",
    "high_risk_precision", "high_risk_recall", "high_risk_f1",
    "high_risk_support", "high_risk_binary_f1",
)

SEVERITY_METRIC_COLUMNS = tuple(
    column for column in REGRESSION_METRIC_COLUMNS
    if column not in {"mae", "rmse", "r2", "spearman"}
)


def _metric_row(metrics: dict[str, Any]) -> dict[str, Any]:
    result = dict(metrics)
    matrix = result.pop("confusion_matrix", None)
    result["confusion_matrix_json"] = (
        json.dumps(matrix, ensure_ascii=False) if matrix is not None else None
    )
    return result
```

- [ ] **Step 7: Scope별 회귀 학습·선택 구현**

`run_asset_score_suite()` 내부에서 `iter_scopes()`와 `split_by_date()`로 분할하고 다음 순서로 회귀 모델을 처리한다.

```python
regression_scores: dict[str, float] = {}
regression_models: dict[str, Pipeline] = {}
for model_name in REGRESSION_MODELS:
    model = build_regression_model(
        model_name, train[features], max_iter=max_iter,
        random_state=random_state,
    )
    model.fit(train[features], train[score_task.target])
    regression_models[model_name] = model
    valid_prediction = np.maximum(model.predict(valid[features]), 0.0)
    regression_scores[model_name] = mean_absolute_error(
        valid[score_task.target], valid_prediction
    )

selected_regression = min(
    LEARNED_REGRESSION_MODELS,
    key=lambda name: (
        regression_scores[name],
        LEARNED_REGRESSION_MODELS.index(name),
    ),
)
```

각 모델의 validation/test 예측에 `score_regression_metrics()`를 적용하되 `selected_model`은 선택된 학습 모델에만 `True`로 기록한다. Dummy는 기준선이므로 선택 대상에서 제외한다.

- [ ] **Step 8: Scope별 4단계 분류 학습·선택 구현**

기존 `build_model()`을 다중분류에 사용한다. 분류 모델의 `classes_` 순서가 고정 등급 순서와 다를 수 있으므로 확률 출력은 다음 함수로 재배열한다.

```python
def ordered_probabilities(model: Pipeline, frame: pd.DataFrame) -> np.ndarray:
    probabilities = model.predict_proba(frame)
    classes = list(model.named_steps["classifier"].classes_)
    ordered = np.zeros((len(frame), len(SEVERITY_LEVELS)), dtype=float)
    for target_index, level in enumerate(SEVERITY_LEVELS):
        if level in classes:
            ordered[:, target_index] = probabilities[:, classes.index(level)]
    return ordered
```

Dummy 기준선은 `DummyClassifier(strategy="prior")`를 사용하고 학습 모델 세 개는 기존 `MODEL_NAMES`와 `build_model()`을 사용한다. 검증 Macro F1이 가장 높은 학습 모델을 선택한다.

```python
severity_scores[model_name] = severity_classification_metrics(
    valid[severity_task.target], model.predict(valid[features])
)["macro_f1"]
selected_severity = max(
    MODEL_NAMES,
    key=lambda name: (
        severity_scores[name],
        -MODEL_NAMES.index(name),
    ),
)
```

- [ ] **Step 9: 선택 예측과 모델 저장 구현**

선택된 두 모델의 테스트 예측을 키 컬럼으로 결합하고 다음 컬럼을 저장한다.

```python
prediction = test[[
    "transaction_date", "label_end_date", "machine_type", "asset_tag",
    "failure_points", "severity_level",
]].rename(columns={
    "failure_points": "actual_failure_points",
    "severity_level": "actual_severity_level",
})
prediction["predicted_failure_points"] = regression_prediction
prediction["regression_severity_level"] = predicted_severity(
    regression_prediction
)
prediction["predicted_severity_level"] = severity_prediction
for index, level in enumerate(SEVERITY_LEVELS):
    prediction[f"prob_{level}"] = severity_probability[:, index]
```

모델 payload는 다음 키를 포함한다.

```python
regression_payload = {
    "pipeline": selected_regression_model,
    "feature_data": features,
    "target_data": score_task.target,
    "selected_model": selected_regression,
    "validation_start": str(validation_start),
    "test_start": str(test_start),
    "verification_predictions": selected_regression_model.predict(
        verification_rows
    ),
}

severity_payload = {
    "pipeline": selected_severity_model,
    "feature_data": features,
    "target_data": severity_task.target,
    "selected_model": selected_severity,
    "class_order": list(SEVERITY_LEVELS),
    "validation_start": str(validation_start),
    "test_start": str(test_start),
    "verification_probabilities": selected_severity_model.predict_proba(
        verification_rows
    ),
}
```

파일명은 `regression__<scope_kind>__<scope_name>__<model>__selected.joblib`과 `severity__<scope_kind>__<scope_name>__<model>__selected.joblib` 형식을 사용한다.

- [ ] **Step 10: Feature importance와 실행 설정 저장 구현**

회귀 선택 모델은 `scoring="neg_mean_absolute_error"`, 분류 선택 모델은 `scoring="f1_macro"`로 검증 구간 최대 1,000행에 permutation importance를 계산한다. `feature_importance.csv`에는 `task_type`, scope, model, feature, 평균과 표준편차를 저장한다.

`run_config.json`에는 다음 값을 저장한다.

```python
config = {
    "regression_models": list(REGRESSION_MODELS),
    "severity_models": ["dummy_classifier", *MODEL_NAMES],
    "severity_levels": list(SEVERITY_LEVELS),
    "observed_score_boundaries": [0, 5, 11, 12],
    "predicted_score_boundaries": [0.5, 5.5, 11.5],
    "scope": scope,
    "machine_type": machine_type,
    "asset_tag": asset_tag,
    "validation_start": str(validation_start),
    "test_start": str(test_start),
    "max_iter": max_iter,
    "random_state": random_state,
}
```

- [ ] **Step 11: 실행기 테스트 통과 확인**

Run: `python -m pytest tests/test_industrial_regression.py -v`

Expected: 모든 테스트 PASS. 임시 폴더에 다섯 CSV/JSON 산출물과 두 선택 모델이 존재한다.

- [ ] **Step 12: Task 3 커밋**

```bash
git add src/industrial_regression.py tests/test_industrial_regression.py
git commit -m "feat: train asset score and severity models"
```

### Task 4: 새 실행 파일과 CLI 계약

**Files:**
- Create: `src/current_asset_score_model.py`
- Modify: `tests/test_industrial_entrypoints.py:9-29`

**Interfaces:**
- Consumes: `load_industrial_data()`, `prepare_asset_score_current()`, `prepare_asset_severity_current()`, `run_asset_score_suite()`
- Produces: `build_parser(default_output=...)`, `main(default_output=...)`, 실행 명령 `python src/current_asset_score_model.py`

- [ ] **Step 1: CLI help와 기본 출력 실패 테스트 작성**

`tests/test_industrial_entrypoints.py`의 parametrized script 목록에 `src/current_asset_score_model.py`를 추가하고 다음 테스트를 작성한다.

```python
from src.current_asset_score_model import (
    DEFAULT_OUTPUT as SCORE_DEFAULT_OUTPUT,
    build_parser as build_score_parser,
)


def test_current_asset_score_defaults():
    args = build_score_parser().parse_args([])
    assert args.output == SCORE_DEFAULT_OUTPUT
    assert args.scope == "all"
    assert args.max_iter == 100
    assert args.validation_start == "2024-01-01"
    assert args.test_start == "2024-07-01"
```

- [ ] **Step 2: CLI 테스트를 실행해 실패 확인**

Run: `python -m pytest tests/test_industrial_entrypoints.py -k "asset_score or entrypoint_help" -v`

Expected: FAIL because `src/current_asset_score_model.py` does not exist.

- [ ] **Step 3: 실행 파일 구현**

`src/current_asset_score_model.py`를 다음 구조로 만든다. 기존 이진 분류용
`add_common_arguments()`는 사용하지 않고 이 실행 파일에 필요한 인자만 등록한다.

```python
"""장비·날짜별 고장점수 회귀와 4단계 위험도 분류 실행 파일이다."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "src"

from .asset_features import (
    prepare_asset_score_current,
    prepare_asset_severity_current,
)
from .industrial_data import load_industrial_data
from .industrial_regression import run_asset_score_suite
from .industrial_training import DEFAULT_DATA, PROJECT_ROOT

DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "asset_score_current"


def build_parser(
    default_output: Path = DEFAULT_OUTPUT,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="장비 당일 고장점수 회귀와 4단계 위험도 분류"
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument(
        "--scope",
        choices=["all", "overall", "machine_type", "asset_tag"],
        default="all",
    )
    parser.add_argument("--machine-type")
    parser.add_argument("--asset-tag")
    parser.add_argument("--validation-start", default="2024-01-01")
    parser.add_argument("--test-start", default="2024-07-01")
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--random-state", type=int, default=42)
    return parser


def main(default_output: Path = DEFAULT_OUTPUT) -> None:
    args = build_parser(default_output).parse_args()
    raw = load_industrial_data(args.data)
    run_asset_score_suite(
        prepare_asset_score_current(raw),
        prepare_asset_severity_current(raw),
        output_dir=args.output,
        scope=args.scope,
        machine_type=args.machine_type,
        asset_tag=args.asset_tag,
        max_iter=args.max_iter,
        validation_start=args.validation_start,
        test_start=args.test_start,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: CLI 테스트 통과 확인**

Run: `python -m pytest tests/test_industrial_entrypoints.py -v`

Expected: 새 실행 파일을 포함한 모든 help 테스트와 기본값 테스트 PASS.

- [ ] **Step 5: 작은 실제 데이터 smoke 실행**

Run:

```bash
python src/current_asset_score_model.py \
  --data dataVerification/synthetic_industrial_machine_data.csv \
  --output /tmp/urankle-asset-score-smoke \
  --scope overall \
  --max-iter 20
```

Expected: exit code 0. `/tmp/urankle-asset-score-smoke`에 `regression_metrics.csv`, `severity_metrics.csv`, `test_predictions.csv`, `feature_importance.csv`, `run_config.json`, `models/`가 생성된다.

- [ ] **Step 6: Task 4 커밋**

```bash
git add src/current_asset_score_model.py tests/test_industrial_entrypoints.py
git commit -m "feat: add asset score model cli"
```

### Task 5: 사용자 문서와 전체 검증

**Files:**
- Modify: `src/RUN_INDUSTRIAL.md`
- Modify: `docs/industrial_code_guide.md`
- Modify: `docs/industrial_feature_guide.md`
- Modify if verification exposes a defect: Task 1-4에서 수정한 파일만

**Interfaces:**
- Consumes: 새 CLI와 실제 출력 스키마
- Produces: 사용자가 그대로 복사해 실행할 수 있는 명령과 결과 해석 설명

- [ ] **Step 1: 실행 가이드에 새 명령과 결과 폴더 추가**

`src/RUN_INDUSTRIAL.md`의 macOS·Linux와 Windows 명령에 다음 실행을 추가한다.

```bash
python src/current_asset_score_model.py
```

과제 표에 다음 행을 추가한다.

```markdown
| `current_asset_score_model.py` | 장비·날짜 | 당일 `failure_points`와 4단계 위험도 | `outputs/asset_score_current` |
```

등급 표와 회귀/분류 산출물 설명을 추가하고 `high_risk`가 실제 기계 정지를 뜻하지 않는다는 제한을 명시한다.

- [ ] **Step 2: 코드·Feature 가이드 갱신**

`docs/industrial_code_guide.md`에 다음 호출 흐름을 추가한다.

```text
current_asset_score_model.py
  ├─ prepare_asset_score_current()
  ├─ prepare_asset_severity_current()
  └─ run_asset_score_suite()
```

`docs/industrial_feature_guide.md`에 다음 수식과 등급을 추가한다.

```markdown
\[
L(S)=
\begin{cases}
\mathrm{normal},&S=0,\\
\mathrm{caution},&1\le S\le5,\\
\mathrm{risk},&6\le S\le11,\\
\mathrm{high\_risk},&S\ge12.
\end{cases}
\]
```

회귀 예측 등급 경계 `0.5`, `5.5`, `11.5`와 기존 이진 모델이 계속 기준선으로 남는다는 점을 함께 설명한다.

- [ ] **Step 3: 새 기능 집중 테스트 실행**

Run:

```bash
python -m pytest \
  tests/test_asset_features.py \
  tests/test_industrial_regression.py \
  tests/test_industrial_entrypoints.py \
  -v
```

Expected: 모든 테스트 PASS.

- [ ] **Step 4: 전체 회귀 테스트 실행**

Run: `python -m pytest -v`

Expected: 기존 xfail만 유지되고 새 실패가 0건이다.

- [ ] **Step 5: 전체 scope 실제 데이터 실행**

Run:

```bash
python src/current_asset_score_model.py \
  --data dataVerification/synthetic_industrial_machine_data.csv \
  --output /tmp/urankle-asset-score-full \
  --scope all \
  --max-iter 100
```

Expected: overall 1개, machine_type 5개, asset_tag 10개 scope의 회귀와 4단계 분류 결과가 생성되고 모든 선택 모델 행의 `status`가 `ok`다.

- [ ] **Step 6: 산출물 무결성 검사**

Run:

```bash
python - <<'PY'
import json
from pathlib import Path

import pandas as pd

output = Path('/tmp/urankle-asset-score-full')
regression = pd.read_csv(output / 'regression_metrics.csv')
severity = pd.read_csv(output / 'severity_metrics.csv')
predictions = pd.read_csv(output / 'test_predictions.csv')
config = json.loads((output / 'run_config.json').read_text(encoding='utf-8'))

assert {'overall', 'machine_type', 'asset_tag'} <= set(regression.scope_kind)
assert {'overall', 'machine_type', 'asset_tag'} <= set(severity.scope_kind)
assert predictions.predicted_failure_points.ge(0).all()
assert set(predictions.actual_severity_level) <= set(config['severity_levels'])
assert set(predictions.predicted_severity_level) <= set(config['severity_levels'])
assert predictions[[
    'prob_normal', 'prob_caution', 'prob_risk', 'prob_high_risk'
]].sum(axis=1).round(10).eq(1.0).all()
assert not predictions.duplicated([
    'transaction_date', 'machine_type', 'asset_tag', 'scope_kind', 'scope_name'
]).any()
print('asset score output verification passed')
PY
```

Expected: `asset score output verification passed` 출력 후 exit code 0.

- [ ] **Step 7: 문서와 최종 구현 커밋**

```bash
git add src/RUN_INDUSTRIAL.md docs/industrial_code_guide.md docs/industrial_feature_guide.md
git commit -m "docs: explain asset severity model"
```

- [ ] **Step 8: 최종 작업트리와 커밋 확인**

Run:

```bash
git status --short --branch
git log --oneline -6
```

Expected: 계획 밖의 변경이 없고 구현 커밋과 문서 커밋이 순서대로 표시된다.
