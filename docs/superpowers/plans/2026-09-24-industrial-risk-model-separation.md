# 산업 장비 위험 모델 분리 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 장비·부품 단위의 당일 탐지와 향후 7일 예측을 네 개의 명시적 실행 흐름으로 분리하고, 누수 없는 Feature·Target 생성과 공통 모델 평가·저장 체계를 구축한다.

**Architecture:** 공통 데이터 계약과 `PreparedTask`를 `industrial_data.py`에 두고, 장비 Feature와 부품 Feature를 각각 `asset_features.py`, `part_features.py`에서 생성한다. `industrial_training.py`는 네 과제 모두에 대해 기준선과 세 모델을 학습하고 검증 Average Precision으로 모델을 선택하며, 세 임계값 정책과 공통 산출물을 저장한다. 네 개의 명시적 CLI를 추가하고 기존 두 CLI는 기존 출력 경로를 유지하는 호환 래퍼로 남긴다.

**Tech Stack:** Python 3.11+, pandas 2.2.3, NumPy 1.26.4, scikit-learn 1.6.1, joblib 1.4.2, pytest 8.3.3

**Spec:** `docs/superpowers/specs/2026-09-24-industrial-risk-model-separation-design.md`

## Global Constraints

- 기본 미래 예측 기간은 7일이며 날짜 \(t\)까지의 센서로 \(t+1\)일부터 \(t+7\)일까지를 예측한다.
- 장비 점수 기준 기본값은 12, 13, 14이며 Target 이름은 실제 연산에 맞춰 `target_ge_12`와 같은 형식을 사용한다.
- `new_risk_7d`는 오늘 위험하지 않은 장비만 포함하고, `any_risk_7d`는 현재 상태와 관계없이 포함한다.
- 미래 센서 Feature는 `asset_tag`별로 계산하고, 부품 고장 이력은 `asset_tag + part_no`별로 계산한다.
- 미래 Target을 완전히 관측할 수 없는 행은 음성으로 채우지 않고 제외한다.
- 기본 검증 시작일은 `2024-01-01`, 테스트 시작일은 `2024-07-01`이다.
- 모델 선택과 확률 임계값 선택에는 검증 데이터만 사용하고 테스트 데이터는 최종 평가에만 사용한다.
- 초기 기본 모델에는 LightGBM을 포함하지 않는다.
- 기존 `src/current_state_model.py`, `src/timeseries_model.py` 명령과 각각의 기존 기본 출력 경로 `outputs/current_state`, `outputs/timeseries`를 유지한다.
- 기존 사용자의 데이터 또는 산출물을 삭제하지 않는다.
- 새로 만들거나 역할을 바꾸는 모든 Python 파일에는 파일의 책임과 호출 관계를 설명하는 한글 모듈 docstring을 둔다.
- 외부에서 호출하는 모든 함수에는 입력값, 반환값, 발생 가능한 예외를 설명하는 한글 docstring을 작성한다.
- 주석은 코드 동작을 그대로 번역하지 않고 미래 라벨 경계, `shift(1)`, 그룹 키처럼 해당 구현이 필요한 이유를 설명한다.

## Review Focus

1. 같은 장비·날짜에 센서값이 서로 다르면 임의 집계하지 않고 어떤 센서와 키가 충돌했는지 포함한 `ValueError`를 내야 한다. Task 2 테스트에서 고정한다.
2. 날짜가 하루라도 빠진 미래 구간은 7개 행이 있다는 이유만으로 유효하다고 판단하지 않고 해당 기준일을 제외해야 한다. Task 3과 Task 4 테스트에서 고정한다.
3. 학습·검증·테스트 중 빈 구간이나 단일 클래스 구간은 전체 실행을 중단시키지 않고 명시적인 `skipped` 행으로 기록해야 한다. Task 6 테스트에서 고정한다.
4. 검증 데이터에 처음 등장하는 `asset_tag`, `part_no` 또는 다른 범주값은 `handle_unknown="ignore"`로 처리되어야 한다. Task 6 테스트에서 고정한다.
5. 검증 세트에서 최소 Precision 조건을 충족하는 임계값이 없으면 임의 임계값으로 가장하지 않고 `cutoff=None`, `status="unavailable"`로 기록해야 한다. Task 5 테스트에서 고정한다.

---

## 파일 책임 지도

### 새로 만드는 파일

- `src/industrial_data.py`: 공통 상수, 원본 로딩, 날짜 Feature, `PreparedTask`, 시간 분할, 범위 순회
- `src/asset_features.py`: 장비·날짜 집계, 장비 당일 Target, 장비 미래 Target과 이력 Feature
- `src/part_features.py`: 부품 당일 Target, 부품 미래 Target과 이력 Feature
- `src/industrial_cli.py`: 네 CLI에서 공유하는 인자와 실행 인자 변환
- `src/current_asset_model.py`: 장비 당일 실행 진입점
- `src/forecast_asset_model.py`: 장비 미래 실행 진입점
- `src/current_part_model.py`: 부품 당일 실행 진입점
- `src/forecast_part_model.py`: 부품 미래 실행 진입점
- `tests/test_industrial_data.py`: 공통 데이터 계약 테스트
- `tests/test_asset_features.py`: 장비 Target·Feature·누수 테스트
- `tests/test_part_features.py`: 부품 Target·Feature·누수 테스트
- `tests/test_industrial_entrypoints.py`: 네 CLI 및 호환 CLI 스모크 테스트

### 수정하는 파일

- `src/industrial_features.py`: 기존 import와 함수명을 유지하는 호환 재노출 모듈로 축소
- `src/industrial_training.py`: 임계값 정책, 순위 지표, 세 학습모델 비교, 모델 선택 및 출력 계약 구현
- `src/current_state_model.py`: `current_part_model.py` 호환 래퍼로 변경
- `src/timeseries_model.py`: `forecast_part_model.py` 호환 래퍼로 변경
- `src/eda.py`: 모델 학습을 제거하고 네 결과 폴더 비교 기능만 유지
- `src/RUN_INDUSTRIAL.md`: macOS·Linux·Windows 실행 예시와 네 과제 설명 갱신
- `docs/industrial_feature_guide.md`: 네 과제별 Feature와 Target 문서화
- `docs/industrial_code_guide.md`: 각 Python 파일의 책임, 호출 흐름, 주요 함수와 실행 예시 설명
- `tests/test_industrial_training.py`: 임계값·순위 지표·학습 선택·출력 계약 테스트 확장

---

### Task 1: 공통 데이터 계약과 시간 분할 추출

**Files:**
- Create: `src/industrial_data.py`
- Create: `tests/test_industrial_data.py`
- Modify: `src/industrial_features.py:1-269`

**Interfaces:**
- Produces: `PreparedTask`, `load_industrial_data()`, `add_calendar_features()`, `split_by_date()`, `iter_scopes()`
- Consumes: 원본 CSV의 필수 컬럼과 pandas `DataFrame`

- [ ] **Step 1: 누락 컬럼, 달력 Feature 및 라벨 종료일 분할 실패 테스트 작성**

```python
# tests/test_industrial_data.py
import numpy as np
import pandas as pd
import pytest

from src.industrial_data import (
    PreparedTask,
    add_calendar_features,
    load_industrial_data,
    split_by_date,
)


def test_add_calendar_features_uses_transaction_date():
    frame = pd.DataFrame({"transaction_date": pd.to_datetime(["2024-01-01", "2024-07-01"])})
    result = add_calendar_features(frame)
    assert result["day_of_week"].tolist() == [0, 0]
    assert np.allclose(result["month_sin"], np.sin(2 * np.pi * np.array([1, 7]) / 12))
    assert np.allclose(result["month_cos"], np.cos(2 * np.pi * np.array([1, 7]) / 12))


def test_split_by_date_excludes_validation_label_crossing_test_boundary():
    frame = pd.DataFrame({
        "transaction_date": pd.to_datetime(["2023-12-20", "2024-01-01", "2024-06-25", "2024-07-01"]),
        "label_end_date": pd.to_datetime(["2023-12-27", "2024-01-08", "2024-07-02", "2024-07-08"]),
    })
    result = split_by_date(frame, "2024-01-01", "2024-07-01")
    assert result["train"]["transaction_date"].dt.strftime("%Y-%m-%d").tolist() == ["2023-12-20"]
    assert result["valid"]["transaction_date"].dt.strftime("%Y-%m-%d").tolist() == ["2024-01-01"]
    assert result["test"]["transaction_date"].dt.strftime("%Y-%m-%d").tolist() == ["2024-07-01"]


def test_prepared_task_rejects_feature_target_overlap():
    with pytest.raises(ValueError, match="Target"):
        PreparedTask(
            frame=pd.DataFrame({"target": [0]}),
            features=("target",),
            target="target",
            grain="asset",
            mode="current",
        )
```

- [ ] **Step 2: 테스트를 실행해 새 모듈 부재로 실패하는지 확인**

Run: `pytest tests/test_industrial_data.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'src.industrial_data'`.

- [ ] **Step 3: 공통 데이터 모듈 최소 구현**

```python
# src/industrial_data.py
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATE_COLUMN = "transaction_date"
ASSET_COLUMN = "asset_tag"
PART_COLUMN = "part_no"
MACHINE_COLUMN = "machine_type"
PLANT_COLUMN = "plant_code"
CURRENT_TARGET = "breakdown_flag"
SENSOR_COLUMNS = [
    "temp_bearing_degC", "temp_motor_degC", "vibration_h_mms",
    "vibration_v_mms", "oil_pressure_bar", "load_pct", "shaft_rpm",
    "power_consumption_kw",
]
REQUIRED_COLUMNS = [
    DATE_COLUMN, ASSET_COLUMN, PART_COLUMN, MACHINE_COLUMN, PLANT_COLUMN,
    "criticality", CURRENT_TARGET, *SENSOR_COLUMNS,
]


@dataclass(frozen=True)
class PreparedTask:
    frame: pd.DataFrame
    features: tuple[str, ...]
    target: str
    grain: str
    mode: str
    risk_definition: str = "current"
    score_threshold: int | None = None
    horizon: int | None = None

    def __post_init__(self) -> None:
        if self.target in self.features:
            raise ValueError("Target 컬럼을 Feature에 포함할 수 없습니다.")
        missing = [column for column in (*self.features, self.target) if column not in self.frame]
        if missing:
            raise ValueError(f"PreparedTask에 필요한 컬럼이 없습니다: {missing}")


def load_industrial_data(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=[DATE_COLUMN])
    missing = [column for column in REQUIRED_COLUMNS if column not in frame]
    if missing:
        raise ValueError(f"CSV에 필요한 컬럼이 없습니다: {missing}")
    frame[CURRENT_TARGET] = pd.to_numeric(frame[CURRENT_TARGET], errors="coerce")
    if frame[CURRENT_TARGET].isna().any():
        raise ValueError("breakdown_flag에 숫자가 아닌 값이 있습니다.")
    return frame


def add_calendar_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    dates = pd.to_datetime(result[DATE_COLUMN])
    result["day_of_week"] = dates.dt.dayofweek
    result["month_sin"] = np.sin(2 * np.pi * dates.dt.month / 12)
    result["month_cos"] = np.cos(2 * np.pi * dates.dt.month / 12)
    return result
```

`split_by_date()`와 `iter_scopes()`는 기존 동작을 그대로 옮기되 `label_end_date` 기준 경계 조건과 빈 필터 오류를 유지한다. `src/industrial_features.py`는 당분간 이 모듈의 공통 이름을 import하여 기존 import 경로가 깨지지 않게 한다.

- [ ] **Step 4: 공통 데이터 테스트와 기존 테스트 실행**

Run: `pytest tests/test_industrial_data.py tests/test_industrial_training.py -v`

Expected: PASS.

- [ ] **Step 5: 커밋**

```bash
git add src/industrial_data.py src/industrial_features.py tests/test_industrial_data.py
git commit -m "refactor: 산업 데이터 공통 계약 분리"
```

---

### Task 2: 장비 단위 당일 Target과 Feature 구현

**Files:**
- Create: `src/asset_features.py`
- Create: `tests/test_asset_features.py`

**Interfaces:**
- Consumes: `PreparedTask`, `SENSOR_COLUMNS`, 원본 부품 단위 DataFrame
- Produces: `build_asset_daily(frame) -> DataFrame`, `prepare_asset_current(frame, score_threshold) -> PreparedTask`

- [ ] **Step 1: 고장점수, `>=` 경계 및 센서 충돌 실패 테스트 작성**

```python
# tests/test_asset_features.py
import pandas as pd
import pytest

from src.asset_features import build_asset_daily, prepare_asset_current


def _rows(sensor_second: float = 1.0) -> pd.DataFrame:
    base = {
        "transaction_date": pd.Timestamp("2024-01-01"),
        "asset_tag": "A-1", "machine_type": "Press", "plant_code": "P1",
        "temp_bearing_degC": 50.0, "temp_motor_degC": 60.0,
        "vibration_h_mms": 1.0, "vibration_v_mms": 1.1,
        "oil_pressure_bar": 5.0, "load_pct": 70.0,
        "shaft_rpm": 1000.0, "power_consumption_kw": 20.0,
    }
    return pd.DataFrame([
        {**base, "part_no": "P-A", "criticality": "A", "breakdown_flag": 1},
        {**base, "part_no": "P-B", "criticality": "A", "breakdown_flag": 1,
         "vibration_h_mms": sensor_second},
        {**base, "part_no": "P-C", "criticality": "A", "breakdown_flag": 1},
    ])


def test_asset_current_uses_greater_than_or_equal_boundary():
    prepared = prepare_asset_current(_rows(), score_threshold=12)
    assert prepared.target == "target_ge_12"
    assert prepared.frame["failure_points"].tolist() == [12]
    assert prepared.frame[prepared.target].tolist() == [1]


def test_asset_daily_rejects_conflicting_sensor_values():
    with pytest.raises(ValueError, match="vibration_h_mms"):
        build_asset_daily(_rows(sensor_second=2.0))
```

- [ ] **Step 2: 실패 확인**

Run: `pytest tests/test_asset_features.py -v`

Expected: FAIL because `src.asset_features` does not exist.

- [ ] **Step 3: 장비 일별 집계와 당일 준비 함수 구현**

```python
# src/asset_features.py 핵심 인터페이스
CRITICALITY_WEIGHTS = {"A": 4, "B": 2, "C": 1}
ASSET_CURRENT_FEATURES = (
    "machine_type", "asset_tag", *SENSOR_COLUMNS,
    "day_of_week", "month_sin", "month_cos",
)


def build_asset_daily(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result[DATE_COLUMN] = pd.to_datetime(result[DATE_COLUMN]).dt.normalize()
    result["criticality_weight"] = (
        result["criticality"].astype("string").str.strip().str.upper()
        .map(CRITICALITY_WEIGHTS)
    )
    if result["criticality_weight"].isna().any():
        raise ValueError("criticality에는 A, B, C만 사용할 수 있습니다.")
    duplicate_keys = [DATE_COLUMN, ASSET_COLUMN, PART_COLUMN]
    if result.duplicated(duplicate_keys).any():
        raise ValueError("같은 날짜·장비·부품의 중복 행이 있습니다.")
    group_keys = [DATE_COLUMN, MACHINE_COLUMN, ASSET_COLUMN]
    counts = result.groupby(group_keys)[SENSOR_COLUMNS].nunique(dropna=False)
    conflicts = counts.gt(1)
    if conflicts.any().any():
        columns = conflicts.columns[conflicts.any()].tolist()
        raise ValueError(f"같은 장비·날짜의 센서값이 다릅니다: {columns}")
    result["failure_points"] = result[CURRENT_TARGET] * result["criticality_weight"]
    daily = result.groupby(group_keys, as_index=False).agg({
        "failure_points": "sum", **{column: "first" for column in SENSOR_COLUMNS},
    })
    daily["label_end_date"] = daily[DATE_COLUMN]
    return add_calendar_features(daily)


def prepare_asset_current(frame: pd.DataFrame, score_threshold: int) -> PreparedTask:
    if score_threshold < 1:
        raise ValueError("score_threshold는 1 이상이어야 합니다.")
    daily = build_asset_daily(frame)
    target = f"target_ge_{score_threshold}"
    daily[target] = (daily["failure_points"] >= score_threshold).astype(int)
    return PreparedTask(
        daily, ASSET_CURRENT_FEATURES, target, "asset", "current",
        score_threshold=score_threshold,
    )
```

- [ ] **Step 4: 장비 당일 테스트 실행**

Run: `pytest tests/test_asset_features.py -v`

Expected: PASS.

- [ ] **Step 5: 커밋**

```bash
git add src/asset_features.py tests/test_asset_features.py
git commit -m "feat: 장비 단위 당일 위험 데이터 추가"
```

---

### Task 3: 장비 단위 향후 7일 신규·전체 위험 구현

**Files:**
- Modify: `src/asset_features.py`
- Modify: `tests/test_asset_features.py`

**Interfaces:**
- Consumes: `build_asset_daily()`, `score_threshold`, `horizon`, `risk_definition`
- Produces: `prepare_asset_forecast(frame, score_threshold, horizon=7, risk_definition="new") -> PreparedTask`

- [ ] **Step 1: 미래 날짜 연속성, 오늘 제외, `new`/`any` 차이 및 누수 실패 테스트 작성**

```python
def test_asset_forecast_looks_only_at_next_seven_calendar_days(asset_history):
    prepared = prepare_asset_forecast(
        asset_history, score_threshold=12, horizon=7, risk_definition="any"
    )
    row = prepared.frame.loc[prepared.frame["transaction_date"].eq("2024-01-01")].iloc[0]
    assert row[prepared.target] == 1  # 1월 4일 점수 12
    assert row["label_end_date"] == pd.Timestamp("2024-01-08")


def test_asset_forecast_excludes_incomplete_calendar_window(asset_history):
    broken = asset_history[asset_history["transaction_date"] != pd.Timestamp("2024-01-05")]
    prepared = prepare_asset_forecast(broken, 12, horizon=7, risk_definition="any")
    assert pd.Timestamp("2024-01-01") not in set(prepared.frame["transaction_date"])


def test_new_risk_excludes_current_positive_but_any_keeps_it(asset_history):
    new = prepare_asset_forecast(asset_history, 12, 7, "new").frame
    any_risk = prepare_asset_forecast(asset_history, 12, 7, "any").frame
    current_positive_date = pd.Timestamp("2024-01-04")
    assert current_positive_date not in set(new["transaction_date"])
    assert current_positive_date in set(any_risk["transaction_date"])


def test_future_sensor_change_does_not_change_past_features(asset_history):
    before = prepare_asset_forecast(asset_history, 12, 7, "any").frame
    changed = asset_history.copy()
    changed.loc[changed["transaction_date"].eq("2024-01-09"), "load_pct"] = 999
    after = prepare_asset_forecast(changed, 12, 7, "any").frame
    columns = [column for column in before if column.startswith("load_pct_")]
    pd.testing.assert_series_equal(
        before.loc[before["transaction_date"].eq("2024-01-01"), columns].iloc[0],
        after.loc[after["transaction_date"].eq("2024-01-01"), columns].iloc[0],
    )
```

`asset_history` fixture는 12일의 연속 날짜, 같은 장비의 부품 3개, 1월 4일
`failure_points=12`가 되도록 `tests/test_asset_features.py`에 작성한다.

```python
@pytest.fixture
def asset_history():
    rows = []
    for offset, date in enumerate(pd.date_range("2024-01-01", periods=12, freq="D")):
        for part in ("P-A", "P-B", "P-C"):
            rows.append({
                "transaction_date": date,
                "asset_tag": "A-1", "machine_type": "Press", "plant_code": "P1",
                "part_no": part, "criticality": "A",
                "breakdown_flag": int(date == pd.Timestamp("2024-01-04")),
                "temp_bearing_degC": 50.0 + offset,
                "temp_motor_degC": 60.0 + offset,
                "vibration_h_mms": 1.0 + offset / 10,
                "vibration_v_mms": 1.1 + offset / 10,
                "oil_pressure_bar": 5.0, "load_pct": 70.0 + offset,
                "shaft_rpm": 1000.0 + offset,
                "power_consumption_kw": 20.0 + offset,
            })
    return pd.DataFrame(rows)
```

- [ ] **Step 2: 실패 확인**

Run: `pytest tests/test_asset_features.py -k forecast -v`

Expected: FAIL because `prepare_asset_forecast` is not defined.

- [ ] **Step 3: 장비 미래 Target과 이력 Feature 구현**

```python
def prepare_asset_forecast(
    frame: pd.DataFrame,
    score_threshold: int,
    horizon: int = 7,
    risk_definition: str = "new",
) -> PreparedTask:
    if horizon < 1:
        raise ValueError("horizon은 1 이상이어야 합니다.")
    if risk_definition not in {"new", "any"}:
        raise ValueError("risk_definition은 new 또는 any여야 합니다.")
    daily = build_asset_daily(frame).sort_values([ASSET_COLUMN, DATE_COLUMN]).copy()
    current_target = f"target_ge_{score_threshold}"
    daily[current_target] = (daily["failure_points"] >= score_threshold).astype(int)
    grouped = daily.groupby(ASSET_COLUMN, sort=False)

    valid = pd.Series(True, index=daily.index)
    future_values = []
    for offset in range(1, horizon + 1):
        future_date = grouped[DATE_COLUMN].shift(-offset)
        valid &= (future_date - daily[DATE_COLUMN]).dt.days.eq(offset)
        future_values.append(grouped[current_target].shift(-offset).astype(float))
    target = f"target_{risk_definition}_risk_{horizon}d_ge_{score_threshold}"
    daily[target] = pd.concat(future_values, axis=1).max(axis=1).where(valid)
    daily["label_end_date"] = (
        daily[DATE_COLUMN] + pd.Timedelta(days=horizon)
    ).where(valid, pd.NaT)

    for column in SENSOR_COLUMNS:
        values = grouped[column]
        daily[f"{column}_current"] = daily[column]
        daily[f"{column}_lag1"] = values.shift(1)
        daily[f"{column}_lag3"] = values.shift(3)
        daily[f"{column}_lag7"] = values.shift(7)
        daily[f"{column}_mean7"] = values.transform(lambda s: s.rolling(7, min_periods=3).mean())
        daily[f"{column}_std7"] = values.transform(lambda s: s.rolling(7, min_periods=3).std())
        daily[f"{column}_diff1"] = values.diff(1)

    shifted_points = grouped["failure_points"].shift(1)
    daily["failure_points_lag1"] = shifted_points
    daily["failure_points_mean7"] = shifted_points.groupby(daily[ASSET_COLUMN]).transform(
        lambda s: s.rolling(7, min_periods=1).mean()
    )
    daily["failure_points_max7"] = shifted_points.groupby(daily[ASSET_COLUMN]).transform(
        lambda s: s.rolling(7, min_periods=1).max()
    )
```

같은 함수에서 과거 `current_target`을 `shift(1)`한 뒤 최근 30일 합으로
`risk_event_count_30d`를 만들고, 마지막 위험 날짜를 forward-fill하여
`days_since_last_risk`를 계산한다. `risk_definition="new"`이면 Target 결측 제거 전에
`current_target == 0`인 행만 남기고, 두 정의 모두 Target 결측을 제거한 뒤 정수로 변환한다.

- [ ] **Step 4: 장비 Feature 전체 테스트 실행**

Run: `pytest tests/test_asset_features.py -v`

Expected: PASS.

- [ ] **Step 5: 커밋**

```bash
git add src/asset_features.py tests/test_asset_features.py
git commit -m "feat: 장비 향후 7일 위험 데이터 추가"
```

---

### Task 4: 부품 단위 당일·미래 Feature 모듈 분리

**Files:**
- Create: `src/part_features.py`
- Create: `tests/test_part_features.py`
- Modify: `src/industrial_features.py:1-269`

**Interfaces:**
- Consumes: 원본 부품 단위 DataFrame, `PreparedTask`
- Produces: `prepare_part_current(frame) -> PreparedTask`, `prepare_part_forecast(frame, horizon=7) -> PreparedTask`
- Compatibility: `industrial_features.prepare_current()`와 `prepare_forecast()`는 기존 3-tuple을 반환

- [ ] **Step 1: 부품 식별 Feature, 그룹 경계 및 미래 누수 실패 테스트 작성**

```python
# tests/test_part_features.py
def test_part_current_includes_part_identity_and_criticality(part_history):
    prepared = prepare_part_current(part_history)
    assert prepared.target == "breakdown_flag"
    assert {"part_no", "criticality", "plant_code"} <= set(prepared.features)
    assert "breakdown_flag" not in prepared.features


def test_part_forecast_target_starts_tomorrow(part_history):
    prepared = prepare_part_forecast(part_history, horizon=7)
    row = prepared.frame.query("asset_tag == 'A-1' and part_no == 'P-1'").iloc[0]
    assert row[prepared.target] == 1
    assert row["label_end_date"] == row["transaction_date"] + pd.Timedelta(days=7)


def test_part_breakdown_history_does_not_cross_parts(part_history):
    prepared = prepare_part_forecast(part_history, horizon=7).frame
    p2 = prepared.query("asset_tag == 'A-1' and part_no == 'P-2'")
    assert (p2["breakdown_count_7d"].fillna(0) == 0).all()


def test_part_forecast_rejects_missing_calendar_day(part_history):
    missing = part_history[~(
        part_history["part_no"].eq("P-1")
        & part_history["transaction_date"].eq(pd.Timestamp("2024-01-05"))
    )]
    prepared = prepare_part_forecast(missing, horizon=7)
    key_rows = prepared.frame.query("asset_tag == 'A-1' and part_no == 'P-1'")
    assert pd.Timestamp("2024-01-01") not in set(key_rows["transaction_date"])
```

`part_history`는 다음 fixture로 고정한다.

```python
@pytest.fixture
def part_history():
    rows = []
    for offset, date in enumerate(pd.date_range("2024-01-01", periods=12, freq="D")):
        for part, criticality in (("P-1", "A"), ("P-2", "C")):
            rows.append({
                "transaction_date": date,
                "asset_tag": "A-1", "machine_type": "Press", "plant_code": "P1",
                "part_no": part, "criticality": criticality,
                "breakdown_flag": int(part == "P-1" and date == pd.Timestamp("2024-01-04")),
                "temp_bearing_degC": 50.0 + offset,
                "temp_motor_degC": 60.0 + offset,
                "vibration_h_mms": 1.0 + offset / 10,
                "vibration_v_mms": 1.1 + offset / 10,
                "oil_pressure_bar": 5.0, "load_pct": 70.0 + offset,
                "shaft_rpm": 1000.0 + offset,
                "power_consumption_kw": 20.0 + offset,
            })
    return pd.DataFrame(rows)
```

- [ ] **Step 2: 실패 확인**

Run: `pytest tests/test_part_features.py -v`

Expected: FAIL because `src.part_features` does not exist.

- [ ] **Step 3: 부품 Feature 모듈 구현**

```python
PART_CURRENT_FEATURES = (
    MACHINE_COLUMN, ASSET_COLUMN, PART_COLUMN, "criticality", PLANT_COLUMN,
    *SENSOR_COLUMNS, "day_of_week", "month_sin", "month_cos",
)


def prepare_part_current(frame: pd.DataFrame) -> PreparedTask:
    result = _validated_part_frame(frame)
    result["label_end_date"] = result[DATE_COLUMN]
    result = add_calendar_features(result)
    return PreparedTask(
        result, PART_CURRENT_FEATURES, CURRENT_TARGET, "part", "current"
    )


def prepare_part_forecast(frame: pd.DataFrame, horizon: int = 7) -> PreparedTask:
    result = _validated_part_frame(frame)
    target = f"target_{horizon}d"
    # Target 연속성은 asset_tag + part_no 그룹에서 t+1 ... t+horizon으로 검사한다.
    # 센서 current/lag/rolling/diff는 asset_tag + transaction_date의 유일 센서 표를
    # 먼저 만든 뒤 계산하여 부품 행에 merge한다.
    # breakdown 이력은 asset_tag + part_no 그룹에서 shift(1) 후 계산한다.
    # 현재 breakdown_flag == 0이고 미래 Target을 완전히 관측한 행만 반환한다.
```

센서 Feature 이름은 장비 미래 모델과 동일한 `*_current`, `*_lag1`, `*_lag3`,
`*_lag7`, `*_mean7`, `*_std7`, `*_diff1` 규칙을 사용한다. 부품 이력으로
`breakdown_lag1`, `breakdown_count_7d`, `breakdown_count_30d`,
`days_since_last_breakdown`을 만든다.

`src/industrial_features.py`에는 다음 호환 함수를 둔다.

```python
def prepare_current(frame):
    prepared = prepare_part_current(frame)
    return prepared.frame, list(prepared.features), prepared.target


def prepare_forecast(frame, horizon=7):
    prepared = prepare_part_forecast(frame, horizon)
    return prepared.frame, list(prepared.features), prepared.target
```

- [ ] **Step 4: 새 부품 테스트와 기존 회귀 테스트 실행**

Run: `pytest tests/test_part_features.py tests/test_industrial_training.py -v`

Expected: PASS.

- [ ] **Step 5: 커밋**

```bash
git add src/part_features.py src/industrial_features.py tests/test_part_features.py
git commit -m "refactor: 부품 위험 Feature 모듈 분리"
```

---

### Task 5: 임계값 정책과 Top-K 평가 지표 구현

**Files:**
- Modify: `src/industrial_training.py:45-115`
- Modify: `tests/test_industrial_training.py:1-23`

**Interfaces:**
- Produces: `ThresholdSelection`, `choose_threshold()`, `choose_thresholds()`, `classification_metrics()`, `ranking_metrics()`
- Consumes: 검증 정답·확률, `min_precision`, `top_fraction`

- [ ] **Step 1: 세 정책, 불가능한 Precision 및 Top-K 지표 실패 테스트 작성**

```python
from src.industrial_training import choose_thresholds, ranking_metrics


def test_choose_thresholds_returns_f1_min_precision_and_top_fraction():
    y = np.array([0, 0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.7, 0.9])
    selected = {item.policy: item for item in choose_thresholds(
        y, scores, min_precision=0.75, top_fraction=0.40
    )}
    assert selected["f1"].status == "ok"
    assert selected["min_precision"].cutoff == 0.7
    assert selected["top_fraction"].cutoff == 0.7


def test_min_precision_is_unavailable_when_no_candidate_satisfies_it():
    selected = {item.policy: item for item in choose_thresholds(
        [1, 0, 0, 0], [0.1, 0.9, 0.8, 0.7], min_precision=0.80
    )}
    assert selected["min_precision"].cutoff is None
    assert selected["min_precision"].status == "unavailable"


def test_ranking_metrics_uses_exact_top_count():
    result = ranking_metrics([1, 0, 1, 0, 0], [0.9, 0.8, 0.7, 0.6, 0.5])
    assert result["precision_at_20pct"] == 1.0
    assert result["recall_at_20pct"] == 0.5
    assert result["lift_at_20pct"] == 2.5
```

- [ ] **Step 2: 실패 확인**

Run: `pytest tests/test_industrial_training.py -k 'thresholds or min_precision or ranking' -v`

Expected: FAIL because the new interfaces do not exist.

- [ ] **Step 3: 임계값 선택 자료형과 정책 구현**

```python
@dataclass(frozen=True)
class ThresholdSelection:
    policy: str
    cutoff: float | None
    status: str


def choose_thresholds(
    y_true: Any,
    scores: Any,
    *,
    min_precision: float = 0.30,
    top_fraction: float = 0.10,
) -> list[ThresholdSelection]:
    if not 0 < min_precision <= 1:
        raise ValueError("min_precision은 0보다 크고 1 이하여야 합니다.")
    if not 0 < top_fraction <= 1:
        raise ValueError("top_fraction은 0보다 크고 1 이하여야 합니다.")
    # f1: 기존 choose_threshold와 같은 tie-break(가장 높은 cutoff)
    # min_precision: 조건을 만족하는 후보 중 recall 최대, 동률이면 cutoff 최대
    # top_fraction: 검증 점수 상위 ceil(n * fraction)번째 점수
    # 빈 배열 또는 단일 클래스는 f1/top_fraction 0.5, min_precision unavailable
```

기존 `choose_threshold(y_true, scores)`는 호환성을 위해 `f1` 정책 cutoff를 반환한다.
`ranking_metrics()`는 5%, 10%, 20% 각각 `ceil(n*fraction)`개의 인덱스를 안정 정렬로
선택하여 Precision, Recall, Lift를 계산한다. 동점 확률이어도 정확히 해당 개수만
평가한다.

- [ ] **Step 4: 임계값·지표 테스트 실행**

Run: `pytest tests/test_industrial_training.py -v`

Expected: PASS, including the existing F1 threshold equivalence test.

- [ ] **Step 5: 커밋**

```bash
git add src/industrial_training.py tests/test_industrial_training.py
git commit -m "feat: 산업 분류 임계값과 Top-K 지표 추가"
```

---

### Task 6: 공통 다중 모델 학습·선택·산출물 구현

**Files:**
- Modify: `src/industrial_training.py:117-381`
- Modify: `tests/test_industrial_training.py`

**Interfaces:**
- Consumes: `Sequence[PreparedTask]`, 출력 경로, 범위, 모델 설정, 날짜 경계, 임계값 설정
- Produces: `run_experiment_suite(tasks, ...) -> DataFrame`, `metrics.csv`, `test_predictions.csv`, `feature_importance.csv`, `run_config.json`, `models/*.joblib`

- [ ] **Step 1: 모델 목록, 검증 AP 선택, 단일 클래스 skip 및 산출물 실패 테스트 작성**

```python
def test_build_model_handles_unseen_category():
    train = pd.DataFrame({"part_no": ["P1", "P2"], "value": [0.0, 1.0]})
    model = build_model("logistic_regression", train, max_iter=20, random_state=42)
    model.fit(train, [0, 1])
    scores = model.predict_proba(pd.DataFrame({"part_no": ["NEW"], "value": [0.5]}))
    assert scores.shape == (1, 2)


def test_suite_selects_model_by_validation_average_precision(tmp_path, prepared_task):
    metrics = run_experiment_suite(
        [prepared_task], output_dir=tmp_path, scope="overall", max_iter=10
    )
    validation = metrics.query(
        "split == 'validation' and threshold_policy == 'f1' and model != 'prior'"
    )
    model_order = {name: index for index, name in enumerate(MODEL_NAMES)}
    expected = (
        validation.assign(_order=validation["model"].map(model_order))
        .sort_values(["average_precision", "_order"], ascending=[False, True])
        .iloc[0]["model"]
    )
    selected = metrics.query("selected_model == True and split == 'test'")
    assert set(selected["model"]) == {expected}
    assert {"f1", "min_precision", "top_fraction"} == set(selected["threshold_policy"])
    assert (tmp_path / "metrics.csv").exists()
    assert (tmp_path / "test_predictions.csv").exists()
    assert (tmp_path / "feature_importance.csv").exists()
    assert (tmp_path / "run_config.json").exists()


def test_suite_records_single_class_scope_as_skipped(tmp_path, single_class_task):
    metrics = run_experiment_suite([single_class_task], output_dir=tmp_path, scope="overall")
    assert set(metrics["status"]) == {"skipped"}
    assert "single training class" in metrics["reason"].iloc[0]
```

`prepared_task`와 `single_class_task`는 다음 helper로 만든다. 세 구간에 모두 두
클래스가 있는 기본 Task와 학습 구간이 단일 클래스인 Task를 명시적으로 분리한다.

```python
def _training_task(single_class_train=False):
    dates = pd.to_datetime([
        "2023-12-01", "2023-12-02", "2023-12-03", "2023-12-04",
        "2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04",
        "2024-07-01", "2024-07-02", "2024-07-03", "2024-07-04",
    ])
    labels = [0, 0, 0, 0] if single_class_train else [0, 1, 0, 1]
    labels += [0, 1, 0, 1, 0, 1, 0, 1]
    frame = pd.DataFrame({
        "transaction_date": dates,
        "label_end_date": dates,
        "machine_type": ["Press"] * len(dates),
        "asset_tag": ["A-1", "A-2"] * 6,
        "signal": np.arange(len(dates), dtype=float),
        "target": labels,
    })
    return PreparedTask(
        frame=frame, features=("machine_type", "asset_tag", "signal"),
        target="target", grain="asset", mode="current",
    )


@pytest.fixture
def prepared_task():
    return _training_task()


@pytest.fixture
def single_class_task():
    return _training_task(single_class_train=True)
```

- [ ] **Step 2: 실패 확인**

Run: `pytest tests/test_industrial_training.py -k 'unseen or suite' -v`

Expected: FAIL because `run_experiment_suite` and new output columns are absent.

- [ ] **Step 3: 모델 생성과 선택 흐름 구현**

```python
MODEL_NAMES = (
    "logistic_regression",
    "random_forest",
    "hist_gradient_boosting",
)


def run_experiment_suite(
    tasks: Sequence[PreparedTask],
    *,
    output_dir: str | Path,
    scope: str = "all",
    machine_type: str | None = None,
    asset_tag: str | None = None,
    max_iter: int = 100,
    validation_start: str = "2024-01-01",
    test_start: str = "2024-07-01",
    threshold_policies: tuple[str, ...] = ("f1", "min_precision", "top_fraction"),
    min_precision: float = 0.30,
    top_fraction: float = 0.10,
    random_state: int = 42,
) -> pd.DataFrame:
    # 각 PreparedTask -> scope -> 모델 순서로 학습한다.
    # prior 기준선은 train 양성률의 상수 점수로 기록한다.
    # 학습모델 3개의 validation AP 중 최댓값을 selected_model=True로 표시한다.
    # 각 모델의 검증 점수에서 세 임계값을 정하고 같은 cutoff를 test에 적용한다.
    # 마지막에 네 산출물과 모델 파일을 한 번에 기록한다.
```

`threshold_policies`에 허용되는 값은 `f1`, `min_precision`, `top_fraction`이며,
그 밖의 값이나 빈 tuple에는 `ValueError`를 낸다. `validate_features()`는 현재 Target
뿐 아니라 `target_`으로 시작하는 다른 정답 컬럼과 원본 `breakdown_flag`가 Feature에
포함되는 것도 차단한다. 단, 부품 당일 모델에서 `breakdown_flag`가 Target인 것은
허용한다.

`build_model()`은 `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`를
유지하고 다음 분류기를 만든다.

```python
LogisticRegression(max_iter=max(1000, max_iter), class_weight="balanced", random_state=random_state)
RandomForestClassifier(
    n_estimators=max(50, max_iter), class_weight="balanced",
    random_state=random_state, n_jobs=-1,
)
HistGradientBoostingClassifier(
    max_iter=max_iter, early_stopping=False, random_state=random_state,
)
```

검증 AP 동률 시 `MODEL_NAMES` 앞쪽 모델을 선택한다. 빈 split, 학습 단일 클래스 또는
검증 단일 클래스는 학습하지 않고 `status="skipped"`, 구체적인 `reason`을 기록한다.
최소 Precision 정책의 cutoff가 `None`이면 해당 정책의 분류 지표는 `None`, 상태는
`unavailable`로 기록하되 AP와 ROC-AUC 같은 임계값 비의존 지표는 보존한다.

- [ ] **Step 4: 출력 계약과 Feature 중요도 구현**

`metrics.csv`는 최소한 다음 컬럼을 가진다.

```text
grain,mode,target,risk_definition,score_threshold,horizon,
scope_kind,scope_name,model,selected_model,split,status,reason,
threshold_policy,probability_cutoff,train_rows,valid_rows,test_rows,
train_positive_rate,valid_positive_rate,test_positive_rate,
accuracy,precision,recall,f1,roc_auc,average_precision,
true_negative,false_positive,false_negative,true_positive,
precision_at_5pct,recall_at_5pct,lift_at_5pct,
precision_at_10pct,recall_at_10pct,lift_at_10pct,
precision_at_20pct,recall_at_20pct,lift_at_20pct
```

`test_predictions.csv`에는 원본 키, Target, `risk_score`, `threshold_policy`,
`probability_cutoff`, `prediction`, 모델·과제 식별자를 쓴다. `feature_importance.csv`는
선택 모델에 대해 검증 데이터에서 `sklearn.inspection.permutation_importance`를
`scoring="average_precision"`, `n_repeats=3`, `random_state`로 계산하며 원본 Feature
이름별 평균·표준편차를 저장한다. 검증 행이 1,000개를 넘으면
`random_state`로 1,000행을 표본 추출한다.

`run_config.json`에는 데이터 단위, 모드, Target 목록, 날짜 경계, 모델 목록,
임계값 정책 인자, `random_state`를 UTF-8 JSON으로 저장한다. 모델 파일명에는
Target, scope, 모델명을 포함하고 저장 payload에는 `PreparedTask` 메타데이터와
각 정책 cutoff를 포함한다. 최종 선택 모델만 예를 들어
`target_ge_12__machine_type__Press__random_forest__selected.joblib`처럼 Target,
범위 종류, 범위 이름, 모델 이름과 `selected` 표시가 포함된 이름으로 저장하여 배포
대상과 비교용 지표를 혼동하지 않게 한다.

다음 저장·재로딩 테스트를 같은 테스트 파일에 추가한다.

```python
def test_saved_selected_model_reloads_with_same_predictions(tmp_path, prepared_task):
    run_experiment_suite([prepared_task], output_dir=tmp_path, scope="overall", max_iter=10)
    model_file = next((tmp_path / "models").glob("*__selected.joblib"))
    payload = joblib.load(model_file)
    rows = prepared_task.frame.loc[:, list(prepared_task.features)].head(3)
    expected = payload["verification_scores"]
    actual = payload["pipeline"].predict_proba(rows)[:, 1]
    assert np.allclose(actual, expected)
```

저장 payload의 `verification_scores`는 저장 직전 `prepared_task.frame`의 앞 3행에
대해 계산한다. 이를 통해 전처리와 분류기가 함께 저장되었는지 확인한다.

- [ ] **Step 5: 공통 학습 테스트 실행**

Run: `pytest tests/test_industrial_training.py -v`

Expected: PASS.

- [ ] **Step 6: 커밋**

```bash
git add src/industrial_training.py tests/test_industrial_training.py
git commit -m "feat: 산업 위험 공통 모델 선택과 산출물 구현"
```

---

### Task 7: 네 CLI와 기존 호환 진입점 구현

**Files:**
- Create: `src/industrial_cli.py`
- Create: `src/current_asset_model.py`
- Create: `src/forecast_asset_model.py`
- Create: `src/current_part_model.py`
- Create: `src/forecast_part_model.py`
- Create: `tests/test_industrial_entrypoints.py`
- Modify: `src/current_state_model.py:1-42`
- Modify: `src/timeseries_model.py:1-43`

**Interfaces:**
- Consumes: 네 `prepare_*` 함수와 `run_experiment_suite()`
- Produces: 독립 실행 가능한 네 명시적 CLI와 두 호환 CLI

- [ ] **Step 1: 기본 인자와 준비 함수 라우팅 실패 테스트 작성**

```python
# tests/test_industrial_entrypoints.py
import subprocess
import sys

import pytest


@pytest.mark.parametrize("script", [
    "src/current_asset_model.py",
    "src/forecast_asset_model.py",
    "src/current_part_model.py",
    "src/forecast_part_model.py",
    "src/current_state_model.py",
    "src/timeseries_model.py",
])
def test_entrypoint_help(script):
    result = subprocess.run(
        [sys.executable, script, "--help"], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


def test_forecast_asset_parser_defaults_to_new_seven_days():
    parser = build_forecast_asset_parser()
    args = parser.parse_args([])
    assert args.horizon == 7
    assert args.risk_definition == "new"
    assert args.score_threshold == [12, 13, 14]
```

- [ ] **Step 2: 실패 확인**

Run: `pytest tests/test_industrial_entrypoints.py -v`

Expected: FAIL because the four explicit entrypoints do not exist.

- [ ] **Step 3: 공통 CLI 인자와 네 진입점 구현**

```python
# src/industrial_cli.py
def add_common_arguments(parser: argparse.ArgumentParser, default_output: Path) -> None:
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--output", default=default_output)
    parser.add_argument("--scope", choices=["all", "overall", "machine_type", "asset_tag"], default="all")
    parser.add_argument("--machine-type")
    parser.add_argument("--asset-tag")
    parser.add_argument("--validation-start", default="2024-01-01")
    parser.add_argument("--test-start", default="2024-07-01")
    parser.add_argument(
        "--threshold-policy", action="append",
        choices=["f1", "min_precision", "top_fraction"],
    )
    parser.add_argument("--min-precision", type=float, default=0.30)
    parser.add_argument("--top-fraction", type=float, default=0.10)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--random-state", type=int, default=42)
```

`--threshold-policy`를 생략하면 세 정책을 모두 실행하며, 한 번 이상 지정하면 지정된
정책만 중복 없이 입력 순서대로 실행한다.

장비 CLI에는 다음 공통 인자를 추가한다.

```python
parser.add_argument(
    "--score-threshold", type=int, action="append", dest="score_threshold"
)
```

파싱 후 값이 `None`이면 `[12, 13, 14]`를 사용한다. 미래 CLI는
`--horizon` 기본값 7을 사용하고, 장비 미래 CLI는
`--risk-definition {new,any}` 기본값 `new`를 사용한다.

각 진입점은 원본을 한 번 로드하고 필요한 `PreparedTask` 목록을 만든 다음
`run_experiment_suite()`를 한 번 호출한다. 새 기본 출력 경로는 각각
`outputs/asset_current`, `outputs/asset_forecast_7d`, `outputs/part_current`,
`outputs/part_forecast_7d`다.

- [ ] **Step 4: 기존 CLI를 출력 경로까지 유지하는 호환 래퍼로 변경**

```python
# src/current_state_model.py
from src.current_part_model import main

if __name__ == "__main__":
    main(default_output=PROJECT_ROOT / "outputs" / "current_state")
```

`src/timeseries_model.py`도 `forecast_part_model.main()`을 호출하되 기본 출력은
`outputs/timeseries`로 유지한다. 직접 실행과 `python -m src.current_part_model` 같은 모듈 실행을 모두
지원하도록 현재의 `sys.path` 보정 패턴을 유지한다.

- [ ] **Step 5: CLI 테스트 실행**

Run: `pytest tests/test_industrial_entrypoints.py -v`

Expected: PASS.

- [ ] **Step 6: 커밋**

```bash
git add src/industrial_cli.py src/current_asset_model.py src/forecast_asset_model.py \
  src/current_part_model.py src/forecast_part_model.py src/current_state_model.py \
  src/timeseries_model.py tests/test_industrial_entrypoints.py
git commit -m "feat: 장비와 부품 위험 모델 CLI 분리"
```

---

### Task 8: `eda.py`를 결과 비교 전용으로 변경하고 문서 갱신

**Files:**
- Modify: `src/eda.py:1-293`
- Modify: `src/RUN_INDUSTRIAL.md:1-31`
- Modify: `docs/industrial_feature_guide.md:1-96`
- Create: `docs/industrial_code_guide.md`
- Create: `tests/test_eda.py`

**Interfaces:**
- Consumes: 네 출력 폴더의 `metrics.csv`
- Produces: `load_metric_outputs(paths) -> DataFrame`, `build_comparison(metrics) -> DataFrame`, `outputs/model_comparison.csv`

- [ ] **Step 1: 결과 비교 실패 테스트 작성**

```python
# tests/test_eda.py
import pandas as pd

from src.eda import build_comparison


def test_build_comparison_keeps_selected_test_rows_only():
    metrics = pd.DataFrame([
        {"grain": "asset", "mode": "current", "model": "random_forest",
         "selected_model": True, "split": "test", "threshold_policy": "f1",
         "average_precision": 0.4},
        {"grain": "asset", "mode": "current", "model": "logistic_regression",
         "selected_model": False, "split": "test", "threshold_policy": "f1",
         "average_precision": 0.3},
        {"grain": "asset", "mode": "current", "model": "random_forest",
         "selected_model": True, "split": "validation", "threshold_policy": "f1",
         "average_precision": 0.5},
    ])
    result = build_comparison(metrics)
    assert len(result) == 1
    assert result.iloc[0]["model"] == "random_forest"
```

- [ ] **Step 2: 실패 확인**

Run: `pytest tests/test_eda.py -v`

Expected: FAIL because importing the current `eda.py` executes training and no comparison function exists.

- [ ] **Step 3: 결과 비교 전용 EDA 구현**

`src/eda.py`에서 학습 코드와 사용하지 않는 LightGBM·그래프 import를 제거한다.
기본 입력은 다음 네 경로로 한다.

```python
DEFAULT_METRICS = [
    PROJECT_ROOT / "outputs" / "asset_current" / "metrics.csv",
    PROJECT_ROOT / "outputs" / "asset_forecast_7d" / "metrics.csv",
    PROJECT_ROOT / "outputs" / "part_current" / "metrics.csv",
    PROJECT_ROOT / "outputs" / "part_forecast_7d" / "metrics.csv",
]
```

`load_metric_outputs()`는 존재하는 파일만 읽되 하나도 없으면 실행해야 할 네 명령을
포함한 `FileNotFoundError`를 낸다. `build_comparison()`은 `split == "test"`,
`selected_model == True`인 행을 선택하고 과제·scope·정책별 AP, Precision, Recall,
F1, Top-10% 지표를 반환한다. CLI는 표를 출력하고 기본적으로
`outputs/model_comparison.csv`에 저장한다.

- [ ] **Step 4: 실행 문서와 Feature 문서 갱신**

`src/RUN_INDUSTRIAL.md`에 macOS/Linux용 다음 예시와 Windows 예시를 모두 쓴다.

```bash
python src/current_asset_model.py
python src/forecast_asset_model.py --horizon 7 --risk-definition new
python src/forecast_asset_model.py --horizon 7 --risk-definition any
python src/current_part_model.py
python src/forecast_part_model.py --horizon 7
python src/eda.py
```

`docs/industrial_feature_guide.md`에는 네 Target 수식, 장비/부품 그룹 키, 미래 센서
Feature가 날짜 \(t\)를 포함한다는 점, 고장 이력이 `shift(1)`부터 시작한다는 점을
반영한다. 교육용 합성 데이터의 한계와 식별자 기억 효과를 유지한다.

`docs/industrial_code_guide.md`는 다음 순서로 작성한다.

```markdown
# 산업 위험 모델 코드 가이드

## 1. 전체 실행 흐름
원본 CSV → Feature/Target 생성 → 시간 분할 → 모델 비교 → 임계값 선택 → 결과 저장

## 2. 실행 파일
각 파일의 목적, 기본 Target, 실행 명령, 주요 선택 인자를 설명한다.

## 3. 공통 모듈
industrial_data.py, asset_features.py, part_features.py,
industrial_training.py, industrial_cli.py의 책임과 공개 함수를 설명한다.

## 4. 데이터 누수 방지
label_end_date, shift(1), 미래 날짜 연속성 검사가 필요한 이유를 예시로 설명한다.

## 5. 결과 파일 읽기
metrics.csv, test_predictions.csv, feature_importance.csv,
run_config.json, models/의 주요 컬럼과 해석 예시를 설명한다.

## 6. 기존 파일 호환성
current_state_model.py, timeseries_model.py, industrial_features.py가
어떤 새 파일로 연결되는지 설명한다.
```

각 Python 파일의 모듈 docstring과 공개 함수 docstring이 존재하는지도 다음 테스트로
고정한다.

```python
def test_public_industrial_modules_have_korean_documentation():
    modules = [
        industrial_data, asset_features, part_features, industrial_training,
        industrial_cli, current_asset_model, forecast_asset_model,
        current_part_model, forecast_part_model,
    ]
    for module in modules:
        assert module.__doc__ and any("가" <= char <= "힣" for char in module.__doc__)
        for name, function in inspect.getmembers(module, inspect.isfunction):
            if function.__module__ == module.__name__ and not name.startswith("_"):
                assert function.__doc__, f"{module.__name__}.{name}에 docstring이 없습니다."
```

- [ ] **Step 5: EDA와 문서 연관 테스트 실행**

Run: `pytest tests/test_eda.py tests/test_industrial_entrypoints.py -v`

Expected: PASS.

- [ ] **Step 6: 커밋**

```bash
git add src/eda.py src/RUN_INDUSTRIAL.md docs/industrial_feature_guide.md \
  docs/industrial_code_guide.md tests/test_eda.py
git commit -m "docs: 산업 위험 모델 실행과 결과 비교 정리"
```

---

### Task 9: 전체 회귀·스모크 검증과 최종 정리

**Files:**
- Modify if required by failures: files already listed in Tasks 1-8 only

**Interfaces:**
- Consumes: 완료된 네 모델 흐름
- Produces: 통과한 전체 테스트, 실제 데이터 스모크 산출물, 깨끗한 작업 트리

- [ ] **Step 1: 전체 테스트 실행**

Run: `pytest -v`

Expected: 모든 비-xfail 테스트 PASS. 기존 `tests/test_features.py`의 사전 존재 xfail은
그대로 허용하되 새 xfail은 추가하지 않는다.

- [ ] **Step 2: 네 새 CLI 도움말과 두 호환 CLI 도움말 확인**

```bash
python src/current_asset_model.py --help
python src/forecast_asset_model.py --help
python src/current_part_model.py --help
python src/forecast_part_model.py --help
python src/current_state_model.py --help
python src/timeseries_model.py --help
```

Expected: 모두 exit code 0이며 장비 CLI에는 `--score-threshold`, 미래 CLI에는
`--horizon`, 장비 미래 CLI에는 `--risk-definition`이 표시된다.

- [ ] **Step 3: 실제 데이터로 네 과제 overall 스모크 실행**

```bash
python src/current_asset_model.py --scope overall --max-iter 20 --output outputs/verification_asset_current
python src/forecast_asset_model.py --scope overall --max-iter 20 --output outputs/verification_asset_forecast
python src/current_part_model.py --scope overall --max-iter 20 --output outputs/verification_part_current
python src/forecast_part_model.py --scope overall --max-iter 20 --output outputs/verification_part_forecast
```

Expected: 네 명령 모두 exit code 0. 각 폴더에 `metrics.csv`,
`test_predictions.csv`, `feature_importance.csv`, `run_config.json`, `models/`가 존재한다.

또한 `docs/industrial_code_guide.md`가 존재하고, 새 산업 모델 모듈의 공개 함수
docstring 검사도 전체 `pytest` 결과에 포함되어 통과해야 한다.

- [ ] **Step 4: 산출물 스키마와 선택 규칙 검사**

```bash
python - <<'PY'
from pathlib import Path
import pandas as pd

roots = [
    Path("outputs/verification_asset_current"),
    Path("outputs/verification_asset_forecast"),
    Path("outputs/verification_part_current"),
    Path("outputs/verification_part_forecast"),
]
required = {
    "grain", "mode", "target", "model", "selected_model", "split",
    "threshold_policy", "average_precision", "precision_at_10pct",
    "recall_at_10pct", "lift_at_10pct",
}
for root in roots:
    metrics = pd.read_csv(root / "metrics.csv")
    assert required <= set(metrics.columns), (root, required - set(metrics.columns))
    assert metrics.query("selected_model == True and split == 'test'").shape[0] > 0
    assert (root / "test_predictions.csv").stat().st_size > 0
    assert (root / "run_config.json").stat().st_size > 0
print("산출물 검증 성공")
PY
```

Expected: `산출물 검증 성공`.

- [ ] **Step 5: 호환 CLI 스모크 실행**

```bash
python src/current_state_model.py --scope overall --max-iter 10 --output outputs/verification_legacy_current
python src/timeseries_model.py --scope overall --max-iter 10 --output outputs/verification_legacy_forecast
```

Expected: 두 명령 모두 exit code 0이고 공통 산출물 계약을 만족한다.

- [ ] **Step 6: diff와 작업 트리 검토**

```bash
git diff --check
git status --short
git log --oneline -10
```

Expected: `git diff --check` 출력 없음. 의도하지 않은 파일과 실행 중 생성된 검증
산출물이 커밋 대상에 포함되지 않음.

- [ ] **Step 7: 검증 중 필요한 최소 수정이 있었다면 최종 커밋**

```bash
git add src/industrial_data.py src/asset_features.py src/part_features.py \
  src/industrial_features.py src/industrial_training.py src/industrial_cli.py \
  src/current_asset_model.py src/forecast_asset_model.py \
  src/current_part_model.py src/forecast_part_model.py \
  src/current_state_model.py src/timeseries_model.py src/eda.py \
  src/RUN_INDUSTRIAL.md docs/industrial_feature_guide.md \
  docs/industrial_code_guide.md \
  tests/test_industrial_data.py tests/test_asset_features.py \
  tests/test_part_features.py tests/test_industrial_training.py \
  tests/test_industrial_entrypoints.py tests/test_eda.py
git commit -m "test: 산업 위험 모델 통합 검증 보완"
```

검증 수정이 없으면 빈 커밋을 만들지 않는다.
