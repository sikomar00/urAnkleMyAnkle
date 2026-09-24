"""산업 장비 데이터의 공통 계약, 날짜 Feature와 시간 분할을 제공한다.

장비·부품 모델은 이 모듈의 컬럼 이름과 :class:`PreparedTask`를 공유한다.
미래 예측에서는 Target 관측 종료일을 기준으로 분할하여 검증 데이터가 테스트
기간의 정답을 미리 보지 않도록 한다.
"""

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
    "temp_bearing_degC",
    "temp_motor_degC",
    "vibration_h_mms",
    "vibration_v_mms",
    "oil_pressure_bar",
    "load_pct",
    "shaft_rpm",
    "power_consumption_kw",
]

REQUIRED_COLUMNS = [
    DATE_COLUMN,
    ASSET_COLUMN,
    PART_COLUMN,
    MACHINE_COLUMN,
    PLANT_COLUMN,
    "criticality",
    CURRENT_TARGET,
    *SENSOR_COLUMNS,
]


@dataclass(frozen=True)
class PreparedTask:
    """학습 준비가 끝난 데이터와 과제 메타데이터를 함께 보관한다.

    Args:
        frame: Feature와 Target이 들어 있는 학습용 데이터.
        features: 모델 입력 컬럼 이름.
        target: 정답 컬럼 이름.
        grain: 데이터 단위. ``asset`` 또는 ``part``를 사용한다.
        mode: ``current`` 또는 ``forecast``.
        risk_definition: 현재·신규·전체 위험을 구분하는 이름.
        score_threshold: 장비 고장점수 판정 기준.
        horizon: 미래 예측 일수.

    Raises:
        ValueError: Target이 Feature에 포함되거나 필요한 컬럼이 없을 때.
    """

    frame: pd.DataFrame
    features: tuple[str, ...]
    target: str
    grain: str
    mode: str
    risk_definition: str = "current"
    score_threshold: int | None = None
    horizon: int | None = None

    def __post_init__(self) -> None:
        """Feature와 Target의 기본 계약을 검사한다."""
        if self.target in self.features:
            raise ValueError("Target 컬럼을 Feature에 포함할 수 없습니다.")
        missing = [
            column
            for column in (*self.features, self.target)
            if column not in self.frame
        ]
        if missing:
            raise ValueError(f"PreparedTask에 필요한 컬럼이 없습니다: {missing}")


def load_industrial_data(path: str | Path) -> pd.DataFrame:
    """원본 CSV를 읽고 공통 필수 컬럼과 고장값을 검증한다.

    Args:
        path: 읽을 CSV 경로.

    Returns:
        날짜가 datetime으로 변환된 원본 데이터 복사본.

    Raises:
        ValueError: 필수 컬럼이 없거나 ``breakdown_flag``가 숫자가 아닐 때.
    """
    frame = pd.read_csv(path, parse_dates=[DATE_COLUMN])
    missing = [column for column in REQUIRED_COLUMNS if column not in frame]
    if missing:
        raise ValueError(f"CSV에 필요한 컬럼이 없습니다: {missing}")
    frame[CURRENT_TARGET] = pd.to_numeric(frame[CURRENT_TARGET], errors="coerce")
    if frame[CURRENT_TARGET].isna().any():
        raise ValueError("breakdown_flag에 숫자가 아닌 값이 있습니다.")
    return frame


def add_calendar_features(frame: pd.DataFrame) -> pd.DataFrame:
    """날짜에서 요일과 월 주기 Feature를 만든다.

    Args:
        frame: ``transaction_date``가 포함된 데이터.

    Returns:
        입력을 변경하지 않고 달력 Feature가 추가된 복사본.
    """
    result = frame.copy()
    dates = pd.to_datetime(result[DATE_COLUMN])
    result["day_of_week"] = dates.dt.dayofweek
    result["month_sin"] = np.sin(2 * np.pi * dates.dt.month / 12)
    result["month_cos"] = np.cos(2 * np.pi * dates.dt.month / 12)
    return result


def split_by_date(
    frame: pd.DataFrame,
    validation_start: str | pd.Timestamp = "2024-01-01",
    test_start: str | pd.Timestamp = "2024-07-01",
) -> dict[str, pd.DataFrame]:
    """Target 관측 종료일을 고려해 학습·검증·테스트를 분리한다.

    Args:
        frame: ``transaction_date``와 ``label_end_date``가 포함된 데이터.
        validation_start: 검증 입력 기간의 시작일.
        test_start: 테스트 입력 기간의 시작일.

    Returns:
        ``train``, ``valid``, ``test`` DataFrame을 담은 딕셔너리.

    Raises:
        ValueError: 날짜 경계가 역전되거나 필수 날짜 컬럼이 없을 때.
    """
    validation_start = pd.Timestamp(validation_start)
    test_start = pd.Timestamp(test_start)
    if validation_start >= test_start:
        raise ValueError("validation_start는 test_start보다 빨라야 합니다.")

    required = {DATE_COLUMN, "label_end_date"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"시간 분할에 필요한 컬럼이 없습니다: {missing}")

    result = frame.copy()
    result[DATE_COLUMN] = pd.to_datetime(result[DATE_COLUMN])
    result["label_end_date"] = pd.to_datetime(result["label_end_date"])

    # 입력 날짜가 검증 구간에 있더라도 정답 관측 기간이 테스트까지 넘어가면
    # 검증 데이터에서 제외하여 테스트 정답 누수를 막는다.
    train = result[result["label_end_date"] < validation_start].copy()
    valid = result[
        (result[DATE_COLUMN] >= validation_start)
        & (result["label_end_date"] < test_start)
    ].copy()
    test = result[result[DATE_COLUMN] >= test_start].copy()
    return {"train": train, "valid": valid, "test": test}


def _filter_frame(
    frame: pd.DataFrame,
    machine_type: str | None = None,
    asset_tag: str | None = None,
) -> pd.DataFrame:
    """요청한 기계 종류와 장비 식별자로 데이터를 제한한다."""
    result = frame
    if machine_type is not None:
        result = result[result[MACHINE_COLUMN].eq(machine_type)]
    if asset_tag is not None:
        result = result[result[ASSET_COLUMN].eq(asset_tag)]
    if result.empty:
        raise ValueError("지정한 machine_type 또는 asset_tag에 해당하는 데이터가 없습니다.")
    return result


def iter_scopes(
    frame: pd.DataFrame,
    scope: str = "all",
    machine_type: str | None = None,
    asset_tag: str | None = None,
) -> Iterator[tuple[str, str, pd.DataFrame]]:
    """전체·기계 종류·장비별 평가용 데이터 조각을 순서대로 반환한다.

    Args:
        frame: 분할할 학습 데이터.
        scope: ``all``, ``overall``, ``machine_type``, ``asset_tag`` 중 하나.
        machine_type: 선택적으로 제한할 기계 종류.
        asset_tag: 선택적으로 제한할 장비 식별자.

    Yields:
        범위 종류, 범위 이름, 해당 DataFrame의 튜플.

    Raises:
        ValueError: 지원하지 않는 scope이거나 필터 결과가 비었을 때.
    """
    valid_scopes = {"all", "overall", "machine_type", "asset_tag"}
    if scope not in valid_scopes:
        raise ValueError(f"scope는 다음 중 하나여야 합니다: {sorted(valid_scopes)}")
    filtered = _filter_frame(frame, machine_type, asset_tag)
    if scope in {"all", "overall"}:
        yield "overall", "all", filtered.copy()
    if scope in {"all", "machine_type"} and asset_tag is None:
        for value in sorted(filtered[MACHINE_COLUMN].dropna().unique()):
            yield (
                "machine_type",
                str(value),
                filtered[filtered[MACHINE_COLUMN].eq(value)].copy(),
            )
    if scope in {"all", "asset_tag"}:
        for value in sorted(filtered[ASSET_COLUMN].dropna().unique()):
            yield (
                "asset_tag",
                str(value),
                filtered[filtered[ASSET_COLUMN].eq(value)].copy(),
            )
