"""화면 ① "현황" KPI·점검 우선순위 표에 필요한 데이터 로딩·가공을 담당한다.

원본 CSV를 :mod:`asset_features`의 기존 집계 함수(``build_asset_daily``,
``add_asset_severity``)로 가공할 뿐, 등급가중 고장점수나 4단계 위험도 판정
로직을 여기서 새로 만들지 않는다.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import pandas as pd

from .asset_features import add_asset_severity, build_asset_daily
from .industrial_data import (
    ASSET_COLUMN,
    CURRENT_TARGET,
    DATE_COLUMN,
    MACHINE_COLUMN,
    PART_COLUMN,
    PLANT_COLUMN,
    SENSOR_COLUMNS,
    load_industrial_data,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = (
    PROJECT_ROOT / "data" / "raw" / "synthetic_industrial_machine_data.csv"
)

# 화면 ③ 세그먼트 컨트롤의 위험 기준선(12/13/14) 중 기본값과 동일하게 고정한다.
# 세그먼트와의 연동은 이번 작업 범위 밖이다.
HIGH_RISK_THRESHOLD = 12

# 화면 ②의 "현재 등급" 표시 전용 — severity_level(영문) 매핑. load_screen1_kpis()/
# load_priority_table()의 high_risk 판정 로직은 severity_level 원문을 그대로 쓴다.
SEVERITY_LABELS_KO = {"normal": "정상", "caution": "주의", "risk": "경계", "high_risk": "위험"}

# 화면 ④ "데이터 사전" 전용 — pandas dtype.kind → 4종 라벨.
DTYPE_LABELS = {"M": "날짜", "O": "문자열", "i": "정수", "f": "실수"}

# 화면 ④ "데이터 사전"의 단위·설명 — 지시서 표 그대로, CSV 컬럼 순서.
DATA_DICTIONARY_META = {
    "transaction_date": ("—", "관측일(일 단위)"),
    "asset_tag": ("—", "기계 고유 ID"),
    "machine_type": ("—", "기계 종류"),
    "plant_code": ("—", "공장 코드"),
    "part_no": ("—", "부품 고유 번호"),
    "part_description": ("—", "부품 설명"),
    "part_family": ("—", "부품 범주"),
    "criticality": ("—", "중요도 ABC (A=높음, C=낮음)"),
    "uom": ("—", "수량 단위 (EA 등)"),
    "unit_cost_inr": ("INR", "부품 단가"),
    "qty_issued": ("EA", "출고 수량 (0=수요 없음)"),
    "issue_value_inr": ("INR", "출고 금액 (수량×단가)"),
    "temp_bearing_degC": ("°C", "베어링 온도"),
    "temp_motor_degC": ("°C", "모터 온도"),
    "vibration_h_mms": ("mm/s", "수평 진동 (RMS)"),
    "vibration_v_mms": ("mm/s", "수직 진동 (RMS)"),
    "oil_pressure_bar": ("bar", "오일·유압"),
    "load_pct": ("%", "정격 대비 부하율"),
    "shaft_rpm": ("rpm", "축 회전 속도"),
    "power_consumption_kw": ("kW", "소비 전력"),
    "breakdown_flag": ("—", "고장 표시 (1=있음, 0=없음)"),
    "wo_type": ("—", "작업지시 유형 (BD·PM·작업 없음). 결측이 아니라 범주"),
}


def _data_path() -> Path:
    """MACHINE_DATA_PATH 환경변수가 있으면 우선하고, 없으면 기본 원본 CSV 경로."""
    override = os.environ.get("MACHINE_DATA_PATH")
    return Path(override) if override else DEFAULT_DATA_PATH


@lru_cache(maxsize=1)
def _load_raw() -> pd.DataFrame:
    return load_industrial_data(_data_path())


@lru_cache(maxsize=1)
def _daily() -> pd.DataFrame:
    daily = build_asset_daily(_load_raw())
    return add_asset_severity(daily, high_risk_threshold=HIGH_RISK_THRESHOLD)


def _last_failure_date_by_asset(daily: pd.DataFrame) -> pd.Series:
    """자산별 failure_points > 0인 가장 최근 날짜 (전체 기간 내)."""
    failed_days = daily[daily["failure_points"] > 0]
    return failed_days.groupby(ASSET_COLUMN)[DATE_COLUMN].max()


def load_screen1_kpis() -> dict:
    """화면 ① KPI 타일 6개의 값을 계산한다 (필터 미반영, 전체 스냅샷)."""
    raw = _load_raw()
    daily = _daily()

    observed_machines = int(raw[ASSET_COLUMN].nunique())
    failure_machine_days = int((daily["failure_points"] > 0).sum())

    latest_date = daily[DATE_COLUMN].max()
    latest = daily[daily[DATE_COLUMN].eq(latest_date)]
    high_risk_count = int((latest["severity_level"] == "high_risk").sum())
    failure_rate_pct = (
        high_risk_count / observed_machines * 100 if observed_machines else 0.0
    )

    return {
        "observed_machines": observed_machines,
        "failure_machine_days": failure_machine_days,
        "failure_rate_pct": failure_rate_pct,
        "avg_power_kw": float(latest["power_consumption_kw"].mean()),
        "max_bearing_temp": float(latest["temp_bearing_degC"].max()),
        "parts_issue_value_inr": float(raw["issue_value_inr"].sum()),
    }


def load_priority_table(sort_by: str = "grade") -> list[dict]:
    """"점검 우선순위" 표의 행 데이터를 만든다.

    Args:
        sort_by: ``"grade"``(등급가중 고장점수 내림차순, 기본) 또는
            ``"threshold"``(기준선 초과 우선, 동률이면 고장점수 내림차순).
    """
    raw = _load_raw()
    daily = _daily()

    latest_date = daily[DATE_COLUMN].max()
    latest = daily[daily[DATE_COLUMN].eq(latest_date)].set_index(ASSET_COLUMN)

    asset_info = (
        raw[[ASSET_COLUMN, MACHINE_COLUMN, PLANT_COLUMN]]
        .drop_duplicates(subset=[ASSET_COLUMN])
        .set_index(ASSET_COLUMN)
    )

    last_failure_date = _last_failure_date_by_asset(daily)

    rows = []
    for asset_tag, info in asset_info.iterrows():
        if asset_tag in latest.index:
            failure_points = float(latest.loc[asset_tag, "failure_points"])
            threshold_exceeded = bool(
                latest.loc[asset_tag, "severity_level"] == "high_risk"
            )
        else:
            failure_points = 0.0
            threshold_exceeded = False

        recent_date = last_failure_date.get(asset_tag)
        if pd.notna(recent_date):
            same_day_failed = raw[
                raw[ASSET_COLUMN].eq(asset_tag)
                & raw[DATE_COLUMN].eq(recent_date)
                & raw[CURRENT_TARGET].eq(1)
            ]
            failed_part_count = int(same_day_failed[PART_COLUMN].nunique())
            last_failure_date_str = recent_date.strftime("%Y-%m-%d")
        else:
            failed_part_count = 0
            last_failure_date_str = None

        rows.append(
            {
                "asset_tag": asset_tag,
                "machine_type": info[MACHINE_COLUMN],
                "plant_code": info[PLANT_COLUMN],
                "failure_points": failure_points,
                "threshold_exceeded": threshold_exceeded,
                "last_failure_date": last_failure_date_str,
                "failed_part_count": failed_part_count,
            }
        )

    if sort_by == "threshold":
        rows.sort(key=lambda row: (not row["threshold_exceeded"], -row["failure_points"]))
    else:
        rows.sort(key=lambda row: -row["failure_points"])

    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def load_asset_list() -> list[str]:
    """전체 자산 태그를 알파벳 오름차순으로 반환한다."""
    raw = _load_raw()
    return sorted(raw[ASSET_COLUMN].unique().tolist())


def load_asset_detail_kpis(asset_tag: str) -> dict:
    """화면 ② 상단 스트립의 값을 계산한다 (선택된 자산 1개 기준).

    Raises:
        ValueError: asset_tag가 데이터에 없을 때.
    """
    assets = load_asset_list()
    if asset_tag not in assets:
        raise ValueError(f"알 수 없는 asset_tag입니다: {asset_tag}")

    raw = _load_raw()
    daily = _daily()
    asset_daily = daily[daily[ASSET_COLUMN].eq(asset_tag)]

    info = raw[raw[ASSET_COLUMN].eq(asset_tag)].iloc[0]

    latest_date = daily[DATE_COLUMN].max()
    latest_row = asset_daily[asset_daily[DATE_COLUMN].eq(latest_date)].iloc[0]

    last_failure_date = _last_failure_date_by_asset(daily).get(asset_tag)
    last_failure_date_str = (
        last_failure_date.strftime("%Y-%m-%d") if pd.notna(last_failure_date) else None
    )

    window_start = latest_date - pd.Timedelta(days=29)
    recent_30d = asset_daily[
        asset_daily[DATE_COLUMN].between(window_start, latest_date, inclusive="both")
    ]

    return {
        "asset_tag": asset_tag,
        "machine_type": info[MACHINE_COLUMN],
        "plant_code": info[PLANT_COLUMN],
        "current_grade": SEVERITY_LABELS_KO.get(
            str(latest_row["severity_level"]), str(latest_row["severity_level"])
        ),
        "risk_score": float(latest_row["failure_points"]),
        "last_failure_date": last_failure_date_str,
        "failure_days_count": int((asset_daily["failure_points"] > 0).sum()),
        "avg_power_30d_kw": float(recent_30d["power_consumption_kw"].mean()),
    }


def load_data_dictionary() -> list[dict]:
    """화면 ④ "데이터 사전" 22행을 CSV 컬럼 순서 그대로 반환한다."""
    raw = _load_raw()
    n = len(raw)

    rows = []
    for column in raw.columns:
        dtype_label = DTYPE_LABELS[raw[column].dtype.kind]
        if column == "wo_type":
            missing_pct = None  # 결측이 아니라 "작업 없음" 범주 — 화면에서 "—"로 표시
        else:
            missing_pct = round(float(raw[column].isna().sum()) / n * 100, 2)
        unit, description = DATA_DICTIONARY_META[column]
        rows.append({
            "column": column,
            "dtype_label": dtype_label,
            "unit": unit,
            "missing_pct": missing_pct,
            "description": description,
        })
    return rows


def load_data_quality_summary() -> dict:
    """화면 ④ "품질 요약" 타일 4개의 값을 계산한다."""
    raw = _load_raw()
    n = len(raw)

    composite_key_duplicates = int(raw.duplicated([DATE_COLUMN, ASSET_COLUMN, PART_COLUMN]).sum())
    full_duplicates = int(raw.duplicated().sum())

    wo_type_blank_count = int(raw["wo_type"].isna().sum())
    wo_type_blank_pct = round(wo_type_blank_count / n * 100, 2)

    period_start = raw[DATE_COLUMN].min()
    period_end = raw[DATE_COLUMN].max()
    period_days = int((period_end - period_start).days) + 1

    group_keys = [DATE_COLUMN, MACHINE_COLUMN, ASSET_COLUMN]
    group_sizes = raw.groupby(group_keys).size()
    sensor_counts = raw.groupby(group_keys)[SENSOR_COLUMNS].nunique(dropna=False)
    identical_sensor_groups = int(sensor_counts.le(1).all(axis=1).sum())

    return {
        "composite_key_duplicates": composite_key_duplicates,
        "full_duplicates": full_duplicates,
        "wo_type_blank_count": wo_type_blank_count,
        "wo_type_blank_pct": wo_type_blank_pct,
        "period_days": period_days,
        "period_start": period_start.strftime("%Y-%m-%d"),
        "period_end": period_end.strftime("%Y-%m-%d"),
        "group_count": int(len(group_sizes)),
        "rows_per_group": int(group_sizes.iloc[0]) if len(group_sizes) else 0,
        "identical_sensor_groups": identical_sensor_groups,
    }


def load_source_info() -> dict:
    """화면 ④ "출처 · 라이선스 · 합성 데이터 한계" 카드의 값을 계산한다."""
    raw = _load_raw()
    row_count, col_count = raw.shape

    return {
        "row_count": row_count,
        "col_count": col_count,
        "source_name": "Kaggle — Machine Demand & Failure Prediction Dataset",
        "access_date_note": "다운로드일: 2026-09-22 이전 (정확한 날짜 미기록)",
        "license": "CC BY-SA 4.0 (2026-09-21 게시 페이지 확인 기준)",
        "limitations": (
            "물리적 정확성이 아닌 통계적 현실성을 목표로 만든 합성 데이터이며 결과를 "
            "실제 설비의 고장 기준·정비 효과로 일반화하지 않는다"
        ),
    }
