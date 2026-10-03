"""화면 ① "현황" KPI·점검 우선순위 표에 필요한 데이터 로딩·가공을 담당한다.

원본 CSV를 :mod:`asset_features`의 기존 집계 함수(``build_asset_daily``,
``add_asset_severity``)로 가공할 뿐, 등급가중 고장점수나 4단계 위험도 판정
로직을 여기서 새로 만들지 않는다.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_curve

from .asset_features import add_asset_severity, build_asset_daily
from .family_features import FAMILY_NAMES, build_family_daily
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

# 화면 ③ "부품군 진단" 과제의 자산별 부품군(9종) 이상탐지 지표 원본.
DEFAULT_FAMILY_METRICS_PATH = (
    PROJECT_ROOT / "outputs" / "family_current" / "metrics.csv"
)
FAMILY_DIAGNOSIS_METRIC_COLUMNS = [
    "support", "positive_rate", "precision", "recall", "average_precision", "roc_auc",
]

# 화면 ③ 과제 ①~③ 모델 비교 — src/model_comparison.py가 만든 결과를 읽기만 한다.
COMPARISON_TASKS = {
    "machine_risk": PROJECT_ROOT / "outputs" / "machine_risk",
    "part_within_7d": PROJECT_ROOT / "outputs" / "pf_within_7d",
    "part_current": PROJECT_ROOT / "outputs" / "current_state",
}
COMPARISON_MODEL_ORDER = ["baseline_a", "baseline_b", "logistic_regression", "random_forest",
                          "hist_gradient_boosting"]
PR_CURVE_MAX_POINTS = 600

# 화면 ③ "부품군 진단" 드릴다운(PR곡선·혼동행렬) 원본 — 일별 예측 로그.
DEFAULT_FAMILY_TEST_PREDICTIONS_PATH = (
    PROJECT_ROOT / "outputs" / "family_current" / "test_predictions.csv"
)
# 화면 ③ "부품군 진단" 드릴다운(변수중요도) 원본.
DEFAULT_FAMILY_FEATURE_IMPORTANCE_PATH = (
    PROJECT_ROOT / "outputs" / "family_current" / "feature_importance.csv"
)

# 화면 ①②의 기계 등급(severity_level "high_risk")에 쓰는 위험 기준선 — 12점 고정.
# 화면 ③의 12/13/14 토글은 모델 비교 결과만 바꾸고 이 값에는 영향을 주지 않는다.
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


def period_start(days: int | None) -> pd.Timestamp | None:
    """필터바 기간 — 데이터 최신일을 포함한 최근 ``days``일의 시작일. None이면 전체 기간."""
    if days is None:
        return None
    return _load_raw()[DATE_COLUMN].max() - pd.Timedelta(days=days - 1)


@lru_cache(maxsize=1)
def load_asset_catalog() -> pd.DataFrame:
    """기계별 공장·기계 종류(asset_tag 오름차순) — 필터바 선택지와 필터 대상 계산용."""
    return (
        _load_raw()[[ASSET_COLUMN, PLANT_COLUMN, MACHINE_COLUMN]]
        .drop_duplicates(subset=[ASSET_COLUMN])
        .sort_values(ASSET_COLUMN)
        .reset_index(drop=True)
    )


def filter_assets(plant: str | None = None, machine_type: str | None = None,
                  machine: str | None = None) -> list[str]:
    """필터바 공장·기계 종류·기계 조건에 맞는 기계 태그(오름차순). None은 조건 없음."""
    catalog = load_asset_catalog()
    for column, value in ((PLANT_COLUMN, plant), (MACHINE_COLUMN, machine_type), (ASSET_COLUMN, machine)):
        if value:
            catalog = catalog[catalog[column].eq(value)]
    return catalog[ASSET_COLUMN].tolist()


def _scoped(frame: pd.DataFrame, assets: list[str] | None = None,
            start: pd.Timestamp | None = None) -> pd.DataFrame:
    """``assets``(None = 전체 기계)·``start``(None = 전체 기간)로 행을 거른다."""
    if assets is not None:
        frame = frame[frame[ASSET_COLUMN].isin(assets)]
    if start is not None:
        frame = frame[frame[DATE_COLUMN] >= start]
    return frame


def _last_failure_date_by_asset(daily: pd.DataFrame) -> pd.Series:
    """자산별 failure_points > 0인 가장 최근 날짜 (전체 기간 내)."""
    failed_days = daily[daily["failure_points"] > 0]
    return failed_days.groupby(ASSET_COLUMN)[DATE_COLUMN].max()


def load_screen1_kpis(assets: list[str] | None = None, start: pd.Timestamp | None = None) -> dict:
    """화면 ① KPI 타일 값을 계산한다.

    ``assets``는 모든 값에, ``start``는 고장 표시 기계·일과 부품 출고 금액에만
    적용한다. 나머지는 데이터 최신일(기준일) 스냅샷이다.
    """
    raw = _scoped(_load_raw(), assets)
    daily = _scoped(_daily(), assets)

    observed_machines = int(raw[ASSET_COLUMN].nunique())
    failure_machine_days = int((_scoped(daily, start=start)["failure_points"] > 0).sum())

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
        "parts_issue_value_inr": float(_scoped(raw, start=start)["issue_value_inr"].sum()),
    }


def load_screen5_kpis() -> dict:
    """화면 ⑤ KPI 1~4 — load_screen1_kpis()가 이미 계산한 값을 그대로 재사용한다."""
    kpis = load_screen1_kpis()
    return {
        "failure_machine_days": kpis["failure_machine_days"],
        "failure_rate_pct": kpis["failure_rate_pct"],
        "parts_issue_value_inr": kpis["parts_issue_value_inr"],
        "avg_power_kw": kpis["avg_power_kw"],
    }


def _comparison_dir(task: str) -> Path:
    if task not in COMPARISON_TASKS:
        raise ValueError(f"알 수 없는 비교 과제입니다: {task}")
    return COMPARISON_TASKS[task]


def _threshold_mask(frame: pd.DataFrame, threshold: int | None) -> pd.Series:
    """기준(12/13/14)이 있는 과제는 그 값으로, 없는 과제는 빈 값 행으로 거른다."""
    if threshold is None:
        return frame["severity_threshold"].isna()
    return frame["severity_threshold"].eq(threshold)


@lru_cache(maxsize=None)
def _comparison_raw(task: str) -> pd.DataFrame:
    return pd.read_csv(_comparison_dir(task) / "comparison.csv")


@lru_cache(maxsize=None)
def _comparison_predictions(task: str) -> pd.DataFrame:
    return pd.read_csv(_comparison_dir(task) / "comparison_predictions.csv")


@lru_cache(maxsize=None)
def load_comparison_config(task: str) -> dict:
    """과제 설명·분할·판정 기준 정책 — comparison_config.json."""
    return json.loads((_comparison_dir(task) / "comparison_config.json").read_text(encoding="utf-8"))


def load_model_comparison(task: str, threshold: int | None = None) -> list[dict]:
    """화면 ③ 비교표 행 — 기준 A, 기준 B, LR, RF, HGB 순서(결과가 있는 모델만).

    Raises:
        ValueError: 과제·기준에 해당하는 결과가 없을 때.
    """
    frame = _comparison_raw(task)
    frame = frame[_threshold_mask(frame, threshold)]
    if frame.empty:
        raise ValueError(f"{task}: 기준 {threshold}의 비교 결과가 없습니다.")
    order = {model: i for i, model in enumerate(COMPARISON_MODEL_ORDER)}
    frame = frame.assign(_order=frame["model"].map(order)).sort_values("_order").drop(columns="_order")
    return frame.to_dict("records")


def load_focus_model(task: str, threshold: int | None = None) -> str:
    """KPI·PR 곡선·혼동행렬·임계값 카드가 보여 줄 모델 — 운영 모델(팀 결정), 없으면 검증 선택 모델."""
    rows = load_model_comparison(task, threshold)
    for flag in ("production", "selected_by_validation"):
        chosen = [row["model"] for row in rows if bool(row[flag])]
        if chosen:
            return chosen[0]
    raise ValueError(f"{task}: 운영·선택 모델이 없습니다.")


def _comparison_scores(task: str, model: str, threshold: int | None) -> tuple[np.ndarray, np.ndarray]:
    frame = _comparison_predictions(task)
    subset = frame[_threshold_mask(frame, threshold) & frame["model"].eq(model)]
    if subset.empty:
        raise ValueError(f"{task}: {model}의 Test 점수가 없습니다.")
    return subset["actual"].to_numpy(dtype=int), subset["score"].to_numpy(dtype=float)


def load_comparison_pr_curve(task: str, model: str, threshold: int | None = None) -> dict:
    """PR 곡선(최대 PR_CURVE_MAX_POINTS점으로 솎음)과 Test 양성 비율."""
    actual, score = _comparison_scores(task, model, threshold)
    precision, recall, _ = precision_recall_curve(actual, score)
    keep = np.unique(np.linspace(0, len(precision) - 1, min(len(precision), PR_CURVE_MAX_POINTS)).astype(int))
    return {"precision_curve": precision[keep].tolist(), "recall_curve": recall[keep].tolist(),
            "positive_rate": float(actual.mean())}


def load_comparison_at_cutoff(task: str, model: str, cutoff: float, threshold: int | None = None) -> dict:
    """판정 기준 ``cutoff``(점수 ≥ cutoff → 경보)를 Test 점수에 다시 적용한 혼동행렬·지표."""
    actual, score = _comparison_scores(task, model, threshold)
    predicted = score >= cutoff
    tp = int((predicted & (actual == 1)).sum())
    fp = int((predicted & (actual == 0)).sum())
    fn = int((~predicted & (actual == 1)).sum())
    tn = int((~predicted & (actual == 0)).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "rows": int(len(actual)), "alerts": tp + fp,
            "alert_rate": (tp + fp) / len(actual), "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0, "positive_rate": float(actual.mean()),
            "cutoff": float(cutoff)}


def load_comparison_feature_importance(task: str, threshold: int | None = None, top: int = 10) -> list[dict]:
    """운영·선택 모델의 검증 구간 permutation importance(scoring=AP) 상위 ``top``개.
    파일이 없으면 빈 목록."""
    path = _comparison_dir(task) / "feature_importance.csv"
    if not path.is_file():
        return []
    frame = pd.read_csv(path)
    frame = frame[_threshold_mask(frame, threshold)]
    return (frame.sort_values("importance_mean", ascending=False).head(top)
            [["feature", "importance_mean", "importance_std", "model"]].to_dict("records"))


@lru_cache(maxsize=1)
def load_failure_trend() -> pd.DataFrame:
    """화면 ⑤ "고장·위험 추세" — 일자별 고장 표시 건수 · 위험 기준선 초과 비율.

    ``_load_raw()``/``_daily()``와 같은 방식으로 캐싱한다 — 테마 전환 시마다
    차트를 다시 그릴 때 이 데이터를 재계산하지 않기 위함(색만 다시 계산).
    """
    daily = _daily()
    total_assets = int(_load_raw()[ASSET_COLUMN].nunique())

    failure_days = (daily["failure_points"] > 0).groupby(daily[DATE_COLUMN]).sum()
    high_risk_count = (daily["severity_level"] == "high_risk").groupby(daily[DATE_COLUMN]).sum()

    trend = pd.DataFrame({
        DATE_COLUMN: failure_days.index,
        "failure_days": failure_days.to_numpy().astype(int),
        "high_risk_pct": (
            high_risk_count.to_numpy().astype(float) / total_assets * 100
            if total_assets else 0.0
        ),
    })
    return trend.sort_values(DATE_COLUMN).reset_index(drop=True)


def load_priority_table(sort_by: str = "grade", direction: str = "desc",
                        assets: list[str] | None = None) -> list[dict]:
    """"점검 우선순위" 표의 행 데이터를 만든다.

    Args:
        sort_by: ``"grade"``(등급가중 고장점수 내림차순, 기본) 또는
            ``"threshold"``(기준선 초과 우선, 동률이면 고장점수 내림차순).
        direction: ``"desc"``(기본) 또는 ``"asc"`` — sort_by 기준으로 정렬한
            뒤 전체 순서를 뒤집는다. rank는 이 최종 순서 기준으로 매긴다.
        assets: 대상 기계 태그 목록. None이면 전체 기계.
    """
    raw = _scoped(_load_raw(), assets)
    daily = _scoped(_daily(), assets)

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
    if direction == "asc":
        rows.reverse()

    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def load_screen1_machine_status(assets: list[str] | None = None) -> list[dict]:
    """화면 ① "기계 상태" 타일 — 자산별 최신 등급·등급가중 고장점수와
    최근 30일 고장점수 스파크라인. asset_tag 오름차순(기계 드롭다운과 동일 순서).
    ``assets``가 None이면 전체 기계.
    """
    daily = _daily()
    latest_date = daily[DATE_COLUMN].max()

    rows = []
    for asset_tag in load_asset_list():
        if assets is not None and asset_tag not in assets:
            continue
        asset_daily = daily[daily[ASSET_COLUMN].eq(asset_tag)].sort_values(DATE_COLUMN)
        info = _load_raw()[_load_raw()[ASSET_COLUMN].eq(asset_tag)].iloc[0]
        latest_row = asset_daily[asset_daily[DATE_COLUMN].eq(latest_date)].iloc[0]
        rows.append({
            "asset_tag": asset_tag,
            "machine_type": info[MACHINE_COLUMN],
            "current_grade": SEVERITY_LABELS_KO.get(
                str(latest_row["severity_level"]), str(latest_row["severity_level"])
            ),
            "risk_score": float(latest_row["failure_points"]),
            "sparkline": asset_daily.tail(30)["failure_points"].astype(float).tolist(),
        })
    return rows


def load_screen1_power_by_machine(assets: list[str] | None = None,
                                  start: pd.Timestamp | None = None) -> list[dict]:
    """화면 ① "기계별 평균 소비 전력" — 기간(``start`` 이후, None이면 전체 기간)
    자산별 평균 소비전력(kW), 내림차순.
    """
    daily = _scoped(_daily(), assets, start)
    avg_power = daily.groupby(ASSET_COLUMN)["power_consumption_kw"].mean()
    avg_power = avg_power.sort_values(ascending=False)
    return [
        {"asset_tag": asset_tag, "avg_power_kw": float(value)}
        for asset_tag, value in avg_power.items()
    ]


def load_asset_failure_heatmap(assets: list[str] | None = None,
                               start: pd.Timestamp | None = None) -> pd.DataFrame:
    """화면 ① "고장 표시 히트맵" — 자산 × 월(YYYY-MM) 그레인, 셀 값은 그 달에
    고장 표시된 부품-일 행 수 합계(CURRENT_TARGET == 1인 원자료 행 수).

    ``assets``(None = 전체 기계)와 ``start``(None = 전체 기간)로 범위를 좁힌다.
    기간이 월 중간에서 시작하면 그 달 칸은 기간 안의 행만 센다.

    Returns:
        asset_tag, period("YYYY-MM"), failed_part_count 3열. 데이터가 없는
        자산×월 조합도 0으로 채워 히트맵 격자에 빈 칸이 생기지 않게 한다.
    """
    raw = _scoped(_load_raw(), start=start)
    periods = sorted(raw[DATE_COLUMN].dt.to_period("M").astype(str).unique())
    assets = [a for a in load_asset_list() if assets is None or a in assets]

    failed = raw.loc[raw[CURRENT_TARGET].eq(1)].copy()
    failed["period"] = failed[DATE_COLUMN].dt.to_period("M").astype(str)
    counts = failed.groupby([ASSET_COLUMN, "period"])[PART_COLUMN].count()

    full_index = pd.MultiIndex.from_product([assets, periods], names=[ASSET_COLUMN, "period"])
    counts = counts.reindex(full_index, fill_value=0)

    heat = counts.reset_index().rename(columns={ASSET_COLUMN: "asset_tag", PART_COLUMN: "failed_part_count"})
    heat["failed_part_count"] = heat["failed_part_count"].astype(int)
    return heat[["asset_tag", "period", "failed_part_count"]]


def load_month_coverage(start: pd.Timestamp | None = None) -> dict[str, tuple[int, int]]:
    """히트맵 월별 (관측 일수, 그 달의 일수). 관측 일수가 더 적으면 '부분 월'이다 —
    합계 건수가 작게 나와도 그 달이 덜 위험했다는 뜻이 아니다."""
    dates = pd.Series(_scoped(_load_raw(), start=start)[DATE_COLUMN].unique())
    months = dates.dt.to_period("M")
    observed = dates.groupby(months).size()
    return {str(month): (int(count), int(month.days_in_month)) for month, count in observed.items()}


def load_asset_list() -> list[str]:
    """전체 자산 태그를 알파벳 오름차순으로 반환한다."""
    raw = _load_raw()
    return sorted(raw[ASSET_COLUMN].unique().tolist())


def load_asset_detail_kpis(asset_tag: str, start: pd.Timestamp | None = None) -> dict:
    """화면 ② 상단 스트립의 값을 계산한다 (선택된 자산 1개 기준).

    고장 표시 일수·평균 소비 전력은 기간(``start`` 이후, None이면 전체 기간)
    기준이고, 등급·위험도·최근 고장 표시일은 기준일 스냅샷이다.

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

    period_daily = _scoped(asset_daily, start=start)

    return {
        "asset_tag": asset_tag,
        "machine_type": info[MACHINE_COLUMN],
        "plant_code": info[PLANT_COLUMN],
        "current_grade": SEVERITY_LABELS_KO.get(
            str(latest_row["severity_level"]), str(latest_row["severity_level"])
        ),
        "risk_score": float(latest_row["failure_points"]),
        "last_failure_date": last_failure_date_str,
        "failure_days_count": int((period_daily["failure_points"] > 0).sum()),
        "avg_power_kw": float(period_daily["power_consumption_kw"].mean()),
    }


def load_asset_sensor_series(asset_tag: str, start: pd.Timestamp | None = None) -> pd.DataFrame:
    """화면 ② "센서 8종 스몰 멀티플" — 선택 자산의 기간(``start`` 이후, None이면
    전체 기간) 센서 시계열.

    Returns:
        transaction_date, 센서 8종(SENSOR_COLUMNS 순서), is_failure_day
        (failure_points > 0), is_high_risk_day(위험 기준선 초과 = severity_level
        "high_risk") 열을 가진 DataFrame. 날짜 오름차순.

    Raises:
        ValueError: asset_tag가 데이터에 없을 때.
    """
    assets = load_asset_list()
    if asset_tag not in assets:
        raise ValueError(f"알 수 없는 asset_tag입니다: {asset_tag}")

    daily = _scoped(_daily(), [asset_tag], start)
    asset_daily = daily.sort_values(DATE_COLUMN)

    series = asset_daily[[DATE_COLUMN, *SENSOR_COLUMNS]].reset_index(drop=True)
    series["is_failure_day"] = (asset_daily["failure_points"] > 0).to_numpy()
    series["is_high_risk_day"] = (asset_daily["severity_level"] == "high_risk").to_numpy()
    return series


def load_asset_parts_history(asset_tag: str, start: pd.Timestamp | None = None) -> pd.DataFrame:
    """화면 ② "부품 출고 이력" — 선택 자산의 기간(``start`` 이후, None이면 전체
    기간) 부품별 출고 금액 합계 상위 10개.

    Returns:
        part_no, part_description, total_issue_value_inr 3열, 금액 내림차순
        상위 10행. 기간 안에 출고 금액이 0인 부품은 넣지 않는다.

    Raises:
        ValueError: asset_tag가 데이터에 없을 때.
    """
    assets = load_asset_list()
    if asset_tag not in assets:
        raise ValueError(f"알 수 없는 asset_tag입니다: {asset_tag}")

    asset_raw = _scoped(_load_raw(), [asset_tag], start)

    totals = (
        asset_raw.groupby([PART_COLUMN, "part_description"])["issue_value_inr"]
        .sum()
        .reset_index()
        .rename(columns={"issue_value_inr": "total_issue_value_inr"})
        .query("total_issue_value_inr > 0")
        .sort_values("total_issue_value_inr", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )
    return totals


def load_asset_peer_comparison(asset_tag: str) -> dict:
    """화면 ② "동종 기계 대비" — 선택 자산과 같은 machine_type인 나머지 1대의
    베어링 온도(temp_bearing_degC) 분포를 비교한다.

    기계 종류마다 자산이 정확히 2대뿐이라는 데이터 특성상 "동종 기계"는
    선택 자산을 제외한 나머지 1대로 계산해서 구한다(하드코딩 금지).

    Returns:
        asset_tag, peer_asset_tag, asset_values(선택 자산의 temp_bearing_degC
        1,095개), peer_values(동종 자산의 temp_bearing_degC 1,095개)를 담은 dict.

    Raises:
        ValueError: asset_tag가 데이터에 없거나, 동종(같은 machine_type) 자산이
            정확히 1대가 아닐 때(0대 또는 2대 이상 — 데이터 가정이 깨진 경우).
    """
    assets = load_asset_list()
    if asset_tag not in assets:
        raise ValueError(f"알 수 없는 asset_tag입니다: {asset_tag}")

    raw = _load_raw()
    machine_type = raw[raw[ASSET_COLUMN].eq(asset_tag)][MACHINE_COLUMN].iloc[0]
    same_type_assets = sorted(raw[raw[MACHINE_COLUMN].eq(machine_type)][ASSET_COLUMN].unique())
    peers = [a for a in same_type_assets if a != asset_tag]
    if len(peers) != 1:
        raise ValueError(
            f"'{asset_tag}'(machine_type={machine_type})의 동종 기계가 정확히 1대가 "
            f"아닙니다(현재 {len(peers)}대): {peers}"
        )
    peer_asset_tag = peers[0]

    daily = _daily()
    asset_values = daily[daily[ASSET_COLUMN].eq(asset_tag)].sort_values(DATE_COLUMN)[
        "temp_bearing_degC"
    ].tolist()
    peer_values = daily[daily[ASSET_COLUMN].eq(peer_asset_tag)].sort_values(DATE_COLUMN)[
        "temp_bearing_degC"
    ].tolist()

    return {
        "asset_tag": asset_tag,
        "peer_asset_tag": peer_asset_tag,
        "asset_values": asset_values,
        "peer_values": peer_values,
    }


def load_asset_failure_onset_trend(asset_tag: str) -> dict:
    """화면 ② "고장 직전 센서 변화" — 선택 자산의 가장 최근 위험 기준선
    시작 에피소드, t-7~t 구간의 베어링 온도.

    Returns:
        episode_start_date("YYYY-MM-DD"), relative_days([-7..0]),
        dates(8개 실제 날짜 문자열), temp_bearing_degC(8개 값)를 담은 dict.

    Raises:
        ValueError: asset_tag가 데이터에 없거나, 위험 기준선 시작 에피소드가
            하나도 없을 때.
    """
    assets = load_asset_list()
    if asset_tag not in assets:
        raise ValueError(f"알 수 없는 asset_tag입니다: {asset_tag}")

    daily = _daily()
    asset_daily = daily[daily[ASSET_COLUMN].eq(asset_tag)].sort_values(DATE_COLUMN).reset_index(drop=True)

    is_high_risk = (asset_daily["severity_level"] == "high_risk").to_numpy()
    is_start = is_high_risk & ~np.r_[False, is_high_risk[:-1]]
    start_indices = np.flatnonzero(is_start)
    if len(start_indices) == 0:
        raise ValueError(f"'{asset_tag}'에 위험 기준선 시작 에피소드가 없습니다.")

    onset_idx = int(start_indices[-1])
    window = asset_daily.iloc[max(0, onset_idx - 7):onset_idx + 1]

    return {
        "episode_start_date": asset_daily.iloc[onset_idx][DATE_COLUMN].strftime("%Y-%m-%d"),
        "relative_days": list(range(-(len(window) - 1), 1)),
        "dates": [d.strftime("%Y-%m-%d") for d in window[DATE_COLUMN]],
        "temp_bearing_degC": window["temp_bearing_degC"].tolist(),
    }


@lru_cache(maxsize=1)
def _family_metrics_raw() -> pd.DataFrame:
    path = DEFAULT_FAMILY_METRICS_PATH
    if not path.exists():
        raise FileNotFoundError(
            "부품군 진단 metrics.csv가 없습니다. 먼저 다음 명령을 실행하세요:\n"
            "python -m src.current_family_diagnosis "
            "--data dataVerification/synthetic_industrial_machine_data.csv"
        )
    return pd.read_csv(path)


def load_asset_family_diagnosis(asset_tag: str) -> list[dict]:
    """화면 ③ "부품군 진단" 과제 — 선택 자산의 부품군 9종별 이상탐지 성능.

    Raises:
        ValueError: asset_tag가 데이터에 없거나, 부품군 9행이 정확히
            나오지 않을 때.
    """
    assets = load_asset_list()
    if asset_tag not in assets:
        raise ValueError(f"알 수 없는 asset_tag입니다: {asset_tag}")

    metrics = _family_metrics_raw()
    selected = metrics.loc[
        metrics["scope_kind"].eq("asset_tag")
        & metrics["scope_name"].eq(asset_tag)
        & metrics["target"].eq("affected")
        & metrics["threshold_policy"].eq("f1")
        & metrics["split"].eq("test")
    ]
    if len(selected) != 9:
        raise ValueError(f"{asset_tag}의 부품군 진단 행이 9개가 아닙니다: {len(selected)}개")

    selected = selected.sort_values("average_precision", ascending=False)
    return [
        {
            "part_family": r["part_family"],
            "model": r["model"],
            "support": int(r["support"]),
            **{col: float(r[col]) for col in FAMILY_DIAGNOSIS_METRIC_COLUMNS if col != "support"},
        }
        for _, r in selected.iterrows()
    ]


def load_family_feature_importance(part_family: str) -> list[dict]:
    """화면 ③ "부품군 진단" 드릴다운 — 선택 부품군의 변수중요도 상위 10개
    (target=="affected" 고정, 자산과 무관한 모델 자체 속성).

    Raises:
        ValueError: part_family가 9종에 없거나, 해당 데이터가 없을 때.
    """
    if part_family not in FAMILY_NAMES:
        raise ValueError(f"알 수 없는 part_family입니다: {part_family}")

    path = DEFAULT_FAMILY_FEATURE_IMPORTANCE_PATH
    if not path.exists():
        raise FileNotFoundError(
            "부품군 변수중요도 feature_importance.csv가 없습니다. 먼저 다음 명령을 실행하세요:\n"
            "python -m src.current_family_diagnosis "
            "--data dataVerification/synthetic_industrial_machine_data.csv"
        )
    importance = pd.read_csv(path)
    selected = importance.loc[
        importance["part_family"].eq(part_family) & importance["target"].eq("affected")
    ]
    if selected.empty:
        raise ValueError(f"{part_family}의 변수중요도 데이터가 없습니다.")

    top10 = selected.sort_values("importance_mean", ascending=False).head(10)
    return [
        {"feature": r["feature"], "importance_mean": float(r["importance_mean"])}
        for _, r in top10.iterrows()
    ]


@lru_cache(maxsize=1)
def _family_test_predictions_raw() -> pd.DataFrame:
    path = DEFAULT_FAMILY_TEST_PREDICTIONS_PATH
    if not path.exists():
        raise FileNotFoundError(
            "부품군 진단 test_predictions.csv가 없습니다. 먼저 다음 명령을 실행하세요:\n"
            "python -m src.current_family_diagnosis "
            "--data dataVerification/synthetic_industrial_machine_data.csv"
        )
    return pd.read_csv(path)


def load_family_pr_curve_and_confusion(asset_tag: str, part_family: str) -> dict:
    """화면 ③ "부품군 진단" 드릴다운 — 선택 자산·부품군의 PR곡선 좌표와
    혼동행렬(임계값은 metrics.csv의 probability_cutoff를 그대로 써서
    load_asset_family_diagnosis()가 이미 보여준 precision/recall과 일치시킨다).

    Raises:
        ValueError: asset_tag/part_family가 유효하지 않거나, 필요한 행이
            정확히 하나가 아닐 때.
    """
    assets = load_asset_list()
    if asset_tag not in assets:
        raise ValueError(f"알 수 없는 asset_tag입니다: {asset_tag}")
    if part_family not in FAMILY_NAMES:
        raise ValueError(f"알 수 없는 part_family입니다: {part_family}")

    metrics = _family_metrics_raw()
    metrics_row = metrics.loc[
        metrics["scope_kind"].eq("asset_tag")
        & metrics["scope_name"].eq(asset_tag)
        & metrics["part_family"].eq(part_family)
        & metrics["target"].eq("affected")
        & metrics["threshold_policy"].eq("f1")
        & metrics["split"].eq("test")
    ]
    if len(metrics_row) != 1:
        raise ValueError(
            f"{asset_tag}/{part_family}의 metrics 행이 1개가 아닙니다: {len(metrics_row)}개"
        )
    cutoff = float(metrics_row.iloc[0]["probability_cutoff"])

    preds = _family_test_predictions_raw()
    sub = preds.loc[
        preds["asset_tag"].eq(asset_tag)
        & preds["part_family"].eq(part_family)
        & preds["target"].eq("affected")
        & preds["threshold_policy"].eq("f1")
    ]
    if sub.empty:
        raise ValueError(f"{asset_tag}/{part_family}의 test_predictions 행이 없습니다.")

    precision, recall, _ = precision_recall_curve(sub["actual"], sub["risk_score"])
    predicted = (sub["risk_score"] >= cutoff).astype(int)
    tn, fp, fn, tp = confusion_matrix(sub["actual"], predicted, labels=[0, 1]).ravel()

    return {
        "precision_curve": precision.tolist(),
        "recall_curve": recall.tolist(),
        "cutoff": cutoff,
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


@lru_cache(maxsize=1)
def _family_daily() -> pd.DataFrame:
    return build_family_daily(_load_raw())


def load_family_recurrence_intervals(asset_tag: str, part_family: str) -> list[int]:
    """화면 ③ "부품군 진단" 드릴다운 — 선택 자산·부품군의 affected 에피소드
    연속 시작일 사이 간격(일수). load_asset_failure_onset_trend()와 같은
    방식으로 에피소드 시작점을 탐지한다.

    Raises:
        ValueError: asset_tag/part_family가 유효하지 않을 때.
    """
    assets = load_asset_list()
    if asset_tag not in assets:
        raise ValueError(f"알 수 없는 asset_tag입니다: {asset_tag}")
    if part_family not in FAMILY_NAMES:
        raise ValueError(f"알 수 없는 part_family입니다: {part_family}")

    daily = _family_daily()
    sub = daily.loc[
        daily[ASSET_COLUMN].eq(asset_tag) & daily["part_family"].eq(part_family)
    ].sort_values(DATE_COLUMN)

    is_affected = sub["affected"].to_numpy().astype(bool)
    is_start = is_affected & ~np.r_[False, is_affected[:-1]]
    start_dates = sub[DATE_COLUMN].to_numpy()[is_start]
    if len(start_dates) < 2:
        return []
    return np.diff(start_dates).astype("timedelta64[D]").astype(int).tolist()


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


def load_data_start_date() -> str:
    """필터바 "적용 기간"(전체) 시작일 — 원본 데이터의 가장 이른 transaction_date."""
    return _load_raw()[DATE_COLUMN].min().strftime("%Y-%m-%d")


def load_data_reference_date() -> str:
    """앱 헤더 "데이터 기준일" — 원본 데이터의 최신 transaction_date."""
    raw = _load_raw()
    return raw[DATE_COLUMN].max().strftime("%Y-%m-%d")


# 화면 ④ "데이터 조회" 표 전용 — (컬럼id, 라벨, 정렬방향, 표시 포맷 종류).
# 포맷 종류: date/text/int/int_comma/float1/float2/wo_type/severity.
RAW_TABLE_COLUMNS = [
    ("transaction_date", "transaction_date", "left", "date"),
    ("asset_tag", "asset_tag", "left", "text"),
    ("machine_type", "machine_type", "left", "text"),
    ("plant_code", "plant_code", "left", "text"),
    ("part_no", "part_no", "left", "text"),
    ("part_description", "part_description", "left", "text"),
    ("part_family", "part_family", "left", "text"),
    ("criticality", "criticality", "left", "text"),
    ("uom", "uom", "left", "text"),
    ("unit_cost_inr", "unit_cost_inr", "right", "int_comma"),
    ("qty_issued", "qty_issued", "right", "int_comma"),
    ("issue_value_inr", "issue_value_inr", "right", "int_comma"),
    ("temp_bearing_degC", "temp_bearing_degC", "right", "float1"),
    ("temp_motor_degC", "temp_motor_degC", "right", "float1"),
    ("vibration_h_mms", "vibration_h_mms", "right", "float2"),
    ("vibration_v_mms", "vibration_v_mms", "right", "float2"),
    ("oil_pressure_bar", "oil_pressure_bar", "right", "float2"),
    ("load_pct", "load_pct", "right", "float1"),
    ("shaft_rpm", "shaft_rpm", "right", "int_comma"),
    ("power_consumption_kw", "power_consumption_kw", "right", "float2"),
    ("breakdown_flag", "breakdown_flag", "right", "int"),
    ("wo_type", "wo_type", "left", "wo_type"),
]

DAILY_TABLE_COLUMNS = [
    ("transaction_date", "transaction_date", "left", "date"),
    ("machine_type", "machine_type", "left", "text"),
    ("asset_tag", "asset_tag", "left", "text"),
    ("failure_points", "failure_points", "right", "int_comma"),
    ("severity_level", "severity_level", "left", "severity"),
    ("temp_bearing_degC", "temp_bearing_degC", "right", "float1"),
    ("temp_motor_degC", "temp_motor_degC", "right", "float1"),
    ("vibration_h_mms", "vibration_h_mms", "right", "float2"),
    ("vibration_v_mms", "vibration_v_mms", "right", "float2"),
    ("oil_pressure_bar", "oil_pressure_bar", "right", "float2"),
    ("load_pct", "load_pct", "right", "float1"),
    ("shaft_rpm", "shaft_rpm", "right", "int_comma"),
    ("power_consumption_kw", "power_consumption_kw", "right", "float2"),
]

# "등급" 열은 표시는 한글 라벨(severity_level)이지만 정렬은 severity_code
# (0~3) 기준이어야 심각도 순서가 유지된다 — 문자열 정렬이면 순서가 깨진다.
_SORT_FIELD_OVERRIDE = {"severity_level": "severity_code"}

_TABLE_SOURCES = {"raw": (_load_raw, RAW_TABLE_COLUMNS), "daily": (_daily, DAILY_TABLE_COLUMNS)}


def _format_table_value(kind: str, value) -> str:
    if kind == "date":
        return value.strftime("%Y-%m-%d")
    if kind == "int":
        return f"{int(value):,}"
    if kind == "int_comma":
        return f"{value:,.0f}"
    if kind == "float1":
        return f"{value:.1f}"
    if kind == "float2":
        return f"{value:.2f}"
    if kind == "wo_type":
        return "작업 없음" if pd.isna(value) else str(value)
    if kind == "severity":
        return SEVERITY_LABELS_KO.get(str(value), str(value))
    return str(value)


def load_table_page(
    dataset: str, sort_col: str | None, sort_dir: str, page: int, page_size: int = 16,
    assets: list[str] | None = None, start: pd.Timestamp | None = None,
) -> dict:
    """화면 ④ "데이터 조회" 표의 한 페이지를 계산한다.

    Args:
        dataset: ``"raw"``(원자료) 또는 ``"daily"``(기계·일 집계).
        sort_col: 정렬 기준 컬럼id, ``None``이면 정렬하지 않는다.
        sort_dir: ``"asc"`` 또는 ``"desc"``.
        page: 0부터 시작하는 페이지 번호(범위를 벗어나면 클램프한다).
        assets, start: 필터바 대상 기계·기간 시작일. None이면 조건 없음.

    Returns:
        ``columns``(id/label/align), ``data``(포맷된 문자열 딕셔너리 리스트),
        ``total_rows``, ``page_count``, ``page``를 담은 dict.

    Raises:
        ValueError: 알 수 없는 dataset일 때.
    """
    if dataset not in _TABLE_SOURCES:
        raise ValueError(f"알 수 없는 dataset입니다: {dataset}")
    loader, columns = _TABLE_SOURCES[dataset]
    frame = _scoped(loader(), assets, start)

    if sort_col:
        sort_field = _SORT_FIELD_OVERRIDE.get(sort_col, sort_col)
        frame = frame.sort_values(sort_field, ascending=(sort_dir != "desc"), kind="stable")

    total_rows = len(frame)
    page_size = max(page_size, 1)
    page_count = max(-(-total_rows // page_size), 1)
    page = min(max(page, 0), page_count - 1)

    start = page * page_size
    page_frame = frame.iloc[start : start + page_size]

    kinds = {col_id: kind for col_id, _, _, kind in columns}
    data = [
        {col_id: _format_table_value(kinds[col_id], record[col_id]) for col_id, _, _, _ in columns}
        for record in page_frame.to_dict("records")
    ]

    return {
        "columns": [{"id": c, "label": l, "align": a} for c, l, a, _ in columns],
        "data": data,
        "total_rows": total_rows,
        "page_count": page_count,
        "page": page,
    }


def export_table_csv(dataset: str, assets: list[str] | None = None,
                     start: pd.Timestamp | None = None) -> tuple[bytes, str]:
    """화면 ④ CSV 내보내기 — 선택한 데이터셋에서 필터(``assets``·``start``)에
    맞는 행 전체(정렬·페이지 무관)를 만든다.

    숫자는 반올림하지 않은 원래 값, 날짜는 YYYY-MM-DD, wo_type 빈 값은
    "작업 없음", 등급은 한글 라벨로 내보낸다. UTF-8 BOM(utf-8-sig)을 쓴다.

    Raises:
        ValueError: 알 수 없는 dataset일 때.
    """
    if dataset not in _TABLE_SOURCES:
        raise ValueError(f"알 수 없는 dataset입니다: {dataset}")
    loader, columns = _TABLE_SOURCES[dataset]
    frame = _scoped(loader(), assets, start)

    export_cols = [col_id for col_id, _, _, _ in columns]
    export_frame = frame[export_cols].copy()
    for col_id, _, _, kind in columns:
        if kind == "date":
            export_frame[col_id] = export_frame[col_id].dt.strftime("%Y-%m-%d")
        elif kind == "wo_type":
            export_frame[col_id] = export_frame[col_id].fillna("작업 없음")
        elif kind == "severity":
            export_frame[col_id] = export_frame[col_id].astype(str).map(
                lambda v: SEVERITY_LABELS_KO.get(v, v)
            )
        # 그 외 숫자 컬럼은 반올림하지 않은 원래 값 그대로 내보낸다.

    csv_bytes = export_frame.to_csv(index=False).encode("utf-8-sig")
    filename = "machine_raw.csv" if dataset == "raw" else "machine_day_aggregate.csv"
    return csv_bytes, filename
