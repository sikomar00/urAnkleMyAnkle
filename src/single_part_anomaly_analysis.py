"""완전 정상 장비일과 특정 부품 단독 고장 장비일의 센서 이상을 비교한다."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import pandas as pd
import sklearn

from .asset_anomaly_features import RobustNormalBaseline
from .family_features import build_family_daily
from .family_residual_features import (
    HEALTH_SENSOR_COLUMNS as RESIDUAL_HEALTH_SENSORS,
    OperatingResidualTransformer,
)
from .industrial_data import (
    ASSET_COLUMN,
    CURRENT_TARGET,
    DATE_COLUMN,
    MACHINE_COLUMN,
    PART_COLUMN,
    SENSOR_COLUMNS,
)

PART_FAMILY_COLUMN = "part_family"
HEALTH_SENSOR_COLUMNS = tuple(SENSOR_COLUMNS[:5])
OPERATING_SENSOR_COLUMNS = tuple(SENSOR_COLUMNS[5:])
_DAY_KEYS = [DATE_COLUMN, MACHINE_COLUMN, ASSET_COLUMN]


def build_single_part_daily(
    raw: pd.DataFrame,
    *,
    expected_parts_per_asset: int | None = None,
) -> pd.DataFrame:
    """부품 행을 완전 정상·단독 고장·다중 고장 장비일로 집계한다."""
    required = {
        *_DAY_KEYS,
        PART_COLUMN,
        PART_FAMILY_COLUMN,
        CURRENT_TARGET,
        *SENSOR_COLUMNS,
    }
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"단독 고장 비교에 필요한 컬럼이 없습니다: {missing}")

    frame = raw.copy()
    frame[DATE_COLUMN] = pd.to_datetime(frame[DATE_COLUMN]).dt.normalize()
    frame[CURRENT_TARGET] = pd.to_numeric(frame[CURRENT_TARGET], errors="coerce")
    if frame[CURRENT_TARGET].isna().any() or not frame[CURRENT_TARGET].isin([0, 1]).all():
        raise ValueError("breakdown_flag는 0 또는 1이어야 합니다.")
    if frame.duplicated([DATE_COLUMN, ASSET_COLUMN, PART_COLUMN]).any():
        raise ValueError("같은 날짜·장비·부품 행이 중복되었습니다.")

    signatures = frame.groupby(
        [MACHINE_COLUMN, ASSET_COLUMN, DATE_COLUMN], observed=True
    ).apply(
        lambda group: frozenset(
            group[[PART_COLUMN, PART_FAMILY_COLUMN]].itertuples(
                index=False, name=None
            )
        ),
        include_groups=False,
    )
    signature_counts = signatures.groupby(
        level=[MACHINE_COLUMN, ASSET_COLUMN]
    ).nunique()
    if signature_counts.gt(1).any():
        raise ValueError("장비의 부품 구성이 날짜에 따라 다릅니다.")
    if expected_parts_per_asset is not None:
        if expected_parts_per_asset < 1:
            raise ValueError("장비별 기대 부품 수는 1 이상이어야 합니다.")
        daily_part_counts = frame.groupby(_DAY_KEYS, observed=True)[PART_COLUMN].nunique()
        if not daily_part_counts.eq(expected_parts_per_asset).all():
            raise ValueError(
                f"장비별 기대 부품 수 {expected_parts_per_asset}개와 "
                "일치하지 않는 장비일이 있습니다."
            )

    value_columns = [*SENSOR_COLUMNS]
    counts = frame.groupby(_DAY_KEYS, observed=True)[value_columns].nunique(
        dropna=False
    )
    if counts.gt(1).any(axis=None):
        raise ValueError("같은 장비·날짜 안에서 센서값이 서로 다릅니다.")

    rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(_DAY_KEYS, observed=True, sort=False):
        failed = group.loc[group[CURRENT_TARGET].eq(1)]
        failed_count = int(len(failed))
        if failed_count == 0:
            status = "clean"
        elif failed_count == 1:
            status = "isolated"
        else:
            status = "multiple"
        first = group.iloc[0]
        row: dict[str, Any] = dict(zip(_DAY_KEYS, keys, strict=True))
        row.update(
            {
                "total_parts": int(group[PART_COLUMN].nunique()),
                "failed_parts": failed_count,
                "failure_status": status,
                "failed_part_no": (
                    failed.iloc[0][PART_COLUMN] if failed_count == 1 else pd.NA
                ),
                "failed_part_family": (
                    failed.iloc[0][PART_FAMILY_COLUMN]
                    if failed_count == 1
                    else pd.NA
                ),
            }
        )
        row.update({sensor: first[sensor] for sensor in SENSOR_COLUMNS})
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        [DATE_COLUMN, ASSET_COLUMN]
    ).reset_index(drop=True)


def add_direct_robust_z(
    daily: pd.DataFrame,
    *,
    validation_start: str | pd.Timestamp = "2024-01-01",
    min_normal_rows: int = 30,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """학습 기간 완전 정상일만으로 직접 센서 Robust Z-score를 계산한다."""
    frame = daily.copy()
    frame[DATE_COLUMN] = pd.to_datetime(frame[DATE_COLUMN])
    frame["failure_points"] = frame["failed_parts"]
    train = frame.loc[frame[DATE_COLUMN].lt(pd.Timestamp(validation_start))]
    baseline = RobustNormalBaseline(min_normal_rows=min_normal_rows).fit(train)
    transformed = baseline.transform(frame)
    return transformed, baseline.baseline_table()


def add_operating_residual_z(
    raw: pd.DataFrame,
    daily: pd.DataFrame,
    *,
    validation_start: str | pd.Timestamp = "2024-01-01",
    min_normal_rows: int = 30,
    max_iter: int = 100,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """운전조건 기대값을 제거한 건강 센서 잔차 Robust Z-score를 추가한다."""
    family_daily = build_family_daily(raw)
    train = family_daily.loc[
        family_daily[DATE_COLUMN].lt(pd.Timestamp(validation_start))
    ]
    transformer = OperatingResidualTransformer(
        min_normal_rows=min_normal_rows,
        max_iter=max_iter,
        random_state=random_state,
    ).fit(train)
    transformed = transformer.transform(family_daily)
    residual_columns = [
        f"{sensor}_residual_robust_z" for sensor in RESIDUAL_HEALTH_SENSORS
    ]
    residual_daily = transformed[[*_DAY_KEYS, *residual_columns]].drop_duplicates(
        _DAY_KEYS
    )
    if residual_daily.duplicated([DATE_COLUMN, ASSET_COLUMN]).any():
        raise ValueError("같은 장비·날짜에 서로 다른 잔차 Z-score가 있습니다.")
    merged = daily.merge(
        residual_daily,
        on=_DAY_KEYS,
        how="left",
        validate="one_to_one",
    )
    return merged, transformer.artifact_table()


def _cliffs_delta(case_values: np.ndarray, control_values: np.ndarray) -> float:
    """단독 고장과 정상의 절대 Z-score에 대한 Cliff's delta를 계산한다."""
    if not len(case_values) or not len(control_values):
        return float("nan")
    comparisons = case_values[:, None] - control_values[None, :]
    return float((np.sum(comparisons > 0) - np.sum(comparisons < 0)) / comparisons.size)


def _stratified_statistics(
    asset_values: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    threshold: float,
    samples: int,
    random_state: int,
) -> dict[str, float | int]:
    """장비 안에서 비교한 효과를 장비별 동일 가중치로 집계한다."""
    usable = [
        (clean, isolated)
        for clean, isolated in asset_values
        if len(clean) and len(isolated)
    ]
    if not usable:
        return {
            "case_assets": 0,
            "clean_rows": 0,
            "isolated_rows": 0,
            "clean_anomaly_rate": float("nan"),
            "isolated_anomaly_rate": float("nan"),
            "anomaly_rate_diff_pp": float("nan"),
            "ci_low_pp": float("nan"),
            "ci_high_pp": float("nan"),
            "median_abs_z_clean": float("nan"),
            "median_abs_z_isolated": float("nan"),
            "cliffs_delta": float("nan"),
        }

    clean_rates = np.array([np.mean(clean >= threshold) for clean, _ in usable])
    isolated_rates = np.array(
        [np.mean(isolated >= threshold) for _, isolated in usable]
    )
    clean_rate = float(clean_rates.mean())
    isolated_rate = float(isolated_rates.mean())
    ci_low = ci_high = float("nan")
    if samples >= 1:
        generator = np.random.default_rng(random_state)
        per_asset_bootstrap = []
        for clean, isolated in usable:
            clean_flags = (clean >= threshold).astype(float)
            isolated_flags = (isolated >= threshold).astype(float)
            clean_draws = generator.choice(
                clean_flags, size=(samples, len(clean_flags)), replace=True
            ).mean(axis=1)
            isolated_draws = generator.choice(
                isolated_flags,
                size=(samples, len(isolated_flags)),
                replace=True,
            ).mean(axis=1)
            per_asset_bootstrap.append(isolated_draws - clean_draws)
        bootstrap_matrix = np.vstack(per_asset_bootstrap)
        asset_indices = generator.integers(
            0, len(usable), size=(samples, len(usable))
        )
        sample_indices = np.arange(samples)[:, None]
        differences = bootstrap_matrix[asset_indices, sample_indices].mean(axis=1)
        ci_low, ci_high = np.quantile(differences * 100.0, [0.025, 0.975])

    return {
        "case_assets": len(usable),
        "clean_rows": sum(len(clean) for clean, _ in usable),
        "isolated_rows": sum(len(isolated) for _, isolated in usable),
        "clean_anomaly_rate": clean_rate,
        "isolated_anomaly_rate": isolated_rate,
        "anomaly_rate_diff_pp": (isolated_rate - clean_rate) * 100.0,
        "ci_low_pp": float(ci_low),
        "ci_high_pp": float(ci_high),
        "median_abs_z_clean": float(
            np.mean([np.median(clean) for clean, _ in usable])
        ),
        "median_abs_z_isolated": float(
            np.mean([np.median(isolated) for _, isolated in usable])
        ),
        "cliffs_delta": float(
            np.mean([_cliffs_delta(isolated, clean) for clean, isolated in usable])
        ),
    }


def _candidate_status(
    *, isolated_rows: int, difference_pp: float, ci_low_pp: float, ci_high_pp: float
) -> str:
    if isolated_rows < 10:
        return "표본 부족"
    if difference_pp <= -5:
        return "역방향"
    if ci_low_pp > 0 and difference_pp >= 15:
        return "강한 후보"
    if difference_pp >= 5:
        return "약한 후보"
    return "관계 미약"


def summarize_part_anomalies(
    daily: pd.DataFrame,
    *,
    period: str,
    analysis_method: str,
    z_columns: Mapping[str, str],
    thresholds: Sequence[float] = (2.0, 3.0),
    bootstrap_samples: int = 1_000,
    random_state: int = 42,
) -> pd.DataFrame:
    """부품·센서별로 동일 장비 정상일과 단독 고장일의 이상치율을 비교한다."""
    required = {
        ASSET_COLUMN,
        "failure_status",
        "failed_part_no",
        "failed_part_family",
        *z_columns.values(),
    }
    missing = sorted(required - set(daily.columns))
    if missing:
        raise ValueError(f"이상치 요약에 필요한 컬럼이 없습니다: {missing}")

    isolated = daily.loc[daily["failure_status"].eq("isolated")].copy()
    rows: list[dict[str, Any]] = []
    for part_no, cases in isolated.groupby("failed_part_no", observed=True):
        family = cases["failed_part_family"].iloc[0]
        for sensor, z_column in z_columns.items():
            asset_values: list[tuple[np.ndarray, np.ndarray]] = []
            for asset_tag, asset_cases in cases.groupby(
                ASSET_COLUMN, observed=True
            ):
                asset_controls = daily.loc[
                    daily["failure_status"].eq("clean")
                    & daily[ASSET_COLUMN].eq(asset_tag)
                ]
                clean_abs = (
                    pd.to_numeric(asset_controls[z_column], errors="coerce")
                    .abs()
                    .dropna()
                    .to_numpy()
                )
                case_abs = (
                    pd.to_numeric(asset_cases[z_column], errors="coerce")
                    .abs()
                    .dropna()
                    .to_numpy()
                )
                if len(clean_abs) and len(case_abs):
                    asset_values.append((clean_abs, case_abs))
            for threshold in thresholds:
                seed = random_state + len(rows)
                statistics = _stratified_statistics(
                    asset_values,
                    threshold=threshold,
                    samples=bootstrap_samples,
                    random_state=seed,
                )
                difference_pp = float(statistics["anomaly_rate_diff_pp"])
                ci_low = float(statistics["ci_low_pp"])
                ci_high = float(statistics["ci_high_pp"])
                rows.append(
                    {
                        "period": period,
                        "analysis_method": analysis_method,
                        "part_no": part_no,
                        "part_family": family,
                        "sensor": sensor,
                        "sensor_group": (
                            "health"
                            if analysis_method == "operating_adjusted_residual_z"
                            or sensor in HEALTH_SENSOR_COLUMNS
                            else "operating_context"
                        ),
                        "z_threshold": float(threshold),
                        **statistics,
                        "candidate_status": _candidate_status(
                            isolated_rows=int(statistics["isolated_rows"]),
                            difference_pp=difference_pp,
                            ci_low_pp=ci_low,
                            ci_high_pp=ci_high,
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _period_frames(
    daily: pd.DataFrame,
    *,
    validation_start: str | pd.Timestamp,
    test_start: str | pd.Timestamp,
) -> dict[str, pd.DataFrame]:
    validation_start = pd.Timestamp(validation_start)
    test_start = pd.Timestamp(test_start)
    if validation_start >= test_start:
        raise ValueError("validation_start는 test_start보다 빨라야 합니다.")
    dates = pd.to_datetime(daily[DATE_COLUMN])
    return {
        "overall": daily,
        "train": daily.loc[dates.lt(validation_start)],
        "valid": daily.loc[dates.ge(validation_start) & dates.lt(test_start)],
        "test": daily.loc[dates.ge(test_start)],
    }


def _part_profile(periods: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    parts = sorted(
        {
            str(value)
            for frame in periods.values()
            for value in frame["failed_part_no"].dropna().unique()
        }
    )
    rows: list[dict[str, Any]] = []
    for period, frame in periods.items():
        family_lookup = (
            frame.dropna(subset=["failed_part_no"])
            .drop_duplicates("failed_part_no")
            .set_index("failed_part_no")["failed_part_family"]
            .to_dict()
        )
        for part_no in parts:
            cases = frame.loc[
                frame["failure_status"].eq("isolated")
                & frame["failed_part_no"].astype("string").eq(part_no)
            ]
            rows.append(
                {
                    "period": period,
                    "part_no": part_no,
                    "part_family": family_lookup.get(part_no, pd.NA),
                    "isolated_rows": len(cases),
                    "case_assets": cases[ASSET_COLUMN].nunique(),
                    "all_clean_rows": int(frame["failure_status"].eq("clean").sum()),
                    "multiple_rows_excluded": int(
                        frame["failure_status"].eq("multiple").sum()
                    ),
                }
            )
    return pd.DataFrame(rows)


def _add_direction_status(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        result = results.copy()
        result["direction_status"] = pd.Series(dtype="object")
        return result
    keys = ["analysis_method", "part_no", "sensor", "z_threshold"]
    statuses: list[dict[str, Any]] = []
    temporal = results.loc[results["period"].isin(["train", "valid", "test"])]
    for values, group in temporal.groupby(keys, observed=True, dropna=False):
        differences = group.loc[
            group["isolated_rows"].gt(0), "anomaly_rate_diff_pp"
        ].dropna()
        signs = set(np.sign(differences.loc[differences.ne(0)]))
        if len(differences) < 2:
            status = "기간 표본 부족"
        elif len(signs) <= 1:
            status = "방향 일치"
        else:
            status = "방향 불일치"
        statuses.append(dict(zip(keys, values, strict=True), direction_status=status))
    status_frame = pd.DataFrame(statuses)
    if status_frame.empty:
        result = results.copy()
        result["direction_status"] = "기간 표본 부족"
        return result
    classified = results.merge(status_frame, on=keys, how="left").assign(
        direction_status=lambda frame: frame["direction_status"].fillna("기간 표본 부족")
    )
    mismatch = classified["direction_status"].eq("방향 불일치") & ~classified[
        "candidate_status"
    ].eq("표본 부족")
    classified.loc[mismatch, "candidate_status"] = "방향 불일치"
    return classified


def run_single_part_analysis(
    raw: pd.DataFrame,
    *,
    output_dir: str | Path,
    validation_start: str | pd.Timestamp = "2024-01-01",
    test_start: str | pd.Timestamp = "2024-07-01",
    min_normal_rows: int = 30,
    bootstrap_samples: int = 1_000,
    random_state: int = 42,
    max_iter: int = 100,
    include_residual: bool = True,
    data_path: str | Path | None = None,
    expected_parts_per_asset: int = 20,
) -> dict[str, pd.DataFrame]:
    """직접·운전조건 보정 이상치 비교를 실행하고 재현 가능한 결과를 저장한다."""
    reproducibility = _reproducibility_metadata(data_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    daily = build_single_part_daily(
        raw, expected_parts_per_asset=expected_parts_per_asset
    )
    if not daily["failure_status"].eq("isolated").any():
        raise ValueError("비교할 특정 부품 단독 고장 사례가 없습니다.")
    direct_daily, direct_baselines = add_direct_robust_z(
        daily,
        validation_start=validation_start,
        min_normal_rows=min_normal_rows,
    )
    periods = _period_frames(
        direct_daily,
        validation_start=validation_start,
        test_start=test_start,
    )

    result_frames: list[pd.DataFrame] = []
    direct_columns = {
        sensor: f"{sensor}_robust_z" for sensor in SENSOR_COLUMNS
    }
    for period, frame in periods.items():
        summary = summarize_part_anomalies(
            frame,
            period=period,
            analysis_method="direct_robust_z",
            z_columns=direct_columns,
            bootstrap_samples=bootstrap_samples,
            random_state=random_state,
        )
        if not summary.empty:
            result_frames.append(summary)

    residual_baselines = pd.DataFrame()
    if include_residual:
        residual_daily, residual_baselines = add_operating_residual_z(
            raw,
            direct_daily,
            validation_start=validation_start,
            min_normal_rows=min_normal_rows,
            max_iter=max_iter,
            random_state=random_state,
        )
        residual_periods = _period_frames(
            residual_daily,
            validation_start=validation_start,
            test_start=test_start,
        )
        residual_columns = {
            sensor: f"{sensor}_residual_robust_z"
            for sensor in HEALTH_SENSOR_COLUMNS
        }
        for period, frame in residual_periods.items():
            summary = summarize_part_anomalies(
                frame,
                period=period,
                analysis_method="operating_adjusted_residual_z",
                z_columns=residual_columns,
                bootstrap_samples=bootstrap_samples,
                random_state=random_state,
            )
            if not summary.empty:
                result_frames.append(summary)
        daily_to_save = residual_daily
    else:
        daily_to_save = direct_daily

    sensor_results = _add_direction_status(
        pd.concat(result_frames, ignore_index=True) if result_frames else pd.DataFrame()
    )
    profile = _part_profile(periods)
    summary_text = render_single_part_summary(
        sensor_results,
        expected_parts_per_asset=expected_parts_per_asset,
    )

    daily_to_save.to_csv(output_path / "daily_groups.csv", index=False)
    profile.to_csv(output_path / "part_profile.csv", index=False)
    sensor_results.to_csv(output_path / "sensor_anomaly_results.csv", index=False)
    direct_baselines.to_csv(output_path / "direct_baselines.csv", index=False)
    residual_path = output_path / "residual_baselines.csv"
    if include_residual:
        residual_baselines.to_csv(residual_path, index=False)
    elif residual_path.exists():
        residual_path.unlink()
    (output_path / "experiment_summary.md").write_text(summary_text, encoding="utf-8")
    config = {
        "validation_start": str(pd.Timestamp(validation_start).date()),
        "test_start": str(pd.Timestamp(test_start).date()),
        "min_normal_rows": min_normal_rows,
        "bootstrap_samples": bootstrap_samples,
        "random_state": random_state,
        "max_iter": max_iter,
        "include_residual": include_residual,
        "z_thresholds": [2.0, 3.0],
        "expected_parts_per_asset": expected_parts_per_asset,
        **reproducibility,
    }
    (output_path / "run_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "daily_groups": daily_to_save,
        "part_profile": profile,
        "sensor_results": sensor_results,
        "direct_baselines": direct_baselines,
        "residual_baselines": residual_baselines,
    }


def _reproducibility_metadata(data_path: str | Path | None) -> dict[str, Any]:
    """입력 파일·Git 상태·핵심 의존성 버전을 결과 설정에 기록한다."""
    resolved_path: Path | None = None
    input_sha256: str | None = None
    if data_path is not None:
        resolved_path = Path(data_path).expanduser().resolve()
        if not resolved_path.is_file():
            raise ValueError(f"입력 데이터 파일을 찾을 수 없습니다: {resolved_path}")
        digest = hashlib.sha256()
        with resolved_path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        input_sha256 = digest.hexdigest()

    project_root = Path(__file__).resolve().parents[1]

    def git_output(*arguments: str) -> str | None:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=False,
        )
        return completed.stdout.strip() if completed.returncode == 0 else None

    git_head = git_output("rev-parse", "HEAD")
    git_status = git_output("status", "--porcelain")
    source_paths = [
        project_root / "src" / "single_part_anomaly_analysis.py",
        project_root / "src" / "current_single_part_anomaly.py",
    ]
    source_hashes: dict[str, str] = {}
    combined_source = hashlib.sha256()
    for source_path in source_paths:
        relative = source_path.relative_to(project_root).as_posix()
        content = source_path.read_bytes()
        source_hashes[relative] = hashlib.sha256(content).hexdigest()
        combined_source.update(relative.encode("utf-8"))
        combined_source.update(b"\0")
        combined_source.update(content)
        combined_source.update(b"\0")
    return {
        "input_path": str(resolved_path) if resolved_path is not None else None,
        "input_sha256": input_sha256,
        "git_head": git_head,
        "working_tree_dirty": bool(git_status) if git_status is not None else None,
        "analysis_source_sha256": combined_source.hexdigest(),
        "analysis_source_files": source_hashes,
        "python_version": sys.version,
        "dependency_versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__,
        },
    }


def render_single_part_summary(
    results: pd.DataFrame,
    *,
    expected_parts_per_asset: int = 20,
) -> str:
    """부품별 이상치 비교 결과를 한국어 Markdown으로 요약한다."""
    test_results = results.loc[results["period"].eq("test")].copy()
    if "sensor_group" not in test_results:
        test_results["sensor_group"] = np.where(
            test_results["sensor"].isin(HEALTH_SENSOR_COLUMNS),
            "health",
            "operating_context",
        )

    def table_for(selected: pd.DataFrame) -> str:
        ranked = selected.sort_values(
            ["anomaly_rate_diff_pp", "isolated_rows"], ascending=[False, False]
        )
        lines = []
        for row in ranked.head(10).itertuples(index=False):
            lines.append(
                f"| {row.part_no} | {row.part_family} | {row.sensor} | "
                f"{row.isolated_rows} | {row.clean_anomaly_rate:.2%} | "
                f"{row.isolated_anomaly_rate:.2%} | {row.anomaly_rate_diff_pp:.2f}%p | "
                f"[{row.ci_low_pp:.2f}, {row.ci_high_pp:.2f}] | "
                f"{row.cliffs_delta:.3f} | {row.direction_status} | "
                f"{row.candidate_status} |"
            )
        return "\n".join(lines) if lines else "| 결과 없음 | - | - | 0 | - | - | - | - | - | - | 표본 부족 |"

    def threshold_sections(selected: pd.DataFrame) -> str:
        sections = []
        for threshold in (2.0, 3.0):
            sections.append(
                f"### |Z| >= {threshold:g}\n\n"
                f"{table_header}\n"
                f"{table_for(selected.loc[selected['z_threshold'].eq(threshold)])}"
            )
        return "\n\n".join(sections)

    def strongest_line(selected: pd.DataFrame, label: str) -> str:
        candidates = selected.loc[
            selected["z_threshold"].eq(2.0)
            & selected["anomaly_rate_diff_pp"].gt(0)
            & ~selected["direction_status"].eq("방향 불일치")
        ]
        if candidates.empty:
            return f"{label}에서는 해석 가능한 양의 후보가 없었습니다."
        row = candidates.sort_values("anomaly_rate_diff_pp", ascending=False).iloc[0]
        return (
            f"{label}의 최대 양의 차이는 {row['part_no']} / {row['sensor']}로 "
            f"{row['anomaly_rate_diff_pp']:.2f}%p였고 판정은 "
            f"{row['candidate_status']}입니다."
        )

    direct = test_results["analysis_method"].eq("direct_robust_z")
    health = test_results.loc[direct & test_results["sensor_group"].eq("health")]
    operating = test_results.loc[
        direct & test_results["sensor_group"].eq("operating_context")
    ]
    residual = test_results.loc[
        test_results["analysis_method"].eq("operating_adjusted_residual_z")
    ]
    health_line = strongest_line(health, "건강 센서 직접 비교")
    operating_line = strongest_line(operating, "운전 조건 비교")
    residual_line = strongest_line(residual, "운전조건 보정 잔차")
    table_header = """| 부품 | Family | 센서 | 단독 고장 수 | 정상 이상치율 | 단독 고장 이상치율 | 차이 | 95% 구간(%p) | Cliff's delta | 기간 방향 | 판정 |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---|"""
    health_sections = threshold_sections(health)
    operating_sections = threshold_sections(operating)
    residual_sections = threshold_sections(residual)
    return f"""# 완전 정상과 특정 부품 단독 고장 이상치 비교

## 세 줄 요약

1. {health_line}
2. {operating_line}
3. {residual_line} 이 분석은 탐색이며 고장 원인을 확정하지 않습니다.

## 테스트 구간 건강 센서 직접 이상치

{health_sections}

## 테스트 구간 운전 조건 이상치

{operating_sections}

`load_pct`, `shaft_rpm`, `power_consumption_kw`는 고장 원인이 아니라 고장 발생 후
달라진 운전상태일 가능성도 있으므로 건강 센서와 분리해 해석합니다.

## 테스트 구간 운전조건 보정 잔차

{residual_sections}

## 한 페이지 요약

이번 분석은 부품 {expected_parts_per_asset}개가 모두 정상인 장비일과 정확히 특정 부품 하나만 고장 난
장비일을 비교합니다. 두 개 이상 고장 난 장비일은 제외하며, 정상 비교군도 해당
부품의 단독 고장이 관측된 동일 장비의 정상일로 제한합니다. 따라서 서로 다른 장비의
기본 센서 수준 차이가 결과를 지배하는 문제를 줄였습니다.

표본이 적은 부품은 `표본 부족`으로 표시하되 결과를 숨기지 않습니다. `|Z| >= 2`와
`|Z| >= 3`을 분리하고, 이상치율 차이·장비 층화 부트스트랩 95% 신뢰구간·Cliff's
delta·기간 방향을 함께 확인해야 합니다. 높은 비율 하나만으로
특정 센서가 고장을 일으켰다고 결론 내릴 수 없습니다. 당일 센서와 당일 고장의
연관성을 탐색한 결과이며, 시간적 선후관계를 확인하기 전까지 고장 원인을 확정하지 않습니다.
"""
