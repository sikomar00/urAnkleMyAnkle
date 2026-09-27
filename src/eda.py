"""네 산업 위험 모델의 저장된 평가 결과를 읽고 비교표를 만드는 모듈이다.

이 파일은 모델을 다시 학습하지 않는다. 먼저 네 학습 실행 파일로 ``metrics.csv``를
만든 뒤 실행하면, 선택된 모델의 테스트 결과만 한 표로 저장한다.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "src"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METRICS = (
    PROJECT_ROOT / "outputs" / "asset_current" / "metrics.csv",
    PROJECT_ROOT / "outputs" / "asset_forecast_7d_new" / "metrics.csv",
    PROJECT_ROOT / "outputs" / "asset_forecast_7d_any" / "metrics.csv",
    PROJECT_ROOT / "outputs" / "part_current" / "metrics.csv",
    PROJECT_ROOT / "outputs" / "part_forecast_7d" / "metrics.csv",
)
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "model_comparison.csv"

COMPARISON_COLUMNS = (
    "grain",
    "mode",
    "risk_definition",
    "target",
    "score_threshold",
    "horizon",
    "scope_kind",
    "scope_name",
    "model",
    "threshold_policy",
    "probability_cutoff",
    "status",
    "test_rows",
    "test_positive_rate",
    "average_precision",
    "roc_auc",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "precision_at_10pct",
    "recall_at_10pct",
    "lift_at_10pct",
)


def load_metric_outputs(paths: Sequence[str | Path] = DEFAULT_METRICS) -> pd.DataFrame:
    """존재하는 평가 CSV를 합쳐 반환한다.

    Args:
        paths: 읽을 ``metrics.csv`` 경로 목록. 존재하지 않는 일부 경로는 건너뛴다.

    Returns:
        파일 출처 컬럼 ``metrics_path``가 추가된 통합 평가표.

    Raises:
        FileNotFoundError: 입력 경로 중 존재하는 평가 파일이 하나도 없을 때.
    """
    frames: list[pd.DataFrame] = []
    for value in paths:
        path = Path(value)
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        frame["metrics_path"] = str(path)
        frames.append(frame)
    if not frames:
        commands = "\n".join(
            [
                "python src/current_asset_model.py",
                "python src/forecast_asset_model.py --horizon 7 --risk-definition new",
                "python src/current_part_model.py",
                "python src/forecast_part_model.py --horizon 7",
            ]
        )
        raise FileNotFoundError(
            "비교할 metrics.csv가 없습니다. 먼저 다음 명령을 실행하세요:\n"
            f"{commands}"
        )
    return pd.concat(frames, ignore_index=True, sort=False)


def build_comparison(metrics: pd.DataFrame) -> pd.DataFrame:
    """검증에서 선택된 모델의 테스트 지표만 비교표로 추린다.

    Args:
        metrics: 공통 학습기가 저장한 하나 이상의 평가 결과.

    Returns:
        과제·범위·임계값 정책별 테스트 성능 비교표.

    Raises:
        ValueError: 선택 여부나 데이터 분할처럼 필수인 컬럼이 없을 때.
    """
    required = {"selected_model", "split", "model"}
    missing = sorted(required - set(metrics.columns))
    if missing:
        raise ValueError(f"평가 결과에 필요한 컬럼이 없습니다: {missing}")
    selected = metrics["selected_model"].astype(str).str.lower().eq("true")
    result = metrics.loc[selected & metrics["split"].eq("test")].copy()
    columns = [column for column in COMPARISON_COLUMNS if column in result.columns]
    if "metrics_path" in result.columns:
        columns.append("metrics_path")
    sort_columns = [
        column
        for column in ("grain", "mode", "target", "scope_kind", "scope_name", "threshold_policy")
        if column in result.columns
    ]
    result = result.loc[:, columns]
    if sort_columns:
        result = result.sort_values(sort_columns, kind="stable")
    return result.reset_index(drop=True)


def build_parser() -> argparse.ArgumentParser:
    """결과 비교 CLI의 입력 파일과 출력 파일 인자를 구성한다."""
    parser = argparse.ArgumentParser(description="산업 위험 모델 테스트 결과 비교")
    parser.add_argument(
        "--metrics",
        type=Path,
        action="append",
        help="비교할 metrics.csv. 반복 지정 가능하며 생략하면 네 기본 출력 사용",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> None:
    """저장된 평가 파일을 읽어 비교표를 출력하고 CSV로 저장한다."""
    args = build_parser().parse_args()
    metrics = load_metric_outputs(args.metrics or DEFAULT_METRICS)
    comparison = build_comparison(metrics)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(args.output, index=False)
    print(comparison.to_string(index=False))
    print(f"\n[저장 완료] {args.output.resolve()}")


if __name__ == "__main__":
    main()
