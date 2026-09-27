"""장비 당일 4단계 위험도 Feature 비교 실험 실행 파일이다."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "src"

from .asset_severity_experiments import run_asset_severity_experiments
from .industrial_data import load_industrial_data
from .industrial_training import DEFAULT_DATA, PROJECT_ROOT

DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "asset_severity_experiments"


def build_parser(default_output: Path = DEFAULT_OUTPUT) -> argparse.ArgumentParser:
    """12·13점 및 A~D 비교 실험의 CLI 인자를 정의한다."""
    parser = argparse.ArgumentParser(
        description="장비 당일 4단계 위험도 센서 이상 Feature 비교"
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument(
        "--high-risk-thresholds",
        nargs="+",
        type=int,
        default=[12, 13],
    )
    parser.add_argument(
        "--feature-sets",
        nargs="+",
        choices=["A", "B", "C", "D"],
        default=["A", "B", "C", "D"],
    )
    parser.add_argument(
        "--scope",
        choices=["all", "overall", "machine_type"],
        default="all",
    )
    parser.add_argument("--machine-type")
    parser.add_argument("--asset-tag")
    parser.add_argument("--validation-start", default="2024-01-01")
    parser.add_argument("--test-start", default="2024-07-01")
    parser.add_argument("--min-normal-rows", type=int, default=30)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--random-state", type=int, default=42)
    return parser


def main(default_output: Path = DEFAULT_OUTPUT) -> None:
    """데이터를 한 번 읽어 임계값·Feature 비교 실험을 실행한다."""
    args = build_parser(default_output).parse_args()
    raw = load_industrial_data(args.data)
    run_asset_severity_experiments(
        raw,
        output_dir=args.output,
        high_risk_thresholds=args.high_risk_thresholds,
        feature_sets=args.feature_sets,
        scope=args.scope,
        machine_type=args.machine_type,
        asset_tag=args.asset_tag,
        validation_start=args.validation_start,
        test_start=args.test_start,
        min_normal_rows=args.min_normal_rows,
        max_iter=args.max_iter,
        random_state=args.random_state,
    )
    print(f"실험 결과 저장 완료: {args.output}")


if __name__ == "__main__":
    main()
