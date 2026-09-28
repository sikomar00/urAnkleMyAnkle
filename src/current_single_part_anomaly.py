"""완전 정상과 특정 부품 단독 고장의 당일 센서 이상을 비교하는 실행 파일."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "src"

from .industrial_data import load_industrial_data
from .industrial_training import DEFAULT_DATA, PROJECT_ROOT
from .single_part_anomaly_analysis import run_single_part_analysis

DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "single_part_anomaly"


def build_parser(default_output: Path = DEFAULT_OUTPUT) -> argparse.ArgumentParser:
    """입력 데이터, 기간, 정상 기준과 결과 저장 옵션을 정의한다."""
    parser = argparse.ArgumentParser(
        description="완전 정상과 특정 부품 단독 고장의 센서 이상치 비교"
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--validation-start", default="2024-01-01")
    parser.add_argument("--test-start", default="2024-07-01")
    parser.add_argument("--min-normal-rows", type=int, default=30)
    parser.add_argument("--bootstrap-samples", type=int, default=1_000)
    parser.add_argument("--expected-parts-per-asset", type=int, default=20)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--skip-residual",
        action="store_true",
        help="운전조건 보정 잔차 분석을 생략하고 직접 Robust Z-score만 계산",
    )
    return parser


def main(default_output: Path = DEFAULT_OUTPUT) -> None:
    """원본 데이터를 읽어 부품별 단독 고장 이상치 비교를 실행한다."""
    args = build_parser(default_output).parse_args()
    raw = load_industrial_data(args.data)
    run_single_part_analysis(
        raw,
        output_dir=args.output,
        validation_start=args.validation_start,
        test_start=args.test_start,
        min_normal_rows=args.min_normal_rows,
        bootstrap_samples=args.bootstrap_samples,
        random_state=args.random_state,
        max_iter=args.max_iter,
        include_residual=not args.skip_residual,
        data_path=args.data,
        expected_parts_per_asset=args.expected_parts_per_asset,
    )
    print(f"부품별 단독 고장 이상치 비교 결과 저장 완료: {args.output}")


if __name__ == "__main__":
    main()
