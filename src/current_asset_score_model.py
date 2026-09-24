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
    """장비 점수 실험에 필요한 CLI 인자를 정의한다."""
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
    """데이터를 읽어 회귀와 4단계 분류 실험을 실행한다."""
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
