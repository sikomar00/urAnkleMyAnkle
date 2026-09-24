"""장비·날짜별 당일 위험을 점수 기준별로 학습·평가하는 실행 파일이다."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "src"

from .asset_features import prepare_asset_current
from .industrial_cli import (
    IndustrialArgumentParser,
    add_common_arguments,
    add_score_threshold_argument,
    run_tasks,
)
from .industrial_data import load_industrial_data
from .industrial_training import PROJECT_ROOT

DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "asset_current"


def build_parser(default_output: Path = DEFAULT_OUTPUT) -> IndustrialArgumentParser:
    """장비 당일 위험 CLI 파서를 만든다."""
    parser = IndustrialArgumentParser(description="장비 단위 당일 위험 탐지 모델")
    add_common_arguments(parser, default_output)
    add_score_threshold_argument(parser)
    return parser


def main(default_output: Path = DEFAULT_OUTPUT) -> None:
    """원본 데이터를 읽어 장비 당일 과제를 점수 기준별로 실행한다."""
    args = build_parser(default_output).parse_args()
    raw = load_industrial_data(args.data)
    tasks = [prepare_asset_current(raw, value) for value in args.score_threshold]
    run_tasks(tasks, args)


if __name__ == "__main__":
    main()
