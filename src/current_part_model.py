"""부품·날짜별 당일 고장 여부를 학습·평가하는 실행 파일이다."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "src"

from .industrial_cli import IndustrialArgumentParser, add_common_arguments, run_tasks
from .industrial_data import load_industrial_data
from .industrial_training import PROJECT_ROOT
from .part_features import prepare_part_current

DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "part_current"


def build_parser(default_output: Path = DEFAULT_OUTPUT) -> IndustrialArgumentParser:
    """부품 당일 고장 CLI 파서를 만든다."""
    parser = IndustrialArgumentParser(description="부품 단위 당일 고장 탐지 모델")
    add_common_arguments(parser, default_output)
    return parser


def main(default_output: Path = DEFAULT_OUTPUT) -> None:
    """원본 데이터를 읽어 부품 당일 고장 탐지 과제를 실행한다."""
    args = build_parser(default_output).parse_args()
    raw = load_industrial_data(args.data)
    run_tasks([prepare_part_current(raw)], args)


if __name__ == "__main__":
    main()
