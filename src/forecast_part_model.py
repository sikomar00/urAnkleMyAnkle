"""오늘 정상인 부품의 향후 기본 7일 고장 위험을 예측하는 실행 파일이다."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "src"

from .industrial_cli import IndustrialArgumentParser, add_common_arguments, run_tasks
from .industrial_data import load_industrial_data
from .industrial_training import PROJECT_ROOT
from .part_features import prepare_part_forecast

DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "part_forecast_7d"


def build_parser(default_output: Path = DEFAULT_OUTPUT) -> IndustrialArgumentParser:
    """부품 미래 위험 CLI 파서를 만들며 기본 예측 기간은 7일이다."""
    parser = IndustrialArgumentParser(description="부품 단위 향후 고장 위험 예측 모델")
    add_common_arguments(parser, default_output)
    parser.add_argument("--horizon", type=int, default=7)
    return parser


def main(default_output: Path = DEFAULT_OUTPUT) -> None:
    """원본 데이터를 읽어 부품 미래 고장 위험 과제를 실행한다."""
    args = build_parser(default_output).parse_args()
    raw = load_industrial_data(args.data)
    run_tasks([prepare_part_forecast(raw, horizon=args.horizon)], args)


if __name__ == "__main__":
    main()
