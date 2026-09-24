"""오늘까지의 장비 정보로 향후 기본 7일 위험을 예측하는 실행 파일이다."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "src"

from .asset_features import prepare_asset_forecast
from .industrial_cli import (
    IndustrialArgumentParser,
    add_common_arguments,
    add_score_threshold_argument,
    run_tasks,
)
from .industrial_data import load_industrial_data
from .industrial_training import PROJECT_ROOT

DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "asset_forecast_7d"


def build_parser(default_output: Path = DEFAULT_OUTPUT) -> IndustrialArgumentParser:
    """장비 미래 위험 CLI 파서를 만들며 기본 기간은 7일, 정의는 신규 위험이다."""
    parser = IndustrialArgumentParser(description="장비 단위 향후 위험 예측 모델")
    add_common_arguments(parser, default_output)
    add_score_threshold_argument(parser)
    parser.add_argument("--horizon", type=int, default=7)
    parser.add_argument("--risk-definition", choices=["new", "any"], default="new")
    return parser


def main(default_output: Path = DEFAULT_OUTPUT) -> None:
    """원본 데이터를 읽어 장비 미래 위험 과제를 점수 기준별로 실행한다."""
    args = build_parser(default_output).parse_args()
    raw = load_industrial_data(args.data)
    tasks = [
        prepare_asset_forecast(
            raw,
            score_threshold=value,
            horizon=args.horizon,
            risk_definition=args.risk_definition,
        )
        for value in args.score_threshold
    ]
    run_tasks(tasks, args)


if __name__ == "__main__":
    main()
