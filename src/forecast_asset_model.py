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

DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs"


def default_output_path(horizon: int, risk_definition: str) -> Path:
    """예측 기간과 위험 정의가 서로 덮어쓰지 않는 기본 출력 경로를 반환한다.

    Raises:
        ValueError: 예측 기간이 1보다 작거나 위험 정의가 new/any가 아닐 때.
    """
    if horizon < 1:
        raise ValueError("horizon은 1 이상이어야 합니다.")
    if risk_definition not in {"new", "any"}:
        raise ValueError("risk_definition은 new 또는 any여야 합니다.")
    return DEFAULT_OUTPUT_ROOT / f"asset_forecast_{horizon}d_{risk_definition}"


def build_parser(default_output: Path | None = None) -> IndustrialArgumentParser:
    """장비 미래 위험 CLI 파서를 만들며 기본 기간은 7일, 정의는 신규 위험이다."""
    parser = IndustrialArgumentParser(description="장비 단위 향후 위험 예측 모델")
    add_common_arguments(parser, default_output)
    add_score_threshold_argument(parser)
    parser.add_argument("--horizon", type=int, default=7)
    parser.add_argument("--risk-definition", choices=["new", "any"], default="new")
    return parser


def main(default_output: Path | None = None) -> None:
    """원본 데이터를 읽어 장비 미래 위험 과제를 점수 기준별로 실행한다."""
    args = build_parser(default_output).parse_args()
    if args.output is None:
        args.output = default_output_path(args.horizon, args.risk_definition)
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
