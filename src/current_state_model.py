"""현재 상태 분류 실행 진입점."""

from __future__ import annotations

import argparse

from .industrial_features import load_industrial_data, prepare_current
from .industrial_training import run_experiments


def main() -> None:
    parser = argparse.ArgumentParser(description="현재 breakdown_flag 분류 모델")
    parser.add_argument("--data", default="data/synthetic_industrial_machine_data.csv")
    parser.add_argument("--output", default="outputs/current_state")
    parser.add_argument("--scope", choices=["all", "overall", "machine_type", "asset_tag"], default="all")
    parser.add_argument("--machine-type")
    parser.add_argument("--asset-tag")
    parser.add_argument("--max-iter", type=int, default=100)
    args = parser.parse_args()
    raw = load_industrial_data(args.data)
    frame, feature_data, target_data = prepare_current(raw)
    run_experiments(
        frame,
        feature_data,
        target_data,
        output_dir=args.output,
        mode="current",
        scope=args.scope,
        machine_type=args.machine_type,
        asset_tag=args.asset_tag,
        max_iter=args.max_iter,
    )


if __name__ == "__main__":
    main()

