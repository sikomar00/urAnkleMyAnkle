"""향후 7일 고장 위험 시계열 모델 실행 진입점."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "src"

from .industrial_features import load_industrial_data, prepare_forecast
from .industrial_training import DEFAULT_DATA, PROJECT_ROOT, run_experiments


def main() -> None:
    parser = argparse.ArgumentParser(description="과거 센서·고장 이력 기반 7일 위험 모델")
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--output", default=PROJECT_ROOT / "outputs" / "timeseries")
    parser.add_argument("--horizon", type=int, default=7)
    parser.add_argument("--scope", choices=["all", "overall", "machine_type", "asset_tag"], default="all")
    parser.add_argument("--machine-type")
    parser.add_argument("--asset-tag")
    parser.add_argument("--max-iter", type=int, default=100)
    args = parser.parse_args()
    raw = load_industrial_data(args.data)
    frame, feature_data, target_data = prepare_forecast(raw, horizon=args.horizon)
    run_experiments(
        frame,
        feature_data,
        target_data,
        output_dir=args.output,
        mode="forecast",
        scope=args.scope,
        machine_type=args.machine_type,
        asset_tag=args.asset_tag,
        max_iter=args.max_iter,
    )


if __name__ == "__main__":
    main()

