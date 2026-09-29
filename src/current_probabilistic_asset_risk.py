"""부품 고장확률 기반 장비 당일 위험 실험 CLI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "src"

from .industrial_data import load_industrial_data
from .industrial_training import DEFAULT_DATA, MODEL_NAMES, PROJECT_ROOT
from .probabilistic_asset_risk import run_probabilistic_asset_risk
from .probabilistic_risk_report import render_probabilistic_risk_summary

DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "probabilistic_asset_risk"


def _threshold(value: str) -> int:
    parsed = int(value)
    if parsed < 7:
        raise argparse.ArgumentTypeError("고위험 기준은 7 이상의 정수여야 합니다.")
    return parsed


def _fpr(value: str) -> float:
    parsed = float(value)
    if not 0 < parsed <= 1:
        raise argparse.ArgumentTypeError("FPR 정책은 0보다 크고 1 이하여야 합니다.")
    return parsed


def build_parser(default_output: Path = DEFAULT_OUTPUT) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="부품 고장확률 기반 장비 당일 위험·점수·Top 3 보조 실험"
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--selection-start", default="2024-01-01")
    parser.add_argument("--calibration-start", default="2024-04-01")
    parser.add_argument("--test-start", default="2024-07-01")
    parser.add_argument("--high-risk-thresholds", nargs="+", type=_threshold, default=[12, 13])
    parser.add_argument("--fpr-policies", nargs="+", type=_fpr, default=[0.05, 0.10])
    parser.add_argument("--model-names", nargs="+", choices=list(MODEL_NAMES), default=list(MODEL_NAMES))
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--random-state", type=int, default=42)
    return parser


def main(default_output: Path = DEFAULT_OUTPUT) -> None:
    args = build_parser(default_output).parse_args()
    timestamps = [pd.Timestamp(value) for value in (
        args.selection_start, args.calibration_start, args.test_start
    )]
    if not timestamps[0] < timestamps[1] < timestamps[2]:
        raise ValueError("selection_start < calibration_start < test_start 순서여야 합니다.")
    raw = load_industrial_data(args.data)
    run = run_probabilistic_asset_risk(
        raw,
        output_dir=args.output,
        selection_start=args.selection_start,
        calibration_start=args.calibration_start,
        test_start=args.test_start,
        high_risk_thresholds=tuple(args.high_risk_thresholds),
        fpr_policies=tuple(args.fpr_policies),
        bootstrap_samples=args.bootstrap_samples,
        max_iter=args.max_iter,
        random_state=args.random_state,
        model_names=tuple(args.model_names),
        source_path=args.data,
    )
    summary = render_probabilistic_risk_summary(run)
    (run.output_dir / "experiment_summary.md").write_text(summary, encoding="utf-8")
    print(f"확률 기반 장비 위험 실험 결과 저장 완료: {run.output_dir}")


if __name__ == "__main__":
    main()
