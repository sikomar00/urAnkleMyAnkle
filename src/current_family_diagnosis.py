"""장비 당일 센서로 부품 Family의 이상·심각 여부를 진단하는 실행 파일."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "src"

from .family_diagnosis_experiments import run_family_experiments
from .family_features import prepare_family_features
from .industrial_data import load_industrial_data
from .industrial_training import DEFAULT_DATA, PROJECT_ROOT

DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "family_current"


def build_parser(default_output: Path = DEFAULT_OUTPUT) -> argparse.ArgumentParser:
    """당일 Family 진단의 입력·Feature·Target·날짜 옵션을 정의한다."""
    parser = argparse.ArgumentParser(
        description="장비 센서 기반 당일 부품 Family 이상·심각 진단"
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument(
        "--feature-sets",
        nargs="+",
        choices=["A", "B", "C"],
        default=["A", "B", "C"],
        help="A=당일, B=과거 이력, C=운전조건 보정 잔차",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        choices=["affected", "severe"],
        default=["affected", "severe"],
    )
    parser.add_argument("--scope", choices=["overall"], default="overall")
    parser.add_argument("--validation-start", default="2024-01-01")
    parser.add_argument("--test-start", default="2024-07-01")
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--random-state", type=int, default=42)
    return parser


def main(default_output: Path = DEFAULT_OUTPUT) -> None:
    """원본 로드부터 Feature 준비·실험·결과 저장까지 순서대로 실행한다."""
    args = build_parser(default_output).parse_args()
    raw = load_industrial_data(args.data)
    prepared = prepare_family_features(
        raw,
        validation_start=args.validation_start,
        feature_sets=tuple(args.feature_sets),
    )
    run_family_experiments(
        prepared,
        output_dir=args.output,
        max_iter=args.max_iter,
        random_state=args.random_state,
        targets=tuple(args.targets),
        validation_start=args.validation_start,
        test_start=args.test_start,
    )
    print(f"당일 Family 진단 결과 저장 완료: {args.output}")


if __name__ == "__main__":
    main()
