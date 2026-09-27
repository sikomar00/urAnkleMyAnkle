"""산업 위험 모델 네 실행 파일이 공유하는 CLI 인자와 실행 변환을 제공한다."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from .industrial_data import PreparedTask
from .industrial_training import DEFAULT_DATA, run_experiment_suite

DEFAULT_THRESHOLD_POLICIES = ("f1", "min_precision", "top_fraction")


class IndustrialArgumentParser(argparse.ArgumentParser):
    """산업 모델의 생략된 목록 인자를 사용자용 기본값으로 정규화한다."""

    def parse_args(self, args=None, namespace=None):  # type: ignore[override]
        """명령행을 파싱하고 점수 기준·임계값 정책 기본값을 채운다."""
        parsed = super().parse_args(args=args, namespace=namespace)
        if hasattr(parsed, "score_threshold") and parsed.score_threshold is None:
            parsed.score_threshold = [12, 13, 14]
        policies = getattr(parsed, "threshold_policy", None)
        parsed.threshold_policy = list(dict.fromkeys(policies or DEFAULT_THRESHOLD_POLICIES))
        return parsed


def add_common_arguments(
    parser: argparse.ArgumentParser,
    default_output: Path | None,
) -> None:
    """네 학습 CLI에 공통 데이터·분할·평가 인자를 추가한다.

    Args:
        parser: 인자를 추가할 ``ArgumentParser``.
        default_output: 해당 과제의 기본 산출물 폴더.
    """
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
    parser.add_argument(
        "--threshold-policy",
        action="append",
        choices=list(DEFAULT_THRESHOLD_POLICIES),
        help="반복 지정 가능. 생략하면 f1, min_precision, top_fraction을 모두 실행",
    )
    parser.add_argument("--min-precision", type=float, default=0.30)
    parser.add_argument("--top-fraction", type=float, default=0.10)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--random-state", type=int, default=42)


def add_score_threshold_argument(parser: argparse.ArgumentParser) -> None:
    """장비 과제에 반복 가능한 고장점수 기준 인자를 추가한다."""
    parser.add_argument(
        "--score-threshold",
        type=int,
        action="append",
        dest="score_threshold",
        help="반복 지정 가능. 생략하면 12, 13, 14를 모두 실행",
    )


def run_tasks(tasks: Sequence[PreparedTask], args: argparse.Namespace) -> None:
    """파싱된 공통 인자를 공통 학습·평가 실행기로 전달한다.

    Args:
        tasks: Feature와 Target 생성이 끝난 하나 이상의 과제.
        args: ``add_common_arguments``가 구성한 명령행 인자.

    Raises:
        ValueError: 공통 학습기가 잘못된 정책이나 데이터 계약을 발견한 경우.
    """
    run_experiment_suite(
        tasks,
        output_dir=args.output,
        scope=args.scope,
        machine_type=args.machine_type,
        asset_tag=args.asset_tag,
        max_iter=args.max_iter,
        validation_start=args.validation_start,
        test_start=args.test_start,
        threshold_policies=tuple(args.threshold_policy),
        min_precision=args.min_precision,
        top_fraction=args.top_fraction,
        random_state=args.random_state,
    )
