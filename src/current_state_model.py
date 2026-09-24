"""기존 명령을 부품 단위 당일 탐지 CLI로 연결하는 호환 실행 파일이다."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "src"

from .current_part_model import main as run_current_part
from .industrial_training import PROJECT_ROOT


def main() -> None:
    """기존 기본 출력 경로를 유지하며 새 부품 당일 CLI를 실행한다."""
    run_current_part(default_output=PROJECT_ROOT / "outputs" / "current_state")


if __name__ == "__main__":
    main()
