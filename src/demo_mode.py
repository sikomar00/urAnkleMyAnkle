"""데모 모드 — DB·로그인 없이 대시보드를 읽기 전용으로 실행한다.

`python -m src.wireframe_app --demo` 또는 `DASHBOARD_MODE=demo`로 켠다.
운영 모드 코드는 이 모듈의 값을 읽기만 하고 동작을 바꾸지 않는다.
"""
import os
import sys

DEMO_BADGE_TEXT = "데모 모드 · 읽기 전용"


def is_demo_mode() -> bool:
    return "--demo" in sys.argv or os.environ.get("DASHBOARD_MODE", "").lower() == "demo"


def noop(*_args, **_kwargs) -> None:
    """감사·로그 기록 함수의 대체. DB에 접속하지 않는다."""


def empty_list(*_args, **_kwargs) -> list:
    return []


def no_session_state(*_args, **_kwargs) -> None:
    return None
