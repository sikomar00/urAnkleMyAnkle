"""SQLAlchemy 트랜잭션으로 대시보드의 세 종류 로그를 저장한다."""

from __future__ import annotations

import hmac
import json
import logging
import os
from threading import Lock
from datetime import date
from pathlib import Path

import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .audit_models import ActionLog, ErrorLog, FailureLog, kst_now
from .dashboard_data import _data_path
from .db_models import AdminInfo, utc_now
from .db_service import get_engine
from .security_service import encrypt_personal_data, hash_password, verify_password

_fallback = logging.getLogger("dashboard_audit_fallback")
_sync_lock = Lock()
_last_synced_file: tuple[str, int, int] | None = None
if not _fallback.handlers:
    _handler = logging.FileHandler(Path(__file__).resolve().parents[1] / "instance" / "audit_fallback.log", encoding="utf-8") if (Path(__file__).resolve().parents[1] / "instance").is_dir() else None
    if _handler is None:
        (Path(__file__).resolve().parents[1] / "instance").mkdir(exist_ok=True)
        _handler = logging.FileHandler(Path(__file__).resolve().parents[1] / "instance" / "audit_fallback.log", encoding="utf-8")
    _fallback.addHandler(_handler)
    _fallback.setLevel(logging.WARNING)


def create_audit_tables() -> None:
    """기존 테이블을 변경하지 않고 필요한 네 테이블만 만든다."""
    engine = get_engine()
    for table in (AdminInfo.__table__, ActionLog.__table__, FailureLog.__table__, ErrorLog.__table__):
        table.create(engine, checkfirst=True)


def _actor(value: str | None) -> str:
    # 대시보드 요청에서는 브라우저가 보낸 작업자 ID 대신 검증된 서버 세션을 사용한다.
    try:
        from flask import has_request_context, session as flask_session
        if has_request_context() and flask_session.get("admin_id"):
            return str(flask_session["admin_id"])[:80]
    except RuntimeError:
        pass
    value = (value or "").strip()
    return value[:80] if value else "ANONYMOUS"


def record_action(event_code: str, action_type: str, *, actor_id: str | None = None,
                  target_type: str | None = None, target_id: str | None = None,
                  detail: dict | None = None, status: str = "SUCCESS",
                  block_reason: str | None = None) -> bool:
    """실제 조작만 기록한다. 개인정보·비밀번호는 detail에 넣지 않는다."""
    try:
        with Session(get_engine()) as session, session.begin():
            session.add(ActionLog(occurred_at=kst_now(), event_code=event_code, action_type=action_type,
                                  actor_id=_actor(actor_id), target_type=target_type,
                                  target_id=str(target_id)[:100] if target_id is not None else None,
                                  action_detail=json.dumps(detail or {}, ensure_ascii=False, default=str)[:2000],
                                  result_status=status, block_reason=block_reason))
        return True
    except Exception as exc:
        _fallback.warning("action log DB write failed: %s %s", event_code, type(exc).__name__)
        return False


def record_error(event_code: str, action_type: str, exc: Exception, *, actor_id: str | None = None,
                 source: str, target_id: str | None = None) -> bool:
    """오류 문자열에 접속 URL·비밀값이 있을 수 있어 종류와 안전한 설명만 저장한다."""
    try:
        with Session(get_engine()) as session, session.begin():
            session.add(ErrorLog(occurred_at=kst_now(), actor_id=_actor(actor_id), action_type=action_type,
                                 error_code=event_code, error_type=type(exc).__name__,
                                 error_message=f"{action_type} 처리 중 {type(exc).__name__} 발생",
                                 source=source, target_id=str(target_id)[:100] if target_id is not None else None))
        return True
    except Exception as db_exc:
        _fallback.warning("error log DB write failed: %s %s", event_code, type(db_exc).__name__)
        return False


def log_failure(event_code: str, action_type: str, exc: Exception, *, actor_id: str | None,
                source: str, target_id: str | None = None) -> None:
    """실패 행동과 오류 상세를 한 트랜잭션으로 함께 저장한다."""
    try:
        with Session(get_engine()) as session, session.begin():
            session.add(ActionLog(occurred_at=kst_now(), event_code=event_code, action_type=action_type,
                                  actor_id=_actor(actor_id), target_id=target_id,
                                  action_detail="{}", result_status="FAILED"))
            session.add(ErrorLog(occurred_at=kst_now(), actor_id=_actor(actor_id), action_type=action_type,
                                 error_code="ERR_" + event_code.removeprefix("ACT_"),
                                 error_type=type(exc).__name__,
                                 error_message=f"{action_type} 처리 중 {type(exc).__name__} 발생",
                                 source=source, target_id=target_id))
    except Exception as db_exc:
        _fallback.warning("failure log DB write failed: %s %s", event_code, type(db_exc).__name__)


def admin_exists() -> bool:
    with Session(get_engine()) as session:
        return bool(session.scalar(select(func.count()).select_from(AdminInfo)))


def ensure_dashboard_admin() -> bool:
    """환경설정의 단일 관리자 계정을 DB에 한 번만 준비한다.

    공개 회원가입이나 대시보드 안의 관리자 등록 화면은 만들지 않는다.
    이미 같은 아이디가 있으면 기존 비밀번호 해시는 절대 바꾸지 않는다.
    """
    login_id = os.environ.get("DASHBOARD_ADMIN_ID", "").strip()[:80]
    password = os.environ.get("DASHBOARD_ADMIN_PASSWORD", "")
    if not login_id or not password:
        raise RuntimeError("DASHBOARD_ADMIN_ID와 DASHBOARD_ADMIN_PASSWORD 설정이 필요합니다.")
    with Session(get_engine()) as session, session.begin():
        existing = session.scalar(select(AdminInfo).where(AdminInfo.login_id == login_id))
        if existing is not None:
            return False
        session.add(AdminInfo(
            login_id=login_id,
            password_hash=hash_password(password),
            # 이 값도 DB에서는 AES-256-GCM 암호문으로만 저장된다.
            name_encrypted=encrypt_personal_data("Dashboard Administrator"),
            phone_encrypted=encrypt_personal_data(""),
            email_encrypted=encrypt_personal_data(""),
        ))
        session.add(ActionLog(
            occurred_at=kst_now(), event_code="ACT_ADMIN_BOOTSTRAPPED",
            action_type="기본 관리자 계정 준비", actor_id=login_id,
            target_type="admin", target_id=login_id,
            action_detail="{}", result_status="SUCCESS",
        ))
    return True


def register_first_admin(setup_token: str, login_id: str, password: str, name: str,
                         phone: str, email: str) -> None:
    """초기 관리자 한 명만 등록한다. 설정 토큰은 DB에 남기지 않는다."""
    expected = os.environ.get("DASHBOARD_SETUP_TOKEN", "")
    if not expected or not hmac.compare_digest(expected, setup_token or ""):
        raise PermissionError("초기 등록 코드가 올바르지 않습니다.")
    if not login_id.strip() or not name.strip() or not email.strip():
        raise ValueError("아이디, 이름, 이메일은 필수입니다.")
    with Session(get_engine()) as session, session.begin():
        if session.scalar(select(func.count()).select_from(AdminInfo)):
            raise ValueError("관리자 계정이 이미 등록되었습니다.")
        session.add(AdminInfo(login_id=login_id.strip()[:80], password_hash=hash_password(password),
                              name_encrypted=encrypt_personal_data(name.strip()),
                              phone_encrypted=encrypt_personal_data(phone.strip()),
                              email_encrypted=encrypt_personal_data(email.strip())))
        session.add(ActionLog(occurred_at=kst_now(), event_code="ACT_ADMIN_REGISTER", action_type="관리자 최초 등록",
                              actor_id=_actor(login_id), target_type="admin", target_id=login_id,
                              action_detail="{}", result_status="SUCCESS"))


def verify_admin(login_id: str, password: str) -> bool:
    with Session(get_engine()) as session:
        row = session.scalar(select(AdminInfo).where(AdminInfo.login_id == login_id))
        return bool(row and verify_password(row.password_hash, password))


def change_admin_password(login_id: str, current_password: str, new_password: str) -> bool:
    """현재 비밀번호를 확인하고 새 Argon2 해시와 성공 로그를 함께 확정한다."""
    if len(new_password) < 4:
        raise ValueError("새 비밀번호는 4자 이상이어야 합니다.")
    with Session(get_engine()) as session, session.begin():
        admin = session.scalar(
            select(AdminInfo).where(AdminInfo.login_id == login_id).with_for_update()
        )
        if admin is None or not verify_password(admin.password_hash, current_password):
            return False
        if current_password == new_password:
            raise ValueError("현재 비밀번호와 다른 새 비밀번호를 입력해 주세요.")
        admin.password_hash = hash_password(new_password)
        admin.updated_at = utc_now()
        session.add(ActionLog(occurred_at=kst_now(), event_code="ACT_PASSWORD_CHANGE",
                              action_type="관리자 비밀번호 변경", actor_id=login_id,
                              target_type="admin", target_id=login_id,
                              action_detail="{}", result_status="SUCCESS"))
    return True


def sync_observed_failures() -> dict:
    """CSV 최신 날짜의 실제 부품 고장 표시만 기록하고 중복은 건너뛴다."""
    raw = pd.read_csv(_data_path(), usecols=["transaction_date", "plant_code", "asset_tag",
                                               "part_no", "part_family", "breakdown_flag"],
                      parse_dates=["transaction_date"])
    latest: date = raw["transaction_date"].max().date()
    failed = raw[(raw["transaction_date"].dt.date == latest) & (raw["breakdown_flag"] == 1)]
    created = 0
    with Session(get_engine()) as session, session.begin():
        existing = set(session.execute(select(FailureLog.asset_tag, FailureLog.part_no).where(
            FailureLog.observation_date == latest,
            FailureLog.event_code == "FAIL_PART_OBSERVED")).all())
        for row in failed.itertuples(index=False):
            key = (str(row.asset_tag), str(row.part_no))
            if key in existing:
                continue
            session.add(FailureLog(occurred_at=kst_now(), event_code="FAIL_PART_OBSERVED", observation_date=latest,
                                   plant_code=str(row.plant_code), asset_tag=key[0], part_no=key[1],
                                   part_family=str(row.part_family), actual_failure=True,
                                   failure_status="OBSERVED", source="synthetic_csv"))
            existing.add(key)
            created += 1
    return {"date": latest.isoformat(), "observed_failures": len(failed), "created": created}


def sync_if_csv_changed() -> dict:
    """여러 브라우저가 열려도 CSV가 실제 바뀌었을 때만 재집계한다."""
    global _last_synced_file
    path = _data_path().resolve()
    stat = path.stat()
    state = (str(path), stat.st_mtime_ns, stat.st_size)
    with _sync_lock:
        if state == _last_synced_file:
            return {"unchanged": True}
        result = sync_observed_failures()
        _last_synced_file = state
        return result
