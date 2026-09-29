"""대시보드 계정·역할과 행동·인증·고장·오류 로그를 SQLAlchemy로 저장한다."""

from __future__ import annotations

import hmac
import json
import logging
import os
from datetime import date
from pathlib import Path
from threading import Lock

import pandas as pd
from sqlalchemy import inspect, select, text, update
from sqlalchemy.orm import Session

from .audit_models import ActionLog, ErrorLog, FailureLog, LoginLog, kst_now
from .dashboard_data import _data_path
from .db_models import AdminInfo, SystemLog, utc_now
from .db_service import get_engine
from .security_service import decrypt_personal_data, encrypt_personal_data, hash_password, verify_password


AUTH_EVENT_CODES = {
    "ACT_LOGIN", "ACT_LOGOUT", "ACT_SESSION_EXTEND", "ACT_SESSION_EXPIRED", "ACT_SESSION_REVOKED",
    "ACT_LOGIN_BLOCKED",
}
VALID_ROLES = {"ADMIN", "USER", "UNKNOWN"}

_fallback = logging.getLogger("dashboard_audit_fallback")
_sync_lock = Lock()
_last_synced_file: tuple[str, int, int] | None = None
if not _fallback.handlers:
    root = Path(__file__).resolve().parents[1]
    (root / "instance").mkdir(exist_ok=True)
    _fallback.addHandler(logging.FileHandler(root / "instance" / "audit_fallback.log", encoding="utf-8"))
    _fallback.setLevel(logging.WARNING)


def _normal_role(value: str | None) -> str:
    """DB·세션에 저장할 역할을 세 가지 허용값으로만 제한한다."""
    role = str(value or "UNKNOWN").upper()
    return role if role in VALID_ROLES else "UNKNOWN"


def _actor_and_role(actor_id: str | None = None, actor_role: str | None = None) -> tuple[str, str]:
    """요청이 있으면 서명된 서버 세션의 계정·역할을 우선 사용한다."""
    try:
        from flask import has_request_context, session as flask_session
        if has_request_context() and flask_session.get("admin_id"):
            return str(flask_session["admin_id"])[:80], _normal_role(flask_session.get("role"))
    except RuntimeError:
        pass
    identity = (actor_id or "").strip()[:80] or "ANONYMOUS"
    return identity, _normal_role(actor_role)


def _add_column_if_missing(connection, table: str, column: str, definition: str) -> None:
    """기존 MySQL·SQLite 테이블에 필요한 컬럼만 한 번 추가한다."""
    existing = {item["name"] for item in inspect(connection).get_columns(table)}
    if column not in existing:
        connection.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {definition}"))


def migrate_account_schema() -> None:
    """기존 단일 관리자·행동 로그 DB를 역할·인증 로그 구조로 안전하게 확장한다."""
    engine = get_engine()
    # 새 설치에서는 create()가 완성된 구조를 만들고, 기존 설치에서는 아래 ALTER가 적용된다.
    AdminInfo.__table__.create(engine, checkfirst=True)
    ActionLog.__table__.create(engine, checkfirst=True)
    LoginLog.__table__.create(engine, checkfirst=True)
    FailureLog.__table__.create(engine, checkfirst=True)
    ErrorLog.__table__.create(engine, checkfirst=True)

    with engine.begin() as connection:
        _add_column_if_missing(connection, "dashboard_admin_info", "role", "VARCHAR(10) NOT NULL DEFAULT 'USER'")
        _add_column_if_missing(connection, "dashboard_admin_info", "is_online", "BOOLEAN NOT NULL DEFAULT 0")
        _add_column_if_missing(connection, "dashboard_admin_info", "session_expires_at", "DATETIME NULL")
        _add_column_if_missing(connection, "dashboard_admin_info", "last_login_at", "DATETIME NULL")
        _add_column_if_missing(connection, "action_logs", "actor_role", "VARCHAR(10) NOT NULL DEFAULT 'UNKNOWN'")
        default_admin = os.environ.get("DASHBOARD_ADMIN_ID", "ankles").strip()[:80]
        connection.execute(text("UPDATE dashboard_admin_info SET role = 'ADMIN' WHERE login_id = :login_id"),
                           {"login_id": default_admin})
        connection.execute(text("UPDATE action_logs SET actor_role = 'ADMIN' "
                                "WHERE actor_id = :login_id AND actor_role = 'UNKNOWN'"),
                           {"login_id": default_admin})

    # 예전 action_logs 안의 인증 이벤트는 한 번만 login_logs로 옮긴 뒤 원본에서 제거한다.
    with Session(engine) as session, session.begin():
        old_rows = session.scalars(select(ActionLog).where(ActionLog.event_code.in_(AUTH_EVENT_CODES))).all()
        for row in old_rows:
            session.add(LoginLog(
                occurred_at=row.occurred_at, actor_id=row.actor_id, actor_role=_normal_role(row.actor_role),
                event_code=row.event_code, result_status=row.result_status,
                block_reason=row.block_reason, source="migrated_action_logs",
            ))
            session.delete(row)


def create_audit_tables() -> None:
    """앱이 사용하는 계정·행동·인증·고장·오류 테이블과 컬럼을 준비한다."""
    migrate_account_schema()


def record_action(event_code: str, action_type: str, *, actor_id: str | None = None,
                  actor_role: str | None = None, target_type: str | None = None,
                  target_id: str | None = None, detail: dict | None = None,
                  status: str = "SUCCESS", block_reason: str | None = None) -> bool:
    """로그인 외의 대시보드 행동만 저장한다. 개인정보·비밀번호는 detail에 넣지 않는다."""
    if event_code in AUTH_EVENT_CODES:
        return record_login_event(event_code, actor_id=actor_id, actor_role=actor_role,
                                  status=status, block_reason=block_reason, source="redirected_from_action")
    try:
        identity, role = _actor_and_role(actor_id, actor_role)
        with Session(get_engine()) as session, session.begin():
            session.add(ActionLog(
                occurred_at=kst_now(), event_code=event_code, action_type=action_type,
                actor_id=identity, actor_role=role, target_type=target_type,
                target_id=str(target_id)[:100] if target_id is not None else None,
                action_detail=json.dumps(detail or {}, ensure_ascii=False, default=str)[:2000],
                result_status=status, block_reason=block_reason,
            ))
        return True
    except Exception as exc:
        _fallback.warning("action log DB write failed: %s %s", event_code, type(exc).__name__)
        return False


def record_login_event(event_code: str, *, actor_id: str | None = None,
                       actor_role: str | None = None, status: str = "SUCCESS",
                       block_reason: str | None = None, source: str) -> bool:
    """로그인·로그아웃·세션 이벤트를 행동 로그와 분리해 저장한다."""
    try:
        identity, role = _actor_and_role(actor_id, actor_role)
        with Session(get_engine()) as session, session.begin():
            session.add(LoginLog(
                occurred_at=kst_now(), actor_id=identity, actor_role=role,
                event_code=event_code, result_status=status,
                block_reason=block_reason, source=source[:120],
            ))
        return True
    except Exception as exc:
        _fallback.warning("login log DB write failed: %s %s", event_code, type(exc).__name__)
        return False


def record_error(event_code: str, action_type: str, exc: Exception, *, actor_id: str | None = None,
                 source: str, target_id: str | None = None) -> bool:
    """오류 문자열에 비밀값이 있을 수 있어 오류 종류와 안전한 설명만 저장한다."""
    try:
        identity, _ = _actor_and_role(actor_id)
        with Session(get_engine()) as session, session.begin():
            session.add(ErrorLog(
                occurred_at=kst_now(), actor_id=identity, action_type=action_type,
                error_code=event_code, error_type=type(exc).__name__,
                error_message=f"{action_type} 처리 중 {type(exc).__name__} 발생",
                source=source, target_id=str(target_id)[:100] if target_id is not None else None,
            ))
        return True
    except Exception as db_exc:
        _fallback.warning("error log DB write failed: %s %s", event_code, type(db_exc).__name__)
        return False


def log_failure(event_code: str, action_type: str, exc: Exception, *, actor_id: str | None,
                source: str, target_id: str | None = None) -> None:
    """실패 행동과 오류를 남긴다. 인증 실패는 login_logs에만 인증 이벤트를 남긴다."""
    identity, role = _actor_and_role(actor_id)
    if event_code in AUTH_EVENT_CODES:
        record_login_event(event_code, actor_id=identity, actor_role=role, status="FAILED",
                           block_reason=type(exc).__name__, source=source)
    else:
        record_action(event_code, action_type, actor_id=identity, actor_role=role,
                      target_id=target_id, status="FAILED")
    record_error("ERR_" + event_code.removeprefix("ACT_"), action_type, exc,
                 actor_id=identity, source=source, target_id=target_id)


def ensure_dashboard_admin() -> bool:
    """환경설정의 기본 계정을 ADMIN 역할로 한 번만 준비한다."""
    login_id = os.environ.get("DASHBOARD_ADMIN_ID", "").strip()[:80]
    password = os.environ.get("DASHBOARD_ADMIN_PASSWORD", "")
    if not login_id or not password:
        raise RuntimeError("DASHBOARD_ADMIN_ID와 DASHBOARD_ADMIN_PASSWORD 설정이 필요합니다.")
    with Session(get_engine()) as session, session.begin():
        existing = session.scalar(select(AdminInfo).where(AdminInfo.login_id == login_id))
        if existing is not None:
            existing.role = "ADMIN"
            return False
        session.add(AdminInfo(
            login_id=login_id, password_hash=hash_password(password), role="ADMIN",
            name_encrypted=encrypt_personal_data("Dashboard Administrator"),
            phone_encrypted=encrypt_personal_data(""), email_encrypted=encrypt_personal_data(""),
        ))
        session.add(ActionLog(
            occurred_at=kst_now(), event_code="ACT_ADMIN_BOOTSTRAPPED", action_type="기본 관리자 계정 준비",
            actor_id=login_id, actor_role="ADMIN", target_type="admin", target_id=login_id,
            action_detail="{}", result_status="SUCCESS",
        ))
    return True


def register_first_admin(setup_token: str, login_id: str, password: str, name: str,
                         phone: str, email: str) -> None:
    """기존 테스트·초기화 도구 호환용 ADMIN 계정 생성 함수다."""
    expected = os.environ.get("DASHBOARD_SETUP_TOKEN", "")
    if not expected or not hmac.compare_digest(expected, setup_token or ""):
        raise PermissionError("초기 등록 코드가 올바르지 않습니다.")
    if not login_id.strip() or not name.strip() or not email.strip():
        raise ValueError("아이디, 이름, 이메일은 필수입니다.")
    with Session(get_engine()) as session, session.begin():
        if session.scalar(select(AdminInfo.id).where(AdminInfo.login_id == login_id.strip()[:80])) is not None:
            raise ValueError("이미 사용 중인 아이디입니다.")
        session.add(AdminInfo(
            login_id=login_id.strip()[:80], password_hash=hash_password(password), role="ADMIN",
            name_encrypted=encrypt_personal_data(name.strip()),
            phone_encrypted=encrypt_personal_data(phone.strip()),
            email_encrypted=encrypt_personal_data(email.strip()),
        ))
        session.add(ActionLog(
            occurred_at=kst_now(), event_code="ACT_ADMIN_REGISTER", action_type="관리자 계정 생성",
            actor_id=login_id.strip()[:80], actor_role="ADMIN", target_type="account",
            target_id=login_id.strip()[:80], action_detail="{}", result_status="SUCCESS",
        ))


def register_account(*, role: str, login_id: str, password: str, password_confirm: str,
                     name: str, phone: str, email: str, invite_code: str = "") -> str:
    """가입 팝업의 입력값으로 USER 또는 초대 코드가 확인된 ADMIN 계정을 만든다."""
    requested_role = _normal_role(role)
    login_id = login_id.strip()[:80]
    name, phone, email = name.strip(), phone.strip(), email.strip()
    if requested_role not in {"ADMIN", "USER"}:
        raise ValueError("계정 종류가 올바르지 않습니다.")
    if not all((login_id, name, phone, email)):
        raise ValueError("이름, 이메일, 전화번호, 아이디는 모두 입력해 주세요.")
    if password != password_confirm:
        raise ValueError("비밀번호 확인이 일치하지 않습니다.")
    if len(password) < 4:
        raise ValueError("비밀번호는 4자 이상으로 입력해 주세요.")
    if requested_role == "ADMIN":
        expected = os.environ.get("DASHBOARD_ADMIN_INVITE_CODE", "")
        if not expected or not hmac.compare_digest(expected, invite_code or ""):
            raise PermissionError("관리인 코드가 올바르지 않습니다.")

    event_code = "ACT_ADMIN_REGISTER" if requested_role == "ADMIN" else "ACT_ACCOUNT_REGISTER"
    with Session(get_engine()) as session, session.begin():
        if session.scalar(select(AdminInfo.id).where(AdminInfo.login_id == login_id)) is not None:
            raise FileExistsError("이미 사용 중인 아이디입니다.")
        session.add(AdminInfo(
            login_id=login_id, password_hash=hash_password(password), role=requested_role,
            name_encrypted=encrypt_personal_data(name),
            phone_encrypted=encrypt_personal_data(phone),
            email_encrypted=encrypt_personal_data(email),
        ))
        session.add(ActionLog(
            occurred_at=kst_now(), event_code=event_code,
            action_type="관리자 계정 생성" if requested_role == "ADMIN" else "일반 계정 생성",
            actor_id=login_id, actor_role=requested_role,
            target_type="account", target_id=login_id, action_detail="{}", result_status="SUCCESS",
        ))
    return requested_role


def authenticate_account(login_id: str, password: str) -> str | None:
    """비밀번호를 검증하고 성공한 계정의 역할만 반환한다."""
    with Session(get_engine()) as session:
        row = session.scalar(select(AdminInfo).where(AdminInfo.login_id == login_id.strip()[:80]))
        if row is None or not verify_password(row.password_hash, password):
            return None
        return _normal_role(row.role)


def verify_admin(login_id: str, password: str) -> bool:
    """기존 호출부 호환용 비밀번호 검증 함수다."""
    return authenticate_account(login_id, password) is not None


def update_account_session(login_id: str, role: str, expires_at, *, online: bool) -> None:
    """로그인·연장·로그아웃 시 현재 접속 여부를 계정 표에 반영한다."""
    with Session(get_engine()) as session, session.begin():
        row = session.scalar(select(AdminInfo).where(AdminInfo.login_id == login_id).with_for_update())
        if row is None:
            return
        row.role = _normal_role(role)
        row.is_online = online
        row.session_expires_at = expires_at if online else None
        if online:
            row.last_login_at = utc_now()


def list_user_accounts() -> list[dict]:
    """관리자 화면에 표시할 일반 계정 아이디·복호화 이름·현재 접속 상태를 반환한다."""
    now = utc_now()
    with Session(get_engine()) as session, session.begin():
        rows = session.scalars(select(AdminInfo).where(AdminInfo.role == "USER").order_by(AdminInfo.login_id)).all()
        result = []
        for row in rows:
            online = bool(row.is_online and row.session_expires_at and row.session_expires_at > now)
            if row.is_online and not online:
                row.is_online = False
            result.append({"login_id": row.login_id, "name": decrypt_personal_data(row.name_encrypted),
                           "is_online": online})
        return result


def account_identity_for_session(login_id: str) -> tuple[int, str] | None:
    """계정 번호와 역할을 확인해 같은 아이디로 재가입해도 옛 세션을 구분한다."""
    if not isinstance(login_id, str) or not login_id:
        return None
    with Session(get_engine()) as db:
        identity = db.execute(select(AdminInfo.id, AdminInfo.role).where(
            AdminInfo.login_id == login_id)).one_or_none()
    return (identity.id, identity.role) if identity and identity.role in {"ADMIN", "USER"} else None


def account_role_for_session(login_id: str) -> str | None:
    """계정이 여전히 존재하고 유효한 역할인지 확인한다."""
    identity = account_identity_for_session(login_id)
    return identity[1] if identity else None


def delete_user_accounts(login_ids: list[str], actor_id: str) -> int:
    """관리자가 선택한 일반 계정만 삭제하고 이력을 같은 트랜잭션에 남긴다."""
    if not isinstance(actor_id, str) or not actor_id.strip() or len(actor_id.strip()) > 80:
        raise PermissionError("관리자 계정을 확인할 수 없습니다.")
    actor_id = actor_id.strip()
    if not isinstance(login_ids, list) or not login_ids:
        raise ValueError("삭제할 일반 계정을 선택해 주세요.")
    if any(not isinstance(value, str) or not value.strip() or len(value.strip()) > 80
           for value in login_ids):
        raise ValueError("삭제할 계정 아이디가 올바르지 않습니다.")
    targets = list(dict.fromkeys(value.strip() for value in login_ids))

    # 요청 중 호출되었다면 화면의 서명된 로그인 세션과 인자로 받은 관리자도 일치해야 한다.
    from flask import has_request_context, session as flask_session

    if has_request_context() and (flask_session.get("admin_id") != actor_id
                                  or flask_session.get("role") != "ADMIN"):
        raise PermissionError("관리자 권한이 필요합니다.")

    with Session(get_engine()) as db, db.begin():
        actor = db.scalar(select(AdminInfo).where(AdminInfo.login_id == actor_id).with_for_update())
        if actor is None or actor.role != "ADMIN":
            raise PermissionError("관리자 권한이 필요합니다.")

        rows = db.scalars(select(AdminInfo).where(AdminInfo.login_id.in_(targets)).with_for_update()).all()
        if len(rows) != len(targets):
            raise ValueError("선택한 계정을 찾을 수 없습니다.")
        if any(row.role != "USER" for row in rows):
            raise PermissionError("일반 계정만 삭제할 수 있습니다.")

        # 이전 버전의 시스템 로그 테이블에는 계정 번호를 참조하는 외래 키가 있다.
        # 로그 행 자체는 보존하고 계정 참조만 해제한다.
        if inspect(db.connection()).has_table(SystemLog.__tablename__):
            db.execute(update(SystemLog).where(SystemLog.admin_id.in_([row.id for row in rows]))
                       .values(admin_id=None))

        for row in rows:
            db.add(ActionLog(
                occurred_at=kst_now(), event_code="ACT_ACCOUNT_DELETE", action_type="일반 계정 삭제",
                actor_id=actor_id, actor_role="ADMIN", target_type="account", target_id=row.login_id,
                action_detail="{}", result_status="SUCCESS",
            ))
            db.delete(row)
    return len(rows)


def change_admin_password(login_id: str, current_password: str, new_password: str) -> bool:
    """현재 비밀번호를 확인하고 새 Argon2 해시와 성공 행동 로그를 함께 확정한다."""
    if len(new_password) < 4:
        raise ValueError("새 비밀번호는 4자 이상으로 입력해 주세요.")
    with Session(get_engine()) as session, session.begin():
        admin = session.scalar(select(AdminInfo).where(AdminInfo.login_id == login_id).with_for_update())
        if admin is None or not verify_password(admin.password_hash, current_password):
            return False
        if current_password == new_password:
            raise ValueError("현재 비밀번호와 다른 새 비밀번호를 입력해 주세요.")
        admin.password_hash = hash_password(new_password)
        admin.updated_at = utc_now()
        session.add(ActionLog(
            occurred_at=kst_now(), event_code="ACT_PASSWORD_CHANGE", action_type="비밀번호 변경",
            actor_id=login_id, actor_role=_normal_role(admin.role), target_type="account",
            target_id=login_id, action_detail="{}", result_status="SUCCESS",
        ))
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
            FailureLog.observation_date == latest, FailureLog.event_code == "FAIL_PART_OBSERVED",
        )).all())
        for row in failed.itertuples(index=False):
            key = (str(row.asset_tag), str(row.part_no))
            if key in existing:
                continue
            session.add(FailureLog(
                occurred_at=kst_now(), event_code="FAIL_PART_OBSERVED", observation_date=latest,
                plant_code=str(row.plant_code), asset_tag=key[0], part_no=key[1],
                part_family=str(row.part_family), actual_failure=True,
                failure_status="OBSERVED", source="synthetic_csv",
            ))
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
