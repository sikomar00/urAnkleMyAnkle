"""대시보드 전용 테이블 저장·조회. 화면 코드는 이 함수들만 호출한다."""

from __future__ import annotations

import os
import threading
from datetime import date
from functools import lru_cache
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv
from sqlalchemy import create_engine, desc, func, select, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session

from .db_models import AdminInfo, FailureAlert, SystemLog, utc_now
from .security_service import (
    decrypt_personal_data,
    encrypt_personal_data,
    hash_password,
    verify_password,
)

load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)

RULE_VERSION = "severity12-v1"
ALERT_EVENT_CODE = "EQUIP_SEVERITY_THRESHOLD_EXCEEDED"
ANY_FAILURE_RULE_VERSION = "any-part-failure-v1"
ANY_FAILURE_EVENT_CODE = "EQUIP_PART_FAILURE_OBSERVED"
_failure_count = 0
_failure_lock = threading.Lock()


def record_db_failure() -> None:
    """DB 자체가 죽어 로그를 쓰지 못할 때의 현재 앱 프로세스 내 실패 건수."""
    global _failure_count
    with _failure_lock:
        _failure_count += 1


def local_failure_count() -> int:
    with _failure_lock:
        return _failure_count


@lru_cache(maxsize=2)
def _engine_for_url(url: str):
    parsed = make_url(url)
    if parsed.drivername != "mysql+pymysql" or parsed.database != "predictive_maintenance":
        raise RuntimeError("MACHINE_DATABASE_URL은 mysql+pymysql 형식의 predictive_maintenance를 가리켜야 합니다.")
    return create_engine(url, pool_pre_ping=True, pool_recycle=1800, connect_args={"connect_timeout": 4})


def get_engine():
    url = os.environ.get("MACHINE_DATABASE_URL", "")
    if not url:
        raise RuntimeError("MACHINE_DATABASE_URL이 설정되지 않았습니다.")
    return _engine_for_url(url)


def create_owned_tables() -> None:
    """현재 남기는 관리자 계정 테이블만 생성한다.

    이전 시범용 경보·시스템 로그 테이블은 더 이상 만들지 않는다.
    """
    AdminInfo.__table__.create(get_engine(), checkfirst=True)


def db_state() -> dict:
    """현재 사용하는 DB 연결 상태만 화면용 값으로 변환한다.

    이전 시범용 dashboard_failure_alerts·dashboard_system_logs는 삭제됐으므로
    이 함수에서 해당 테이블을 조회하지 않는다.
    """
    try:
        with get_engine().connect() as connection:
            connection.execute(text("SELECT 1"))
        return {"connected": True, "message": "연결됨", "unreviewed": None,
                "saved_today": None, "recent": None, "failures": local_failure_count()}
    except Exception:
        return {"connected": False, "message": "연결 또는 테이블 확인 필요", "unreviewed": None,
                "saved_today": None, "recent": None, "failures": local_failure_count()}


def list_alerts(limit: int = 100) -> list[dict]:
    with Session(get_engine()) as session:
        alerts = session.scalars(select(FailureAlert).order_by(desc(FailureAlert.created_at), desc(FailureAlert.id)).limit(limit)).all()
        return [{"id": a.id, "observed_on": a.observed_on.isoformat(), "asset_tag": a.asset_tag,
                 "failure_points": a.failure_points, "threshold": a.threshold, "status": a.status,
                 "condition_label": ("부품 1개 이상 고장 표시" if a.rule_version == ANY_FAILURE_RULE_VERSION
                                     else "심각도 12점 이상" if a.rule_version == RULE_VERSION
                                     else a.rule_version),
                 "status_label": "미확인" if a.status == "unreviewed" else "확인됨",
                 "event_code": a.event_code, "source": a.source, "rule_version": a.rule_version,
                 "created_at": a.created_at.isoformat(sep=" ", timespec="seconds")}
                for a in alerts]


def list_system_logs(limit: int = 100) -> list[dict]:
    with Session(get_engine()) as session:
        logs = session.scalars(select(SystemLog).order_by(desc(SystemLog.id)).limit(limit)).all()
        return [{"id": row.id, "occurred_at": row.occurred_at.isoformat(sep=" ", timespec="seconds"),
                 "event_code": row.event_code, "outcome": row.outcome,
                 "outcome_label": {"success": "성공", "denied": "거부", "failure": "실패"}.get(row.outcome, row.outcome),
                 "asset_tag": row.asset_tag or "—", "detail": row.detail or "—"} for row in logs]


def try_log_system_event(event_code: str, outcome: str, detail: str | None = None) -> bool:
    """기존 화면 기능을 막지 않는 부가 로그. 실패하면 현재 프로세스 실패 건수에 반영."""
    try:
        with Session(get_engine()) as session, session.begin():
            session.add(SystemLog(event_code=event_code, outcome=outcome, detail=detail))
        return True
    except Exception:
        record_db_failure()
        return False


def save_observed_alerts(observed_on: date, rows: list[dict], threshold: int,
                         rule_version: str = RULE_VERSION,
                         event_code: str = ALERT_EVENT_CODE) -> dict:
    """최신 CSV 관측일의 설비 경보를 저장한다. 원본 관측일과 저장 시각은 별개다."""
    run_id = str(uuid4())
    created = 0
    existing = 0
    try:
        with Session(get_engine()) as session, session.begin():
            for row in rows:
                asset = str(row["asset_tag"])
                old = session.scalar(select(FailureAlert).where(
                    FailureAlert.observed_on == observed_on,
                    FailureAlert.asset_tag == asset,
                    FailureAlert.rule_version == rule_version,
                ))
                if old is not None:
                    existing += 1
                    continue
                alert = FailureAlert(observed_on=observed_on, asset_tag=asset,
                                     failure_points=int(row["failure_points"]), threshold=threshold,
                                     rule_version=rule_version, event_code=event_code)
                session.add(alert)
                session.flush()
                session.add(SystemLog(event_code="ALERT_CREATED", outcome="success", run_id=run_id,
                                      asset_tag=asset, alert_id=alert.id,
                                      detail=(f"observed_on={observed_on.isoformat()}, rule={rule_version}, "
                                              f"score={alert.failure_points}")))
                created += 1
            session.add(SystemLog(event_code="ALERT_SCAN_FINISHED", outcome="success", run_id=run_id,
                                  detail=(f"observed_on={observed_on.isoformat()}, rule={rule_version}, "
                                          f"created={created}, existing={existing}")))
    except Exception:
        record_db_failure()
        raise
    return {"created": created, "existing": existing, "observed_on": observed_on.isoformat(), "run_id": run_id}


def _admin(session: Session, login_id: str) -> AdminInfo | None:
    return session.scalar(select(AdminInfo).where(AdminInfo.login_id == login_id))


def create_first_admin(login_id: str, password: str, name: str, phone: str, email: str) -> None:
    """초기 관리자 1명은 대시보드 밖에서만 생성한다."""
    if not login_id.strip() or not name.strip():
        raise ValueError("관리자 아이디와 이름은 필수입니다.")
    password_hash = hash_password(password)
    encrypted = [encrypt_personal_data(value.strip()) for value in (name, phone, email)]
    with Session(get_engine()) as session, session.begin():
        if session.scalar(select(func.count()).select_from(AdminInfo)):
            raise ValueError("관리자가 이미 있습니다. 최초 등록은 한 번만 가능합니다.")
        admin = AdminInfo(login_id=login_id.strip(), password_hash=password_hash,
                          name_encrypted=encrypted[0], phone_encrypted=encrypted[1], email_encrypted=encrypted[2])
        session.add(admin)
        session.flush()
        session.add(SystemLog(event_code="ADMIN_CREATED", outcome="success", admin_id=admin.id))


def read_admin_info(login_id: str, password: str) -> dict | None:
    """매 조회마다 비밀번호를 검사한다. 브라우저의 상태 저장소에 인증 정보를 넣지 않는다."""
    with Session(get_engine()) as session, session.begin():
        admin = _admin(session, login_id.strip())
        if admin is None or not verify_password(admin.password_hash, password):
            session.add(SystemLog(event_code="ADMIN_AUTH_FAILED", outcome="denied"))
            return None
        details = {"name": decrypt_personal_data(admin.name_encrypted),
                   "phone": decrypt_personal_data(admin.phone_encrypted),
                   "email": decrypt_personal_data(admin.email_encrypted)}
        session.add(SystemLog(event_code="ADMIN_INFO_VIEWED", outcome="success", admin_id=admin.id))
        return details


def update_admin_info(login_id: str, password: str, name: str, phone: str, email: str,
                      new_password: str = "") -> bool:
    if not name.strip():
        raise ValueError("관리자 이름은 비울 수 없습니다.")
    # 검증에 실패해도 개인정보 원문이나 입력 비밀번호는 로그에 남기지 않는다.
    with Session(get_engine()) as session, session.begin():
        admin = _admin(session, login_id.strip())
        if admin is None or not verify_password(admin.password_hash, password):
            session.add(SystemLog(event_code="ADMIN_AUTH_FAILED", outcome="denied"))
            return False
        admin.name_encrypted = encrypt_personal_data(name.strip())
        admin.phone_encrypted = encrypt_personal_data(phone.strip())
        admin.email_encrypted = encrypt_personal_data(email.strip())
        if new_password:
            admin.password_hash = hash_password(new_password)
        session.add(SystemLog(event_code="ADMIN_INFO_UPDATED", outcome="success", admin_id=admin.id))
        return True


def review_alert(alert_id: int, login_id: str, password: str) -> bool:
    with Session(get_engine()) as session, session.begin():
        admin = _admin(session, login_id.strip())
        if admin is None or not verify_password(admin.password_hash, password):
            session.add(SystemLog(event_code="ADMIN_AUTH_FAILED", outcome="denied"))
            return False
        alert = session.get(FailureAlert, alert_id)
        if alert is None:
            raise ValueError("선택한 경보를 찾을 수 없습니다.")
        if alert.status != "reviewed":
            alert.status = "reviewed"
            alert.reviewed_at = utc_now()
            session.add(SystemLog(event_code="ALERT_REVIEWED", outcome="success", admin_id=admin.id,
                                  alert_id=alert.id, asset_tag=alert.asset_tag))
        return True
