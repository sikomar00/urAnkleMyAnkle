"""관리자 전용 일반 계정 삭제와 이미 로그인한 계정 차단을 검증한다."""

import base64
import re
from datetime import datetime, timedelta, timezone

import pytest
from flask import session as flask_session
from sqlalchemy import create_engine, event, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from src import audit_service
from src.audit_models import ActionLog, LoginLog
from src.db_models import AdminInfo, Base, SystemLog
from src.wireframe_app import app


def _database(monkeypatch):
    engine = create_engine("sqlite://", poolclass=StaticPool)
    Base.metadata.create_all(engine)
    monkeypatch.setattr(audit_service, "get_engine", lambda: engine)
    monkeypatch.setenv("DASHBOARD_AES_KEY", base64.b64encode(bytes(range(32))).decode())
    monkeypatch.setenv("DASHBOARD_ADMIN_INVITE_CODE", "test-invite")
    for role, login_id in (("ADMIN", "admin01"), ("USER", "user01"), ("USER", "user02")):
        audit_service.register_account(
            role=role, login_id=login_id, password="test-password", password_confirm="test-password",
            name=login_id, phone="010-0000-0000", email=f"{login_id}@example.com",
            invite_code="test-invite" if role == "ADMIN" else "",
        )
    return engine


def test_admin_deletes_only_users_and_preserves_audit_history(monkeypatch):
    engine = _database(monkeypatch)
    with Session(engine) as db, db.begin():
        target_id = db.scalar(select(AdminInfo.id).where(AdminInfo.login_id == "user01"))
        db.add(SystemLog(event_code="LEGACY_EVENT", outcome="success", admin_id=target_id))
        db.add(LoginLog(actor_id="user01", actor_role="USER", event_code="ACT_LOGIN",
                        result_status="SUCCESS", source="test"))

    assert audit_service.delete_user_accounts(["user01", "user02"], "admin01") == 2
    assert audit_service.account_role_for_session("user01") is None
    assert audit_service.account_role_for_session("admin01") == "ADMIN"
    with Session(engine) as db:
        assert db.scalars(select(AdminInfo.login_id)).all() == ["admin01"]
        assert db.scalar(select(SystemLog.admin_id).where(SystemLog.event_code == "LEGACY_EVENT")) is None
        assert db.scalar(select(LoginLog.actor_id).where(LoginLog.event_code == "ACT_LOGIN")) == "user01"
        rows = db.scalars(select(ActionLog).where(ActionLog.event_code == "ACT_ACCOUNT_DELETE")
                          .order_by(ActionLog.log_id)).all()
        assert [(row.actor_id, row.actor_role, row.target_id, row.result_status) for row in rows] == [
            ("admin01", "ADMIN", "user01", "SUCCESS"),
            ("admin01", "ADMIN", "user02", "SUCCESS"),
        ]
        assert all(row.action_detail == "{}" for row in rows)


def test_deletion_rejects_invalid_requests_without_partial_changes(monkeypatch):
    engine = _database(monkeypatch)
    with pytest.raises(ValueError):
        audit_service.delete_user_accounts([], "admin01")
    with pytest.raises(ValueError):
        audit_service.delete_user_accounts(["user01", "missing"], "admin01")
    with pytest.raises(PermissionError):
        audit_service.delete_user_accounts(["user01", "admin01"], "admin01")
    with pytest.raises(PermissionError):
        audit_service.delete_user_accounts(["user01"], "user02")
    with app.server.test_request_context("/"):
        flask_session["admin_id"] = "user02"
        flask_session["role"] = "USER"
        with pytest.raises(PermissionError):
            audit_service.delete_user_accounts(["user01"], "admin01")
    with Session(engine) as db:
        assert set(db.scalars(select(AdminInfo.login_id)).all()) == {"admin01", "user01", "user02"}
        assert not db.scalars(select(ActionLog).where(ActionLog.event_code == "ACT_ACCOUNT_DELETE")).all()


def test_audit_write_failure_rolls_back_account_deletion(monkeypatch):
    engine = _database(monkeypatch)
    with Session(engine) as db, db.begin():
        target_id = db.scalar(select(AdminInfo.id).where(AdminInfo.login_id == "user01"))
        db.add(SystemLog(event_code="LEGACY_EVENT", outcome="success", admin_id=target_id))

    def fail_delete_log(_conn, _cursor, statement, _parameters, _context, _executemany):
        if statement.lower().startswith("insert into action_logs"):
            raise RuntimeError("simulated audit failure")

    event.listen(engine, "before_cursor_execute", fail_delete_log)
    try:
        with pytest.raises(RuntimeError, match="simulated audit failure"):
            audit_service.delete_user_accounts(["user01"], "admin01")
    finally:
        event.remove(engine, "before_cursor_execute", fail_delete_log)
    with Session(engine) as db:
        assert db.scalar(select(AdminInfo.id).where(AdminInfo.login_id == "user01")) == target_id
        assert db.scalar(select(SystemLog.admin_id).where(SystemLog.event_code == "LEGACY_EVENT")) == target_id


def test_deletion_works_without_legacy_system_log_table(monkeypatch):
    engine = create_engine("sqlite://", poolclass=StaticPool)
    AdminInfo.__table__.create(engine)
    ActionLog.__table__.create(engine)
    monkeypatch.setattr(audit_service, "get_engine", lambda: engine)
    monkeypatch.setenv("DASHBOARD_AES_KEY", base64.b64encode(bytes(range(32))).decode())
    monkeypatch.setenv("DASHBOARD_ADMIN_INVITE_CODE", "test-invite")
    for role, login_id in (("ADMIN", "admin01"), ("USER", "user01")):
        audit_service.register_account(
            role=role, login_id=login_id, password="test-password", password_confirm="test-password",
            name=login_id, phone="010-0000-0000", email=f"{login_id}@example.com",
            invite_code="test-invite" if role == "ADMIN" else "",
        )
    assert audit_service.delete_user_accounts(["user01"], "admin01") == 1


def test_deleted_user_signed_session_is_rejected_on_next_request(monkeypatch):
    engine = _database(monkeypatch)
    client = app.server.test_client()
    with client.session_transaction() as browser_session:
        browser_session["admin_id"] = "user01"
        browser_session["role"] = "USER"
        browser_session["account_pk"] = audit_service.account_identity_for_session("user01")[0]
        browser_session["expires_at"] = (datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat()
    assert client.get("/_dash-layout").status_code == 200

    assert audit_service.delete_user_accounts(["user01"], "admin01") == 1
    assert client.get("/_dash-layout").status_code == 401
    with client.session_transaction() as browser_session:
        assert "admin_id" not in browser_session
    assert client.get("/").status_code == 302
    with Session(engine) as db:
        assert db.scalar(select(LoginLog.event_code).where(LoginLog.actor_id == "user01",
                                                       LoginLog.event_code == "ACT_SESSION_REVOKED")) == (
                                                           "ACT_SESSION_REVOKED")


def test_old_session_stays_invalid_if_same_login_id_is_registered_again(monkeypatch):
    _database(monkeypatch)
    client = app.server.test_client()
    login_page = client.get("/login")
    csrf = re.search(rb'name="csrf_token" value="([^"]+)"', login_page.data).group(1).decode()
    assert client.post("/login", data={"csrf_token": csrf, "login_id": "user01",
                                       "password": "test-password"}).status_code == 302
    with client.session_transaction() as browser_session:
        old_account_pk = browser_session["account_pk"]

    assert audit_service.delete_user_accounts(["user01"], "admin01") == 1
    audit_service.register_account(
        role="USER", login_id="user01", password="different-password",
        password_confirm="different-password", name="new user", phone="010-0000-0000",
        email="new-user@example.com",
    )
    assert audit_service.account_identity_for_session("user01")[0] != old_account_pk
    assert client.get("/_dash-layout").status_code == 401
