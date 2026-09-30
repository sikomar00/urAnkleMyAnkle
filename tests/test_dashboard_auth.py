"""관리자 한 명 로그인과 서버 측 Dash 접근 차단 검증."""

import base64
import re
from datetime import datetime, timedelta, timezone

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool
from flask import session

from src import audit_service
from src.audit_models import ActionLog, LoginLog
from src.db_models import Base
from src.wireframe_app import app


def _csrf(html: bytes) -> str:
    match = re.search(rb'name="csrf_token" value="([^"]+)"', html)
    assert match is not None
    return match.group(1).decode()


def test_login_blocks_dash_and_uses_registered_admin(monkeypatch):
    engine = create_engine("sqlite://", poolclass=StaticPool)
    Base.metadata.create_all(engine)
    monkeypatch.setattr(audit_service, "get_engine", lambda: engine)
    monkeypatch.setenv("DASHBOARD_AES_KEY", base64.b64encode(bytes(range(32))).decode())
    monkeypatch.setenv("DASHBOARD_SETUP_TOKEN", "test-only-setup")
    audit_service.register_first_admin("test-only-setup", "admin01", "long-password-123",
                                       "테스트", "010", "test@example.com")

    client = app.server.test_client()
    assert client.get("/").status_code == 302
    assert client.get("/_dash-layout").status_code == 401
    assert client.post("/_dash-update-component", json={}).status_code == 401
    csrf = _csrf(client.get("/login").data)
    wrong = client.post("/login", data={"csrf_token": csrf, "login_id": "admin01",
                                         "password": "wrong"})
    assert wrong.status_code == 401
    assert b"login-error-modal" in wrong.data
    assert "아이디 또는 비밀번호가 올바르지 않습니다.".encode() in wrong.data
    success = client.post("/login", data={"csrf_token": csrf, "login_id": "admin01",
                                           "password": "long-password-123"})
    assert success.status_code == 302
    assert client.get("/").status_code == 200
    assert client.get("/_dash-layout").status_code == 200
    profile_output = next(key for key in app.callback_map if key.startswith("..profile-display.children"))
    profile = client.post("/_dash-update-component", json={
        "output": profile_output,
        "outputs": [{"id": component, "property": property_name} for component, property_name in (
            ("profile-display", "children"), ("profile-login-id", "children"),
                ("session-state", "data"), ("account-manage-btn", "style"))],
        "inputs": [{"id": "screen-tabs", "property": "value", "value": "1"}],
        "state": [], "changedPropIds": ["screen-tabs.value"],
    })
    assert profile.status_code == 200
    assert profile.json["response"]["profile-display"]["children"] == "AD"
    with app.server.test_request_context("/_dash-update-component"):
        session["admin_id"] = "admin01"
        assert audit_service.record_action("ACT_TAB_OPEN", "화면 이동", actor_id="forged-id")
    with Session(engine) as db:
        rows = db.scalars(select(LoginLog).where(LoginLog.event_code.in_(("ACT_LOGIN", "ACT_LOGIN_BLOCKED")))
                          .order_by(LoginLog.log_id)).all()
        assert [(row.actor_id, row.result_status) for row in rows] == [
            ("admin01", "BLOCKED"), ("admin01", "SUCCESS")]
        assert db.scalar(select(ActionLog.actor_id).where(ActionLog.event_code == "ACT_TAB_OPEN")) == "admin01"
    logout_open_output = next(
        key for key, spec in app.callback_map.items()
        if key.startswith("logout-confirm-modal.style")
    )
    logout_open = client.post("/_dash-update-component", json={
        "output": logout_open_output,
        "outputs": {"id": "logout-confirm-modal", "property": "style"},
        "inputs": [{"id": "logout-btn", "property": "n_clicks", "value": 1},
                   {"id": "logout-confirm-no-btn", "property": "n_clicks", "value": 0}],
        "state": [], "changedPropIds": ["logout-btn.n_clicks"],
    })
    assert logout_open.status_code == 200
    assert client.get("/_dash-layout").status_code == 200
    logout_output = next(
        key for key, spec in app.callback_map.items()
        if key.startswith("auth-redirect.href@") and spec["inputs"][0]["id"] == "logout-confirm-yes-btn"
    )
    logout = client.post("/_dash-update-component", json={
        "output": logout_output,
        "outputs": {"id": "auth-redirect", "property": "href"},
        "inputs": [{"id": "logout-confirm-yes-btn", "property": "n_clicks", "value": 1},
                   {"id": "session-modal-logout-btn", "property": "n_clicks", "value": 0},
                   {"id": "password-success-confirm-btn", "property": "n_clicks", "value": 0}],
        "state": [], "changedPropIds": ["logout-confirm-yes-btn.n_clicks"],
    })
    assert logout.status_code == 200
    assert client.get("/_dash-layout").status_code == 401


def test_password_change_rehashes_and_does_not_log_secrets(monkeypatch):
    engine = create_engine("sqlite://", poolclass=StaticPool)
    Base.metadata.create_all(engine)
    monkeypatch.setattr(audit_service, "get_engine", lambda: engine)
    monkeypatch.setenv("DASHBOARD_AES_KEY", base64.b64encode(bytes(range(32))).decode())
    monkeypatch.setenv("DASHBOARD_SETUP_TOKEN", "test-only-setup")
    audit_service.register_first_admin("test-only-setup", "admin01", "long-password-123",
                                       "테스트", "010", "test@example.com")
    assert not audit_service.change_admin_password("admin01", "wrong", "new-password-123")
    assert audit_service.change_admin_password("admin01", "long-password-123", "new-password-123")
    assert not audit_service.verify_admin("admin01", "long-password-123")
    assert audit_service.verify_admin("admin01", "new-password-123")
    with Session(engine) as db:
        row = db.scalar(select(ActionLog).where(ActionLog.event_code == "ACT_PASSWORD_CHANGE"))
        assert row.actor_id == "admin01"
        assert "new-password-123" not in row.action_detail


def test_session_extend_and_expiry_are_logged_once(monkeypatch):
    engine = create_engine("sqlite://", poolclass=StaticPool)
    Base.metadata.create_all(engine)
    monkeypatch.setattr(audit_service, "get_engine", lambda: engine)
    monkeypatch.setenv("DASHBOARD_AES_KEY", base64.b64encode(bytes(range(32))).decode())
    monkeypatch.setenv("DASHBOARD_SETUP_TOKEN", "test-only-setup")
    audit_service.register_first_admin("test-only-setup", "admin01", "long-password-123",
                                       "테스트", "010", "test@example.com")
    client = app.server.test_client()
    csrf = _csrf(client.get("/login").data)
    assert client.post("/login", data={"csrf_token": csrf, "login_id": "admin01",
                                        "password": "long-password-123"}).status_code == 302
    with client.session_transaction() as browser_session:
        browser_session["expires_at"] = (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()
        before_extend = datetime.fromisoformat(browser_session["expires_at"])

    extend_output = next(
        key for key, spec in app.callback_map.items()
        if any(item["id"] == "session-extend-btn" for item in spec["inputs"])
    )
    response = client.post("/_dash-update-component", json={
        "output": extend_output,
        "outputs": [{"id": component, "property": property_name} for component, property_name in (
            ("session-state", "data"), ("session-prompt-state", "data"),
            ("session-expiry-modal", "style"))],
        "inputs": [{"id": "session-extend-btn", "property": "n_clicks", "value": 1},
                   {"id": "session-modal-extend-btn", "property": "n_clicks", "value": 0}],
        "state": [], "changedPropIds": ["session-extend-btn.n_clicks"],
    })
    assert response.status_code == 200
    with client.session_transaction() as browser_session:
        assert datetime.fromisoformat(browser_session["expires_at"]) > before_extend
        browser_session["expires_at"] = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()

    assert client.get("/").status_code == 302
    assert client.get("/_dash-layout").status_code == 401
    with Session(engine) as db:
        codes = db.scalars(select(LoginLog.event_code).order_by(LoginLog.log_id)).all()
        assert codes.count("ACT_SESSION_EXTEND") == 1
        assert codes.count("ACT_SESSION_EXPIRED") == 1


def test_profile_password_change_shows_completion_then_logs_out(monkeypatch):
    engine = create_engine("sqlite://", poolclass=StaticPool)
    Base.metadata.create_all(engine)
    monkeypatch.setattr(audit_service, "get_engine", lambda: engine)
    monkeypatch.setenv("DASHBOARD_AES_KEY", base64.b64encode(bytes(range(32))).decode())
    monkeypatch.setenv("DASHBOARD_SETUP_TOKEN", "test-only-setup")
    audit_service.register_first_admin("test-only-setup", "admin01", "long-password-123",
                                       "테스트", "010", "test@example.com")
    client = app.server.test_client()
    csrf = _csrf(client.get("/login").data)
    assert client.post("/login", data={"csrf_token": csrf, "login_id": "admin01",
                                       "password": "long-password-123"}).status_code == 302
    callback_output = next(key for key in app.callback_map if key.startswith("..password-result.children"))
    response = client.post("/_dash-update-component", json={
        "output": callback_output,
        "outputs": [{"id": component, "property": property_name} for component, property_name in (
            ("password-result", "children"), ("password-current", "value"),
            ("password-new", "value"), ("password-confirm", "value"),
            ("password-success-modal", "style"))],
        "inputs": [{"id": "password-save-btn", "property": "n_clicks", "value": 1}],
        "state": [{"id": component, "property": "value", "value": value} for component, value in (
            ("password-current", "long-password-123"),
            ("password-new", "new-password-123"),
            ("password-confirm", "new-password-123"))],
        "changedPropIds": ["password-save-btn.n_clicks"],
    })
    assert response.status_code == 200
    assert response.json["response"]["password-success-modal"]["style"]["display"] == "flex"
    assert client.get("/_dash-layout").status_code == 200
    assert not audit_service.verify_admin("admin01", "long-password-123")
    assert audit_service.verify_admin("admin01", "new-password-123")
    logout_output = next(
        key for key, spec in app.callback_map.items()
        if key.startswith("auth-redirect.href@") and spec["inputs"][0]["id"] == "logout-confirm-yes-btn"
    )
    confirmed = client.post("/_dash-update-component", json={
        "output": logout_output,
        "outputs": {"id": "auth-redirect", "property": "href"},
        "inputs": [{"id": "logout-confirm-yes-btn", "property": "n_clicks", "value": 0},
                   {"id": "session-modal-logout-btn", "property": "n_clicks", "value": 0},
                   {"id": "password-success-confirm-btn", "property": "n_clicks", "value": 1}],
        "state": [], "changedPropIds": ["password-success-confirm-btn.n_clicks"],
    })
    assert confirmed.status_code == 200
    assert client.get("/_dash-layout").status_code == 401
