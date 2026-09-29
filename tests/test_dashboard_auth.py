"""관리자 한 명 로그인과 서버 측 Dash 접근 차단 검증."""

import base64
import re

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool
from flask import session

from src import audit_service
from src.audit_models import ActionLog
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
    success = client.post("/login", data={"csrf_token": csrf, "login_id": "admin01",
                                           "password": "long-password-123"})
    assert success.status_code == 302
    assert client.get("/").status_code == 200
    assert client.get("/_dash-layout").status_code == 200
    profile = client.post("/_dash-update-component", json={
        "output": "profile-display.children",
        "outputs": {"id": "profile-display", "property": "children"},
        "inputs": [{"id": "screen-tabs", "property": "value", "value": "1"}],
        "state": [], "changedPropIds": ["screen-tabs.value"],
    })
    assert profile.status_code == 200
    assert profile.json["response"]["profile-display"]["children"] == "AD"
    with app.server.test_request_context("/_dash-update-component"):
        session["admin_id"] = "admin01"
        assert audit_service.record_action("ACT_TAB_OPEN", "화면 이동", actor_id="forged-id")
    with Session(engine) as db:
        rows = db.scalars(select(ActionLog).where(ActionLog.event_code == "ACT_LOGIN")
                          .order_by(ActionLog.log_id)).all()
        assert [(row.actor_id, row.result_status) for row in rows] == [
            ("ANONYMOUS", "BLOCKED"), ("admin01", "SUCCESS")]
        assert db.scalar(select(ActionLog.actor_id).where(ActionLog.event_code == "ACT_TAB_OPEN")) == "admin01"
    logout_output = next(
        key for key, spec in app.callback_map.items()
        if key.startswith("auth-redirect.href@") and spec["inputs"][0]["id"] == "logout-btn"
    )
    logout = client.post("/_dash-update-component", json={
        "output": logout_output,
        "outputs": {"id": "auth-redirect", "property": "href"},
        "inputs": [{"id": "logout-btn", "property": "n_clicks", "value": 1}],
        "state": [], "changedPropIds": ["logout-btn.n_clicks"],
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


def test_profile_password_change_callback_logs_out(monkeypatch):
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
            ("auth-redirect", "href"))],
        "inputs": [{"id": "password-save-btn", "property": "n_clicks", "value": 1}],
        "state": [{"id": component, "property": "value", "value": value} for component, value in (
            ("password-current", "long-password-123"),
            ("password-new", "new-password-123"),
            ("password-confirm", "new-password-123"))],
        "changedPropIds": ["password-save-btn.n_clicks"],
    })
    assert response.status_code == 200
    assert response.json["response"]["auth-redirect"]["href"] == "/login"
    assert client.get("/_dash-layout").status_code == 401
    assert not audit_service.verify_admin("admin01", "long-password-123")
    assert audit_service.verify_admin("admin01", "new-password-123")
