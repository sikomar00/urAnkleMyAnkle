"""계정 생성, 역할, 로그인 로그 분리와 개인정보 보호를 검증한다."""

import base64
import re
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from src import audit_service
from src.audit_models import ActionLog, LoginLog
from src.db_models import AdminInfo, Base
from src.wireframe_app import app


def _csrf(html: bytes) -> str:
    match = re.search(rb'name="csrf_token" value="([^"]+)"', html)
    assert match is not None
    return match.group(1).decode()


def _engine(monkeypatch):
    engine = create_engine("sqlite://", poolclass=StaticPool)
    Base.metadata.create_all(engine)
    monkeypatch.setattr(audit_service, "get_engine", lambda: engine)
    monkeypatch.setenv("DASHBOARD_AES_KEY", base64.b64encode(bytes(range(32))).decode())
    monkeypatch.setenv("DASHBOARD_ADMIN_INVITE_CODE", "invite-code")
    return engine


def test_registers_roles_and_encrypts_personal_data(monkeypatch):
    engine = _engine(monkeypatch)
    assert audit_service.register_account(
        role="USER", login_id="user01", password="user-password", password_confirm="user-password",
        name="일반 사용자", phone="010-0000-0000", email="user@example.com",
    ) == "USER"
    assert audit_service.register_account(
        role="ADMIN", login_id="admin02", password="admin-password", password_confirm="admin-password",
        name="두번째 관리자", phone="010-1111-1111", email="admin@example.com", invite_code="invite-code",
    ) == "ADMIN"
    with pytest.raises(PermissionError):
        audit_service.register_account(
            role="ADMIN", login_id="blocked", password="admin-password", password_confirm="admin-password",
            name="차단", phone="010", email="blocked@example.com", invite_code="wrong",
        )
    with Session(engine) as session:
        user = session.scalar(select(AdminInfo).where(AdminInfo.login_id == "user01"))
        assert user.role == "USER"
        assert "일반 사용자" not in user.name_encrypted
        assert "user-password" not in user.password_hash
        assert session.scalar(select(func.count()).select_from(ActionLog).where(
            ActionLog.event_code == "ACT_ACCOUNT_REGISTER")) == 1


def test_login_events_are_not_written_to_action_logs(monkeypatch):
    engine = _engine(monkeypatch)
    assert audit_service.record_login_event("ACT_LOGIN", actor_id="user01", actor_role="USER",
                                            source="test")
    assert audit_service.record_action("ACT_TAB_OPEN", "화면 이동", actor_id="user01", actor_role="USER")
    with Session(engine) as session:
        login = session.scalar(select(LoginLog))
        action = session.scalar(select(ActionLog))
        assert (login.actor_id, login.actor_role, login.event_code) == ("user01", "USER", "ACT_LOGIN")
        assert (action.actor_id, action.actor_role, action.event_code) == ("user01", "USER", "ACT_TAB_OPEN")
        assert session.scalar(select(func.count()).select_from(ActionLog).where(
            ActionLog.event_code == "ACT_LOGIN")) == 0


def test_account_list_returns_decrypted_name_and_online_state(monkeypatch):
    engine = _engine(monkeypatch)
    audit_service.register_account(
        role="USER", login_id="user01", password="user-password", password_confirm="user-password",
        name="일반 사용자", phone="010-0000-0000", email="user@example.com",
    )
    audit_service.update_account_session(
        "user01", "USER", datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(minutes=10), online=True
    )
    assert audit_service.list_user_accounts() == [
        {"login_id": "user01", "name": "일반 사용자", "is_online": True}
    ]


def test_register_popup_creates_user_account(monkeypatch):
    engine = _engine(monkeypatch)
    client = app.server.test_client()
    csrf = _csrf(client.get("/login").data)
    response = client.post("/register", data={
        "csrf_token": csrf, "role": "USER", "name": "가입 사용자", "email": "join@example.com",
        "phone": "010-2222-2222", "login_id": "join01", "password": "join-password",
        "password_confirm": "join-password", "invite_code": "",
    })
    assert response.status_code == 200
    assert b"register-success-modal" in response.data
    with Session(engine) as session:
        account = session.scalar(select(AdminInfo).where(AdminInfo.login_id == "join01"))
        assert account.role == "USER"


def test_user_cannot_open_report_summary_tab(monkeypatch):
    """일반 계정이 ⑤ 탭을 누르면 마지막 허용 탭으로 돌아가고 차단 로그를 남긴다."""
    engine = _engine(monkeypatch)
    client = app.server.test_client()
    with client.session_transaction() as browser_session:
        browser_session["admin_id"] = "user01"
        browser_session["role"] = "USER"
        browser_session["expires_at"] = (datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat()

    output = next(key for key in app.callback_map if key.startswith("..screen-content.children"))
    response = client.post("/_dash-update-component", json={
        "output": output,
        "outputs": [
            {"id": "screen-content", "property": "children"},
            {"id": "screen-tabs", "property": "value"},
            {"id": "last-allowed-tab-store", "property": "data"},
            {"id": "report-access-modal", "property": "style"},
        ],
        "inputs": [
            {"id": "screen-tabs", "property": "value", "value": "5"},
            {"id": "seg-store", "property": "data", "value": {}},
            {"id": "report-audience-dd", "property": "value", "value": "mgr"},
            {"id": "prio-sort-store", "property": "data", "value": "grade"},
            {"id": "selected-asset-store", "property": "data", "value": "AST-001"},
            {"id": "report-access-modal-close", "property": "n_clicks", "value": 0},
        ],
        "state": [
            {"id": "last-allowed-tab-store", "property": "data", "value": "2"},
            {"id": "actor-id-input", "property": "value", "value": ""},
        ],
        "changedPropIds": ["screen-tabs.value"],
    })

    assert response.status_code == 200
    payload = response.json["response"]
    assert payload["screen-tabs"]["value"] == "2"
    assert payload["report-access-modal"]["style"]["display"] == "flex"
    with Session(engine) as session:
        log = session.scalar(select(ActionLog).where(ActionLog.event_code == "ACT_REPORT_ACCESS_BLOCKED"))
        assert (log.actor_id, log.actor_role, log.result_status) == ("user01", "USER", "BLOCKED")
