"""계정 생성, 역할, 로그인 로그 분리와 개인정보 보호를 검증한다."""

import base64
import re
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from src import audit_service, wireframe_app
from src.audit_models import ActionLog, ErrorLog, LoginLog
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


def _create_test_account(role: str, login_id: str) -> None:
    """세션 유효성 검증을 통과할 최소 계정을 만든다."""
    audit_service.register_account(
        role=role, login_id=login_id, password="test-password", password_confirm="test-password",
        name=f"{login_id} 이름", phone="010-0000-0000", email=f"{login_id}@example.com",
        invite_code="invite-code" if role == "ADMIN" else "",
    )


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


def _export_callback_response(client, *, export_button: str, audience: str = "mgr"):
    """PDF·Excel 버튼 콜백을 브라우저 요청과 같은 형식으로 호출한다."""
    output = next(key for key in app.callback_map if key.startswith("..report-download.data"))
    pdf_clicks = 1 if export_button == "export-pdf-btn" else 0
    excel_clicks = 1 if export_button == "export-xlsx-btn" else 0
    return client.post("/_dash-update-component", json={
        "output": output,
        "outputs": [
            {"id": "report-download", "property": "data"},
            {"id": "action-echo", "property": "children"},
            {"id": "export-access-modal", "property": "style"},
        ],
        "inputs": [
            {"id": "export-pdf-btn", "property": "n_clicks", "value": pdf_clicks},
            {"id": "export-xlsx-btn", "property": "n_clicks", "value": excel_clicks},
            {"id": "export-access-modal-close", "property": "n_clicks", "value": 0},
        ],
        "state": [
            {"id": "report-audience-dd", "property": "value", "value": audience},
            {"id": "filter-store", "property": "data", "value": {}},
            {"id": "seg-store", "property": "data", "value": {}},
            {"id": "actor-id-input", "property": "value", "value": ""},
        ],
        "changedPropIds": [f"{export_button}.n_clicks"],
    })


def test_user_cannot_export_pdf_or_excel(monkeypatch):
    """일반 계정은 직접 콜백을 호출해도 PDF·Excel 파일을 만들 수 없다."""
    engine = _engine(monkeypatch)
    _create_test_account("USER", "user01")
    monkeypatch.setattr(wireframe_app, "build_report_pdf", lambda *_args: pytest.fail("PDF를 만들면 안 됩니다."))
    monkeypatch.setattr(wireframe_app, "build_report_xlsx", lambda *_args: pytest.fail("Excel을 만들면 안 됩니다."))
    client = app.server.test_client()
    with client.session_transaction() as browser_session:
        browser_session["admin_id"] = "user01"
        browser_session["role"] = "USER"
        browser_session["expires_at"] = (datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat()

    pdf_response = _export_callback_response(client, export_button="export-pdf-btn")
    excel_response = _export_callback_response(client, export_button="export-xlsx-btn")
    assert pdf_response.status_code == 200
    assert excel_response.status_code == 200
    assert pdf_response.json["response"]["export-access-modal"]["style"]["display"] == "flex"
    assert excel_response.json["response"]["export-access-modal"]["style"]["display"] == "flex"
    with Session(engine) as session:
        logs = session.scalars(select(ActionLog).where(
            ActionLog.event_code == "ACT_EXPORT_ACCESS_BLOCKED").order_by(ActionLog.log_id)).all()
        assert [(log.actor_id, log.actor_role, log.target_id, log.result_status) for log in logs] == [
            ("user01", "USER", "pdf", "BLOCKED"),
            ("user01", "USER", "excel", "BLOCKED"),
        ]


def test_admin_can_export_excel(monkeypatch):
    """관리자 계정은 서버 측 역할 확인을 통과하고 Excel 파일을 받는다."""
    engine = _engine(monkeypatch)
    _create_test_account("ADMIN", "admin01")
    monkeypatch.setattr(wireframe_app, "build_report_xlsx", lambda *_args: b"excel-content")
    client = app.server.test_client()
    with client.session_transaction() as browser_session:
        browser_session["admin_id"] = "admin01"
        browser_session["role"] = "ADMIN"
        browser_session["expires_at"] = (datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat()

    response = _export_callback_response(client, export_button="export-xlsx-btn")
    assert response.status_code == 200
    payload = response.json["response"]
    assert payload["report-download"]["data"]["filename"].endswith(".xlsx")
    assert payload["export-access-modal"]["style"]["display"] == "none"
    with Session(engine) as session:
        log = session.scalar(select(ActionLog).where(ActionLog.event_code == "ACT_REPORT_EXPORT"))
        assert (log.actor_id, log.actor_role, log.target_type, log.target_id, log.result_status) == (
            "admin01", "ADMIN", "export", "excel", "SUCCESS")


def test_export_failure_writes_action_and_error_logs(monkeypatch):
    """관리자 내보내기 실패는 행동 로그 FAILED와 오류 로그를 함께 남긴다."""
    engine = _engine(monkeypatch)
    _create_test_account("ADMIN", "admin01")
    monkeypatch.setattr(wireframe_app, "build_report_xlsx",
                        lambda *_args: (_ for _ in ()).throw(RuntimeError("export test failure")))
    client = app.server.test_client()
    with client.session_transaction() as browser_session:
        browser_session["admin_id"] = "admin01"
        browser_session["role"] = "ADMIN"
        browser_session["expires_at"] = (datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat()

    response = _export_callback_response(client, export_button="export-xlsx-btn")
    assert response.status_code == 200
    with Session(engine) as session:
        action = session.scalar(select(ActionLog).where(ActionLog.event_code == "ACT_REPORT_EXPORT"))
        error = session.scalar(select(ErrorLog).where(ErrorLog.error_code == "ERR_REPORT_EXPORT"))
        assert (action.actor_id, action.actor_role, action.result_status) == ("admin01", "ADMIN", "FAILED")
        assert (error.actor_id, error.action_type, error.error_type) == ("admin01", "보고서 내보내기", "RuntimeError")
