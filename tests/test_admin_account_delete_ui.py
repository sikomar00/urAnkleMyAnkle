"""관리자 계정 관리 화면의 삭제 선택·확인 흐름을 검증한다."""

import base64
from datetime import datetime, timedelta, timezone

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from src import audit_service
from src.audit_models import ActionLog
from src.db_models import AdminInfo, Base
from src.wireframe_app import app


def _setup(monkeypatch):
    engine = create_engine("sqlite://", poolclass=StaticPool)
    Base.metadata.create_all(engine)
    monkeypatch.setattr(audit_service, "get_engine", lambda: engine)
    monkeypatch.setenv("DASHBOARD_AES_KEY", base64.b64encode(bytes(range(32))).decode())
    monkeypatch.setenv("DASHBOARD_ADMIN_INVITE_CODE", "test-invite")
    for role, login_id in (("ADMIN", "admin01"), ("USER", "user01"), ("USER", "user02")):
        audit_service.register_account(
            role=role, login_id=login_id, password="test-password", password_confirm="test-password",
            name=f"{login_id} 이름", phone="010-0000-0000", email=f"{login_id}@example.com",
            invite_code="test-invite" if role == "ADMIN" else "",
        )
    return engine


def _client(role: str, login_id: str):
    client = app.server.test_client()
    account_id, stored_role = audit_service.account_identity_for_session(login_id)
    assert stored_role == role
    with client.session_transaction() as browser_session:
        browser_session["admin_id"] = login_id
        browser_session["role"] = role
        browser_session["account_pk"] = account_id
        browser_session["expires_at"] = (datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat()
    return client


def _callback(client, prefix, outputs, inputs, state, changed):
    output = next(key for key in app.callback_map if key.startswith(prefix))
    output_items = [{"id": component, "property": prop} for component, prop in outputs]
    return client.post("/_dash-update-component", json={
        "output": output,
        "outputs": output_items if len(output_items) > 1 else output_items[0],
        "inputs": [{"id": component, "property": prop, "value": value} for component, prop, value in inputs],
        "state": [{"id": component, "property": prop, "value": value} for component, prop, value in state],
        "changedPropIds": [changed],
    })


DELETE_OUTPUTS = (
    ("account-delete-confirm-modal", "style"),
    ("account-delete-confirm-message", "children"),
    ("account-delete-refresh", "data"),
    ("account-delete-feedback", "children"),
)


def _component_ids(value):
    if isinstance(value, dict):
        props = value.get("props", {})
        if isinstance(props.get("id"), str):
            yield props["id"]
        for child in value.values():
            yield from _component_ids(child)
    elif isinstance(value, list):
        for child in value:
            yield from _component_ids(child)


def _delete_callback(client, *, clicked, selected, refresh=0):
    clicks = {"account-delete-confirm-btn": 0, "account-delete-no-btn": 0,
              "account-delete-yes-btn": 0, "account-manage-btn": 0,
              "account-management-close-btn": 0}
    clicks[clicked] = 1
    return _callback(
        client, "..account-delete-confirm-modal.style", DELETE_OUTPUTS,
        [(component, "n_clicks", value) for component, value in clicks.items()],
        [("account-delete-selection", "value", selected),
         ("account-delete-refresh", "data", refresh)],
        f"{clicked}.n_clicks",
    )


def test_account_delete_controls_render_without_replacing_dashboard(monkeypatch):
    _setup(monkeypatch)
    client = _client("ADMIN", "admin01")
    layout = client.get("/_dash-layout")
    dependencies = client.get("/_dash-dependencies")
    assert layout.status_code == 200
    assert dependencies.status_code == 200
    ids = set(_component_ids(layout.json))
    assert {"screen-tabs", "session-timer", "account-manage-btn", "account-delete-mode-btn",
            "account-delete-selection", "account-delete-confirm-modal"} <= ids

    opened = _callback(
        client, "..account-management-modal.style",
        [("account-management-modal", "style"), ("account-list-body", "children")],
        [("account-manage-btn", "n_clicks", 1),
         ("account-management-close-btn", "n_clicks", 0),
         ("account-delete-refresh", "data", 0)],
        [], "account-manage-btn.n_clicks",
    )
    assert opened.status_code == 200
    assert opened.json["response"]["account-management-modal"]["style"]["display"] == "flex"

    mode = _callback(
        client, "..account-delete-mode.data",
        [("account-delete-mode", "data"), ("account-delete-selection", "value")],
        [("account-manage-btn", "n_clicks", 1),
         ("account-management-close-btn", "n_clicks", 0),
         ("account-delete-mode-btn", "n_clicks", 1),
         ("account-delete-refresh", "data", 0)],
        [("account-delete-mode", "data", False)], "account-delete-mode-btn.n_clicks",
    )
    assert mode.status_code == 200
    assert mode.json["response"]["account-delete-mode"]["data"] is True

    rendered = _callback(
        client, "..account-list-body.style",
        [("account-list-body", "style"), ("account-delete-header", "style"),
         ("account-delete-selection", "style"), ("account-delete-confirm-btn", "style"),
         ("account-delete-mode-btn", "children")],
        [("account-delete-mode", "data", True)], [], "account-delete-mode.data",
    )
    assert rendered.status_code == 200
    assert rendered.json["response"]["account-delete-selection"]["style"]["display"] == "block"
    assert rendered.json["response"]["account-delete-confirm-btn"]["style"]["display"] == "inline-block"


def test_admin_delete_requires_selection_and_second_confirmation(monkeypatch):
    engine = _setup(monkeypatch)
    client = _client("ADMIN", "admin01")

    choices = _callback(
        client, "account-delete-selection.options",
        [("account-delete-selection", "options")],
        [("account-manage-btn", "n_clicks", 1), ("account-delete-refresh", "data", 0)],
        [], "account-manage-btn.n_clicks",
    )
    assert choices.status_code == 200
    assert [row["value"] for row in choices.json["response"]["account-delete-selection"]["options"]] == [
        "user01", "user02"]

    empty = _delete_callback(client, clicked="account-delete-confirm-btn", selected=[])
    assert empty.status_code == 200
    assert empty.json["response"]["account-delete-feedback"]["children"] == "삭제할 계정을 선택해 주세요."

    confirm = _delete_callback(client, clicked="account-delete-confirm-btn", selected=["user01"])
    assert confirm.status_code == 200
    assert confirm.json["response"]["account-delete-confirm-modal"]["style"]["display"] == "flex"
    with Session(engine) as db:
        assert db.scalar(select(AdminInfo.id).where(AdminInfo.login_id == "user01")) is not None

    cancelled = _delete_callback(client, clicked="account-delete-no-btn", selected=["user01"])
    assert cancelled.status_code == 200
    assert cancelled.json["response"]["account-delete-confirm-modal"]["style"]["display"] == "none"
    with Session(engine) as db:
        assert db.scalar(select(AdminInfo.id).where(AdminInfo.login_id == "user01")) is not None

    skipped = _delete_callback(client, clicked="account-delete-yes-btn", selected=["user01"])
    assert skipped.status_code == 200
    assert "account-delete-refresh" not in skipped.json["response"]
    with Session(engine) as db:
        assert db.scalar(select(AdminInfo.id).where(AdminInfo.login_id == "user01")) is not None

    confirm = _delete_callback(client, clicked="account-delete-confirm-btn", selected=["user01"])
    assert confirm.status_code == 200
    completed = _delete_callback(client, clicked="account-delete-yes-btn", selected=["user01"])
    assert completed.status_code == 200
    assert completed.json["response"]["account-delete-refresh"]["data"] == 1
    with Session(engine) as db:
        assert db.scalar(select(AdminInfo.id).where(AdminInfo.login_id == "user01")) is None
        assert db.scalar(select(AdminInfo.id).where(AdminInfo.login_id == "user02")) is not None
        assert db.scalar(select(AdminInfo.id).where(AdminInfo.login_id == "admin01")) is not None
        log = db.scalar(select(ActionLog).where(ActionLog.event_code == "ACT_ACCOUNT_DELETE"))
        assert (log.actor_id, log.actor_role, log.result_status) == ("admin01", "ADMIN", "SUCCESS")


def test_user_cannot_delete_via_direct_callback(monkeypatch):
    engine = _setup(monkeypatch)
    client = _client("USER", "user01")
    response = _delete_callback(client, clicked="account-delete-yes-btn", selected=["user02"])
    assert response.status_code == 200
    assert response.json["response"]["account-delete-feedback"]["children"] == "접근 권한이 필요합니다."
    with Session(engine) as db:
        assert db.scalar(select(AdminInfo.id).where(AdminInfo.login_id == "user02")) is not None
