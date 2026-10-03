"""실제 testdb를 건드리지 않고 SQLite 메모리 DB로 저장 규칙을 확인한다."""

import base64
from datetime import date

from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from src import db_service
from src.db_models import AdminInfo, Base, FailureAlert, SystemLog
from src.security_service import decrypt_personal_data, encrypt_personal_data, hash_password, verify_password


def test_hash_and_encryption_are_different(monkeypatch):
    monkeypatch.setenv("DASHBOARD_AES_KEY", base64.b64encode(bytes(range(32))).decode())
    hashed = hash_password("a-long-test-password")
    assert "a-long-test-password" not in hashed
    assert verify_password(hashed, "a-long-test-password")
    assert not verify_password(hashed, "incorrect-password")
    first = encrypt_personal_data("홍길동")
    second = encrypt_personal_data("홍길동")
    assert first != second  # 매 암호화마다 새 nonce 사용
    assert decrypt_personal_data(first) == "홍길동"


def test_alert_dedup_and_admin_permissions(monkeypatch):
    engine = create_engine("sqlite://", poolclass=StaticPool)
    Base.metadata.create_all(engine)
    monkeypatch.setattr(db_service, "get_engine", lambda: engine)
    monkeypatch.setenv("DASHBOARD_AES_KEY", base64.b64encode(bytes(range(32))).decode())

    rows = [{"asset_tag": "AST-1", "failure_points": 17}]
    first = db_service.save_observed_alerts(date(2025, 1, 1), rows, 12)
    second = db_service.save_observed_alerts(date(2025, 1, 1), rows, 12)
    assert (first["created"], second["created"], second["existing"]) == (1, 0, 1)
    any_failure = db_service.save_observed_alerts(
        date(2025, 1, 1), rows, 1,
        rule_version=db_service.ANY_FAILURE_RULE_VERSION,
        event_code=db_service.ANY_FAILURE_EVENT_CODE,
    )
    assert any_failure["created"] == 1

    db_service.create_first_admin("admin01", "a-long-test-password", "홍길동", "010", "test@example.com")
    assert db_service.read_admin_info("admin01", "wrong") is None
    assert db_service.read_admin_info("admin01", "a-long-test-password")["name"] == "홍길동"
    with Session(engine) as session:
        admin = session.scalar(select(AdminInfo))
        assert admin.name_encrypted != "홍길동"
        assert "a-long-test-password" not in admin.password_hash
        alert_id = session.scalar(select(FailureAlert.id))
    assert not db_service.review_alert(alert_id, "admin01", "wrong")
    assert db_service.review_alert(alert_id, "admin01", "a-long-test-password")
    assert db_service.update_admin_info("admin01", "a-long-test-password", "새 이름", "", "", "")
    assert db_service.read_admin_info("admin01", "a-long-test-password")["name"] == "새 이름"
    with Session(engine) as session:
        assert session.scalar(select(func.count()).select_from(FailureAlert)) == 2
        assert session.get(FailureAlert, alert_id).status == "reviewed"
        assert session.scalar(select(func.count()).select_from(SystemLog)) >= 4


def test_four_operational_screens_remain_available():
    from src import wireframe_app

    assert len(wireframe_app.SCREENS) == 4
    assert set(wireframe_app.SCREEN_BUILDERS) == {"1", "2", "3", "4"}
    client = wireframe_app.app.server.test_client()
    assert client.get("/").status_code == 302
    assert client.get("/login").status_code == 200
    assert client.get("/_dash-layout").status_code == 401
