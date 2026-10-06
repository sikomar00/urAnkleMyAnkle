"""실제 MySQL을 건드리지 않는 로그 저장·암호화 검증."""

import base64
from datetime import datetime, timedelta, timezone

from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from src import audit_service
from src.audit_models import ActionLog, ErrorLog, FailureLog, kst_now
from src.db_models import AdminInfo, Base


def test_three_log_tables_and_admin_security(monkeypatch, tmp_path):
    engine = create_engine("sqlite://", poolclass=StaticPool)
    Base.metadata.create_all(engine)
    monkeypatch.setattr(audit_service, "get_engine", lambda: engine)
    monkeypatch.setenv("DASHBOARD_AES_KEY", base64.b64encode(bytes(range(32))).decode())
    monkeypatch.setenv("DASHBOARD_SETUP_TOKEN", "test-setup-code")

    assert audit_service.record_action("ACT_TAB_OPEN", "화면 이동", actor_id="KYS01",
                                       target_type="tab", target_id="2")
    assert audit_service.record_error("ERR_REPORT_EXPORT", "보고서", ValueError("secret"),
                                      source="test")
    audit_service.register_first_admin("test-setup-code", "admin01", "long-password-123",
                                       "홍길동", "010-0000-0000", "test@example.com")
    assert audit_service.verify_admin("admin01", "long-password-123")
    assert not audit_service.verify_admin("admin01", "wrong")
    with Session(engine) as session:
        assert session.scalar(select(func.count()).select_from(ActionLog)) == 2
        logged_at = session.scalar(select(ActionLog.occurred_at).where(ActionLog.event_code == "ACT_TAB_OPEN"))
        expected = datetime.now(timezone(timedelta(hours=9))).replace(tzinfo=None)
        assert abs((logged_at - expected).total_seconds()) < 5
        assert session.scalar(select(func.count()).select_from(ErrorLog)) == 1
        assert "secret" not in session.scalar(select(ErrorLog.error_message))
        admin = session.scalar(select(AdminInfo))
        assert "long-password-123" not in admin.password_hash
        assert "홍길동" not in admin.name_encrypted


def test_log_timestamp_uses_korean_local_time():
    expected = datetime.now(timezone(timedelta(hours=9))).replace(tzinfo=None)
    assert abs((kst_now() - expected).total_seconds()) < 2


def test_default_dashboard_admin_is_created_once(monkeypatch):
    engine = create_engine("sqlite://", poolclass=StaticPool)
    Base.metadata.create_all(engine)
    monkeypatch.setattr(audit_service, "get_engine", lambda: engine)
    monkeypatch.setenv("DASHBOARD_AES_KEY", base64.b64encode(bytes(range(32))).decode())
    monkeypatch.setenv("DASHBOARD_ADMIN_ID", "ankles")
    monkeypatch.setenv("DASHBOARD_ADMIN_PASSWORD", "1234")

    assert audit_service.ensure_dashboard_admin() is True
    assert audit_service.ensure_dashboard_admin() is False
    assert audit_service.verify_admin("ankles", "1234")


def test_observed_failure_sync_is_idempotent(monkeypatch, tmp_path):
    engine = create_engine("sqlite://", poolclass=StaticPool)
    Base.metadata.create_all(engine)
    monkeypatch.setattr(audit_service, "get_engine", lambda: engine)
    csv = tmp_path / "sample.csv"
    csv.write_text(
        "transaction_date,plant_code,asset_tag,part_no,part_family,breakdown_flag\n"
        "2025-01-01,P1,A1,B1,Bearing,0\n"
        "2025-01-02,P1,A1,B1,Bearing,1\n"
        "2025-01-02,P1,A1,B2,Belt,1\n", encoding="utf-8")
    monkeypatch.setattr(audit_service, "_data_path", lambda: csv)
    monkeypatch.setattr(audit_service, "_last_synced_file", None)
    assert audit_service.sync_if_csv_changed()["created"] == 2
    assert audit_service.sync_if_csv_changed() == {"unchanged": True}
    assert audit_service.sync_observed_failures()["created"] == 0
    with Session(engine) as session:
        assert session.scalar(select(func.count()).select_from(FailureLog)) == 2
