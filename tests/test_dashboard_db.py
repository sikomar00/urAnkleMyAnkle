"""실제 testdb를 건드리지 않고 SQLite 메모리 DB로 저장 규칙을 확인한다."""

import base64
from datetime import date

from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from src import alert_service, db_service
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


def test_auto_scan_runs_only_when_csv_changes(tmp_path, monkeypatch):
    import pandas as pd

    csv_path = tmp_path / "observations.csv"
    csv_path.write_text("first", encoding="utf-8")
    monkeypatch.setattr(alert_service, "_data_path", lambda: csv_path)
    daily = pd.DataFrame({
        "transaction_date": pd.to_datetime(["2025-01-01"] * 3),
        "asset_tag": ["AST-1", "AST-2", "AST-3"],
        "failure_points": [0, 4, 17],
    })

    def fake_daily():
        return daily

    fake_daily.cache_clear = lambda: None

    def fake_raw():
        return None

    fake_raw.cache_clear = lambda: None
    monkeypatch.setattr(alert_service, "_daily", fake_daily)
    monkeypatch.setattr(alert_service, "_load_raw", fake_raw)
    calls = []

    def fake_save(observed_on, rows, threshold, **kwargs):
        calls.append((threshold, [row["asset_tag"] for row in rows], kwargs))
        return {"created": len(rows), "existing": 0}

    monkeypatch.setattr(alert_service, "save_observed_alerts", fake_save)
    monkeypatch.setattr(alert_service, "_last_saved_file", None)

    first = alert_service.scan_when_data_changes()
    assert first["created"] == 3
    assert [(threshold, assets) for threshold, assets, _ in calls] == [
        (1, ["AST-2", "AST-3"]), (12, ["AST-3"]),
    ]
    assert alert_service.scan_when_data_changes() is None
    assert len(calls) == 2

    csv_path.write_text("second observation", encoding="utf-8")
    assert alert_service.scan_when_data_changes()["created"] == 3
    assert len(calls) == 4


def test_original_five_screens_remain_available():
    from src import wireframe_app

    assert len(wireframe_app.SCREENS) == 5
    assert set(wireframe_app.SCREEN_BUILDERS) == {"1", "2", "3", "4", "5"}
    client = wireframe_app.app.server.test_client()
    assert client.get("/").status_code == 302
    assert client.get("/login").status_code == 200
    assert client.get("/_dash-layout").status_code == 401
