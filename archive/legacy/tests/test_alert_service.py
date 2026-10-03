"""구 alert_service(5탭 대시보드 경보 자동 적재) 테스트 — archive/legacy/src/alert_service.py와 함께 보관.

pytest 기본 경로(tests/)에서 빠진다. 다시 쓰려면 모듈을 src/로 되돌린다.
"""

from src import alert_service


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
