"""앱 조립 — 로그인 없이 열리고, 콜백 출력이 겹치지 않고, 보고서 내보내기가 동작한다."""

import json
import os
import subprocess
import sys
from pathlib import Path

from src import wireframe_app
from src.ui.base import SCREENS
from src.wireframe_app import app

ROOT = Path(__file__).resolve().parents[1]


def test_wireframe_registers_each_callback_output_once():
    """같은 Dash Output 콜백이 중복 등록되면 브라우저 hydration이 실패한다."""
    outputs = [callback["output"] for callback in app._callback_list]
    duplicates = sorted({output for output in outputs if outputs.count(output) > 1})
    assert duplicates == []


def test_four_screens_open_without_login():
    assert len(SCREENS) == 4
    assert set(wireframe_app.SCREEN_BUILDERS) == {"1", "2", "3", "4"}
    client = app.server.test_client()
    assert client.get("/").status_code == 200
    assert client.get("/_dash-layout").status_code == 200
    # Dash는 정의되지 않은 경로에도 같은 페이지를 준다 — 로그인 폼이 없다는 것만 확인한다.
    assert 'type="password"' not in client.get("/login").get_data(as_text=True)


def test_app_imports_without_database_settings():
    """DB·.env 설정이 하나도 없어도 앱이 뜨고 계정 메뉴가 없다."""
    probe = ("import json; from src.wireframe_app import app; c = app.server.test_client(); "
             "r = c.get('/_dash-layout'); print(json.dumps({'status': r.status_code, 'text': r.get_data(as_text=True)}))")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("MACHINE_DATABASE", "DASHBOARD_"))}
    out = subprocess.run([sys.executable, "-c", probe], cwd=ROOT, env=env, capture_output=True, text=True,
                         timeout=120)
    result = json.loads(out.stdout.strip().splitlines()[-1])
    assert result["status"] == 200, out.stderr
    for removed in ("profile-menu-toggle", "account-manage-btn", "session-timer", "demo-badge"):
        assert removed not in result["text"]


def _export_callback_response(client, *, export_format: str, audience: str = "mgr", clicks: int = 1):
    """내보내기 선택 후 실행 버튼 콜백을 브라우저 요청과 같은 형식으로 호출한다."""
    output = next(key for key in app.callback_map if key.startswith("..report-download.data"))
    return client.post("/_dash-update-component", json={
        "output": output,
        "outputs": [
            {"id": "report-download", "property": "data"},
            {"id": "action-echo", "property": "children"},
        ],
        "inputs": [{"id": "export-run-btn", "property": "n_clicks", "value": clicks}],
        "state": [
            {"id": "export-dd", "property": "value", "value": f"{export_format}:{audience}"},
            {"id": "filter-store", "property": "data", "value": {}},
            {"id": "seg-store", "property": "data", "value": {}},
        ],
        "changedPropIds": ["export-run-btn.n_clicks"],
    })


def test_export_excel_and_repeat_pdf(monkeypatch):
    monkeypatch.setattr(wireframe_app, "build_report_xlsx", lambda *_args: b"excel-content")
    monkeypatch.setattr(wireframe_app, "build_report_pdf", lambda *_args: b"pdf-content")
    client = app.server.test_client()
    excel = _export_callback_response(client, export_format="xlsx")
    assert excel.status_code == 200
    assert excel.json["response"]["report-download"]["data"]["filename"].endswith(".xlsx")
    # 선택값을 바꾸지 않아도 실행 버튼으로 같은 보고서를 연속 내보낼 수 있다.
    for clicks in (1, 2):
        pdf = _export_callback_response(client, export_format="pdf", clicks=clicks)
        assert pdf.status_code == 200
        assert pdf.json["response"]["report-download"]["data"]["filename"].endswith(".pdf")


def test_export_failure_shows_reason(monkeypatch):
    monkeypatch.setattr(wireframe_app, "build_report_xlsx",
                        lambda *_args: (_ for _ in ()).throw(RuntimeError("export test failure")))
    response = _export_callback_response(app.server.test_client(), export_format="xlsx")
    assert response.status_code == 200
    assert response.json["response"]["action-echo"]["children"] == "내보내기 실패 — RuntimeError"
