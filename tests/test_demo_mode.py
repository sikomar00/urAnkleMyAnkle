"""데모 모드: DB·로그인 없이 뜨고, 운영 모드에서는 배지가 숨겨진다."""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PROBE = """
import json
from src.wireframe_app import app, DEMO_MODE, record_action
client = app.server.test_client()
layout = client.get("/_dash-layout")
def find(node, target):
    if isinstance(node, dict):
        if node.get("props", {}).get("id") == target:
            return node["props"]
        return next((f for v in node.values() if (f := find(v, target))), None)
    if isinstance(node, list):
        return next((f for v in node if (f := find(v, target))), None)

badge = find(layout.get_json(), "demo-badge") if layout.status_code == 200 else None
print(json.dumps({"demo": DEMO_MODE, "status": layout.status_code,
                  "noop": record_action("ACT_TEST") is None if DEMO_MODE else None,
                  "badge": badge and badge["style"]["display"]}, ensure_ascii=False))
"""


def _probe(env_extra):
    env = {**os.environ, "MACHINE_DATABASE_URL": "mysql+pymysql://x:y@127.0.0.1:1/predictive_maintenance",
           **env_extra}
    env.pop("DASHBOARD_MODE", None) if "DASHBOARD_MODE" not in env_extra else None
    out = subprocess.run([sys.executable, "-c", PROBE], cwd=ROOT, env=env, capture_output=True,
                         text=True, timeout=120)
    return out, json.loads(out.stdout.strip().splitlines()[-1])


def test_demo_mode_serves_layout_without_db_or_login():
    out, result = _probe({"DASHBOARD_MODE": "demo"})
    assert result["demo"] and result["status"] == 200 and result["noop"] and result["badge"] == "inline-flex", out.stderr


def test_default_mode_is_not_demo():
    _, result = _probe({})
    assert result["demo"] is False
