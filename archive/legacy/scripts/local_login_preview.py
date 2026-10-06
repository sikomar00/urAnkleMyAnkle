"""로컬 로그인 미리보기 — MySQL 없이 운영 모드(로그인·계정 메뉴·계정 관리)를 띄운다.

    python -m scripts.local_login_preview [--port 8052]

- 운영 코드는 바꾸지 않는다. 이 프로세스 안에서만 DB 엔진을 로컬 SQLite 파일
  (.cache/local_preview.sqlite3)로 바꾼다. 테스트(tests/test_account_roles.py)와 같은 방식이다.
- 실행할 때마다 DB를 새로 만들고 고정 계정 두 개를 만든다.
      관리자  admin / admin1234
      일반    user  / user1234
  127.0.0.1에서만 열고 이 로컬 SQLite에만 있는 계정이라 고정값으로 둔다.
- 배포·운영에는 쓰지 않는다. 실제 운영은 .env의 MACHINE_DATABASE_URL(MySQL)을 쓴다.
"""
import argparse
import base64
import os
import secrets
import sys
from pathlib import Path

from sqlalchemy import create_engine

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / ".cache" / "local_preview.sqlite3"
ADMIN_ID, ADMIN_PASSWORD = "admin", "admin1234"
USER_ID, USER_PASSWORD = "user", "user1234"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8052)
    args = parser.parse_args()

    if "--demo" in sys.argv or os.environ.get("DASHBOARD_MODE", "").lower() == "demo":
        raise SystemExit("로그인 미리보기는 운영 모드로만 띄운다. DASHBOARD_MODE=demo를 지우고 실행한다.")
    os.environ.update({
        "DASHBOARD_ADMIN_ID": ADMIN_ID,
        "DASHBOARD_ADMIN_PASSWORD": ADMIN_PASSWORD,
        "DASHBOARD_AES_KEY": base64.b64encode(secrets.token_bytes(32)).decode(),
        "DASHBOARD_ADMIN_INVITE_CODE": secrets.token_urlsafe(9),
        "DASHBOARD_SESSION_SECRET": secrets.token_urlsafe(48),
        "DASHBOARD_COOKIE_SECURE": "0",
    })
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    DB_PATH.unlink(missing_ok=True)
    engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})

    from src import audit_service, db_service
    db_service.get_engine = lambda: engine
    audit_service.get_engine = lambda: engine

    from src import wireframe_app as w  # 엔진을 바꾼 뒤에 앱을 불러온다.

    audit_service.create_audit_tables()
    audit_service.ensure_dashboard_admin()
    audit_service.register_account(
        role="USER", login_id=USER_ID, password=USER_PASSWORD, password_confirm=USER_PASSWORD,
        name="미리보기 사용자", phone="010-0000-0000", email="preview@example.com", invite_code="",
    )
    print("\n로컬 로그인 미리보기 (SQLite: .cache/local_preview.sqlite3, 실행할 때마다 새로 만든다)")
    print(f"  주소     http://127.0.0.1:{args.port}/login")
    print(f"  관리자   {ADMIN_ID} / {ADMIN_PASSWORD}")
    print(f"  일반     {USER_ID}  / {USER_PASSWORD}\n", flush=True)
    w.app.run(host="127.0.0.1", port=args.port, debug=False)


if __name__ == "__main__":
    main()
