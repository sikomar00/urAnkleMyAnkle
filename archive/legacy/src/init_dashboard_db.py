"""현재 5탭 대시보드용 MySQL 로그 테이블을 초기화한다.

실행: python -m src.init_dashboard_db
관리자 계정은 대시보드 실행 시 .env의 단일 계정으로 자동 준비된다.
"""

from .audit_service import create_audit_tables


def main() -> None:
    create_audit_tables()
    print("MySQL의 action_logs, failure_logs, error_logs, dashboard_admin_info를 확인했습니다.")


if __name__ == "__main__":
    main()
