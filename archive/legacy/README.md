# archive/legacy — 현재 대시보드에서 쓰지 않는 파일 보관

현재 앱은 `src/wireframe_app.py`(화면 4개, 로그인·감사 로그)다. 아래는 그 이전 단계의 산출물로,
팀원 작업 이력을 남기기 위해 지우지 않고 옮겨 둔 것이다(git 이력은 `git log --follow`로 이어진다).
pytest 기본 경로(`tests/`)에 포함되지 않고, 앱·CI에서 import하지 않는다.

| 경로 | 무엇 | 옮긴 이유 |
|---|---|---|
| `app/` | 구 5탭 Dash 앱(`python -m app.app`) | `src/wireframe_app.py`로 대체 |
| `src/dashboard.py`, `src/dashboard_assets/` | 이전 대시보드 시안 | 위와 같음 |
| `src/ai4i_train.py`, `src/features.py`, `src/validate.py`, `notebooks/ai4i_model.ipynb` | AI4I 2020 데이터 실험 | 현재 데이터(합성 산업 설비)와 무관 |
| `src/energy_train.py`, `notebooks/energy_eda.ipynb`, `outputs/energy_*.csv` | 전력 예측 실험 | 현재 화면·보고서에서 쓰지 않음 |
| `scripts/make_dummy.py` | AI4I·에너지 더미 데이터 생성 | 위 실험 전용 |
| `src/alert_service.py`, `src/init_dashboard_db.py` | 5탭 대시보드용 경보·로그 테이블 | 현재 앱은 `audit_service`를 쓴다 |
| `docs/dashboard-handoff.md`, `docs/dashboard-preview.md` | 이전 대시보드 문서 | 위와 같음 |
| `tests/test_dashboard.py`, `tests/test_features.py`, `tests/test_alert_service.py` | 위 모듈의 테스트 | 모듈과 함께 보관 |

`src/db_inga.py`(하드코딩 DB 계정, import만 해도 테이블 생성·행 쓰기)는 보관하지 않고 삭제했다.

## 로그인·계정·감사 로그 (2026-10-03 앱에서 제거)

DB(MySQL)가 팀원 컴퓨터에만 있어 다른 팀원이 로그인할 수 없어서, 앱에서 로그인 기능을 떼어 냈다.
지금 앱(`src/wireframe_app.py`)은 DB·로그인 없이 실행되고 누구나 모든 화면과 내보내기를 쓸 수 있다.

| 경로 | 무엇 |
|---|---|
| `src/dashboard_auth.py` | 로그인·회원가입 화면, Flask 세션·CSRF, 10분 세션 만료 |
| `src/audit_service.py`, `src/audit_models.py` | 계정·역할, 행동·로그인·고장·오류 로그(SQLAlchemy ORM) |
| `src/security_service.py` | Argon2id 비밀번호 해시, AES-256-GCM 개인정보 암호화 |
| `src/db_service.py`, `src/db_models.py` | MySQL 연결(`MACHINE_DATABASE_URL`), 관리자 정보 표 |
| `src/demo_mode.py` | DB 없이 띄우던 데모 모드(로그인이 없어져 필요 없음) |
| `scripts/local_login_preview.py` | MySQL 대신 로컬 SQLite로 로그인을 띄우던 미리보기 |
| `.env.example` | DB 주소·암호화 키·관리자 계정 환경 변수 |
| `docs/DASHBOARD_LOGIN_GUIDE.md` 외 6개 | 로그인·세션·계정 역할·감사 로그 안내 |
| `tests/test_dashboard_auth.py` 외 6개 | 위 모듈의 테스트 31개 |
| `sql/recent_logs.sql` | 감사 로그 조회용 SELECT 문(MySQL Workbench용) — 최상위 `sql/`에 남아 있던 걸 뒤늦게 이동 |

다시 쓰려면 이 파일들을 원래 위치로 옮기고, `requirements.txt`에서 뺀 패키지
(`SQLAlchemy>=2.0,<2.1`, `PyMySQL>=1.1,<2`, `argon2-cffi>=23,<26`, `cryptography>=43,<47`,
`python-dotenv>=1.0,<2`)를 되돌린 뒤, `wireframe_app.py`의 계정 메뉴·모달·콜백을 커밋
`7ad86c0`(제거 직전) 상태에서 가져온다.
