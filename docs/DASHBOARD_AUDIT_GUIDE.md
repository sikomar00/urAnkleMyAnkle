# 4탭 대시보드의 MySQL 로그 기록 안내

현재 브랜치 `KYS-dashboard`의 `src/wireframe_app.py`는 ①~④ 탭을 유지합니다. 별도의 DB 관리 탭 없이 사용자의 의미 있는 조작을 `action_logs`, CSV에 표시된 부품 고장을 `failure_logs`, 프로그램 오류를 `error_logs`에 저장합니다. 이 로그 테이블과 계정 테이블은 MySQL 데이터베이스 `predictive_maintenance`에 있습니다. MySQL Workbench는 그 DB를 조회하는 프로그램이며 Python은 SQLAlchemy ORM과 PyMySQL로 MySQL 서버에 접속합니다.

이전 시범 구조의 `dashboard_failure_alerts`, `dashboard_system_logs`는 사용하지 않아 삭제했습니다. 현재 로그인·행동·고장·오류 로그는 각각 `dashboard_admin_info`, `action_logs`, `failure_logs`, `error_logs`를 사용합니다.

## 실행

```powershell
cd C:\urAnkleMyAnkle
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe src\wireframe_app.py
```

앱이 시작할 때 `create_audit_tables()`가 계정·로그 테이블을 만들고 `.env`의 기본 관리자 계정을 준비한다. 예전 `src/init_dashboard_db.py`는 `archive/legacy/src/`로 옮겼다. DB 없이 화면만 보려면 `python -m src.wireframe_app --demo`, 로그인까지 보려면 `python -m scripts.local_login_preview`.

브라우저에서 `http://127.0.0.1:8052/`를 열면 로그인 화면부터 표시됩니다. 실행 중인 대시보드 코드를 수정한 뒤에는 **기존 터미널 서버를 Ctrl+C로 종료하고 다시 실행**해야 수정이 반영됩니다. DB URL, AES 키, 로그인 세션 서명 키와 단일 관리자 계정은 Git에서 제외된 `.env`에 둡니다. `.env.example`은 변수 이름만 보여줍니다. 로그인 화면에는 공개 회원가입이나 관리자 등록 버튼이 없습니다. 자세한 절차는 [`DASHBOARD_LOGIN_GUIDE.md`](DASHBOARD_LOGIN_GUIDE.md)를 확인해 주세요.

## 테이블과 컬럼

| 테이블 | 용도 | 주요 컬럼 |
| --- | --- | --- |
| `action_logs` | 누가 어떤 화면 조작을 했고 성공·차단·실패했는지 | `log_id`, `occurred_at`, `actor_id`, `event_code`, `action_type`, `target_type`, `target_id`, `action_detail`, `result_status`, `block_reason` |
| `failure_logs` | 합성 CSV 최신 관측일에 고장 표시된 개별 부품 | `log_id`, `occurred_at`, `event_code`, `observation_date`, `target_date`, `plant_code`, `asset_tag`, `part_no`, `part_family`, `risk_score`, `risk_level`, `main_signal`, `model_version`, `actual_failure`, `failure_status`, `source` |
| `error_logs` | 앱·다운로드·DB 처리 중 발생한 오류 | `log_id`, `occurred_at`, `actor_id`, `action_type`, `error_code`, `error_type`, `error_message`, `source`, `target_id`, `resolved` |
| `dashboard_admin_info` | 최초 관리자 한 명의 보안 정보 | `id`, `login_id`, `password_hash`, `name_encrypted`, `phone_encrypted`, `email_encrypted`, `created_at`, `updated_at` |

`actor_id`는 로그인 성공 후 **서버 세션에서 확인한 관리자 아이디**입니다. 브라우저가 직접 보낸 작업자 ID는 믿지 않습니다. 로그인 실패처럼 인증 전 사건은 `ANONYMOUS`로 남습니다. 세 로그 테이블의 `occurred_at`은 **한국 표준시(KST, UTC+9)**로 저장합니다. 기존에 UTC로 기록된 행은 2026-09-29에 한 번만 9시간 보정했습니다. 현재 `failure_logs`의 `FAIL_PART_OBSERVED`는 **실제 공장 실시간 사고나 모델 예측이 아니라 합성 CSV의 `breakdown_flag=1` 표시**입니다. 아직 모델 경보를 생성하지 않으므로 `target_date`, `risk_score`, `model_version` 등은 비어 있습니다. 이는 데이터를 꾸며 넣지 않기 위한 설계입니다. `observation_date + asset_tag + part_no + event_code` 유일 제약으로 중복 저장을 막습니다. 앱 시작 시와 브라우저가 열린 동안 1분마다 CSV 파일의 수정 시각을 확인하고, 바뀌었을 때만 최신 날짜를 다시 읽습니다.

## 이벤트 코드

| 구분 | 코드 | 발생 조건 |
| --- | --- | --- |
| 행동 | `ACT_TAB_OPEN` | ①~④ 탭 이동 |
| 행동 | `ACT_FILTER_CHANGE` | 공장·종류·설비·기간 선택 또는 초기화 |
| 행동 | `ACT_ASSET_NAVIGATE` | 이전·다음 설비 이동 |
| 행동 | `ACT_SEGMENT_CHANGE` | 분석 항목 전환 |
| 행동 | `ACT_AUDIENCE_CHANGE` | 보고서 대상 선택 |
| 행동 | `ACT_THEME_CHANGE` | 테마 전환 |
| 행동 | `ACT_PRIORITY_SORT` | 현황 KPI를 눌러 우선순위 정렬 |
| 행동 | `ACT_DATA_QUERY` | 데이터 표 페이지·정렬 조작 |
| 행동 | `ACT_REPORT_EXPORT`, `ACT_CSV_EXPORT` | 파일 내보내기 성공·실패 |
| 차단 | `ACT_UNAVAILABLE` | 아직 구현하지 않은 버튼 클릭 |
| 관리자 | `ACT_ADMIN_REGISTER`, `ACT_ADMIN_VERIFY` | 최초 등록·비밀번호 검증 |
| 로그인 | `ACT_LOGIN`, `ACT_LOGOUT`, `ACT_PASSWORD_CHANGE` | 로그인·로그아웃·비밀번호 변경 |
| 고장 | `FAIL_PART_OBSERVED` | CSV 최신 관측일 부품 고장 표시 |
| 오류 | `ERR_REPORT_EXPORT`, `ERR_CSV_EXPORT`, `ERR_DATA_QUERY`, `ERR_SCREEN_RENDER`, `ERR_FAILURE_SYNC`, `ERR_DB_STARTUP`, `ERR_DASH_CALLBACK` | 해당 작업 중 오류 |

화면 내부의 자동 재렌더, 차트 재색칠, 데이터 로드 자체는 사용자 행동으로 기록하지 않습니다. 다운로드 등 실패 시 행동 기록과 오류 기록은 한 DB 트랜잭션으로 함께 저장합니다. MySQL 자체가 끊기면 같은 MySQL에 오류를 쓸 수 없으므로 `instance/audit_fallback.log`에 비밀값을 뺀 오류 종류만 기록합니다. 해당 파일은 Git에서 제외됩니다. SQL은 문자열 결합으로 만들지 않고 ORM을 사용하므로 사용자 입력을 SQL 문법으로 실행하지 않습니다.

## 관리자 보안 확인

`security_service.py`는 비밀번호를 Argon2id 단방향 해시로 저장하고, 이름·전화번호·이메일을 AES-256-GCM으로 암호화합니다. AES 키는 `.env`에만 있고 DB에는 저장하지 않습니다. 앱 시작 시 `.env`의 단일 관리자 계정을 DB에 자동으로 한 번 준비합니다. Workbench에서 아래처럼 확인할 수 있습니다. `password_hash`는 원문 비밀번호와 달라야 하며, `name_encrypted` 등은 원문 대신 `v1:`로 시작하는 암호문이어야 합니다. 비밀번호 검증은 로그인·비밀번호 변경 때 서버에서 수행하며 개인정보 원문을 로그에 쓰지 않습니다.

```sql
USE predictive_maintenance;
SELECT log_id, occurred_at, actor_id, event_code, target_id, result_status
FROM action_logs ORDER BY log_id DESC LIMIT 30;

SELECT log_id, observation_date, plant_code, asset_tag, part_no,
       event_code, actual_failure, source
FROM failure_logs ORDER BY log_id DESC LIMIT 30;

SELECT log_id, occurred_at, error_code, error_type, source
FROM error_logs ORDER BY log_id DESC LIMIT 30;

SELECT login_id, password_hash, name_encrypted, phone_encrypted, email_encrypted
FROM dashboard_admin_info;
```

Workbench에서 `SELECT * FROM predictive_maintenance.action_logs;`만 실행하면 최신순이 보장되지 않습니다. 위처럼 `ORDER BY log_id DESC`를 붙여 실행하면 가장 최근 저장 건이 맨 위에 표시됩니다. 같은 방식으로 `failure_logs`와 `error_logs`도 조회할 수 있습니다. 이 쿼리는 [`recent_logs.sql`](../sql/recent_logs.sql)에도 저장해 두었습니다. Workbench에서 해당 파일을 열고 원하는 SELECT 문을 실행하시면 됩니다.

정상 동작만 했다면 `error_logs`는 비어 있는 것이 맞습니다. 현재 ①~⑤의 일부 버튼은 원래 와이어프레임 기능으로 남아 있으므로 클릭하면 `BLOCKED`로 기록합니다. Dash 데이터 콜백은 로그인 세션 없이 호출하면 401로 차단됩니다. 이 구현은 교육용 프로젝트의 단일 관리자 로그인·로그·해시·암호화 검증 범위입니다.

## 코드 위치

- `src/wireframe_app.py`: 5탭 화면, 프로필 메뉴, 비밀번호 변경·로그아웃, 조작별 기록 호출
- `src/dashboard_auth.py`: 로그인 화면과 서버 측 Dash 접근 차단
- `src/audit_models.py`: 세 로그 테이블의 SQLAlchemy ORM
- `src/audit_service.py`: 트랜잭션, 중복 방지, CSV 변경 확인, 관리자 등록
- `src/security_service.py`: Argon2id와 AES-256-GCM
- `src/db_service.py`: 기존 MySQL 연결 설정 재사용
- `src/audit_service.py`의 `create_audit_tables()`: 앱 시작 시 로그·관리자 테이블 생성(예전 `init_dashboard_db.py`는 `archive/legacy/`)
- `tests/test_audit_service.py`: SQLite 테스트 DB로 저장·보안·중복 검증

기존 `dashboard_failure_alerts`, `dashboard_system_logs` 테이블과 구 DB 시범 코드는 이전 실험의 흔적이며 현재 5탭 화면의 로그는 위 새 테이블로 들어갑니다.
