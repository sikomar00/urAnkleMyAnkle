# 계정 역할·로그인 로그 기능 안내

## 현재 DB 테이블

MySQL 데이터베이스는 `predictive_maintenance`입니다.

| 테이블 | 용도 |
| --- | --- |
| `dashboard_admin_info` | ADMIN·USER 계정, 해시 비밀번호, 암호화 개인정보, 접속 상태 |
| `action_logs` | 탭 이동, 필터 변경, 보고서 다운로드, 계정 생성 등 일반 행동 |
| `login_logs` | 로그인, 로그아웃, 세션 연장·만료, 로그인 차단 |
| `failure_logs` | CSV에서 확인한 부품 고장 표시 |
| `error_logs` | 프로그램·DB 처리 오류 |

이전 시범용 `dashboard_failure_alerts`, `dashboard_system_logs`는 현재 사용하지 않습니다.

## 같은 네트워크에서 대시보드 열기

대시보드를 실행한 PC와 같은 사내·가정 네트워크에 있다면 `http://192.168.219.110:8052/login`으로 접속할 수 있습니다. 실행 코드는 기본으로 `0.0.0.0`에서 요청을 받도록 되어 있으며, Windows 방화벽에는 TCP 8052 인바운드 규칙이 필요합니다.

처음 한 번은 **관리자 권한 PowerShell**에서 아래처럼 규칙을 등록합니다.

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\enable_dashboard_lan.ps1
```

이 주소는 같은 네트워크 전용입니다. 인터넷 어디서나 공개하려면 고정 공인 IP 또는 도메인, 공유기 포트 설정, HTTPS와 별도 배포 환경이 필요합니다.

## 계정 테이블의 역할·세션 컬럼

`dashboard_admin_info`에는 기존 암호화·해시 컬럼 외에 아래 값이 있습니다.

| 컬럼 | 내용 |
| --- | --- |
| `role` | `ADMIN` 또는 `USER` |
| `is_online` | 현재 로그인 세션이 있는지 여부 |
| `session_expires_at` | 로그인 세션 만료 예정 시각 |
| `last_login_at` | 가장 최근 로그인 시각 |

기존 `ankles` 계정은 `ADMIN`으로 이전됩니다. 일반 계정은 계정 생성 화면에서 만들 수 있습니다.

## 역할별 기능

| 기능 | ADMIN | USER |
| --- | --- | --- |
| 대시보드 ①~④ 탭 | 가능 | 가능 |
| PDF·Excel 내보내기 | 가능 | 불가 — 접근 권한 안내 표시 |
| 비밀번호 변경·로그아웃·세션 연장 | 가능 | 가능 |
| 일반 계정 목록 조회 | 가능 | 불가 |
| 일반 계정의 복호화된 이름 조회 | 가능 | 불가 |
| 관리자 계정 생성 | 관리인 코드가 맞을 때 가능 | 로그인 화면에서 관리인 코드가 맞을 때 가능 |

관리자 프로필에는 `계정 관리` 버튼이 표시됩니다. 이 창은 USER 역할 계정의 아이디·복호화된 이름·접속 상태만 표시합니다. 초록색 원은 현재 세션이 유효한 일반 계정을 뜻합니다.

## 가입 흐름

1. 로그인 화면 아래의 **계정 생성**을 누릅니다.
2. 일반 계정 생성 또는 관리자 계정 생성을 고릅니다.
3. 이름, 이메일, 전화번호, 아이디, 비밀번호, 비밀번호 확인을 입력합니다.
4. 관리자 계정 생성일 때만 관리인 코드를 추가로 입력합니다.
5. 성공하면 계정 생성 완료 창이 표시되고 로그인 화면에서 새 계정으로 로그인합니다.

관리인 코드는 프로젝트 루트 `.env`의 `DASHBOARD_ADMIN_INVITE_CODE` 값입니다. 실제 코드는 Git에 올리지 않으며, `.env.example`에는 변수 이름만 둡니다.

## 해싱과 암호화

- `password_hash`: Argon2 해시입니다. 비밀번호 원문을 복원하지 않고 입력값이 맞는지만 확인합니다.
- `name_encrypted`, `phone_encrypted`, `email_encrypted`: AES-256-GCM 암호문입니다. AES 키는 `.env`의 `DASHBOARD_AES_KEY`에만 있습니다.
- 비밀번호, 관리인 코드, AES 키, 복호화된 개인정보는 `action_logs`와 `login_logs`에 저장하지 않습니다.

## 이벤트 코드

### `login_logs`

| 코드 | 시점 |
| --- | --- |
| `ACT_LOGIN` | 로그인 성공 |
| `ACT_LOGIN_BLOCKED` | 잘못된 아이디·비밀번호 로그인 |
| `ACT_LOGOUT` | 로그아웃 |
| `ACT_SESSION_EXTEND` | 10분 세션 초기화 |
| `ACT_SESSION_EXPIRED` | 10분 세션 만료 |

### `action_logs`

| 코드 | 시점 |
| --- | --- |
| `ACT_ACCOUNT_REGISTER` | 일반 계정 생성 성공 |
| `ACT_ADMIN_REGISTER` | 관리자 계정 생성 성공 |
| `ACT_ACCOUNT_REGISTER_BLOCKED` | 계정 생성 차단 |
| `ACT_ACCOUNT_LIST_VIEW` | 관리자 계정 목록 조회 |
| `ACT_PASSWORD_CHANGE` | 비밀번호 변경 |
| `ACT_TAB_OPEN` | 탭 이동 |
| `ACT_FILTER_CHANGE` | 조회 조건 변경 |
| `ACT_REPORT_EXPORT` | 보고서 다운로드 |
| `ACT_EXPORT_ACCESS_BLOCKED` | 일반 계정의 PDF·Excel 내보내기 차단 |
| `ACT_CSV_EXPORT` | CSV 다운로드 |

PDF·Excel 내보내기 성공과 권한 차단은 `action_logs`에 남습니다. 파일 생성 중 실제 오류가 발생하면 같은 내보내기 행동은 `FAILED`로 기록되고, 오류 원인은 `error_logs`에 `ERR_REPORT_EXPORT`로 함께 남습니다.

두 로그 테이블 모두 `actor_id`와 `actor_role`을 저장합니다. 로그인과 로그아웃 이벤트는 `action_logs`에 중복 저장하지 않습니다.

## MySQL Workbench 확인 SQL

```sql
USE predictive_maintenance;

SELECT login_id, role, is_online, session_expires_at, last_login_at
FROM dashboard_admin_info
ORDER BY login_id;

SELECT occurred_at, actor_id, actor_role, event_code, result_status, block_reason, source
FROM login_logs
ORDER BY log_id DESC;

SELECT occurred_at, actor_id, actor_role, event_code, action_type, result_status
FROM action_logs
ORDER BY log_id DESC;

SELECT login_id, password_hash, name_encrypted, phone_encrypted, email_encrypted
FROM dashboard_admin_info;
```

마지막 조회에서는 비밀번호가 `$argon2`로 시작하고 개인정보가 `v1:`로 시작하는 암호문으로 보여야 합니다.

## 실행

```powershell
& .\.venv\Scripts\python.exe src\wireframe_app.py
```

기본 접속 주소는 `http://127.0.0.1:8052/login`입니다. 같은 내부망에서 접속하려면 실행 시 `host="0.0.0.0"`으로 열고 Windows 방화벽에서 8052 포트를 허용해야 합니다.
