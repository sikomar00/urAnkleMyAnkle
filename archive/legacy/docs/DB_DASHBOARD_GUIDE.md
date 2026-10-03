# 이전 ⑥ 화면 시범 안내 (현행 아님)

현재 5탭 대시보드에는 ⑥ 화면이 없습니다. 현행 구조와 실행 방법은
[`DASHBOARD_AUDIT_GUIDE.md`](DASHBOARD_AUDIT_GUIDE.md)를 확인해 주세요.

아래 내용은 이전 시범 구현 기록입니다. 여기 나오는 `src/alert_service.py`·`src/init_dashboard_db.py`는 `archive/legacy/src/`로 옮겼고, `src/db_inga.py`는 삭제했습니다(2026-10-03, `1ca1000`).

이 문서는 `KYS-dashboard` 브랜치의 설비 대시보드에 추가한 DB 기능을 설명합니다. 기존 ①~⑤ 화면은 CSV 기반 조회를 유지하고, ⑥ 화면에서 대시보드 전용 MySQL 표를 조회합니다. 사진으로 만든 화면 시안의 숫자는 코드에 사용하지 않았습니다.

## 현재 데이터로 실제로 하는 일

- 원본: `data/raw/synthetic_industrial_machine_data.csv` (합성 교육용 데이터)
- 분석 단위: `asset_tag × transaction_date`당 설비 1대의 하루 기록 1행
- 점수: 그날 `breakdown_flag=1`인 부품에 `criticality` A=4, B=2, C=1을 적용해 합산한 `failure_points`
- 경보 기준 ①: 최신 **관측일**에 부품이 1개 이상 고장 표시된 설비 (`failure_points > 0`)
- 경보 기준 ②: 같은 관측일의 고장 심각도 점수가 12 이상인 설비 (`HIGH_RISK_THRESHOLD=12`)
- 대시보드를 열면 첫 확인을 바로 실행하고, 열린 동안 10초마다 CSV 파일의 수정 여부를 확인합니다. 파일이 바뀐 경우에만 새 기록을 읽고 두 경보 기준을 자동 저장합니다. 화면을 다시 그리거나 필터를 바꾸는 것만으로는 DB에 다시 쓰지 않습니다. ⑥ 화면의 `지금 다시 확인` 버튼은 선택 사항입니다.

이것은 **합성 CSV의 같은 날 고장 표시를 저장한 경보**입니다. 실제 공장에서 방금 발생한 고장 신호도 아니고, 다음 날 고장 예측이나 예측 확률도 아닙니다. 센서값이 점수의 원인이라는 뜻도 아닙니다. 현재 화면의 ③ 과제·임계값 선택은 저장 기준을 바꾸지 않습니다. 12점은 기존 대시보드의 실험 기준이며 현장 운영 기준으로 확정된 값이 아닙니다. 같은 설비가 두 조건에 모두 걸리면 **기준별로 1건씩** 저장합니다.

## 파일별 역할

| 파일 | 역할 |
| --- | --- |
| `src/wireframe_app.py` | ① 현황과 ⑥ DB·관리 화면, CSV 변경 감지용 Dash 주기 확인 연결 |
| `src/alert_service.py` | CSV 변경 시 최신 관측일의 두 고장 기준을 각각 선택해 저장 |
| `src/db_models.py` | SQLAlchemy ORM으로 대시보드 전용 표 3개 정의 |
| `src/db_service.py` | MySQL 연결, 표 생성, 경보·시스템 로그 저장, 조회, 관리자 정보 처리 |
| `src/security_service.py` | Argon2id 비밀번호 해싱·검증과 AES-256-GCM 개인정보 암복호화 |
| `src/init_dashboard_db.py` | 표 생성과 최초 관리자 1명 대화식 등록 |
| `src/db_inga.py` | 기존 연습 예제. 이번 구현에서 수정·실행하지 않음 |
| `.env.example` | 로컬 `.env`에 필요한 환경변수의 이름만 표시 |
| `tests/test_dashboard_db.py` | 실제 MySQL을 건드리지 않고 핵심 규칙 검증 |

## MySQL 구조

앱은 SQLAlchemy와 PyMySQL로 로컬 MySQL의 **`testdb`**에 연결합니다. Workbench는 DB를 확인하는 관리 화면입니다. `Base.metadata.create_all()`은 아래 `dashboard_` 표만 생성하며 기존 `student` 표를 변경하지 않습니다. `create_all()`은 이미 존재하는 표의 구조를 자동 변경하지 않으므로 향후 컬럼 변경은 별도 마이그레이션이 필요합니다.

### `dashboard_failure_alerts`

한 행이 한 설비의 한 관측일·한 고장 기준 경보입니다. 주요 컬럼은 `id`(자동 번호), `observed_on`(CSV 관측 날짜), `asset_tag`(설비 ID), `failure_points`(고장 심각도 점수), `threshold`(1 또는 12), `rule_version`(`any-part-failure-v1` 또는 `severity12-v1`), `event_code`, `status`(`unreviewed`/`reviewed`), `source`(`synthetic_csv`), `created_at`(DB 저장 시각), `reviewed_at`입니다.

`observed_on + asset_tag + rule_version`에 유일 제약을 걸어 같은 기준의 경보가 중복 등록되지 않게 했습니다. 새 경보와 그 `ALERT_CREATED` 시스템 로그는 **한 트랜잭션**에서 확정됩니다. 하나라도 실패하면 함께 되돌립니다. 동시 요청 충돌도 DB의 유일 제약이 막으며, 충돌 시 작업이 실패했다고 알립니다.

### `dashboard_system_logs`

한 행이 한 시스템 동작입니다. 컬럼은 `id`, `occurred_at`, `event_code`, `outcome`, `run_id`(같은 분석 실행의 묶음 ID), `asset_tag`, `alert_id`, `admin_id`, `detail`입니다. 개인정보 원문·비밀번호·DB 접속 비밀번호·암호화 키는 넣지 않습니다.

| 이벤트 코드 | 실제 기록 시점 |
| --- | --- |
| `ALERT_CREATED` | 새 관측일 경보가 저장됨 |
| `ALERT_SCAN_FINISHED` | 최신 관측일 분석 종료, 새 경보·기존 경보 수 기록 |
| `ALERT_REVIEWED` | 관리자 비밀번호를 검증하고 경보를 확인 처리함 |
| `ADMIN_CREATED` | 대화식 최초 관리자 등록 성공 |
| `ADMIN_AUTH_FAILED` | 관리자 정보 조회·수정·경보 확인 비밀번호 검증 실패 |
| `ADMIN_INFO_VIEWED` | 비밀번호 검증 후 관리자 정보 조회 |
| `ADMIN_INFO_UPDATED` | 비밀번호 검증 후 관리자 정보 변경 |
| `REPORT_EXPORTED`, `REPORT_EXPORT_FAILED` | 기존 PDF/Excel 내보내기 결과 |
| `DATA_EXPORTED`, `DATA_EXPORT_FAILED` | 기존 ④ 화면 CSV 내보내기 결과 |

DB가 완전히 끊어지면 **같은 DB 안에 실패 로그를 남길 수 없습니다.** 이 경우 ⑥ 화면은 연결 오류를 표시하고, 실패 건수는 현재 앱 프로세스의 메모리에서만 셉니다. 앱을 재시작하면 이 숫자는 초기화됩니다. 영구적인 DB 장애 추적은 별도 파일/외부 로그 저장소가 필요합니다.

### `dashboard_admin_info`

최초 관리자 1명을 저장합니다. 컬럼은 `id`, `login_id`(조회 식별자, 평문), `password_hash`(Argon2id 결과), `name_encrypted`, `phone_encrypted`, `email_encrypted`, `created_at`, `updated_at`입니다. 전화번호와 이메일은 선택 입력이지만 저장할 때 암호화합니다.

대시보드 전체를 잠그는 로그인 화면은 없습니다. ⑥ 탭에서 관리자 정보를 **조회하거나 수정할 때마다** 현재 비밀번호를 확인합니다. 경보 확인 처리도 관리자 비밀번호를 요구합니다. 브라우저의 `dcc.Store`에는 비밀번호나 복호화된 개인정보를 저장하지 않으며, 비밀번호 입력칸은 처리 후 비웁니다. 사용자가 조회 후 화면을 열어 둔 상태의 정보는 화면에 보일 수 있으므로 자리를 비울 때 탭을 닫아야 합니다. 이것은 완전한 다중 사용자 인증 시스템이 아닙니다.

## 보안 처리

- 비밀번호는 Argon2id로 해싱해 저장합니다. 원문을 다시 읽거나 복호화하지 않습니다. 최초 비밀번호·새 비밀번호는 12자 이상입니다.
- 이름·전화번호·이메일은 AES-256-GCM으로 각각 암호화합니다. 암호화마다 새 nonce를 만들고 암호문과 함께 저장합니다. 키는 `.env`에만 두며 DB에 저장하지 않습니다. 키를 잃으면 기존 개인정보를 복호화할 수 없습니다.
- `MACHINE_DATABASE_URL`은 앱의 MySQL 접속 정보이며 관리자 비밀번호와 다릅니다. 실제 운영 시에는 root 대신 대시보드 표에 필요한 권한만 가진 전용 MySQL 사용자를 권장합니다.
- 모든 조회·저장은 ORM 표현식으로 값을 전달하며 사용자 입력을 SQL 문자열에 이어 붙이지 않습니다.
- `.env`는 Git에서 제외됩니다. `.env.example`에는 예시 이름만 있습니다. 연습용 `db_inga.py`에 직접 적힌 접속 정보는 이번 앱에서 가져다 쓰지 않습니다.

## 설치·처음 설정·실행 (Windows PowerShell)

프로젝트 폴더 `C:\urAnkleMyAnkle`에서 다음을 실행합니다.

```powershell
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

`.env.example`을 참고해 프로젝트 루트의 `.env`에 `MACHINE_DATABASE_URL`과 `DASHBOARD_AES_KEY`를 설정합니다. 이 컴퓨터에는 이번 검증을 위해 Git에서 제외되는 로컬 `.env`가 생성되어 있습니다. 키를 새로 만들 때는 다음 명령의 결과를 `.env`의 `DASHBOARD_AES_KEY`에 입력합니다. **이미 암호화해 저장한 정보가 있으면 키를 임의로 교체하지 마십시오.**

```powershell
& .\.venv\Scripts\python.exe -c "import base64,secrets; print(base64.b64encode(secrets.token_bytes(32)).decode())"
```

표 생성:

```powershell
& .\.venv\Scripts\python.exe -m src.init_dashboard_db
```

관리자 최초 등록은 관리자 본인이 터미널에서 수행합니다. 비밀번호는 입력할 때 화면에 표시되지 않습니다. **현재는 아직 관리자를 등록하지 않았습니다.**

```powershell
& .\.venv\Scripts\python.exe -m src.init_dashboard_db --admin
```

대시보드 실행:

VS Code에서는 **실행 및 디버그(F5)**에서 `대시보드 실행 (.venv)`을 선택하면 프로젝트의 `.venv`로 실행됩니다. 설정은 `.vscode/launch.json`에 저장되어 있어, 기존에 다른 Python 환경을 선택했더라도 이 실행 구성에서는 `.venv`를 사용합니다. 편집기 우측 상단의 일반 `Python 파일 실행` 버튼은 VS Code에 저장된 인터프리터 선택을 따르므로, 그 버튼을 사용하려면 Python 인터프리터도 `.venv`로 선택해야 합니다.

```powershell
& .\.venv\Scripts\python.exe .\src\wireframe_app.py
```

표시되는 `http://127.0.0.1:8050/`을 브라우저에서 열면 ① 현황을 볼 수 있습니다. DB 자동 저장에 ⑥ 탭을 열 필요는 없습니다. ⑥ 탭은 경보 이력·시스템 로그·관리자 정보 확인용입니다. 이미 8050 포트를 쓰는 앱이 있다면 기존 앱을 먼저 종료하고 다시 실행해야 최신 화면이 나타납니다.

## Workbench에서 확인

`Local instance MySQL80`을 열고 왼쪽 `SCHEMAS → testdb → Tables`에서 새로고침하면 세 표가 보입니다. SQL 편집기에서 다음 조회를 각각 실행할 수 있습니다. 아래 문장은 **조회 전용**입니다.

```sql
SELECT id, observed_on, asset_tag, failure_points, threshold, status, created_at
FROM testdb.dashboard_failure_alerts ORDER BY id DESC LIMIT 20;

SELECT id, occurred_at, event_code, outcome, asset_tag, detail
FROM testdb.dashboard_system_logs ORDER BY id DESC LIMIT 20;

SELECT id, login_id, password_hash, name_encrypted, email_encrypted
FROM testdb.dashboard_admin_info;
```

마지막 조회에서 비밀번호는 해시, 개인정보는 암호문으로 보여야 합니다. 비밀번호·AES 키 자체를 Workbench에 넣어 조회하지 마십시오.

## 수행한 검증과 남은 범위

- `testdb` 연결 확인, 기존 `student` 테이블 유지, 새 `dashboard_` 표 3개 생성 확인
- 실제 CSV의 최신 관측일 `2025-01-01`: 설비 10대 중 부품 고장 표시 10대, 12점 이상 1대(`AST-2031`, 17점)를 **별도 기준**으로 저장
- 동일 분석 재실행: 새 경보 0건, 기존 경보 11건으로 기준별 중복 방지 확인
- 비밀번호 해싱·검증, AES-GCM 암복호화, 관리자 접근 검증, 경보 상태 변경, DB 미연결 화면 렌더 테스트
- 자동 저장 경로의 파일 변경 감지와 두 기준 구분을 테스트로 확인

현재 **최초 관리자 계정은 등록하지 않았으므로** 관리자 개인정보의 실 MySQL 저장·수정은 아직 사용자 비밀번호로 시험하지 않았습니다. 10초마다 파일 **변경 여부만** 확인하는 기능은 대시보드가 브라우저에서 열려 있을 때 작동합니다. 새 CSV 관측 기록이 파일에 들어오지 않으면 새 고장은 감지되지 않습니다. 브라우저가 닫혀 있어도 실행되는 별도 작업, 실시간 장비 수신, 다음 날 고장 예측 모델 연결, 대규모 트래픽용 비동기 처리는 아직 구현하지 않았습니다.
