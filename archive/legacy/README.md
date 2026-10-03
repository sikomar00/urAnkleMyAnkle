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
