# 프로젝트명 (TODO: 팀에서 정한 이름으로 교체)

전력·생산·기상 데이터로 에너지 사용량·수요를 분석하고, AI4I 2020
예지보전 데이터로 설비 고장 위험을 예측한 뒤, 두 결과를 Plotly
Dash 대시보드로 통합한 프로젝트입니다.

레포 폴더 구조 전체 설명은 루트의 `CONTRIBUTING.md`를 먼저 읽어주세요.

## 팀 구성 및 역할 (TODO)

| 이름 | 담당 | 기여도 |
|---|---|---|
| | 에너지 트랙 | |
| | AI4I 트랙 | |
| | 통합·문서 | |

## 기술 스택 (TODO)

- 언어/라이브러리: Python, pandas, scikit-learn, Prophet, Dash
- AI 도구: (사용한 도구와 용도를 적으세요. 예: Claude Code — 코드
  초안·디버깅 보조. 모든 코드는 팀원이 검토·이해한 상태로 병합함)

## 실행 순서

1. 가상환경 생성 및 활성화

   **macOS / Linux**
   ```bash
   python3.11 -m venv venv
   source venv/bin/activate
```

   **Windows (PowerShell)**
   ```powershell
   py -3.11 -m venv venv
   venv\Scripts\activate
   ```
   > `python`을 쳤을 때 Microsoft Store가 열리면: 설정 → 앱 → 고급 앱 설정
   > → 앱 실행 별칭 에서 python.exe / python3.exe를 끄세요.
   > `py -3.11`이 없다고 나오면 python.org에서 3.11을 설치하세요.
2. 패키지 설치
   ```
   pip install -r requirements.txt
   ```
3. 원본 데이터 다운로드 → `data/raw/`에 위치 (출처: DATA_SOURCES.md)
4. 모델 학습 (outputs/, models/ 생성)
   ```
   python -m src.energy_train
   python -m src.ai4i_train
   ```
5. Dash 앱 실행
   ```
   python -m app.app
   ```

> ⚠️ **반드시 레포 루트에서 `-m` 옵션으로 실행하세요.**
> `python app/app.py` 처럼 실행하면 `sys.path[0]`이 `app/` 폴더가 되어
> `from src...` import가 전부 `ModuleNotFoundError`로 실패합니다.

## 재현 시 자동 생성되는 파일 (Git에 없는 것이 정상)

- `data/raw/`, `data/processed/` 내부 파일 — 3번 단계에서 생성
- `models/rf.joblib` — 4번 단계에서 생성
- `outputs/*.csv`의 실제 값 — 4번 단계 실행 시 더미 데이터가
  실제 결과로 덮어써짐 (더미는 `python -m scripts.make_dummy`로 재생성)

## 문서

- 데이터 출처: `DATA_SOURCES.md`
- 데이터 계약(outputs 컬럼 구조): `docs/data_contract.md`
- 트러블슈팅 기록: `docs/troubleshooting_log.md`
- 의사결정 기록: `docs/decision_log.md`

## 한계 및 향후 개선 (TODO)

-

## 라이선스

- **코드**: MIT (`LICENSE` 참조)
- **데이터**: 각 출처의 라이선스를 따릅니다 (`DATA_SOURCES.md` 참조).
  본 저장소의 MIT 라이선스는 데이터에 적용되지 않습니다.
  AI4I 2020은 CC BY 4.0 (출처 표기 조건).
