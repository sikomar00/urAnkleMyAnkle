# 프로젝트명 (TODO: 팀에서 정한 이름으로 교체)

전력·생산·기상 데이터로 에너지 사용량·수요를 분석하고, AI4I 2020
예지보전 데이터로 설비 고장 위험을 예측한 뒤, 두 결과를 Plotly
Dash 대시보드로 통합한 프로젝트입니다.

레포 폴더 구조 전체 설명은 루트의 `가이드.txt`를 먼저 읽어주세요.

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
   ```
   python -m venv venv
   source venv/bin/activate   # Windows는 venv\Scripts\activate
   ```
2. 패키지 설치
   ```
   pip install -r requirements.txt
   ```
3. 원본 데이터 다운로드 → `data/raw/`에 위치 (출처: DATA_SOURCES.md)
4. 모델 학습 (outputs/, models/ 생성)
   ```
   python src/energy_train.py
   python src/ai4i_train.py
   ```
5. Dash 앱 실행
   ```
   python app/app.py
   ```

## 재현 시 자동 생성되는 파일 (Git에 없는 것이 정상)

- `data/raw/`, `data/processed/` 내부 파일 — 3번 단계에서 생성
- `models/rf.joblib` — 4번 단계에서 생성
- `outputs/*.csv`의 실제 값 — 4번 단계 실행 시 더미 데이터가
  실제 결과로 덮어써짐

## 문서

- 데이터 출처: `DATA_SOURCES.md`
- 데이터 계약(outputs 컬럼 구조): `docs/data_contract.md`
- 트러블슈팅 기록: `docs/troubleshooting_log.md`
- 의사결정 기록: `docs/decision_log.md`

## 한계 및 향후 개선 (TODO)

-
