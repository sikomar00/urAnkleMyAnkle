# 산업 모델 실행

PowerShell에서 프로젝트의 가상환경 Python을 사용합니다. 활성화는 필요 없습니다.

```powershell
cd C:\urAnkleMyAnkle

# 현재 상태 분류
.\venv\Scripts\python.exe src\current_state_model.py

# 향후 7일 고장 위험 예측
.\venv\Scripts\python.exe src\timeseries_model.py
```

공통 실행기로도 같은 작업을 실행할 수 있습니다.

```powershell
.\venv\Scripts\python.exe -m src.industrial_training --mode current
.\venv\Scripts\python.exe -m src.industrial_training --mode forecast
```

기본 입력은 프로젝트 기준 `data/raw/synthetic_industrial_machine_data.csv`입니다.
현재 상태 결과는 `outputs/current_state`, 미래 예측 결과는 `outputs/timeseries`에 저장됩니다.
각 폴더에는 `metrics.csv`, `test_predictions.csv`, `models/`가 생성됩니다.
같은 출력 폴더로 재실행하면 해당 결과 파일을 갱신합니다.

기본 범위는 전체 + machine_type별 + asset_tag별입니다. `[Training]`은 진행,
`[Done]`은 결과 저장 완료를 뜻합니다. `skipped`는 날짜별 데이터가 부족하거나
학습 정답이 한 종류여서 건너뛴 경우입니다.

`industrial_features.py`는 공통 함수 모음이므로 따로 실행하지 않습니다.
Windows에서는 CPU 사용량을 기본 최대 4개로 설정하며, 사용자가 지정한
`LOKY_MAX_CPU_COUNT` 값은 존중합니다.

다른 데이터/출력 폴더가 필요할 때만 `--data`, `--output`을 지정합니다.
기본 경로는 실행 위치와 무관하고, 직접 전달하는 상대 경로는 현재 터미널 위치 기준입니다.

주의: 현재 날짜 분할은 학습 2024년 이전, 검증 2024년 1~6월, 테스트 2024년 7월 이후입니다.
다른 기간의 데이터를 쓰려면 분할 기준도 검토해야 합니다.
미래 모델은 과거 변수로 미래 라벨을 예측하는 분류 모델이며, EDA 그래프 생성은 별도 작업입니다.
