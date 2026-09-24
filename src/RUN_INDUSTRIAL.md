# 산업 위험 모델 실행 방법

이 프로젝트는 장비/부품과 당일/미래를 섞지 않고 네 과제로 따로 실행합니다.
터미널의 현재 위치는 프로젝트 루트(`urAnkleMyAnkle`)여야 합니다.

## 1. macOS·Linux

가상환경을 활성화했다면 다음처럼 실행합니다.

```bash
python src/current_asset_model.py
python src/forecast_asset_model.py --horizon 7 --risk-definition new
python src/forecast_asset_model.py --horizon 7 --risk-definition any
python src/current_part_model.py
python src/forecast_part_model.py --horizon 7
python src/eda.py
```

가상환경을 활성화하지 않았다면 이 프로젝트의 환경 경로를 직접 사용해도 됩니다.

```bash
/Users/kodohyeon/Documents/project_LS/venv/bin/python src/current_asset_model.py
```

## 2. Windows PowerShell

```powershell
.\venv\Scripts\python.exe src\current_asset_model.py
.\venv\Scripts\python.exe src\forecast_asset_model.py --horizon 7 --risk-definition new
.\venv\Scripts\python.exe src\forecast_asset_model.py --horizon 7 --risk-definition any
.\venv\Scripts\python.exe src\current_part_model.py
.\venv\Scripts\python.exe src\forecast_part_model.py --horizon 7
.\venv\Scripts\python.exe src\eda.py
```

테스트 파일은 직접 `python tests/test_....py`로 실행하지 말고 프로젝트 루트에서
다음처럼 pytest 모듈로 실행합니다.

```bash
python -m pytest -v
```

## 3. 네 과제와 기본 출력

| 실행 파일 | 한 행의 단위 | 예측 대상 | 기본 출력 |
|---|---|---|---|
| `current_asset_model.py` | 장비·날짜 | 당일 고장점수 12/13/14 이상 | `outputs/asset_current` |
| `forecast_asset_model.py` | 장비·날짜 | 향후 7일 신규 또는 전체 위험 | `outputs/asset_forecast_7d` |
| `current_part_model.py` | 장비·부품·날짜 | 당일 `breakdown_flag` | `outputs/part_current` |
| `forecast_part_model.py` | 장비·부품·날짜 | 향후 7일 신규 고장 | `outputs/part_forecast_7d` |

`new`는 오늘 이미 위험한 장비를 제외하고 새로 위험해질 장비를 찾습니다. `any`는
현재 위험의 지속도 포함합니다. 운영상 당일 탐지 후 예방점검 대상을 고를 때는
`new`가 기본이며, 전체 위험 상태를 보고 싶을 때만 `any`를 비교합니다.

## 4. 자주 쓰는 선택 인자

```bash
# 장비 점수 기준 하나만 실행
python src/current_asset_model.py --score-threshold 12

# 전체 범위만 빠르게 실행
python src/forecast_part_model.py --scope overall --max-iter 20

# Precision 0.50 이상 중 Recall이 가장 높은 임계값 정책만 실행
python src/current_part_model.py --threshold-policy min_precision --min-precision 0.50

# 별도 입력과 출력 사용
python src/current_asset_model.py --data data/raw/my_data.csv --output outputs/my_run
```

`--threshold-policy`는 `f1`, `min_precision`, `top_fraction`을 반복 지정할 수 있고,
생략하면 세 정책을 모두 평가합니다. 기본 날짜 분할은 학습 2024년 이전, 검증
2024-01-01~2024-06-30, 테스트 2024-07-01 이후입니다.

## 5. 사용 모델과 결과 파일

각 과제는 다음 세 모델을 학습하여 검증 Average Precision이 가장 높은 하나를 고릅니다.

1. `LogisticRegression`: 선형 기준모델, `class_weight="balanced"`
2. `RandomForestClassifier`: 비선형 트리 앙상블
3. `HistGradientBoostingClassifier`: 비선형 부스팅 모델

LightGBM은 macOS의 `libomp` 같은 네이티브 의존성 문제를 피하기 위해 기본 흐름에서
제외했습니다. 출력 폴더에는 다음 파일이 생깁니다.

- `metrics.csv`: 검증·테스트 지표와 선택 모델 표시
- `test_predictions.csv`: 선택 모델의 테스트 위험점수와 예측
- `feature_importance.csv`: 선택 모델의 permutation importance
- `run_config.json`: 실행 설정
- `models/`: 다시 불러올 수 있는 선택 모델

`src/eda.py`는 이 결과들을 읽어 선택된 테스트 결과만
`outputs/model_comparison.csv`로 합칩니다. 모델을 다시 학습하지 않습니다.

## 6. 기존 명령 호환

기존 `current_state_model.py`는 새 `current_part_model.py`로,
`timeseries_model.py`는 새 `forecast_part_model.py`로 연결됩니다. 기존 기본 출력 경로인
`outputs/current_state`, `outputs/timeseries`도 유지합니다.
