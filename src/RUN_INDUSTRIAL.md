# 산업 위험 모델 실행 방법

이 프로젝트는 장비/부품과 당일/미래를 섞지 않고 네 과제로 따로 실행합니다.
터미널의 현재 위치는 프로젝트 루트(`urAnkleMyAnkle`)여야 합니다.

## 1. macOS·Linux

가상환경을 활성화했다면 다음처럼 실행합니다.

```bash
python src/current_asset_model.py
python src/current_asset_score_model.py
python src/current_asset_severity_experiments.py \
  --data dataVerification/synthetic_industrial_machine_data.csv \
  --high-risk-thresholds 12 13 --feature-sets A B C D --scope all --max-iter 100
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
.\venv\Scripts\python.exe src\current_asset_score_model.py
.\venv\Scripts\python.exe src\current_asset_severity_experiments.py `
  --data dataVerification\synthetic_industrial_machine_data.csv `
  --high-risk-thresholds 12 13 --feature-sets A B C D --scope all --max-iter 100
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
| `current_asset_score_model.py` | 장비·날짜 | 당일 `failure_points`와 4단계 위험도 | `outputs/asset_score_current` |
| `current_asset_severity_experiments.py` | 장비·날짜 | 12·13점 4단계와 센서 Feature A~D 비교 | `outputs/asset_severity_experiments` |
| `forecast_asset_model.py` | 장비·날짜 | 향후 7일 신규 또는 전체 위험 | `outputs/asset_forecast_7d_new` 또는 `_any` |
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

### 장비 점수·4단계 모델만 실행하기

```bash
python src/current_asset_score_model.py \
  --data dataVerification/synthetic_industrial_machine_data.csv \
  --scope all \
  --max-iter 100
```

장비 일별 고장점수 (S)는 `normal`((S=0)), `caution`((1\le S\le5)),
`risk`((6\le S\le11)), `high_risk`((S\ge12))로 표시합니다.
`high_risk`는 부품별 `breakdown_flag`에 중요도 가중치를 적용한 대리 지표이며,
실제 기계 정지·생산 손실이 확인됐다는 뜻은 아닙니다.

## 5. 사용 모델과 결과 파일

각 과제는 Dummy 양성률 기준선과 다음 세 학습모델을 비교합니다. 최종 모델은 세
학습모델 중 검증 Average Precision이 가장 높은 하나를 고릅니다.

1. `DummyClassifier`: 학습 양성률만 사용하는 기준선
2. `LogisticRegression`: 표준화된 수치 Feature를 쓰는 선형 모델
3. `RandomForestClassifier`: 비선형 트리 앙상블
4. `HistGradientBoostingClassifier`: 비선형 부스팅 모델

LightGBM은 macOS의 `libomp` 같은 네이티브 의존성 문제를 피하기 위해 기본 흐름에서
제외했습니다. 출력 폴더에는 다음 파일이 생깁니다.

- `metrics.csv`: 검증·테스트 지표와 선택 모델 표시
- `test_predictions.csv`: 선택 모델의 테스트 위험점수와 예측
- `feature_importance.csv`: 선택 모델의 permutation importance
- `run_config.json`: 실행 설정
- `models/`: 다시 불러올 수 있는 선택 모델

`src/eda.py`는 이 결과들을 읽어 선택된 테스트 결과만
`outputs/model_comparison.csv`로 합칩니다. 모델을 다시 학습하지 않습니다.

`current_asset_score_model.py`는 별도로 다음을 생성합니다.

- `regression_metrics.csv`: 점수 회귀의 MAE, RMSE, (R^2), Spearman과 4단계 환산 지표
- `severity_metrics.csv`: 4단계 분류의 Macro/Weighted F1과 등급별 지표
- `test_predictions.csv`: 실제·예측 점수, 등급, 네 등급의 예측 확률
- `feature_importance.csv`: 회귀와 분류 선택 모델의 permutation importance
- `run_config.json`, `models/`: 실행 설정과 다시 불러올 수 있는 선택 모델

이 실험에서는 Accuracy만으로 판단하지 말고 `macro_f1`과
`high_risk_precision`, `high_risk_recall`, `high_risk_binary_f1`을 함께 봐야 합니다.

### 센서 이상·이력 Feature 비교 실험

```bash
/Users/kodohyeon/Documents/project_LS/venv/bin/python src/current_asset_severity_experiments.py \
  --data dataVerification/synthetic_industrial_machine_data.csv \
  --high-risk-thresholds 12 13 \
  --feature-sets A B C D \
  --scope all \
  --max-iter 100
```

이 명령은 12점과 13점 고위험 기준을 각각 A~D Feature로 평가합니다. 결과는
`outputs/asset_severity_experiments/`에 저장되며, 핵심 파일은 다음과 같습니다.

- `experiment_summary.md`: 12·13점, A~D, 기계·장비별 결과를 정리한 한글 보고서
- `experiment_metrics.csv`: 모든 모델의 검증·테스트 지표와 선택 여부
- `class_metrics.csv`, `confusion_matrices.csv`: 등급별 지표와 혼동행렬
- `machine_failure_profile.csv`: 서로 구분한 세 가지 발생률
- `test_predictions.csv`, `feature_importance.csv`: 테스트 예측과 Feature 중요도
- `zscore_baselines.csv`: 학습 정상행으로 고정한 센서별 기준
- `run_config.json`, `models/`: 실행 설정과 다시 사용할 수 있는 모델

`machine_failure_profile.csv`의 세 비율은 의미가 다릅니다. 부품 행 기준
`part_breakdown_rate`, 점수가 1 이상인 장비일 기준 `asset_issue_day_rate`, 임계값
이상인 장비일 기준 `asset_high_risk_day_rate`이며, 어느 것도 실제 기계 정지율로
간주하면 안 됩니다.

## 6. 기존 명령 호환

기존 `current_state_model.py`는 새 `current_part_model.py`로,
`timeseries_model.py`는 새 `forecast_part_model.py`로 연결됩니다. 기존 기본 출력 경로인
`outputs/current_state`, `outputs/timeseries`도 유지합니다.
