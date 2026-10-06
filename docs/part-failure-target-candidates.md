# 부품 고장 예측 정답 후보 6가지

이번 실험은 `synthetic_industrial_machine_data.csv`의 **개별 설비·부품·날짜별 `breakdown_flag`**를 정답으로 사용합니다. 부품 고장 표시와 실제 설비 전체 정지는 다릅니다. 목적은 새 입력변수를 발굴하는 것이 아니라 **같은 입력변수와 전처리로 예측할 정답의 정의를 비교**하는 것입니다.

## 실행 파일과 정답

| 실행 파일 | 관측 단위 | 정답 |
| --- | --- | --- |
| `src/pf_next_day.py` | 설비 × 부품 × 오늘 | 내일 해당 부품의 고장 표시 여부 |
| `src/pf_within_3d.py` | 설비 × 부품 × 오늘 | 향후 1~3일 중 해당 부품의 고장 표시가 한 번이라도 있는지 |
| `src/pf_within_7d.py` | 설비 × 부품 × 오늘 | 향후 1~7일 중 해당 부품의 고장 표시가 한 번이라도 있는지 |
| `src/pf_count_7d.py` | 설비 × 부품 × 오늘 | 향후 1~7일 동안 고장 표시된 **날짜 수**(0~7). 실제 수리 횟수가 아님 |
| `src/pf_first_day_class.py` | 설비 × 부품 × 오늘 | 첫 표시 시점: 0=7일 내 없음, 1=4~7일 뒤, 2=2~3일 뒤, 3=내일 |
| `src/pfam_within_3d.py` | 설비 × 부품군 × 오늘 | 오늘 해당 부품군의 모든 세부 부품이 정상이고 향후 1~3일 중 **어느 세부 부품이든** 고장 표시되는지 |

각 파일은 독립적으로 실행할 수 있습니다. 공통 계산은 `src/pf_common.py`에 있으며, 기존 `src/data_huijae2.py`의 C 실험(부품 정보·이전 고장 이력·당일 센서)에서 **입력변수와 로지스틱 회귀 설정**을 재사용합니다. 부품 단위 입력에는 단가, 이전 7·30일 고장 표시일 수, 마지막 표시 이후 경과일, 이전 표시 유무, 당일 센서 8개와 부품 번호·부품군·중요도·단위가 포함됩니다. `asset_tag`, 당일 `breakdown_flag`, `wo_type`, 출고 정보와 미래 정보는 모델 입력에서 제외합니다. 부품군 실험은 부품 번호가 더는 한 행을 가리키지 않으므로, 부품군·설비 종류·공장과 부품군의 과거 이력·단가 합계·당일 센서를 사용합니다. 부품군 9개를 한 모델에서 함께 학습합니다.

## 날짜와 평가

원본의 날짜 순서와 설비·부품(군)별 달력 날짜 누락을 먼저 확인합니다. 미래 정답은 **전체 시계열에 실제 날짜를 맞춰 만든 뒤** 오늘 고장 표시가 0인 행을 선택합니다. 중간 날짜가 빠졌거나 미래 확인 기간이 끝까지 없는 행에는 정답을 붙이지 않습니다. 기존 실험과 같은 달력 날짜 경계로 Train 70%, Validation 15%, Test 15%를 나누고, 미래 정답 기간이 다음 구간으로 넘어가는 행은 제외합니다. 이전 30일 기록이 완전하지 않은 초기 행도 기존 C 실험과 같은 입력 조건을 위해 제외합니다. 터미널과 `exclusions.csv`에 구간별 제외 이유가 나옵니다.

결측 대체·표준화·범주 변환은 Train에서만 학습합니다. 이진 분류는 기존 설정의 로지스틱 회귀와 랜덤 포레스트를 사용하며, 경보 확률 기준은 모두 **0.5로 고정**합니다. Validation 또는 Test 성능에 맞춰 기준을 변경하지 않았습니다. 회귀는 Train 평균 예측, 릿지 회귀, 랜덤 포레스트 회귀를 비교합니다. 4단계 분류는 로지스틱 회귀와 랜덤 포레스트를 비교합니다. 모델은 모든 설비의 Train을 합쳐 학습하고, 설비별 성능은 Test 결과를 나눠 표시합니다.

## 결과 폴더와 파일

각 파일의 결과 폴더는 `outputs/pf_next_day`, `outputs/pf_within_3d`, `outputs/pf_within_7d`, `outputs/pf_count_7d`, `outputs/pf_first_day_class`, `outputs/pf_family_within_3d`입니다.

- `overall_results.csv`: 전체 Test 모델 성능입니다. 이진 분류는 Accuracy·Precision·Recall·F1·PR-AUC·TN·FP·FN·TP, 회귀는 MAE·RMSE·R², 4단계 분류는 Accuracy·Macro F1을 담습니다.
- `asset_results.csv`: 같은 Test 결과를 `asset_tag`별로 나눈 표입니다. 모델을 설비별로 다시 학습한 결과가 아닙니다.
- `predictions.csv`: Test의 실제 정답과 모델별 예측값 또는 확률입니다.
- `date_splits.csv`, `exclusions.csv`, `label_distribution.csv`: 사용 날짜·행 수, 제외 이유, 각 구간의 정답 분포입니다.
- 4단계 분류에만 `class_distribution.csv`, `per_class_results.csv`, `confusion_matrix.csv`가 추가됩니다.

현재 원본의 Test에서 다음 날 부품 고장 표시 비율은 약 **9.64%**이며, 확률 0.5 기준 로지스틱 회귀는 경보를 한 건도 내지 않아 Recall과 F1이 0입니다. 향후 3일은 약 **26.22%**, 7일은 약 **50.62%**입니다. 7일 고장 표시일 수 회귀의 릿지 모델은 MAE **0.634**, R² **0.054**로 설명력이 낮습니다. 4단계 분류의 가장 높은 Macro F1도 약 **0.239**입니다. 성능을 좋게 보이도록 Test에 맞춰 변수를 바꾸거나 경보 기준을 조정하지 않았습니다.

부품군 3일 정답의 Test 고장 비율은 약 **47.38%**로, 개별 부품 3일 정답의 약 **26.22%**와 관측 단위가 다릅니다. 이 둘의 Accuracy나 F1을 그대로 비교해 어느 모델이 우수하다고 결론 내릴 수 없습니다. 동일한 설비·부품이 여러 날짜에 반복되고 미래 기간도 겹치므로 행별 성능이 독립적인 고장 사건 수를 뜻하지 않습니다. 이전 실험에서도 같은 Test 기간을 본 이력이 있어 완전히 새로운 외부 시험 자료는 아닙니다.

## 실행

프로젝트 폴더에서 아래 명령을 하나씩 실행합니다. 다른 CSV는 뒤에 `--data "C:\경로\파일.csv"`를 붙입니다.

```powershell
& .\.venv\Scripts\python.exe .\src\pf_next_day.py
& .\.venv\Scripts\python.exe .\src\pf_within_3d.py
& .\.venv\Scripts\python.exe .\src\pf_within_7d.py
& .\.venv\Scripts\python.exe .\src\pf_count_7d.py
& .\.venv\Scripts\python.exe .\src\pf_first_day_class.py
& .\.venv\Scripts\python.exe .\src\pfam_within_3d.py
```

`tests/test_pf_common.py`는 달력 날짜 누락, 첫 고장 표시 시점, 7일 표시일 수와 부품군의 세부 부품 집계를 검사합니다. 향후 몇 일 앞서 경보를 내는 방식과 실제 정비 의미는 이 후보 결과를 보고 별도로 정해야 합니다.
