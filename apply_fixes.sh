#!/usr/bin/env bash
# =====================================================================
# urAnkleMyAnkle 레포 구조 수정 스크립트
#
# 사용법: 레포 루트에서 실행
#     bash apply_fixes.sh
#
# 이 스크립트가 하는 일:
#   1. .DS_Store 추적 해제
#   2. requirements.txt 버전 고정
#   3. import 경로 문제 해결 (__init__.py, pyproject.toml)
#   4. pytest 실행 가능하게 수정
#   5. 데이터 계약 확장 (요건 공백 메우기)
#   6. 더미 CSV 60행으로 확장
#   7. 폴더설명.txt -> README.md 변환
#   8. README 실행 순서 수정
#   9. .gitattributes, .python-version, CI 추가
#
# 하지 않는 일: 모델링 로직 구현 (TODO 그대로 둠 — 학습 대상)
# =====================================================================
set -e

if [ ! -f "requirements.txt" ] || [ ! -d "app/pages" ]; then
  echo "❌ 레포 루트가 아닙니다. urAnkleMyAnkle 폴더에서 실행하세요."
  exit 1
fi

echo "▶ 1/9  .DS_Store 추적 해제"
git rm --cached -q .DS_Store app/.DS_Store data/.DS_Store 2>/dev/null || true
find . -name ".DS_Store" -not -path "./.git/*" -delete 2>/dev/null || true

echo "▶ 2/9  requirements.txt 버전 고정"
cat > requirements.txt << 'REQ'
# =====================================================================
# 버전 고정 원칙
#   - 팀원이 서로 다른 날 설치해도 같은 버전이 깔려야 함
#   - joblib 모델은 scikit-learn 버전이 다르면 로드 시 경고/오류 발생
#   - 2026년 9월 현재 최신은 pandas 3.0 / dash 4.4 / numpy 2.5 이지만,
#     교재 코드 호환을 위해 2024년 말 스냅샷으로 고정함
#
# ⚠️ 교재가 dash 3.x 이상 기준이면 아래 두 줄만 교체:
#       dash==3.4.0
#       plotly==6.5.2
#    (dash 3.0부터 app.run_server() 제거 -> app.run() 사용)
#
# Python 3.11 전제 (.python-version 참조)
# =====================================================================

# === 데이터 처리 ===
numpy==1.26.4
pandas==2.2.3
scipy==1.13.1

# === 머신러닝 ===
scikit-learn==1.6.1
joblib==1.4.2

# === 시계열 예측 ===
prophet==1.1.6
holidays==0.58          # prophet 의존. 핀 없으면 prophet을 깨뜨린 전력 있음

# === 시각화 / Dash ===
plotly==5.24.1
dash==2.18.2

# === 노트북 ===
ipykernel==6.29.5

# === 협업 도구 ===
nbstripout==0.7.1
pytest==8.3.3
REQ

echo "▶ 3/9  import 경로 수정"
touch app/__init__.py app/pages/__init__.py
cat > pyproject.toml << 'PYP'
# import 경로 문제 해결용 최소 설정
#
# 이 파일이 없으면:
#   - `pytest` 실행 시 ModuleNotFoundError: No module named 'src'
#   - `python app/app.py` 실행 시 같은 에러
# pythonpath = ["."] 가 프로젝트 루트를 import 경로에 넣어줍니다.

[project]
name = "ls-jumpup-pdm"
version = "0.1.0"
requires-python = ">=3.11"

[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]
PYP
echo "3.11" > .python-version

echo "▶ 4/9  app/app.py 탭 전환 콜백 연결"
cat > app/app.py << 'APP'
"""
Dash 앱 진입점. 화면 5개(app/pages/*)를 탭으로 묶어서 실행한다.

실행 (반드시 레포 루트에서):
    python -m app.app

⚠️ `python app/app.py` 로 실행하면 sys.path[0]이 app/ 폴더가 되어
   `from src...` import가 전부 실패합니다. -m 옵션을 쓰면 현재
   작업 디렉터리가 import 경로에 들어갑니다.
"""
from dash import Dash, Input, Output, dcc, html

from app.pages import ai4i, data_lookup, energy_status, forecast, report_summary

# 탭 value -> (표시 이름, 해당 page 모듈)
TABS = {
    "tab-energy": ("에너지 현황", energy_status),
    "tab-forecast": ("예측", forecast),
    "tab-ai4i": ("AI4I 예지보전", ai4i),
    "tab-data": ("데이터 조회", data_lookup),
    "tab-report": ("보고서 요약", report_summary),
}

app = Dash(__name__, suppress_callback_exceptions=True)
server = app.server

app.layout = html.Div([
    html.H1("예지보전·에너지 통합 대시보드"),
    dcc.Tabs(
        id="main-tabs",
        value="tab-energy",
        children=[dcc.Tab(label=name, value=key) for key, (name, _) in TABS.items()],
    ),
    html.Div(id="tab-content"),
])


@app.callback(Output("tab-content", "children"), Input("main-tabs", "value"))
def render_tab(value):
    """선택된 탭의 page 모듈에서 layout을 꺼내 화면에 꽂는다."""
    _, module = TABS[value]
    return module.layout


# 각 page가 register_callbacks(app)을 정의했으면 등록.
# 아직 없는 page는 건너뛰므로, 구현 순서에 상관없이 앱이 뜬다.
for _name, _module in TABS.values():
    _register = getattr(_module, "register_callbacks", None)
    if callable(_register):
        _register(app)


if __name__ == "__main__":
    app.run(debug=True)
APP

echo "▶ 5/9  tests 수정 (구현 전까지 xfail)"
cat > tests/test_features.py << 'TST'
"""
src/features.py 함수의 최소 동작 확인용 테스트.

실행 (레포 루트에서):
    pytest

xfail 표시는 make_ai4i_features 구현이 끝나면 제거하세요.
제거 후 이 테스트가 통과해야 파생변수 로직이 계약대로 동작한다는
증거가 됩니다.
"""
import pandas as pd
import pytest

from src.features import make_ai4i_features


@pytest.mark.xfail(reason="make_ai4i_features 구현 전", strict=False)
def test_make_ai4i_features_adds_expected_columns():
    df = pd.DataFrame({
        "air_k": [300.0],
        "process_k": [310.0],
        "rpm": [1500],
        "torque": [40.0],
        "tool_wear": [10],
    })
    result = make_ai4i_features(df)
    for col in ["temp_diff", "power_w", "strain"]:
        assert col in result.columns


@pytest.mark.xfail(reason="make_ai4i_features 구현 전", strict=False)
def test_make_ai4i_features_does_not_mutate_input():
    """원본 DataFrame을 건드리지 않아야 한다 (학습/추론 재현성)."""
    df = pd.DataFrame({
        "air_k": [300.0], "process_k": [310.0],
        "rpm": [1500], "torque": [40.0], "tool_wear": [10],
    })
    before = df.columns.tolist()
    make_ai4i_features(df)
    assert df.columns.tolist() == before
TST

echo "▶ 6/9  데이터 계약 확장"
cat > docs/data_contract.md << 'CON'
# 데이터 계약 (outputs/ 폴더 파일 스키마)

**이 문서가 outputs/ 폴더의 실제 csv 파일 구조와 항상 일치해야
합니다.** 컬럼을 바꾸려면 이 문서를 먼저 수정하고 팀에 공유한
다음 코드를 바꾸세요.

> ⚠️ Dash는 `outputs/` 만 읽습니다. `data/processed/` 는 .gitignore로
> 제외되므로, 화면에 필요한 결과는 반드시 `outputs/` 에 저장하세요.

---

## 에너지 트랙

### outputs/energy_daily.csv — 일 단위 집계 (에너지 현황 화면)

| 컬럼 | 타입 | 설명 |
|---|---|---|
| date | date | 일자 |
| region | string | 지역/공장 구분 (필터용) |
| usage_kwh | float | 일 총 사용량 |
| peak_kw | float | 일 최대부하 |
| load_factor | float | 부하율 = 평균부하 / 최대부하 |
| avg_temp | float | 평균기온 (기상 병합 결과) |

### outputs/energy_forecast.csv — 예측 결과

| 컬럼 | 타입 | 설명 |
|---|---|---|
| ds | date | 날짜 |
| y_actual | float | 실제값 (test 기간만, train 기간은 결측) |
| yhat | float | 예측값 |
| yhat_lower | float | 예측구간 하한 (baseline/linreg는 결측 가능) |
| yhat_upper | float | 예측구간 상한 |
| model | string | `baseline` / `linreg` / `prophet` |

### outputs/energy_scores.csv — 모델 성능 비교

| 컬럼 | 타입 | 설명 |
|---|---|---|
| model | string | `baseline` / `linreg` / `prophet` |
| MAE | float | |
| RMSE | float | |
| R2 | float | |
| n_test | int | 평가에 쓴 행 수 |
| horizon_days | int | 예측 지평(일). baseline과 동일해야 공정한 비교 |

> **요건 대응**: "회귀(기준모델 대비 MAE/RMSE/R²)"는 `linreg` vs
> `baseline` 비교, "Prophet 시계열 예측"은 `prophet` 행입니다.
> 셋 다 있어야 요건을 채웁니다.

---

## AI4I 트랙

### outputs/ai4i_metrics.csv — 분류 성능 비교 + 임계값 변화표

| 컬럼 | 타입 | 설명 |
|---|---|---|
| model | string | `baseline` / `tree` / `rf` |
| threshold | float | 판정 임계값 (0.5, 0.3, 0.2 …) |
| precision | float | test 세트 기준 |
| recall | float | test 세트 기준 |
| f1 | float | test 세트 기준 |
| tn / fp / fn / tp | int | 혼동행렬 |
| n_train | int | 학습 행 수 |
| n_test | int | 평가 행 수 |
| pos_rate_train | float | 학습셋 고장 비율 |
| pos_rate_test | float | 평가셋 고장 비율 (stratify 검증용) |

> `model` + `threshold` 조합이 행의 식별자입니다. 임계값을 바꿔
> 여러 행을 넣을 때 `model` 만으로는 중복이 되어 Dash 필터가 꼬입니다.

### outputs/ai4i_scored.csv — 행 단위 결과

| 컬럼 | 타입 | 설명 |
|---|---|---|
| udi | int | 식별자 (표시용, 모델 입력 아님) |
| type | string | L/M/H |
| air_k | float | Air temperature [K] |
| process_k | float | Process temperature [K] |
| rpm | float | Rotational speed |
| torque | float | Torque [Nm] |
| tool_wear | float | Tool wear [min] |
| failure | int | 실제 Machine failure (0/1) |
| split | string | train / test |
| proba | float | RF 모델의 고장 확률 예측값 |
| cluster | int | K-Means 군집 번호 |
| anomaly | int | 거리 기반 이상치 여부 (0/1) |

### outputs/ai4i_kmeans_selection.csv — k 선택 근거

| 컬럼 | 타입 | 설명 |
|---|---|---|
| k | int | 군집 개수 |
| inertia | float | 관성(SSE). 엘보우 판단용 |
| silhouette | float | 실루엣 계수 |

> **발표 대비**: "k를 왜 이 값으로 정했나"는 비지도 파트에서
> 가장 높은 확률로 들어오는 질문입니다. 이 파일이 답변 근거입니다.

### outputs/ai4i_cluster_profile.csv — 군집별 특징

| 컬럼 | 타입 | 설명 |
|---|---|---|
| cluster | int | 군집 번호 |
| n | int | 소속 행 수 |
| failure_rate | float | 군집 내 실제 고장 비율 |
| dist_mean | float | 중심까지 평균 거리 |
| air_k_mean / process_k_mean / rpm_mean / torque_mean / tool_wear_mean | float | 군집 중심 해석용 |

> **요건 대응**: "정상/이상 군집의 특징과 한계 해석" 서술의 근거표.

### outputs/ai4i_classification_report.txt

sklearn `classification_report` 출력을 그대로 저장. 요건 명시 항목.

---

## 업로드 CSV 형식 (데이터 조회 화면)

`src/validate.py` 의 `REQUIRED_COLUMNS` 는 **개명 후** 컬럼명
(`air_k`, `process_k` …)을 기준으로 합니다. AI4I 원본 CSV의 컬럼명은
`Air temperature [K]` 형태라서 그대로 올리면 검증에 실패합니다.

**팀 결정 (TODO: 택 1 하고 나머지 줄 삭제)**
- [ ] **A. 관대하게** — `validate_ai4i_csv` 가 원본 컬럼명도 받아서
      내부에서 rename. 심사자가 원본 파일을 그냥 올려도 동작.
- [ ] **B. 엄격하게** — `outputs/sample_upload.csv` 를 제공하고
      화면에 다운로드 버튼 + "이 형식으로 올리세요" 안내.

컬럼 매핑표는 `src/features.py` 의 `COLUMN_RENAME_MAP` 에 있습니다.

---

## 변경 이력

| 날짜 | 변경 내용 | 변경자 |
|---|---|---|
| 2026-09-18 | 최초 작성 | |
| 2026-09-18 | threshold·클래스비율 컬럼 추가, energy_daily / kmeans_selection / cluster_profile / classification_report 신설, linreg 추가, 업로드 형식 절 추가 | |
CON

echo "▶ 7/9  더미 데이터 생성 스크립트 + 실행"
mkdir -p scripts
cat > scripts/make_dummy.py << 'DUM'
"""
더미(가짜) 결과 파일 생성기.

왜 필요한가:
  Dash 담당자는 모델 결과가 나오기 전에도 화면을 만들어야 합니다.
  4행짜리 더미로는 차트 레이아웃을 잡을 수 없어서 60행으로 만듭니다.
  컬럼 구조는 docs/data_contract.md 와 완전히 동일합니다.

실행 (레포 루트에서):
    python -m scripts.make_dummy

실제 모델 결과가 나오면 같은 파일명으로 덮어쓰면 됩니다.
Dash 코드는 손댈 필요 없습니다.
"""
from pathlib import Path

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)   # 시드 고정 = 팀원 모두 같은 더미
OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

N_DAYS, N_TRAIN = 60, 45


def energy():
    ds = pd.date_range("2026-07-01", periods=N_DAYS, freq="D")
    # 주간 주기(7일)를 넣어야 Prophet 결과가 붙었을 때의 모양이 비슷해짐
    base = 12500 + 800 * np.sin(np.arange(N_DAYS) * 2 * np.pi / 7)
    y = base + RNG.normal(0, 300, N_DAYS)
    is_test = np.arange(N_DAYS) >= N_TRAIN

    rows = []
    for model, err, band in [("baseline", 850, None), ("linreg", 640, None),
                             ("prophet", 520, 700)]:
        yhat = y + RNG.normal(0, err / 3, N_DAYS)
        rows.append(pd.DataFrame({
            "ds": ds,
            "y_actual": np.where(is_test, y, np.nan),
            "yhat": yhat.round(1),
            "yhat_lower": (yhat - band).round(1) if band else np.nan,
            "yhat_upper": (yhat + band).round(1) if band else np.nan,
            "model": model,
        }))
    pd.concat(rows).to_csv(OUT / "energy_forecast.csv", index=False)

    pd.DataFrame({
        "model": ["baseline", "linreg", "prophet"],
        "MAE": [850.2, 640.1, 520.7],
        "RMSE": [1020.5, 812.4, 690.3],
        "R2": [0.41, 0.55, 0.68],
        "n_test": [N_DAYS - N_TRAIN] * 3,
        "horizon_days": [N_DAYS - N_TRAIN] * 3,
    }).to_csv(OUT / "energy_scores.csv", index=False)

    daily = []
    for region in ["A공장", "B공장"]:
        peak = base / 20 + RNG.normal(0, 30, N_DAYS)
        usage = y * (1.0 if region == "A공장" else 0.7)
        daily.append(pd.DataFrame({
            "date": ds, "region": region,
            "usage_kwh": usage.round(1),
            "peak_kw": peak.round(1),
            "load_factor": (usage / 24 / peak).round(3),
            "avg_temp": (26 + 4 * np.sin(np.arange(N_DAYS) / 9)).round(1),
        }))
    pd.concat(daily).to_csv(OUT / "energy_daily.csv", index=False)


def ai4i():
    n = 600
    failure = (RNG.random(n) < 0.034).astype(int)   # 실제 고장률 3.4%
    split = np.where(np.arange(n) < int(n * 0.8), "train", "test")
    proba = np.clip(RNG.beta(1.5, 20, n) + failure * 0.6, 0, 1)
    pd.DataFrame({
        "udi": np.arange(1, n + 1),
        "type": RNG.choice(["L", "M", "H"], n, p=[0.5, 0.3, 0.2]),
        "air_k": (298 + RNG.normal(0, 2, n)).round(1),
        "process_k": (308 + RNG.normal(0, 1, n)).round(1),
        "rpm": (1500 + RNG.normal(0, 180, n)).round(0),
        "torque": (40 + RNG.normal(0, 10, n)).round(1),
        "tool_wear": RNG.integers(0, 250, n),
        "failure": failure,
        "split": split,
        "proba": proba.round(3),
        "cluster": RNG.integers(0, 4, n),
        "anomaly": (proba > 0.5).astype(int),
    }).to_csv(OUT / "ai4i_scored.csv", index=False)

    rows = []
    for model, (p, r) in {"baseline": (0.0, 0.0), "tree": (0.55, 0.62),
                          "rf": (0.61, 0.70)}.items():
        for th in [0.5, 0.3, 0.2]:
            rr = min(r + (0.5 - th) * 0.6, 0.95) if model != "baseline" else 0.0
            pp = max(p - (0.5 - th) * 0.5, 0.05) if model != "baseline" else 0.0
            tp = int(70 * rr)
            fn = 70 - tp
            fp = int(tp / pp - tp) if pp > 0 else 0
            rows.append({
                "model": model, "threshold": th,
                "precision": round(pp, 3), "recall": round(rr, 3),
                "f1": round(2 * pp * rr / (pp + rr), 3) if pp + rr else 0.0,
                "tn": 1930 - fp, "fp": fp, "fn": fn, "tp": tp,
                "n_train": 8000, "n_test": 2000,
                "pos_rate_train": 0.0339, "pos_rate_test": 0.035,
            })
    pd.DataFrame(rows).to_csv(OUT / "ai4i_metrics.csv", index=False)

    pd.DataFrame({
        "k": [2, 3, 4, 5, 6],
        "inertia": [4820.1, 3610.4, 2980.7, 2705.2, 2540.8],
        "silhouette": [0.41, 0.46, 0.52, 0.49, 0.44],
    }).to_csv(OUT / "ai4i_kmeans_selection.csv", index=False)

    pd.DataFrame({
        "cluster": [0, 1, 2, 3],
        "n": [180, 205, 132, 83],
        "failure_rate": [0.011, 0.018, 0.045, 0.121],
        "dist_mean": [1.02, 0.94, 1.31, 1.88],
        "air_k_mean": [297.8, 298.4, 299.1, 300.2],
        "process_k_mean": [307.9, 308.6, 309.2, 310.4],
        "rpm_mean": [1602.4, 1480.1, 1390.7, 1298.3],
        "torque_mean": [33.2, 40.8, 46.1, 55.7],
        "tool_wear_mean": [62.1, 118.4, 165.9, 214.3],
    }).to_csv(OUT / "ai4i_cluster_profile.csv", index=False)

    (OUT / "ai4i_classification_report.txt").write_text(
        "(더미) src/ai4i_train.py 실행 시 classification_report 출력으로 교체됨\n",
        encoding="utf-8")


if __name__ == "__main__":
    energy()
    ai4i()
    print("더미 생성 완료:")
    for f in sorted(OUT.glob("*")):
        if f.is_file():
            print(" -", f)
DUM
touch scripts/__init__.py
python -m scripts.make_dummy

echo "▶ 7.5   src/features.py 에 컬럼 매핑표 추가"
cat > src/features.py << 'FEA'
"""
파생변수 생성 함수 모음.

노트북(notebooks/ai4i_model.ipynb)과 Dash 앱
(app/pages/data_lookup.py) 양쪽에서 똑같이 이 함수를 불러다
씁니다. 학습 때와 앱에서 전처리 방식이 달라지지 않도록 하기
위함입니다 (자세한 이유는 src/README.md 참고).
"""
import numpy as np
import pandas as pd

# AI4I 원본 CSV의 컬럼명 -> 프로젝트 표준 snake_case
#
# 원본을 그대로 읽으면 컬럼명에 공백·대괄호가 들어 있어서
# df.air_k 같은 접근이 안 되고, src/validate.py의 REQUIRED_COLUMNS
# 와도 맞지 않습니다. 로드 직후 한 번만 적용하세요:
#     df = pd.read_csv(path).rename(columns=COLUMN_RENAME_MAP)
COLUMN_RENAME_MAP = {
    "UDI": "udi",
    "Product ID": "product_id",
    "Type": "type",
    "Air temperature [K]": "air_k",
    "Process temperature [K]": "process_k",
    "Rotational speed [rpm]": "rpm",
    "Torque [Nm]": "torque",
    "Tool wear [min]": "tool_wear",
    "Machine failure": "failure",
    "TWF": "twf", "HDF": "hdf", "PWF": "pwf",
    "OSF": "osf", "RNF": "rnf",
}

# 모델 입력에서 제외할 컬럼과 그 근거 (발표에서 설명할 항목)
#   udi, product_id : 단순 식별자. 학습하면 행 번호를 외우는 꼴
#   twf~rnf         : 고장의 '원인 유형' = 타깃이 정해진 뒤에야 알 수 있음
#                     -> 누수(leakage). 반드시 제외
LEAKAGE_COLUMNS = ["udi", "product_id", "twf", "hdf", "pwf", "osf", "rnf"]


def make_ai4i_features(df: pd.DataFrame) -> pd.DataFrame:
    """AI4I 데이터프레임에 파생변수를 추가해 **새 객체로** 반환한다.

    입력은 COLUMN_RENAME_MAP 적용 후 상태를 가정한다.

    TODO:
    - temp_diff = process_k - air_k
    - power_w = torque * rpm * 2 * pi / 60
    - strain = tool_wear * torque
    - 원본 df를 수정하지 말고 df.copy()로 시작할 것
    """
    raise NotImplementedError("TODO: 파생변수 로직 구현")
FEA

echo "▶ 7.6   README.md 실행 순서 수정"
python - << 'PY'
from pathlib import Path
p = Path("README.md")
t = p.read_text(encoding="utf-8")

t = t.replace("""   python -m venv venv
   source venv/bin/activate   # Windows는 venv\\Scripts\\activate""",
"""   python3.11 -m venv venv
   source venv/bin/activate   # Windows는 venv\\Scripts\\activate""")

t = t.replace("""   python src/energy_train.py
   python src/ai4i_train.py""",
"""   python -m src.energy_train
   python -m src.ai4i_train""")

t = t.replace("   python app/app.py", "   python -m app.app")

t = t.replace("레포 폴더 구조 전체 설명은 루트의 `가이드.txt`를 먼저 읽어주세요.",
              "레포 폴더 구조 전체 설명은 루트의 `CONTRIBUTING.md`를 먼저 읽어주세요.")

note = """
> ⚠️ **반드시 레포 루트에서 `-m` 옵션으로 실행하세요.**
> `python app/app.py` 처럼 실행하면 `sys.path[0]`이 `app/` 폴더가 되어
> `from src...` import가 전부 `ModuleNotFoundError`로 실패합니다.
"""
t = t.replace("\n## 재현 시 자동 생성되는 파일", note + "\n## 재현 시 자동 생성되는 파일")

t = t.replace("""- `outputs/*.csv`의 실제 값 — 4번 단계 실행 시 더미 데이터가
  실제 결과로 덮어써짐""",
"""- `outputs/*.csv`의 실제 값 — 4번 단계 실행 시 더미 데이터가
  실제 결과로 덮어써짐 (더미는 `python -m scripts.make_dummy`로 재생성)""")

t = t.rstrip() + """

## 라이선스

- **코드**: MIT (`LICENSE` 참조)
- **데이터**: 각 출처의 라이선스를 따릅니다 (`DATA_SOURCES.md` 참조).
  본 저장소의 MIT 라이선스는 데이터에 적용되지 않습니다.
  AI4I 2020은 CC BY 4.0 (출처 표기 조건).
"""
p.write_text(t, encoding="utf-8")
print("README 수정 완료")
PY

echo "▶ 8/9  폴더설명.txt -> README.md, 가이드.txt -> CONTRIBUTING.md"
for d in src app app/assets app/pages report notebooks outputs models tests data/raw data/processed docs; do
  if [ -f "$d/폴더설명.txt" ]; then
    # git mv 가 안 되는 경우(gitignore 대상 폴더)는 일반 mv로 대체
    git mv "$d/폴더설명.txt" "$d/README.md" 2>/dev/null \
      || { mv "$d/폴더설명.txt" "$d/README.md"; git rm --cached -q "$d/폴더설명.txt" 2>/dev/null || true; }
  fi
done
if [ -f "가이드.txt" ]; then
  git mv "가이드.txt" "CONTRIBUTING.md" 2>/dev/null || mv "가이드.txt" "CONTRIBUTING.md"
fi

echo "▶ 9/9  .gitattributes, CI, .gitignore 보강"
cat > .gitattributes << 'ATR'
# 노트북 출력이 Git에 안 남게 함 (병합 충돌 방지)
# 각자 한 번씩 `nbstripout --install` 도 실행해야 실제로 동작합니다.
*.ipynb filter=nbstripout
*.ipynb diff=ipynb

# 줄바꿈 통일 (윈도우 팀원 대비)
* text=auto eol=lf
ATR

mkdir -p .github/workflows
cat > .github/workflows/ci.yml << 'CI'
name: ci
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: pip
      - run: pip install -r requirements.txt
      - run: python -m scripts.make_dummy     # 더미 생성이 깨지지 않는지
      - run: python -c "import app.app"       # Dash 앱 import 가능한지
      - run: pytest -q
CI

python - << 'PY'
from pathlib import Path
p = Path(".gitignore")
t = p.read_text(encoding="utf-8")
add = """
# AI4I 원본은 CC BY 4.0 + 500KB 수준이라 재현성을 위해 커밋 허용
!data/raw/ai4i2020.csv

# 폴더 설명 문서는 GitHub에서 보여야 하므로 예외
!data/raw/README.md
!data/processed/README.md
"""
if "ai4i2020.csv" not in t:
    p.write_text(t.rstrip() + "\n" + add, encoding="utf-8")
PY

echo
echo "✅ 완료. 아래로 확인하세요:"
echo "    python -m pytest -q"
echo "    python -c 'import app.app'"
echo "    git status"
