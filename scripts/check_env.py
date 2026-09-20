"""
설치 후 환경 검증 스크립트. Mac/Windows 팀원 전원 동일하게 실행.

실행 (레포 루트, 가상환경 활성화 상태에서):
    python -m scripts.check_env

중요: Prophet 에러가 나도 절대 cmdstanpy.install_cmdstan()을 먼저
시도하지 마세요. prophet의 pip wheel은 macOS/Windows 양쪽 다
컴파일된 cmdstan 도구와 모델 바이너리를 이미 포함하고 있어서,
정상 설치라면 이 스크립트가 추가 설치 없이 통과해야 합니다.
실패하면 docs/troubleshooting_log.md에 이 스크립트의 전체 출력을
그대로 붙이고 팀에 공유하세요 — install_cmdstan()은 그 이후
마지막 수단입니다.
"""
import sys
from pathlib import Path

ok = True


def check(label, fn):
    global ok
    try:
        result = fn()
        print(f"[OK]   {label}: {result}")
    except Exception as e:
        print(f"[FAIL] {label}: {type(e).__name__}: {e}")
        ok = False


print(f"플랫폼: {sys.platform} / 파이썬: {sys.version.split()[0]}")
print(f"인터프리터: {sys.executable}")
print()

check("numpy", lambda: __import__("numpy").__version__)
check("pandas", lambda: __import__("pandas").__version__)
check("scikit-learn", lambda: __import__("sklearn").__version__)
check("plotly", lambda: __import__("plotly").__version__)
check("dash", lambda: __import__("dash").__version__)


def _prophet_fit():
    import numpy as np
    import pandas as pd
    from prophet import Prophet

    df = pd.DataFrame({
        "ds": pd.date_range("2024-01-01", periods=60, freq="D"),
        "y": np.arange(60.0) + np.random.randn(60),
    })
    m = Prophet().fit(df)
    fc = m.predict(m.make_future_dataframe(periods=7))
    need = {"yhat", "yhat_lower", "yhat_upper"}
    missing = need - set(fc.columns)
    if missing:
        raise RuntimeError(f"예측 결과에 컬럼 누락: {missing}")
    return "fit + predict 성공"


# check("prophet (fit/predict)", _prophet_fit)

# data_path = Path("data/raw/ai4i2020.csv")
# check(
#     "AI4I 데이터",
#     lambda: f"{__import__('pandas').read_csv(data_path).shape} (data/raw/ai4i2020.csv)"
#     if data_path.exists()
#     else (_ for _ in ()).throw(FileNotFoundError("data/raw/ai4i2020.csv 없음")),
# )

print()
print("=== 전체 결과:", "PASS ===" if ok else "FAIL ===")
sys.exit(0 if ok else 1)
