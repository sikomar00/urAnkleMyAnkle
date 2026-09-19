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
