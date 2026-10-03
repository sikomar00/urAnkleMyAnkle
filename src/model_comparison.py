"""과제별 모델 비교표 — 기준 A·B와 LR·RF·HGB를 같은 분할·평가 구간에서 비교한다.

    python -m src.model_comparison --task all [--data data/raw/synthetic_industrial_machine_data.csv]

과제
  machine_risk    기계 고위험일 판별(당일 고장점수 ≥ 12/13/14). src/huijae_example.py와 같은
                  날짜 70/15/15 분할·전처리·피처로 LR·RF·HGB를 학습한다. → outputs/machine_risk/
  part_within_7d  부품 7일 내 고장 표시. src/pf_within_7d.py의 Test 예측(LR·RF)에 기준 A·B를
                  더하고, 영향 변수 계산을 위해 같은 설정의 RF를 다시 학습한다. → outputs/pf_within_7d/
  part_current    부품 당일 고장 표시 판별. outputs/current_state의 기존 예측(prior·HGB)에 기준 B를
                  더하고, 저장된 HGB로 영향 변수를 계산한다(재학습하지 않는다). → outputs/current_state/

공통 산출물(과제 폴더마다): comparison.csv, comparison_predictions.csv(Test 점수), baselines.csv,
feature_importance.csv(운영·선택 모델, 검증 구간 permutation importance, scoring=AP), comparison_config.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "src"

from .baselines import fit_constant_baseline, fit_group_rate_baseline, score_group_rate_baseline  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "raw" / "synthetic_industrial_machine_data.csv"
OUTPUTS = ROOT / "outputs"
RANDOM_STATE = 42
PRODUCTION_MODEL = "random_forest"  # 팀 결정(DESIGN.md §9). 근거: TODO(사용자 확인: RF production 근거)
MODEL_LABELS = {
    "baseline_a": "기준 A · 학습 구간 양성 비율",
    "logistic_regression": "로지스틱 회귀",
    "random_forest": "랜덤 포레스트",
    "hist_gradient_boosting": "HGB",
}


def top_share_precision(actual, score, share: float) -> float:
    """점수 상위 ``share`` 비율 행에 경보를 낼 때의 정밀도.

    경계 점수에 동점이 걸리면 그 동점 묶음에서 무작위로 고른다고 보고 기댓값을 쓴다 —
    점수가 모두 같은 기준 A는 양성 비율과 같아진다(행 순서에 좌우되지 않는다).
    """
    actual = np.asarray(actual, dtype=float)
    score = np.asarray(score, dtype=float)
    k = max(1, int(np.ceil(len(score) * share)))
    kth = np.sort(score)[::-1][k - 1]
    above = score > kth
    tied = score == kth
    taken_from_tie = k - int(above.sum())
    hits = actual[above].sum() + taken_from_tie * actual[tied].mean()
    return float(hits / k)


def comparison_metrics(actual, score, cutoff: float) -> dict:
    """같은 Test 행에 대한 순위 지표와 판정 기준 적용 지표."""
    actual = np.asarray(actual, dtype=int)
    score = np.asarray(score, dtype=float)
    predicted = (score >= cutoff).astype(int)
    tn, fp, fn, tp = confusion_matrix(actual, predicted, labels=[0, 1]).ravel()
    positive_rate = float(actual.mean())
    ap = float(average_precision_score(actual, score))
    return {
        "test_rows": int(len(actual)), "positive_rate": positive_rate,
        "average_precision": ap, "ap_lift": ap / positive_rate if positive_rate else np.nan,
        "roc_auc": float(roc_auc_score(actual, score)),
        "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
        "recall": float(tp / (tp + fn)) if tp + fn else 0.0,
        "alert_rate": float(predicted.mean()),
        "top10_precision": top_share_precision(actual, score, 0.10),
        "accuracy": float((tp + tn) / len(actual)),
        "always_negative_accuracy": 1.0 - positive_rate,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def _row(task, threshold, model, label, role, cutoff, policy, actual, score, **extra) -> dict:
    return {"task": task, "severity_threshold": threshold, "model": model, "model_label": label,
            "role": role, "cutoff": float(cutoff), "cutoff_policy": policy,
            **comparison_metrics(actual, score, cutoff), **extra}


def _importance(model, frame, target, columns, threshold, model_name) -> pd.DataFrame:
    result = permutation_importance(model, frame[columns], frame[target], scoring="average_precision",
                                    n_repeats=5, random_state=RANDOM_STATE, n_jobs=-1)
    return (pd.DataFrame({"severity_threshold": threshold, "model": model_name, "feature": columns,
                          "importance_mean": result.importances_mean, "importance_std": result.importances_std,
                          "split": "validation", "scoring": "average_precision"})
            .sort_values("importance_mean", ascending=False).reset_index(drop=True))


def _predictions(threshold, model, actual, score) -> pd.DataFrame:
    return pd.DataFrame({"severity_threshold": threshold, "model": model,
                         "actual": np.asarray(actual, dtype=int), "score": np.asarray(score, dtype=float)})


def _write(output: Path, rows, predictions, baselines, importance, config) -> None:
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output / "comparison.csv", index=False, encoding="utf-8-sig")
    pd.concat(predictions, ignore_index=True).to_csv(output / "comparison_predictions.csv", index=False)
    pd.concat(baselines, ignore_index=True).to_csv(output / "baselines.csv", index=False, encoding="utf-8-sig")
    pd.concat(importance, ignore_index=True).to_csv(output / "feature_importance.csv", index=False)
    (output / "comparison_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2, default=str),
                                                   encoding="utf-8")


# ------------------------------------------------------------
# 기계 고위험일 판별
# ------------------------------------------------------------
def build_machine_risk(raw: pd.DataFrame, output: Path, thresholds=(12, 13, 14)) -> pd.DataFrame:
    from .huijae_example import (CATEGORICAL, TEMPORAL_NUMERIC, choose_validation_cutoff,
                                 current_day_pipeline, prepare_machine_data, split_by_date)

    machine = prepare_machine_data(raw)
    groups, split_summary = split_by_date(machine)
    train, valid, test = groups["Train"], groups["Validation"], groups["Test"]
    columns = TEMPORAL_NUMERIC + CATEGORICAL
    estimators = {
        "logistic_regression": None,  # huijae_example의 LR(가중치 없음)과 같다
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
        "hist_gradient_boosting": HistGradientBoostingClassifier(max_iter=100, early_stopping=False,
                                                                 random_state=RANDOM_STATE),
    }
    rows, predictions, baselines, importance, type_rows = [], [], [], [], []
    for threshold in thresholds:
        target = f"severity_{threshold}"
        scores: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        rate_a = fit_constant_baseline(train, target)
        scores["baseline_a"] = (np.full(len(valid), rate_a), np.full(len(test), rate_a))
        table = fit_group_rate_baseline(train, ["asset_tag"], target)
        baselines.append(table.assign(severity_threshold=threshold, fallback_rate=rate_a))
        scores["baseline_b"] = (score_group_rate_baseline(table, valid, ["asset_tag"], rate_a),
                                score_group_rate_baseline(table, test, ["asset_tag"], rate_a))
        fitted = {}
        for name, estimator in estimators.items():
            pipeline = current_day_pipeline(TEMPORAL_NUMERIC, CATEGORICAL, None)
            if estimator is not None:
                pipeline.set_params(classifier=estimator)
            pipeline.fit(train[columns], train[target])
            fitted[name] = pipeline
            scores[name] = (pipeline.predict_proba(valid[columns])[:, 1], pipeline.predict_proba(test[columns])[:, 1])
        validation_ap = {name: float(average_precision_score(valid[target], scores[name][0])) for name in scores}
        selected = max(estimators, key=lambda name: validation_ap[name])
        for name, (valid_score, test_score) in scores.items():
            cutoff, _ = choose_validation_cutoff(valid[target], valid_score)
            role = "baseline" if name.startswith("baseline") else "model"
            label = "기준 B · 기계별 과거 비율" if name == "baseline_b" else MODEL_LABELS[name]
            rows.append(_row("machine_risk", threshold, name, label, role, cutoff, "validation_f1",
                             test[target], test_score, validation_ap=validation_ap[name],
                             selected_by_validation=name == selected, production=name == PRODUCTION_MODEL))
            predictions.append(_predictions(threshold, name, test[target], test_score))
        importance.append(_importance(fitted[PRODUCTION_MODEL], valid, target, columns, threshold, PRODUCTION_MODEL))
        # 참고: 기계 종류별로 따로 학습한 RF와, 전체 RF를 같은 종류 행으로 잘라 본 값.
        for machine_type, train_type in train.groupby("machine_type"):
            test_type = test[test.machine_type.eq(machine_type)]
            pipeline = current_day_pipeline(TEMPORAL_NUMERIC, CATEGORICAL, None)
            pipeline.set_params(classifier=RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1))
            pipeline.fit(train_type[columns], train_type[target])
            overall_score = fitted["random_forest"].predict_proba(test_type[columns])[:, 1]
            type_rows.append({
                "severity_threshold": threshold, "machine_type": machine_type,
                "test_rows": len(test_type), "test_positives": int(test_type[target].sum()),
                "positive_rate": float(test_type[target].mean()),
                "ap_rf_by_type": float(average_precision_score(test_type[target],
                                                               pipeline.predict_proba(test_type[columns])[:, 1])),
                "ap_rf_overall": float(average_precision_score(test_type[target], overall_score)),
            })
    config = {
        "task": "machine_risk", "label": "당일 고장점수(A=4·B=2·C=1) ≥ 기준", "thresholds": list(thresholds),
        "grain": "기계·일", "split": split_summary.to_dict("records"), "features": columns,
        "test_period": f"{test.transaction_date.min():%Y-%m-%d} ~ {test.transaction_date.max():%Y-%m-%d}",
        "feature_time": "당일 센서 8종 + 전일까지 이력(diff1·mean7·std7·vs_mean30) + 기계 종류·공장",
        "cutoff_policy": "검증 구간 F1 최대(huijae_example.choose_validation_cutoff)",
        "selection": "검증 구간 AP 최대(학습 모델 중)", "production": PRODUCTION_MODEL,
        "production_note": "팀 결정. TODO(사용자 확인: RF production 근거)",
        "baseline_b_keys": ["asset_tag"], "source": "src/model_comparison.py",
    }
    _write(output, rows, predictions, baselines, importance, config)
    pd.DataFrame(type_rows).to_csv(output / "machine_type_reference.csv", index=False, encoding="utf-8-sig")
    return pd.DataFrame(rows)


# ------------------------------------------------------------
# 부품 7일 내 고장 표시
# ------------------------------------------------------------
def build_part_within_7d(raw: pd.DataFrame, output: Path) -> pd.DataFrame:
    from . import pf_common

    (_, _, _, _, splits, dates, _, _, columns, numeric, categorical) = pf_common.prepare_experiment(raw, "within_7d")
    train, valid, test = splits["train"], splits["validation"], splits["test"]
    keys = ["transaction_date", "asset_tag", "part_no"]
    stored = pd.read_csv(output / "predictions.csv", parse_dates=["transaction_date"])
    test = test.reset_index(drop=True)
    stored_scores = {}
    for model_name, key in [("로지스틱 회귀", "logistic_regression"), ("랜덤 포레스트", "random_forest")]:
        part = stored[stored.model.eq(model_name)][keys + ["probability"]]
        merged = test[keys].merge(part, on=keys, how="left", validate="one_to_one")
        if merged.probability.isna().any():
            raise RuntimeError(f"{model_name}: 저장된 Test 예측과 Test 행이 맞지 않습니다.")
        stored_scores[key] = merged.probability.to_numpy()

    rate_a = fit_constant_baseline(train, "target")
    table = fit_group_rate_baseline(train, ["asset_tag", "part_no"], "target")
    scores = {"baseline_a": np.full(len(test), rate_a),
              "baseline_b": score_group_rate_baseline(table, test, ["asset_tag", "part_no"], rate_a),
              **stored_scores}
    # 원 실험은 판정 기준을 0.5로 고정했고 검증 구간을 쓰지 않았다 — 모든 행에 같은 기준을 쓴다.
    rows, predictions = [], []
    for name, score in scores.items():
        role = "baseline" if name.startswith("baseline") else "model"
        label = "기준 B · 기계×부품별 과거 비율" if name == "baseline_b" else MODEL_LABELS[name]
        rows.append(_row("part_within_7d", None, name, label, role, 0.5, "fixed_0.5", test.target, score,
                         validation_ap=np.nan, selected_by_validation=False, production=name == PRODUCTION_MODEL))
        predictions.append(_predictions(None, name, test.target, score))

    # 영향 변수 — 같은 설정(random_state=42)으로 RF를 다시 학습해 저장된 Test 예측과 같은지 확인한다.
    model = pf_common.make_model("랜덤 포레스트", numeric, categorical)
    model.fit(train[columns], train.target)
    reproduced = model.predict_proba(test[columns])[:, 1]
    max_diff = float(np.max(np.abs(reproduced - stored_scores["random_forest"])))
    importance = _importance(model, valid, "target", columns, None, PRODUCTION_MODEL)
    config = {
        "task": "part_within_7d", "label": "향후 1~7일 안에 고장 표시 1회 이상(오늘 고장 표시 행 제외)",
        "grain": "기계·부품·일", "split": dates.to_dict("records"), "features": columns,
        "test_period": f"{test.transaction_date.min():%Y-%m-%d} ~ {test.transaction_date.max():%Y-%m-%d}",
        "feature_time": "당일 센서 + 전일까지 고장 이력 + 부품 정보",
        "cutoff_policy": "0.5 고정(원 실험이 검증 구간을 쓰지 않음)", "selection": "검증 미사용",
        "production": PRODUCTION_MODEL, "production_note": "팀 결정. TODO(사용자 확인: RF production 근거)",
        "baseline_b_keys": ["asset_tag", "part_no"],
        "rf_reproduction_max_abs_diff": max_diff, "source": "src/model_comparison.py",
    }
    _write(output, rows, predictions, [table.assign(fallback_rate=rate_a)], [importance], config)
    return pd.DataFrame(rows)


# ------------------------------------------------------------
# 부품 당일 고장 표시 판별 (재학습하지 않는다)
# ------------------------------------------------------------
def build_part_current(raw: pd.DataFrame, output: Path) -> pd.DataFrame:
    from .industrial_training import choose_threshold

    stored = pd.read_csv(output / "test_predictions.csv", parse_dates=["transaction_date"])
    stored = stored[stored.scope_kind.eq("overall")]
    artifact = joblib.load(output / "models" / "overall__all__hist_gradient_boosting.joblib")
    features, target = artifact["feature_data"], artifact["target_data"]
    frame = raw.copy()
    frame["transaction_date"] = pd.to_datetime(frame["transaction_date"])
    train = frame[frame.transaction_date.lt("2024-01-01")]
    valid = frame[frame.transaction_date.ge("2024-01-01") & frame.transaction_date.lt("2024-07-01")]
    keys = ["transaction_date", "asset_tag", "part_no"]
    hgb = stored[stored.model.eq("hist_gradient_boosting")].reset_index(drop=True)
    prior = stored[stored.model.eq("prior")].reset_index(drop=True)
    if not hgb[keys].equals(prior[keys]):
        raise RuntimeError("prior와 HGB의 Test 행 순서가 다릅니다.")
    test = hgb[keys + [target]]

    rate_a = fit_constant_baseline(train, target)
    if not np.isclose(rate_a, prior.risk_score.iloc[0]):
        raise RuntimeError("저장된 prior 점수와 학습 구간 양성 비율이 다릅니다.")
    table = fit_group_rate_baseline(train, ["asset_tag", "part_no"], target)
    valid_b = score_group_rate_baseline(table, valid, ["asset_tag", "part_no"], rate_a)
    hgb_valid = artifact["pipeline"].predict_proba(valid[features])[:, 1]
    entries = [
        ("baseline_a", "prior(학습 구간 양성 비율)", prior.risk_score.to_numpy(), float(prior.threshold.iloc[0]), np.nan),
        ("baseline_b", "기준 B · 기계×부품별 과거 비율", score_group_rate_baseline(table, test, ["asset_tag", "part_no"], rate_a),
         choose_threshold(valid[target], valid_b), float(average_precision_score(valid[target], valid_b))),
        ("hist_gradient_boosting", MODEL_LABELS["hist_gradient_boosting"], hgb.risk_score.to_numpy(),
         float(artifact["threshold"]), float(average_precision_score(valid[target], hgb_valid))),
    ]
    rows, predictions = [], []
    for name, label, score, cutoff, valid_ap in entries:
        role = "baseline" if name.startswith("baseline") else "model"
        rows.append(_row("part_current", None, name, label, role, cutoff, "validation_f1", test[target], score,
                         validation_ap=valid_ap, selected_by_validation=name == "hist_gradient_boosting",
                         production=False))
        predictions.append(_predictions(None, name, test[target], score))
    importance = _importance(artifact["pipeline"], valid, target, list(features), None, "hist_gradient_boosting")
    config = {
        "task": "part_current", "label": "당일 breakdown_flag", "grain": "기계·부품·일",
        "split": {"train_end": "2023-12-31", "validation": "2024-01-01~2024-06-30", "test": "2024-07-01~2025-01-01"},
        "features": list(features), "feature_time": "당일 센서 8종 + 기계 종류(부품 식별 정보 없음)",
        "test_period": f"{test.transaction_date.min():%Y-%m-%d} ~ {test.transaction_date.max():%Y-%m-%d}",
        "cutoff_policy": "검증 구간 F1 최대(industrial_training.choose_threshold)",
        "selection": "학습 모델이 HGB 하나", "production": None,
        "reproduce": "git checkout 02d4489 && python -m src.industrial_training --mode current --scope all",
        "baseline_b_keys": ["asset_tag", "part_no"], "source": "src/model_comparison.py",
    }
    _write(output, rows, predictions, [table.assign(fallback_rate=rate_a)], [importance], config)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=["machine_risk", "part_within_7d", "part_current", "all"], default="all")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    args = parser.parse_args()
    raw = pd.read_csv(args.data)
    builders = {
        "machine_risk": lambda: build_machine_risk(raw, OUTPUTS / "machine_risk"),
        "part_within_7d": lambda: build_part_within_7d(raw, OUTPUTS / "pf_within_7d"),
        "part_current": lambda: build_part_current(raw, OUTPUTS / "current_state"),
    }
    for name in (builders if args.task == "all" else [args.task]):
        result = builders[name]()
        print(f"\n[{name}]")
        print(result[["severity_threshold", "model", "average_precision", "ap_lift", "roc_auc", "precision",
                      "recall", "alert_rate", "top10_precision"]].round(3).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
