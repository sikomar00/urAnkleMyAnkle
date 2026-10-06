"""모델 비교표의 기준선 — 학습 구간만으로 적합하고 평가 구간 정답은 읽지 않는다.

- 기준 A: 학습 구간 양성 비율을 모든 행에 같은 점수로 준다(순위 정보 없음).
- 기준 B: 학습 구간에서 계산한 그룹별(기계별 또는 기계×부품별) 과거 양성 비율을
  그 그룹의 점수로 준다. 학습 구간에 없던 그룹은 기준 A 값으로 채운다.

모델이 기준 B를 넘지 못하면 "기계·부품마다 원래 잦은 고장"을 다시 말할 뿐이다.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def fit_constant_baseline(train: pd.DataFrame, target: str) -> float:
    """기준 A — 학습 구간 양성 비율."""
    if train.empty:
        raise ValueError("학습 구간이 비어 있어 기준선을 적합할 수 없습니다.")
    return float(train[target].mean())


def fit_group_rate_baseline(train: pd.DataFrame, keys: Sequence[str], target: str) -> pd.DataFrame:
    """기준 B — 학습 구간의 그룹별 행 수와 양성 비율 표."""
    if train.empty:
        raise ValueError("학습 구간이 비어 있어 기준선을 적합할 수 없습니다.")
    return (train.groupby(list(keys), sort=True)[target]
            .agg(train_rows="size", train_positive_rate="mean")
            .reset_index())


def score_group_rate_baseline(table: pd.DataFrame, frame: pd.DataFrame, keys: Sequence[str],
                              fallback: float) -> np.ndarray:
    """기준 B 점수 — 각 행의 그룹 과거 비율. 표에 없는 그룹은 ``fallback``(기준 A)."""
    merged = frame[list(keys)].merge(table, on=list(keys), how="left", validate="many_to_one")
    return merged["train_positive_rate"].fillna(fallback).to_numpy(dtype=float)
