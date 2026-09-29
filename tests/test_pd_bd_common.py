"""고장 후보의 경계와 실제 전날 기록만 사용하는지 확인합니다."""

import numpy as np
import pandas as pd

from src.huijae_example import SENSORS, prepare_machine_data
from src.pd_bd_common import build_labels


def example():
    rows = []
    dates = ['2024-01-01', '2024-01-02', '2024-01-04', '2024-01-05']
    failures = [[], ['A1', 'B1', 'C1'], ['A1', 'A2'], ['A1', 'B1']]
    parts = [('A1', 'A', 1000), ('A2', 'A', 2000),
             ('B1', 'B', 2500), ('C1', 'C', 3000)]
    for day, date in enumerate(dates):
        for part, grade, cost in parts:
            rows.append({'asset_tag': 'M1', 'transaction_date': date,
                         'part_no': part, 'criticality': grade,
                         'breakdown_flag': int(part in failures[day]),
                         'unit_cost_inr': cost, 'machine_type': 'Type 1',
                         'plant_code': 'Plant 1',
                         **{sensor: float(day) for sensor in SENSORS}})
    raw = pd.DataFrame(rows)
    return raw, prepare_machine_data(raw)


def test_count_a_and_cost_rules():
    raw, machine = example()
    count, _, _ = build_labels(raw, machine, 'CO')
    assert count.failed_parts_ge_3.tolist() == [0, 1, 0, 0]
    assert count.failed_parts_ge_4.sum() == 0
    grade_a, _, _ = build_labels(raw, machine, 'A')
    assert grade_a.A_ge_1.tolist() == [0, 1, 1, 1]
    assert grade_a.A_ge_2.tolist() == [0, 0, 1, 0]
    assert grade_a.A_ge_1_and_total_ge_2.tolist() == [0, 1, 1, 1]
    assert grade_a.A_ge_2_or_total_ge_4.tolist() == [0, 0, 1, 0]
    cost, _, note = build_labels(raw, machine, 'COST')
    assert cost.failed_unit_cost_inr.tolist() == [0, 6500, 3000, 3500]
    assert cost.failed_unit_cost_ge_5000.tolist() == [0, 1, 0, 0]
    assert '실제 수리비' in note


def test_start_and_consecutive_exclude_first_day_and_date_gap():
    raw, machine = example()
    start, _, note = build_labels(raw, machine, 'START')
    assert start.start_parts_ge_3.isna().tolist() == [True, False, True, False]
    assert start.start_parts_ge_3.dropna().tolist() == [1, 0]
    assert start.start_severity_ge_10.dropna().tolist() == [0, 0]
    assert '첫날 1행' in note and '날짜 공백 1행' in note
    consecutive, _, _ = build_labels(raw, machine, 'CONSEC')
    assert np.isnan(consecutive.consec_parts_ge_1.iloc[2])
    assert consecutive.consec_parts_ge_1.dropna().tolist() == [0, 1]
    assert consecutive.consec_parts_ge_2.dropna().tolist() == [0, 1]
