"""부품 미래 정답이 실제 달력 날짜와 가장 빠른 고장 표시를 따르는지 검사합니다."""

import numpy as np
import pandas as pd
import pytest

from src.pf_common import (FAMILY_CATEGORIES, PART_CATEGORIES, check_raw_dates,
                           future_labels, make_family_frame)
from src.prepare_machine_data import SENSORS


def test_future_labels_use_calendar_days_and_complete_windows():
    rows = []
    for asset, days, faults in [
        ('M1', range(1, 10), {2, 3}),
        ('M2', [1, 2, 4, 5, 6, 7, 8, 9], {4}),
    ]:
        for day in days:
            rows.append({'asset_tag': asset, 'part_no': 'P1',
                         'transaction_date': pd.Timestamp(2024, 1, day),
                         'breakdown_flag': int(day in faults)})
    frame = pd.DataFrame(rows)
    keys = ['asset_tag', 'part_no']
    binary, complete, missing = future_labels(frame, keys, 3, 'binary')
    counts, count_complete, _ = future_labels(frame, keys, 7, 'count')
    classes, class_complete, _ = future_labels(frame, keys, 7, 'class')
    assert missing == 1
    assert binary[0] == 1 and complete[0]  # 내일과 모레에 고장 표시가 있습니다.
    assert counts[0] == 2 and count_complete[0]
    assert classes[0] == 3 and class_complete[0]  # 가장 빠른 표시는 다음 날입니다.
    assert classes[1] == 3  # 오늘이 고장인 행에서도 미래 정답을 먼저 계산합니다.
    second_asset = frame.index[frame.asset_tag.eq('M2')][0]
    assert not complete[second_asset] and np.isnan(binary[second_asset])
    assert not count_complete[second_asset] and np.isnan(counts[second_asset])
    assert check_raw_dates(frame)['missing_calendar_days'] == 1


def test_family_frame_aggregates_any_part_failure_without_asset_feature():
    rows = []
    for day in [1, 2]:
        for part in ['P1', 'P2']:
            rows.append({'asset_tag': 'M1', 'part_no': part, 'part_family': 'Bearing',
                         'transaction_date': pd.Timestamp(2024, 1, day),
                         'breakdown_flag': int(day == 2 and part == 'P2'),
                         'unit_cost_inr': 10, 'machine_type': 'Type',
                         'plant_code': 'Plant',
                         **{sensor: 1.0 for sensor in SENSORS}})
    family = make_family_frame(pd.DataFrame(rows))
    assert len(family) == 2
    assert family.breakdown_flag.tolist() == [0, 1]
    assert family.unit_cost_inr.tolist() == [20, 20]
    assert 'asset_tag' not in PART_CATEGORIES + FAMILY_CATEGORIES
    incomplete = pd.DataFrame(rows).drop(index=3)
    with pytest.raises(ValueError, match='세부 부품이 빠져'):
        make_family_frame(incomplete)
