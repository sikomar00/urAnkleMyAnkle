"""부품군 전체 고장 규칙과 누락 자료 검사를 확인합니다."""

import pandas as pd
import pytest

from src.PF1_BD import TARGET, build_family_labels


def sample_parts():
    rows = []
    # A 부품군 두 개, B 부품군 세 개: 일부 고장과 전체 고장을 구별합니다.
    for day, failed in [(1, {'A1'}), (2, {'A1', 'A2'}), (3, {'B1', 'B2', 'B3'})]:
        for family, parts in [('A', ['A1', 'A2']), ('B', ['B1', 'B2', 'B3'])]:
            for part in parts:
                rows.append({'asset_tag': 'M1', 'transaction_date': f'2024-01-0{day}',
                             'part_family': family, 'part_no': part,
                             'breakdown_flag': int(part in failed)})
    return pd.DataFrame(rows)


def test_full_family_requires_every_part_and_at_least_one_family():
    labels, families = build_family_labels(sample_parts())
    assert labels[TARGET].tolist() == [0, 1, 1]
    assert labels.fully_failed_families.tolist() == [0, 1, 1]
    assert families.loc[families.transaction_date.eq(pd.Timestamp('2024-01-01')),
                        'all_parts_failed'].sum() == 0


@pytest.mark.parametrize('part', ['A2', 'B1'])
def test_missing_part_cannot_be_mistaken_for_full_failure(part):
    raw = sample_parts()
    raw = raw.loc[~(raw.transaction_date.eq('2024-01-02') & raw.part_no.eq(part))]
    with pytest.raises(ValueError, match='누락'):
        build_family_labels(raw)


def test_missing_whole_family_is_rejected():
    raw = sample_parts()
    raw = raw.loc[~(raw.transaction_date.eq('2024-01-02') & raw.part_family.eq('B'))]
    with pytest.raises(ValueError, match='부품군 전체가 누락'):
        build_family_labels(raw)
