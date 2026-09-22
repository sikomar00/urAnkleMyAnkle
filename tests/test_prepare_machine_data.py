import pandas as pd
import pytest

from src.prepare_machine_data import SENSORS, prepare


def sample():
    rows = []
    for day, date in enumerate(pd.date_range('2024-01-01', periods=100)):
        for part in ['A', 'B']:
            rows.append(dict(asset_tag='M1', part_no=part, transaction_date=date,
                             machine_type='test', plant_code='P1', part_description=part,
                             part_family='bearing', criticality='A', uom='EA', unit_cost_inr=10,
                             qty_issued=int(day == 40), issue_value_inr=10 * int(day == 40),
                             breakdown_flag=int(part == 'A' and day in [35, 40]),
                             wo_type=None, **{col: float(day + 1) for col in SENSORS}))
    return pd.DataFrame(rows)


def test_grain_target_and_strict_past():
    raw = sample()
    original = raw.copy(deep=True)
    machine, part, schema, _ = prepare(raw)
    pd.testing.assert_frame_equal(raw, original)
    assert len(machine) == 100 and len(part) == 200
    a = part[part.part_no.eq('A')].set_index('transaction_date')
    assert a.iloc[34].target_breakdown_next_7d == 1
    assert pd.isna(a.iloc[35].target_breakdown_next_7d)
    assert a.iloc[35].breakdown_days_prev_7d == 0
    assert a.iloc[36].breakdown_days_prev_7d == 1
    assert a.iloc[40].qty_issued_prev_7d == 0
    assert a.iloc[41].qty_issued_prev_7d == 1
    assert a.iloc[36].days_since_last_observed_breakdown == 1
    assert a.iloc[-7:].target_breakdown_next_7d.isna().all()
    assert not set(['breakdown_flag', 'wo_type', 'qty_issued', 'issue_value_inr', 'target_breakdown_next_7d']) & set(schema['numeric_features'] + schema['categorical_features'])


def test_missing_calendar_day_is_not_filled_as_normal():
    raw = sample()
    raw = raw[~(raw.part_no.eq('A') & raw.transaction_date.eq(pd.Timestamp('2024-02-15')))]
    _, part, _, _ = prepare(raw)
    a = part[part.part_no.eq('A')].set_index('transaction_date')
    assert pd.isna(a.loc['2024-02-14', 'target_breakdown_next_7d'])
    assert not a.loc['2024-02-16', 'history_30d_complete']
    assert pd.isna(a.loc['2024-02-16', 'qty_issued_prev_7d'])


def test_split_purges_horizon_crossing_boundary():
    _, part, _, report = prepare(sample())
    train_end = pd.Timestamp(report['split_dates']['train_end'])
    train = part[part.split.eq('train')]
    assert not train[train.transaction_date.gt(train_end - pd.Timedelta(days=7))].eligible_for_model.any()
    assert (train[train.eligible_for_model].target_window_end <= train_end).all()
    assert part[part.part_no.eq('B') & part.eligible_for_target].target_breakdown_next_7d.eq(0).all()


def test_inconsistent_sensors_rejected():
    raw = sample()
    raw.loc[0, SENSORS[0]] = 999
    with pytest.raises(ValueError, match='센서'):
        prepare(raw)
