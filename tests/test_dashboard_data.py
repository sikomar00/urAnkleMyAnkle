"""dashboard_data.py 화면 ⑤ 로더(load_screen5_kpis/load_failure_trend) 검증."""
import os
from pathlib import Path

import pytest

root = Path(__file__).resolve().parents[1]
name = 'synthetic_industrial_machine_data.csv'
paths = ([Path(os.environ['MACHINE_DATA_PATH'])] if os.environ.get('MACHINE_DATA_PATH')
         else [root / 'data/raw' / name, Path.home() / 'Downloads' / name])
if not any(path.is_file() for path in paths):
    pytest.skip('화면 ⑤ 데이터 로더 검증용 원본 CSV가 없습니다.', allow_module_level=True)

from src import dashboard_data as dd


def test_screen5_kpis_share_values_with_screen1():
    s1, s5 = dd.load_screen1_kpis(), dd.load_screen5_kpis()
    assert s5['failure_machine_days'] == s1['failure_machine_days']
    assert s5['failure_rate_pct'] == pytest.approx(s1['failure_rate_pct'])
    assert s5['parts_issue_value_inr'] == pytest.approx(s1['parts_issue_value_inr'])
    assert s5['avg_power_kw'] == pytest.approx(s1['avg_power_kw'])


def test_failure_trend_shape_and_dtypes():
    trend = dd.load_failure_trend()
    assert len(trend) == 1095
    assert list(trend.columns) == ['transaction_date', 'failure_days', 'high_risk_pct']
    assert trend['transaction_date'].is_monotonic_increasing
    assert trend['failure_days'].dtype.kind == 'i'
    assert trend['high_risk_pct'].dtype.kind == 'f'
    assert (trend['high_risk_pct'] >= 0).all() and (trend['high_risk_pct'] <= 100).all()


def test_failure_trend_last_30_days_sum():
    trend = dd.load_failure_trend()
    assert trend.tail(30)['failure_days'].sum() == 211
