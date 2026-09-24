import subprocess
import sys

import pytest

from src.forecast_asset_model import build_parser


@pytest.mark.parametrize("script", [
    "src/current_asset_model.py", "src/forecast_asset_model.py",
    "src/current_part_model.py", "src/forecast_part_model.py",
    "src/current_state_model.py", "src/timeseries_model.py",
])
def test_entrypoint_help(script):
    result = subprocess.run([sys.executable, script, "--help"], capture_output=True,
                            text=True, check=False)
    assert result.returncode == 0, result.stderr


def test_forecast_asset_defaults():
    args = build_parser().parse_args([])
    assert args.horizon == 7
    assert args.risk_definition == "new"
