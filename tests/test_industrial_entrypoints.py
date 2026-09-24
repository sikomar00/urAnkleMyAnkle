import subprocess
import sys

import pytest

from src.forecast_asset_model import build_parser, default_output_path


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


def test_forecast_asset_risk_definitions_use_different_default_outputs():
    new_output = default_output_path(7, "new")
    any_output = default_output_path(7, "any")
    assert new_output != any_output
    assert new_output.name == "asset_forecast_7d_new"
    assert any_output.name == "asset_forecast_7d_any"
