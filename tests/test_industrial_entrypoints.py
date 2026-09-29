import subprocess
import sys

import pytest

from src.current_asset_score_model import (
    DEFAULT_OUTPUT as SCORE_DEFAULT_OUTPUT,
    build_parser as build_score_parser,
)
from src.current_asset_severity_experiments import (
    DEFAULT_OUTPUT as SEVERITY_EXPERIMENT_OUTPUT,
    build_parser as build_severity_experiment_parser,
)
from src.current_family_diagnosis import (
    DEFAULT_OUTPUT as FAMILY_DEFAULT_OUTPUT,
    build_parser as build_family_parser,
)
from src.current_single_part_anomaly import (
    DEFAULT_OUTPUT as SINGLE_PART_DEFAULT_OUTPUT,
    build_parser as build_single_part_parser,
)
from src.industrial_training import DEFAULT_DATA
from src.forecast_asset_model import build_parser, default_output_path


@pytest.mark.parametrize("script", [
    "src/current_asset_model.py", "src/current_asset_score_model.py",
    "src/forecast_asset_model.py",
    "src/current_asset_severity_experiments.py",
    "src/current_family_diagnosis.py",
    "src/current_single_part_anomaly.py",
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


def test_current_asset_score_defaults():
    args = build_score_parser().parse_args([])
    assert args.output == SCORE_DEFAULT_OUTPUT
    assert args.scope == "all"
    assert args.max_iter == 100
    assert args.validation_start == "2024-01-01"
    assert args.test_start == "2024-07-01"


def test_asset_severity_experiment_defaults():
    args = build_severity_experiment_parser().parse_args([])
    assert args.output == SEVERITY_EXPERIMENT_OUTPUT
    assert args.high_risk_thresholds == [12, 13]
    assert args.feature_sets == ["A", "B", "C", "D"]
    assert args.scope == "all"
    assert args.min_normal_rows == 30


def test_current_family_diagnosis_defaults():
    args = build_family_parser().parse_args([])

    assert args.data == DEFAULT_DATA
    assert args.output == FAMILY_DEFAULT_OUTPUT
    assert args.feature_sets == ["A", "B", "C"]
    assert args.targets == ["affected", "severe"]
    assert args.max_iter == 100
    assert args.random_state == 42
    assert args.validation_start == "2024-01-01"
    assert args.test_start == "2024-07-01"


def test_single_part_anomaly_defaults():
    args = build_single_part_parser().parse_args([])

    assert args.data == DEFAULT_DATA
    assert args.output == SINGLE_PART_DEFAULT_OUTPUT
    assert args.validation_start == "2024-01-01"
    assert args.test_start == "2024-07-01"
    assert args.min_normal_rows == 30
    assert args.bootstrap_samples == 1_000
    assert args.expected_parts_per_asset == 20
    assert args.skip_residual is False


@pytest.mark.parametrize(
    "arguments",
    [
        ["--feature-sets", "Z"],
        ["--targets", "all_failed"],
    ],
)
def test_current_family_diagnosis_rejects_unknown_choices(arguments):
    with pytest.raises(SystemExit):
        build_family_parser().parse_args(arguments)


def test_current_family_diagnosis_module_help():
    result = subprocess.run(
        [sys.executable, "-m", "src.current_family_diagnosis", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Family" in result.stdout
    assert "당일" in result.stdout


def test_forecast_asset_risk_definitions_use_different_default_outputs():
    new_output = default_output_path(7, "new")
    any_output = default_output_path(7, "any")
    assert new_output != any_output
    assert new_output.name == "asset_forecast_7d_new"
    assert any_output.name == "asset_forecast_7d_any"
