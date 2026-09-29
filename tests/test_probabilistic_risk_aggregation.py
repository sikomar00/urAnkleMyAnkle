import numpy as np
import pandas as pd
import pytest

from src.probabilistic_risk_aggregation import (
    aggregate_asset_risk,
    rank_part_probabilities,
    severity_probabilities,
    weighted_score_distribution,
)


def test_weighted_distribution_matches_two_part_hand_calculation():
    result = weighted_score_distribution([0.25, 0.50], [4, 2])

    assert result[0] == pytest.approx(0.375)
    assert result[2] == pytest.approx(0.375)
    assert result[4] == pytest.approx(0.125)
    assert result[6] == pytest.approx(0.125)
    assert result.sum() == pytest.approx(1.0)
    assert np.dot(np.arange(len(result)), result) == pytest.approx(2.0)


def test_weighted_distribution_handles_deterministic_probabilities():
    all_healthy = weighted_score_distribution([0.0, 0.0], [4, 2])
    all_failed = weighted_score_distribution([1.0, 1.0], [4, 2])

    assert all_healthy.tolist() == [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert all_failed.tolist() == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]


@pytest.mark.parametrize(
    ("probabilities", "weights", "message"),
    [
        ([0.2, np.nan], [1, 1], "확률"),
        ([0.2, np.inf], [1, 1], "확률"),
        ([-0.1, 0.2], [1, 1], "0과 1"),
        ([1.1, 0.2], [1, 1], "0과 1"),
        ([0.2, 0.3], [1, 0], "가중치"),
        ([0.2, 0.3], [1, 1.5], "가중치"),
        ([0.2], [1, 2], "개수"),
    ],
)
def test_weighted_distribution_rejects_invalid_inputs(
    probabilities, weights, message
):
    with pytest.raises(ValueError, match=message):
        weighted_score_distribution(probabilities, weights)


def part_probability_frame() -> pd.DataFrame:
    criticality = ["A"] * 6 + ["B"] * 9 + ["C"] * 5
    weights = [4] * 6 + [2] * 9 + [1] * 5
    probabilities = [0.9, 0.5, 0.5, 0.4] + [0.05] * 16
    return pd.DataFrame(
        {
            "transaction_date": pd.Timestamp("2024-07-01"),
            "machine_type": "Machine-1",
            "asset_tag": "AST-1",
            "model_variant": "A1",
            "part_no": [f"P{index:02d}" for index in range(1, 21)],
            "part_family": [f"Family-{(index - 1) // 4 + 1}" for index in range(1, 21)],
            "criticality": criticality,
            "criticality_weight": weights,
            "breakdown_flag": [1, 0, 0, 0, 0, 0, 1] + [0] * 13,
            "failure_probability": probabilities,
        }
    )


def test_rank_part_probabilities_uses_deterministic_tie_breaks():
    ranked = rank_part_probabilities(part_probability_frame())
    by_probability = ranked.sort_values("probability_rank")

    assert by_probability["part_no"].head(4).tolist() == ["P01", "P02", "P03", "P04"]
    assert by_probability["probability_rank"].tolist() == list(range(1, 21))
    assert ranked["expected_score_contribution"].equals(
        ranked["failure_probability"] * ranked["criticality_weight"]
    )
    assert sorted(ranked["contribution_rank"].tolist()) == list(range(1, 21))


def test_rank_part_probabilities_rejects_duplicate_parts():
    frame = part_probability_frame()
    frame.loc[1, "part_no"] = "P01"

    with pytest.raises(ValueError, match="중복"):
        rank_part_probabilities(frame)


def test_severity_probability_tie_chooses_higher_risk_state():
    distribution = np.zeros(48)
    distribution[0] = 0.5
    distribution[12] = 0.5

    result = severity_probabilities(distribution, 12)

    assert result["prob_normal"] == pytest.approx(0.5)
    assert result["prob_high_risk"] == pytest.approx(0.5)
    assert result["predicted_severity"] == "high_risk"


def test_aggregate_asset_risk_builds_score_and_two_threshold_rows():
    ranked = rank_part_probabilities(part_probability_frame())

    result = aggregate_asset_risk(ranked, high_risk_thresholds=(12, 13))

    assert result["high_risk_threshold"].tolist() == [12, 13]
    score_columns = [f"prob_score_{score}" for score in range(48)]
    assert np.allclose(result[score_columns].sum(axis=1), 1.0)
    expected = float(
        (ranked["failure_probability"] * ranked["criticality_weight"]).sum()
    )
    assert result["expected_failure_points"].tolist() == pytest.approx(
        [expected, expected]
    )
    assert result["actual_failure_points"].tolist() == [6, 6]
    assert result.loc[0, "prob_ge_13"] <= result.loc[0, "prob_ge_12"]
    severity_columns = [
        "prob_normal",
        "prob_caution",
        "prob_risk",
        "prob_high_risk",
    ]
    assert np.allclose(result[severity_columns].sum(axis=1), 1.0)
    assert result["actual_severity"].tolist() == ["risk", "risk"]
    assert result.loc[0, "top3_probability_parts"] == "P01;P02;P03"
    assert len(set(result.loc[0, "top3_probability_parts"].split(";"))) == 3


def test_aggregate_asset_risk_requires_exactly_twenty_parts():
    ranked = rank_part_probabilities(part_probability_frame().iloc[:-1])

    with pytest.raises(ValueError, match="20개"):
        aggregate_asset_risk(ranked)

