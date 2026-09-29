import math

import numpy as np
import pandas as pd
import pytest

from src.probabilistic_risk_evaluation import (
    SmoothedBaseRateModel,
    build_ranking_metrics,
    decide_localization_status,
    paired_hit_rate_bootstrap,
    random_ranking_expectations,
)


def test_smoothed_base_rate_matches_beta_binomial_hand_calculation():
    train = pd.DataFrame(
        {
            "asset_tag": ["A", "A", "A", "A", "A", "A", "B", "B"],
            "part_no": ["P1", "P1", "P1", "P1", "P2", "P2", "P1", "P1"],
            "breakdown_flag": [1, 1, 0, 0, 0, 0, 0, 0],
        }
    )
    model = SmoothedBaseRateModel(prior_strength=20).fit(train)

    # 전체 8건 중 2건 고장: alpha=5, beta=15.
    expected_a_p1 = (2 + 5) / (4 + 20)
    predicted = model.predict_proba(
        pd.DataFrame(
            {
                "asset_tag": ["A", "A", "NEW"],
                "part_no": ["P1", "P2", "P9"],
                "breakdown_flag": [0, 1, 1],
            }
        )
    )
    assert model.global_rate_ == pytest.approx(0.25)
    assert predicted.tolist() == pytest.approx(
        [expected_a_p1, (0 + 5) / (2 + 20), 0.25]
    )

    changed_target = pd.DataFrame(
        {
            "asset_tag": ["A", "A", "NEW"],
            "part_no": ["P1", "P2", "P9"],
            "breakdown_flag": [1, 0, 0],
        }
    )
    assert model.predict_proba(changed_target).tolist() == pytest.approx(
        predicted.tolist()
    )


def ranking_frame(model_variant: str = "A1") -> pd.DataFrame:
    rows = []
    orders = [
        ["P1", "P2", "P3", "P4", "P5"],
        ["P2", "P1", "P3", "P5", "P4"],
        ["P5", "P4", "P3", "P2", "P1"],
    ]
    failures = [{"P1", "P4"}, {"P3"}, set()]
    criticality = {"P1": "A", "P2": "A", "P3": "C", "P4": "B", "P5": "C"}
    family = {"P1": "Bearing", "P2": "Bearing", "P3": "Filter", "P4": "Motor", "P5": "Motor"}
    for day, (order, failed) in enumerate(zip(orders, failures, strict=True), start=1):
        rank_by_part = {part: rank for rank, part in enumerate(order, start=1)}
        for part in order:
            rows.append(
                {
                    "transaction_date": pd.Timestamp(f"2024-04-0{day}"),
                    "asset_tag": "AST-1",
                    "model_variant": model_variant,
                    "split": "calibration",
                    "part_no": part,
                    "part_family": family[part],
                    "criticality": criticality[part],
                    "breakdown_flag": int(part in failed),
                    "probability_rank": rank_by_part[part],
                }
            )
    return pd.DataFrame(rows)


def test_ranking_metrics_match_multi_failure_hand_calculation():
    metrics = build_ranking_metrics(ranking_frame(), ks=(1, 3, 5))
    overall = metrics.query("scope_kind == 'overall'").iloc[0]

    assert overall["positive_asset_days"] == 2
    assert overall["total_asset_days"] == 3
    assert overall["positive_parts"] == 3
    assert overall["hit_rate_at_1"] == pytest.approx(0.5)
    assert overall["hit_rate_at_3"] == pytest.approx(1.0)
    assert overall["hit_rate_at_5"] == pytest.approx(1.0)
    assert overall["recall_at_1"] == pytest.approx(1 / 3)
    assert overall["recall_at_3"] == pytest.approx(2 / 3)
    assert overall["recall_at_5"] == pytest.approx(1.0)
    assert overall["precision_at_3"] == pytest.approx(2 / 9)
    assert overall["mrr"] == pytest.approx((1 + 1 / 3) / 2)
    assert overall["unique_top3_sets"] == 2
    assert overall["most_common_top3_set_share"] == pytest.approx(2 / 3)

    criticality = metrics.query("scope_kind == 'criticality'").set_index("scope_value")
    assert criticality.loc["A", "recall_at_3"] == pytest.approx(1.0)
    assert criticality.loc["B", "recall_at_3"] == pytest.approx(0.0)
    assert criticality.loc["C", "recall_at_3"] == pytest.approx(1.0)
    families = metrics.query("scope_kind == 'part_family'").set_index("scope_value")
    assert families.loc["Bearing", "recall_at_3"] == pytest.approx(1.0)
    assert families.loc["Motor", "recall_at_3"] == pytest.approx(0.0)


def test_random_ranking_expectation_uses_each_asset_day_failure_count():
    result = random_ranking_expectations(ranking_frame(), ks=(1, 3))
    row = result.iloc[0]

    expected_hit_1 = ((2 / 5) + (1 / 5)) / 2
    day1_hit_3 = 1 - math.comb(3, 3) / math.comb(5, 3)
    day2_hit_3 = 1 - math.comb(4, 3) / math.comb(5, 3)
    assert row["expected_hit_rate_at_1"] == pytest.approx(expected_hit_1)
    assert row["expected_hit_rate_at_3"] == pytest.approx(
        (day1_hit_3 + day2_hit_3) / 2
    )
    assert row["expected_recall_at_1"] == pytest.approx(1 / 5)
    assert row["expected_recall_at_3"] == pytest.approx(3 / 5)


def test_paired_bootstrap_of_identical_rankings_is_exactly_zero():
    frame = ranking_frame()
    baseline = frame.assign(model_variant="B0")

    result = paired_hit_rate_bootstrap(
        frame, baseline, k=3, samples=200, random_state=7
    )

    assert result["asset_days"] == 2
    assert result["mean_difference"] == pytest.approx(0.0)
    assert result["ci_lower"] == pytest.approx(0.0)
    assert result["ci_upper"] == pytest.approx(0.0)


def test_localization_status_requires_all_validation_conditions():
    approved = {
        "split": "calibration",
        "model_hit_rate_at_3": 0.70,
        "baseline_hit_rate_at_3": 0.55,
        "random_hit_rate_at_3": 0.40,
        "model_recall_at_3": 0.45,
        "baseline_recall_at_3": 0.40,
        "bootstrap_ci_lower": 0.01,
        "unique_top3_sets": 4,
        "test_hit_rate_at_3": 0.0,
    }

    result = decide_localization_status(approved)
    assert result["localization_status"] == "approved_on_validation"
    assert result["display_label"] == "우선 점검 후보"
    assert result["reasons"] == []

    rejected = {**approved, "bootstrap_ci_lower": 0.0, "unique_top3_sets": 1}
    result = decide_localization_status(rejected)
    assert result["localization_status"] == "insufficient_evidence"
    assert result["display_label"] == "참고용 위험 순위"
    assert len(result["reasons"]) == 2


def test_localization_status_rejects_non_calibration_input():
    with pytest.raises(ValueError, match="보정 구간"):
        decide_localization_status({"split": "test"})
