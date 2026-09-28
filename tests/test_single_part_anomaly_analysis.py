import json

import pandas as pd
import pytest

from src.single_part_anomaly_analysis import (
    _add_direction_status,
    add_direct_robust_z,
    build_single_part_daily,
    render_single_part_summary,
    run_single_part_analysis,
    summarize_part_anomalies,
)


def _raw_part_rows() -> pd.DataFrame:
    rows = []
    failures = {
        ("A-1", "2024-01-01"): set(),
        ("A-1", "2024-01-02"): {"P-1"},
        ("A-1", "2024-01-03"): {"P-1", "P-2"},
        ("A-2", "2024-01-01"): set(),
        ("A-2", "2024-01-02"): {"P-2"},
        ("A-2", "2024-01-03"): set(),
    }
    families = {"P-1": "Bearing", "P-2": "Filter", "P-3": "Fastener"}
    for (asset_tag, date), failed_parts in failures.items():
        day = pd.Timestamp(date).day
        for part_no, family in families.items():
            rows.append(
                {
                    "transaction_date": pd.Timestamp(date),
                    "asset_tag": asset_tag,
                    "machine_type": "Press",
                    "part_no": part_no,
                    "part_family": family,
                    "criticality": "A" if part_no == "P-1" else "C",
                    "breakdown_flag": int(part_no in failed_parts),
                    "temp_bearing_degC": 50.0 + day,
                    "temp_motor_degC": 60.0 + day,
                    "vibration_h_mms": 2.0 + day,
                    "vibration_v_mms": 1.0 + day,
                    "oil_pressure_bar": 5.0,
                    "load_pct": 70.0,
                    "shaft_rpm": 1_500.0,
                    "power_consumption_kw": 100.0,
                }
            )
    return pd.DataFrame(rows)


def test_daily_groups_keep_clean_and_exactly_one_failed_part_separate():
    """다중 고장일이 단독 고장 사례로 섞이는 회귀를 방지한다."""
    daily = build_single_part_daily(_raw_part_rows())

    statuses = daily.set_index(["asset_tag", "transaction_date"])["failure_status"]
    isolated = daily.loc[daily["failure_status"].eq("isolated")]

    assert statuses.loc[("A-1", pd.Timestamp("2024-01-01"))] == "clean"
    assert statuses.loc[("A-1", pd.Timestamp("2024-01-02"))] == "isolated"
    assert statuses.loc[("A-1", pd.Timestamp("2024-01-03"))] == "multiple"
    assert isolated.set_index("asset_tag")["failed_part_no"].to_dict() == {
        "A-1": "P-1",
        "A-2": "P-2",
    }


def test_daily_groups_reject_missing_part_rows():
    """부품 행이 누락된 날이 완전 정상으로 오분류되는 회귀를 방지한다."""
    raw = _raw_part_rows()
    missing = raw.loc[
        ~(
            raw["asset_tag"].eq("A-1")
            & raw["transaction_date"].eq("2024-01-01")
            & raw["part_no"].eq("P-3")
        )
    ]

    with pytest.raises(ValueError, match="부품 구성이 날짜에 따라 다릅니다"):
        build_single_part_daily(missing)


def test_daily_groups_can_enforce_expected_part_count():
    """모든 날에 같은 누락이 있어도 완전 정상으로 처리되는 회귀를 방지한다."""
    with pytest.raises(ValueError, match="장비별 기대 부품 수 20개"):
        build_single_part_daily(_raw_part_rows(), expected_parts_per_asset=20)


def test_part_summary_uses_clean_controls_from_case_assets_only():
    """다른 장비의 정상행이 부품별 정상 이상치율을 왜곡하는 회귀를 방지한다."""
    daily = pd.DataFrame(
        [
            ("A-1", "clean", pd.NA, pd.NA, 0.0),
            ("A-1", "clean", pd.NA, pd.NA, 3.0),
            ("A-1", "isolated", "P-1", "Bearing", 4.0),
            ("A-2", "clean", pd.NA, pd.NA, 4.0),
            ("A-2", "multiple", pd.NA, pd.NA, 4.0),
        ],
        columns=[
            "asset_tag",
            "failure_status",
            "failed_part_no",
            "failed_part_family",
            "vibration_h_mms_robust_z",
        ],
    )

    result = summarize_part_anomalies(
        daily,
        period="test",
        analysis_method="direct_robust_z",
        z_columns={"vibration_h_mms": "vibration_h_mms_robust_z"},
        thresholds=(2.0,),
        bootstrap_samples=200,
        random_state=7,
    ).iloc[0]

    assert result["clean_rows"] == 2
    assert result["isolated_rows"] == 1
    assert result["clean_anomaly_rate"] == pytest.approx(0.5)
    assert result["isolated_anomaly_rate"] == pytest.approx(1.0)
    assert result["anomaly_rate_diff_pp"] == pytest.approx(50.0)
    assert result["candidate_status"] == "표본 부족"


def test_part_summary_weights_each_case_asset_equally():
    """사례가 많은 한 장비가 부품별 이상치 차이를 지배하는 회귀를 방지한다."""
    rows = [
        ("A-1", "clean", pd.NA, pd.NA, 0.0),
        ("A-1", "isolated", "P-1", "Bearing", 3.0),
        ("A-2", "clean", pd.NA, pd.NA, 3.0),
    ]
    rows.extend(
        [("A-2", "isolated", "P-1", "Bearing", 0.0)] * 9
    )
    daily = pd.DataFrame(
        rows,
        columns=[
            "asset_tag",
            "failure_status",
            "failed_part_no",
            "failed_part_family",
            "vibration_h_mms_robust_z",
        ],
    )

    result = summarize_part_anomalies(
        daily,
        period="test",
        analysis_method="direct_robust_z",
        z_columns={"vibration_h_mms": "vibration_h_mms_robust_z"},
        thresholds=(2.0,),
        bootstrap_samples=100,
        random_state=7,
    ).iloc[0]

    assert result["case_assets"] == 2
    assert result["clean_anomaly_rate"] == pytest.approx(0.5)
    assert result["isolated_anomaly_rate"] == pytest.approx(0.5)
    assert result["anomaly_rate_diff_pp"] == pytest.approx(0.0)


def test_korean_summary_states_exploratory_limitations():
    """작은 표본 결과가 확정적 고장 원인처럼 보고되는 회귀를 방지한다."""
    results = pd.DataFrame(
        {
            "period": ["test", "test"],
            "analysis_method": ["direct_robust_z", "direct_robust_z"],
            "part_no": ["P-1", "P-1"],
            "part_family": ["Bearing", "Bearing"],
            "sensor": ["vibration_h_mms", "vibration_h_mms"],
            "sensor_group": ["health", "health"],
            "z_threshold": [2.0, 3.0],
            "clean_rows": [20, 20],
            "isolated_rows": [3, 3],
            "clean_anomaly_rate": [0.2, 0.1],
            "isolated_anomaly_rate": [2 / 3, 1 / 3],
            "anomaly_rate_diff_pp": [46.6667, 23.3333],
            "ci_low_pp": [-5.0, -10.0],
            "ci_high_pp": [90.0, 60.0],
            "median_abs_z_clean": [0.8, 0.8],
            "median_abs_z_isolated": [2.5, 2.5],
            "cliffs_delta": [0.6, 0.6],
            "candidate_status": ["표본 부족", "표본 부족"],
            "direction_status": ["방향 일치", "방향 일치"],
        }
    )

    text = render_single_part_summary(results)

    for phrase in (
        "세 줄 요약",
        "한 페이지 요약",
        "건강 센서",
        "운전 조건",
        "운전조건 보정 잔차",
        "|Z| >= 2",
        "|Z| >= 3",
        "Cliff",
        "기간 방향",
        "표본 부족",
        "고장 원인을 확정하지 않습니다",
        "완전 정상",
        "단독 고장",
    ):
        assert phrase in text


def test_direct_robust_z_fits_only_pre_validation_clean_days():
    """검증·테스트 값이나 고장일이 정상 기준에 들어가는 누수를 방지한다."""
    daily = build_single_part_daily(_raw_part_rows())
    earlier_clean = daily.loc[
        daily["asset_tag"].eq("A-1")
        & daily["transaction_date"].eq("2024-01-01")
    ].copy()
    earlier_clean["transaction_date"] = pd.Timestamp("2023-12-31")
    earlier_clean["vibration_h_mms"] = 2.0
    daily = pd.concat([daily, earlier_clean], ignore_index=True)
    daily.loc[daily["transaction_date"].eq("2024-01-03"), "vibration_h_mms"] = 999.0

    transformed, baseline = add_direct_robust_z(
        daily,
        validation_start="2024-01-03",
        min_normal_rows=1,
    )

    row = baseline.query(
        "asset_tag == 'A-1' and sensor == 'vibration_h_mms'"
    ).iloc[0]
    test_value = transformed.loc[
        transformed["asset_tag"].eq("A-1")
        & transformed["transaction_date"].eq("2024-01-03"),
        "vibration_h_mms_robust_z",
    ].iloc[0]

    assert row["normal_rows"] == 2
    assert row["median"] == pytest.approx(2.5)
    assert test_value > 100


def test_runner_writes_reproducible_analysis_artifacts(tmp_path):
    """실행 흐름이 핵심 결과 CSV와 한국어 요약을 빠뜨리는 회귀를 방지한다."""
    raw = _raw_part_rows()
    data_path = tmp_path / "input.csv"
    raw.to_csv(data_path, index=False)
    stale_residual = tmp_path / "residual_baselines.csv"
    stale_residual.write_text("stale", encoding="utf-8")

    artifacts = run_single_part_analysis(
        raw,
        output_dir=tmp_path,
        validation_start="2024-01-02",
        test_start="2024-01-03",
        min_normal_rows=1,
        bootstrap_samples=20,
        include_residual=False,
        data_path=data_path,
        expected_parts_per_asset=3,
    )

    assert not artifacts["sensor_results"].empty
    assert set(artifacts["sensor_results"]["period"]) == {"overall", "valid"}
    for filename in (
        "daily_groups.csv",
        "part_profile.csv",
        "sensor_anomaly_results.csv",
        "direct_baselines.csv",
        "experiment_summary.md",
        "run_config.json",
    ):
        assert (tmp_path / filename).exists(), filename
    config = json.loads((tmp_path / "run_config.json").read_text(encoding="utf-8"))
    assert config["input_path"] == str(data_path.resolve())
    assert len(config["input_sha256"]) == 64
    assert "git_head" in config
    assert "working_tree_dirty" in config
    assert len(config["analysis_source_sha256"]) == 64
    assert set(config["analysis_source_files"]) == {
        "src/single_part_anomaly_analysis.py",
        "src/current_single_part_anomaly.py",
    }
    assert config["python_version"]
    assert set(config["dependency_versions"]) == {"numpy", "pandas", "scikit-learn"}
    assert not stale_residual.exists()
    summary = (tmp_path / "experiment_summary.md").read_text(encoding="utf-8")
    assert "부품 3개가 모두 정상" in summary
    assert "부품 20개가 모두 정상" not in summary


def test_temporal_direction_mismatch_overrides_strong_candidate_label():
    """테스트 한 구간만 강한 결과가 재현 가능한 후보처럼 표시되는 회귀를 방지한다."""
    results = pd.DataFrame(
        {
            "period": ["train", "test"],
            "analysis_method": ["direct_robust_z", "direct_robust_z"],
            "part_no": ["P-1", "P-1"],
            "sensor": ["vibration_h_mms", "vibration_h_mms"],
            "z_threshold": [2.0, 2.0],
            "isolated_rows": [20, 20],
            "anomaly_rate_diff_pp": [-20.0, 40.0],
            "candidate_status": ["강한 후보", "강한 후보"],
        }
    )

    classified = _add_direction_status(results)

    assert classified["direction_status"].eq("방향 불일치").all()
    assert classified["candidate_status"].eq("방향 불일치").all()


def test_lower_case_anomaly_rate_is_labeled_reverse_not_strong():
    """고장군 이상치율 감소가 강한 고장 후보로 해석되는 회귀를 방지한다."""
    rows = []
    for index in range(10):
        rows.append(("A-1", "clean", pd.NA, pd.NA, 3.0))
        rows.append(("A-1", "isolated", "P-1", "Bearing", 0.0))
    daily = pd.DataFrame(
        rows,
        columns=[
            "asset_tag",
            "failure_status",
            "failed_part_no",
            "failed_part_family",
            "vibration_h_mms_robust_z",
        ],
    )

    result = summarize_part_anomalies(
        daily,
        period="test",
        analysis_method="direct_robust_z",
        z_columns={"vibration_h_mms": "vibration_h_mms_robust_z"},
        thresholds=(2.0,),
        bootstrap_samples=50,
    ).iloc[0]

    assert result["anomaly_rate_diff_pp"] == pytest.approx(-100.0)
    assert result["candidate_status"] == "역방향"


def test_runner_rejects_data_without_isolated_failures(tmp_path):
    """단독 고장 사례가 없을 때 불명확한 KeyError로 종료되는 회귀를 방지한다."""
    raw = _raw_part_rows()
    raw["breakdown_flag"] = 0

    with pytest.raises(ValueError, match="단독 고장 사례가 없습니다"):
        run_single_part_analysis(
            raw,
            output_dir=tmp_path,
            validation_start="2024-01-02",
            test_start="2024-01-03",
            min_normal_rows=1,
            bootstrap_samples=20,
            include_residual=False,
            expected_parts_per_asset=3,
        )


def test_summary_headline_skips_temporally_inconsistent_candidate():
    """기간 방향 불일치 후보가 세 줄 요약의 대표 결과가 되는 회귀를 방지한다."""
    results = pd.DataFrame(
        {
            "period": ["test", "test"],
            "analysis_method": ["direct_robust_z", "direct_robust_z"],
            "part_no": ["P-1", "P-2"],
            "part_family": ["Bearing", "Filter"],
            "sensor": ["vibration_h_mms", "temp_motor_degC"],
            "sensor_group": ["health", "health"],
            "z_threshold": [2.0, 2.0],
            "clean_rows": [20, 20],
            "isolated_rows": [9, 10],
            "clean_anomaly_rate": [0.1, 0.1],
            "isolated_anomaly_rate": [0.6, 0.3],
            "anomaly_rate_diff_pp": [50.0, 20.0],
            "ci_low_pp": [10.0, 1.0],
            "ci_high_pp": [80.0, 40.0],
            "cliffs_delta": [0.5, 0.3],
            "candidate_status": ["표본 부족", "강한 후보"],
            "direction_status": ["방향 불일치", "방향 일치"],
        }
    )

    summary = render_single_part_summary(results)
    headline = summary.split("## 테스트 구간", maxsplit=1)[0]

    assert "P-2 / temp_motor_degC" in headline
    assert "P-1 / vibration_h_mms" not in headline
