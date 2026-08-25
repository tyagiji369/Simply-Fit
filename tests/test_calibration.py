import pytest

from src.calibration import run_nhanes_calibration_test


def test_calibration_against_real_extract():
    """The calibration report must run against the committed real NHANES data."""
    res = run_nhanes_calibration_test(n_users=150, seed=42)

    assert res["n_nhanes"] == 5434, "real NHANES extract should be used"

    # KS statistics are in [0, 1]
    for k in ["weight_ks_before", "weight_ks_after", "age_ks_before", "age_ks_after"]:
        assert 0.0 <= res[k] <= 1.0

    # calibration must genuinely improve alignment vs the naive baseline
    assert res["weight_ks_after"] < res["weight_ks_before"]
    assert res["age_ks_after"] < res["age_ks_before"]
    assert res["weight_ks_improvement_pct"] > 0
    assert res["age_ks_improvement_pct"] > 0

    # means should be close to the real population
    assert abs(res["mean_synthetic_weight"] - res["mean_nhanes_weight"]) < 5
    assert abs(res["mean_synthetic_age"] - res["mean_nhanes_age"]) < 5
