import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.skipif(
    not (ROOT / "data/public/nhanes_reference.csv").exists(),
    reason="Real NHANES reference file not present",
)
def test_calibration_uses_real_data():
    from src.calibration import run_nhanes_calibration_test

    res = run_nhanes_calibration_test(n_users=200)
    assert res["n_reference"] > 5000, "must compare against a real sample"
    assert res["weight_ks_after"] < res["weight_ks_before"], (
        "calibration must reduce the KS distance to the real NHANES sample"
    )
    assert res["weight_p_value"] > 0 and res["age_p_value"] > 0
    # Means should be within a few kg / years of the real population
    assert abs(res["mean_synthetic_weight"] - res["mean_reference_weight"]) < 6
    assert abs(res["mean_synthetic_age"] - res["mean_reference_age"]) < 5
    # p<0.05 means 'still different' — the honest framing must be exposed
    assert isinstance(res["distributions_still_significantly_different"], bool)


def test_nhanes_reference_loads():
    from src.calibration import load_nhanes_reference

    ref = load_nhanes_reference()
    assert len(ref) > 5000
    assert ref["age"].between(18, 80).all()
    assert ref["weight_kg"].notna().all()
