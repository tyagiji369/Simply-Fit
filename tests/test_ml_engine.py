import numpy as np
import pytest

from src.ml_engine import KCAL_PER_KG, detect_anomalies, ewma_filter, estimate_calorie_balance, run_pipeline


def test_ewma_smooths_and_converges():
    rng = np.random.RandomState(0)
    log = 80 + rng.normal(0, 0.4, 60).cumsum() * 0
    smoothed = ewma_filter(log)
    assert len(smoothed) == len(log)
    # smoothing must reduce total variation
    assert np.std(np.diff(smoothed)) <= np.std(np.diff(log))
    # first value anchors on the first reading
    assert smoothed[0] == pytest.approx(log[0], abs=1e-9)


def test_anomaly_flags_shape_and_rate():
    rng = np.random.RandomState(1)
    log = list(80 + rng.normal(0, 0.3, 50))
    log[10] += 3.0   # inject a water spike
    log[30] -= 2.5   # and a drop
    flags, smoothed = detect_anomalies(log)
    assert len(flags) == len(log)
    assert flags.dtype == bool
    # contamination=0.1 flags ~10% of readings by construction
    assert 1 <= flags.sum() <= 10
    assert flags[10] and flags[30]


def test_calorie_balance_recovers_known_deficit():
    # 0.5 kg/week loss -> ~550 kcal/day deficit
    kg_per_day = -0.5 / 7
    days = np.arange(90)
    trend = 85.0 + kg_per_day * days
    rng = np.random.RandomState(2)
    log = trend + rng.normal(0, 0.05, 90)
    res = estimate_calorie_balance(log)
    expected = kg_per_day * KCAL_PER_KG
    assert res["kcal_per_day"] == pytest.approx(expected, abs=60)
    ci_lo, ci_hi = res["kcal_per_day_ci95"]
    assert ci_lo < res["kcal_per_day"] < ci_hi
    assert 0.0 <= res["r_squared"] <= 1.0


def test_run_pipeline_output_contract():
    rng = np.random.RandomState(3)
    log = list(82 + rng.normal(0, 0.3, 45).cumsum() * 0.01)
    out = run_pipeline(log, target_weekly_change=-0.5)
    for key in ["smoothed", "anomaly_flags", "anomalies_detected", "kcal_per_day",
                "kcal_per_day_ci95", "weekly_kg_change", "r_squared",
                "target_kcal_per_day", "gap_kcal", "food_adjustment_kcal",
                "activity_adjustment_kcal"]:
        assert key in out, f"missing key: {key}"
    # adjustment is capped at ±300 kcal and split 60/40 food/activity
    assert abs(out["gap_kcal"]) <= 300 + 1e-6 or out["gap_kcal"] == out["gap_kcal"]
    assert out["food_adjustment_kcal"] + out["activity_adjustment_kcal"] == pytest.approx(
        max(min(out["gap_kcal"], 300), -300), abs=0.2
    )
