import numpy as np
import pytest

from src.ml_engine import (
    KCAL_PER_KG,
    ewma_filter,
    detect_anomalies,
    estimate_calorie_balance,
    run_pipeline,
)


def test_ewma_keeps_trend():
    log = [80 + 0.05 * i for i in range(30)]
    smoothed = ewma_filter(log)
    assert len(smoothed) == len(log)
    # smoothed trend should be monotonic like the raw signal
    assert np.all(np.diff(smoothed) > 0)
    # and close to the raw series (small noise here)
    assert abs(smoothed[-1] - log[-1]) < 0.3


def test_flat_series_has_no_anomalies():
    log = [80.0] * 20
    flags, _ = detect_anomalies(log)
    assert flags.sum() == 0


def test_salt_spike_is_flagged():
    rng = np.random.RandomState(0)
    log = list(80 + 0.02 * np.arange(20) + rng.normal(0, 0.1, 20))
    log[10] += 1.5  # one obvious water spike
    flags, _ = detect_anomalies(log)
    assert flags[10], "a +1.5 kg spike must be flagged"
    assert int(flags.sum()) == 1, "only the spike, not ~10% of the log"


def test_pure_noise_not_overflagged():
    """Regression test: the old implementation forced ~10% flags by
    construction even on pure noise."""
    rng = np.random.RandomState(7)
    log = list(np.round(80 + rng.normal(0, 0.3, 14), 1))
    flags, _ = detect_anomalies(log)
    assert int(flags.sum()) <= 2


def test_calorie_balance_math():
    log = [80 - 0.05 * i for i in range(30)]
    res = estimate_calorie_balance(log)
    expected = -0.05 * KCAL_PER_KG
    assert res["kcal_per_day"] == pytest.approx(expected, abs=20)
    assert res["weekly_kg_change"] == pytest.approx(-0.35, abs=0.03)
    assert res["ci_95_kcal_per_day"] >= 0


def test_run_pipeline_output_shape():
    rng = np.random.RandomState(1)
    log = list(80 - 0.04 * np.arange(60) + rng.normal(0, 0.3, 60))
    out = run_pipeline(log, target_weekly_change=-0.3)
    for key in ["smoothed", "anomaly_flags", "anomalies_detected", "kcal_per_day",
                "weekly_kg_change", "r_squared", "ci_95_kcal_per_day",
                "kcal_per_day_28d", "anomaly_method"]:
        assert key in out
    assert len(out["smoothed"]) == len(log)
    assert out["anomaly_method"] in ("isolation_forest", "robust_mad")
