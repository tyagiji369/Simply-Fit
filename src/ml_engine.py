"""
Classical ML layer for Simply-Fit.

Pipeline for inferring calorie balance from a daily weight log:

  1. EWMA smoothing (span=7) separates the fat-change trend from daily
     water/glycogen fluctuation.
  2. Isolation Forest flags anomalous readings (residuals vs the smoothed
     trend), which are replaced by their smoothed value so a salty-dinner
     spike cannot corrupt the calorie estimate.
  3. A linear regression on the smoothed trend gives kg/day, which converts
     to a daily calorie balance via the energy-balance constant
     (7,700 kcal per kg of fat tissue — the classic Wishnofsky rule;
     modern estimates vary, so this is a modeling approximation).

Design note: the 7,700 kcal/kg constant is applied uniformly. A truly
personal coefficient cannot be identified from weight data alone (the same
weight slope is consistent with many intake/expenditure combinations), so
the personalization in this system comes from fitting the trend to the
user's own data — not from learning a per-user kcal/kg constant.

Uncertainty: `kcal_per_day_ci95` is the nominal 95% interval from the
regression slope's standard error. Because the input series is EWMA-smoothed
(residuals are autocorrelated), this interval is anti-conservative — treat
it as a lower bound on uncertainty.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression

KCAL_PER_KG = 7700


def ewma_filter(weight_log, span=7):
    """
    Applies exponential weighted moving average to extract
    the true fat-change trend from noisy daily weight readings.
    """
    return pd.Series(weight_log).ewm(span=span, adjust=False).mean().values


def estimate_calorie_balance(weight_log, kcal_per_kg=KCAL_PER_KG):
    """
    Fits a linear regression on the smoothed weight trend.
    Returns the inferred daily calorie balance with a nominal 95% CI,
    weekly kg change, and regression fit quality.
    """
    smoothed   = ewma_filter(weight_log)
    days       = np.arange(len(smoothed))
    model      = LinearRegression()
    model.fit(days.reshape(-1, 1), smoothed)

    kg_per_day   = model.coef_[0]
    kcal_per_day = kg_per_day * kcal_per_kg
    r_squared    = model.score(days.reshape(-1, 1), smoothed)

    # Standard error of the slope -> 95% CI on kcal_per_day.
    n_days    = len(days)
    residuals = smoothed - model.predict(days.reshape(-1, 1))
    dof       = max(n_days - 2, 1)
    slope_se  = (
        np.sqrt((residuals ** 2).sum() / dof / ((days - days.mean()) ** 2).sum())
        if n_days > 2 else 0.0
    )
    t_crit    = stats.t.ppf(0.975, dof) if n_days > 2 else 0.0
    ci_half   = t_crit * slope_se * kcal_per_kg

    return {
        "kcal_per_day":     round(kcal_per_day, 1),
        "kcal_per_day_ci95": (round(kcal_per_day - ci_half, 1),
                              round(kcal_per_day + ci_half, 1)),
        "kg_per_day":       round(kg_per_day, 4),
        "weekly_kg_change": round(kg_per_day * 7, 3),
        "r_squared":        round(r_squared, 3),
    }


def detect_anomalies(weight_log, contamination=0.1):
    """
    Runs Isolation Forest on residuals between raw readings and the EWMA
    trend. Returns a boolean array (True = anomaly) plus the smoothed trend.
    Anomalous readings are excluded from calorie estimation to prevent
    water-retention spikes from corrupting results.

    Note: contamination=0.1 means roughly 10% of readings are flagged by
    construction — this rate is a deliberate design choice (typical daily
    water-weight volatility), not a learned quantity.
    """
    smoothed  = ewma_filter(weight_log)
    residuals = np.array(weight_log) - smoothed
    iso       = IsolationForest(contamination=contamination, random_state=42)
    flags     = iso.fit_predict(residuals.reshape(-1, 1)) == -1
    return flags, smoothed


def run_pipeline(weight_log, target_weekly_change=None):
    """
    Full ML pipeline — filter, estimate, detect, recommend.
    Returns a dict with all outputs for the Streamlit app.
    """
    flags, smoothed = detect_anomalies(weight_log)
    clean_weights   = np.where(flags, smoothed, weight_log)
    result          = estimate_calorie_balance(clean_weights)

    output = {
        "smoothed":            smoothed,
        "anomaly_flags":       flags,
        "anomalies_detected":  int(flags.sum()),
        "kcal_per_day":        result["kcal_per_day"],
        "kcal_per_day_ci95":   result["kcal_per_day_ci95"],
        "weekly_kg_change":    result["weekly_kg_change"],
        "r_squared":           result["r_squared"],
    }

    if target_weekly_change is not None:
        target_kcal  = (target_weekly_change * KCAL_PER_KG) / 7
        gap          = target_kcal - result["kcal_per_day"]
        adjustment   = max(min(gap, 300), -300)
        output["target_kcal_per_day"]      = round(target_kcal, 1)
        output["gap_kcal"]                 = round(gap, 1)
        output["food_adjustment_kcal"]     = round(adjustment * 0.6, 1)
        output["activity_adjustment_kcal"] = round(adjustment * 0.4, 1)

    return output
