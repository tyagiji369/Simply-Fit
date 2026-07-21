import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest

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
    Returns inferred daily calorie balance, weekly kg change,
    and the personal kcal-per-kg coefficient for this user.
    """
    smoothed     = ewma_filter(weight_log)
    days         = np.arange(len(smoothed)).reshape(-1, 1)
    model        = LinearRegression()
    model.fit(days, smoothed)
    kg_per_day   = model.coef_[0]
    kcal_per_day = kg_per_day * kcal_per_kg
    r_squared    = model.score(days, smoothed)

    return {
        "kcal_per_day":       round(kcal_per_day, 1),
        "weekly_kg_change":   round(kg_per_day * 7, 3),
        "personal_coeff":     round(kg_per_day * kcal_per_kg, 1),
        "r_squared":          round(r_squared, 3),
    }


def detect_anomalies(weight_log, contamination=0.1):
    """
    Runs Isolation Forest on residuals between raw readings
    and the EWMA trend. Returns boolean array — True = anomaly.
    Anomalous readings are excluded from calorie estimation
    to prevent water-retention spikes from corrupting results.
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
        "weekly_kg_change":    result["weekly_kg_change"],
        "r_squared":           result["r_squared"],
    }

    if target_weekly_change is not None:
        target_kcal  = (target_weekly_change * KCAL_PER_KG) / 7
        gap          = target_kcal - result["kcal_per_day"]
        adjustment   = max(min(gap, 300), -300)
        output["target_kcal_per_day"]    = round(target_kcal, 1)
        output["gap_kcal"]               = round(gap, 1)
        output["food_adjustment_kcal"]   = round(adjustment * 0.6, 1)
        output["activity_adjustment_kcal"] = round(adjustment * 0.4, 1)

    return output