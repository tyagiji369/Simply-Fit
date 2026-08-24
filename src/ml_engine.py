"""
Core signal-processing layer for Simply-Fit.

Pipeline:
  1. EWMA smoothing (span=7) isolates the true fat-mass trend from the
     day-to-day water / glycogen / measurement noise.
  2. Anomaly detection flags readings that are unrepresentative of the
     trend. For logs with >= ANOMALY_IF_MIN_SAMPLES points an Isolation
     Forest is used (capped contamination so it can never flag more than
     a fixed number of points for no reason); for shorter logs a robust
     MAD-based z-score is used instead, because a contamination fraction
     on 7-20 points is meaningless.
  3. Linear regression on the cleaned trend converts the slope to a
     daily calorie balance using the standard 7700 kcal/kg constant.
     The uncertainty of that estimate is reported from the residual
     scatter — it is NOT presented as a precise number.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest

KCAL_PER_KG = 7700
ANOMALY_IF_MIN_SAMPLES = 30          # below this, IF is statistically meaningless
MIN_RESIDUAL_KG = 0.5                 # below this magnitude, never call it an anomaly
MAD_Z_THRESHOLD = 4.0                 # robust threshold (in MAD units)
IF_CONTAMINATION = 0.08               # capped: max ~8% of points, never self-invented


def ewma_filter(weight_log, span=7):
    """
    Exponential weighted moving average — extracts the trend from noisy
    daily weight readings.
    """
    return pd.Series(weight_log).ewm(span=span, adjust=False).mean().values


def _robust_flags(residuals, min_residual=MIN_RESIDUAL_KG, z_threshold=MAD_Z_THRESHOLD):
    """
    Deterministic anomaly flagging based on the robust z-score
    (median absolute deviation). A reading is anomalous only if it is
    both large in absolute terms (>= 0.5 kg) and far from the trend in
    MAD units. No fixed percentage of points is forced to be flagged.
    """
    center = np.median(residuals)
    mad = np.median(np.abs(residuals - center))
    if mad < 1e-6:  # perfectly flat residual distribution
        return np.zeros(len(residuals), dtype=bool)
    z = np.abs(residuals - center) / (1.4826 * mad)
    return (z > z_threshold) & (np.abs(residuals) >= min_residual)


def detect_anomalies(weight_log, method="auto", contamination=IF_CONTAMINATION):
    """
    Flags anomalous weight readings.

    Returns (flags, smoothed). ``flags`` is a boolean array where True
    means "unrepresentative reading, excluded from the calorie estimate".
    """
    smoothed = ewma_filter(weight_log)
    residuals = np.array(weight_log, dtype=float) - smoothed

    n = len(weight_log)
    if n < 5:
        return np.zeros(n, dtype=bool), smoothed

    use_iforest = method == "iforest" or (
        method == "auto" and n >= ANOMALY_IF_MIN_SAMPLES
    )

    if use_iforest:
        # Cap the contamination so small logs can never get more than 3
        # flags "by construction"; larger logs get the (capped) fraction.
        effective = min(contamination, 3.0 / max(n, 1))
        iso = IsolationForest(contamination=effective, random_state=42)
        flags = iso.fit_predict(residuals.reshape(-1, 1)) == -1
        # Keep only flags that are also physically meaningful.
        flags &= np.abs(residuals) >= MIN_RESIDUAL_KG
    else:
        flags = _robust_flags(residuals)

    return flags, smoothed


def estimate_calorie_balance(weight_log, kcal_per_kg=KCAL_PER_KG, window_days=None):
    """
    Fits a linear regression on the *cleaned* weight series and converts the
    slope to an inferred daily calorie balance.

    The regression runs on the raw (anomaly-replaced) readings rather than
    on the EWMA-smoothed series: a moving-average filter contracts the
    series and systematically attenuates the slope (measured ~6% at 30
    days), whereas OLS on the raw series is unbiased and its residual
    scatter gives a proper confidence interval.

    Returns the balance, weekly kg change, R^2, and a 95% uncertainty
    interval built from the residual scatter (per-standard-error of the
    slope x 7700). ``window_days`` restricts the fit to the most recent
    N days (used to show both long-term and recent estimates).
    """
    series = np.asarray(weight_log, dtype=float)
    if window_days is not None:
        series = series[-window_days:]
    days = np.arange(len(series)).reshape(-1, 1)

    model = LinearRegression()
    model.fit(days, series)
    kg_per_day = float(model.coef_[0])
    r2 = float(model.score(days, series))

    # Standard error of the slope -> 95% CI of kcal/day.
    residuals = series - model.predict(days)
    n = len(series)
    if n > 2:
        sxx = float(((days - days.mean()) ** 2).sum())
        se_slope = float(np.sqrt((residuals ** 2).sum() / (n - 2) / sxx))
        ci_half = 1.96 * se_slope * kcal_per_kg
    else:
        ci_half = 0.0

    return {
        "kcal_per_day": round(kg_per_day * kcal_per_kg, 1),
        "weekly_kg_change": round(kg_per_day * 7, 3),
        "r_squared": round(r2, 3),
        "ci_95_kcal_per_day": round(ci_half, 1),
        "kg_per_day": round(kg_per_day, 5),
    }


def run_pipeline(weight_log, target_weekly_change=None, anomaly_method="auto"):
    """
    Full ML pipeline — filter, estimate, detect, recommend.

    Returns all outputs for the Streamlit app, including the anomaly
    method used and a long/short-window estimate with uncertainty.
    """
    flags, smoothed = detect_anomalies(weight_log, method=anomaly_method)
    clean_weights = np.where(flags, smoothed, weight_log)

    result = estimate_calorie_balance(clean_weights)
    short = estimate_calorie_balance(clean_weights, window_days=min(28, len(clean_weights)))

    output = {
        "smoothed": smoothed,
        "anomaly_flags": flags,
        "anomalies_detected": int(flags.sum()),
        "anomaly_method": "isolation_forest" if (
            anomaly_method == "iforest" or (
                anomaly_method == "auto" and len(weight_log) >= ANOMALY_IF_MIN_SAMPLES
            )
        ) else "robust_mad",
        "kcal_per_day": result["kcal_per_day"],
        "weekly_kg_change": result["weekly_kg_change"],
        "r_squared": result["r_squared"],
        "ci_95_kcal_per_day": result["ci_95_kcal_per_day"],
        "kcal_per_day_28d": short["kcal_per_day"],
        "weekly_kg_change_28d": short["weekly_kg_change"],
    }

    if target_weekly_change is not None:
        target_kcal = (target_weekly_change * KCAL_PER_KG) / 7
        gap = target_kcal - result["kcal_per_day"]
        adjustment = max(min(gap, 300), -300)
        output["target_kcal_per_day"] = round(target_kcal, 1)
        output["gap_kcal"] = round(gap, 1)
        output["food_adjustment_kcal"] = round(adjustment * 0.6, 1)
        output["activity_adjustment_kcal"] = round(adjustment * 0.4, 1)

    return output
