"""
Distribution validation of the synthetic cohort against real CDC NHANES data.

Reference population
--------------------
data/public/nhanes_adults.csv — 5,434 US adults (18-80) extracted directly
from the CDC NHANES 2017-2018 public files:

  * DEMO_J.XPT (demographics: RIAGENDR, RIDAGEYR)
  * BMX_J.XPT  (body measures: BMXWT, BMXHT, BMXBMI)
  * https://wwwn.cdc.gov/nchs/nhanes/

Rows with missing values or implausible ranges (weight outside 30-250 kg,
height outside 120-220 cm, age outside 18-80) were dropped. The extract is
committed so this check is fully reproducible.

What is compared
----------------
Two synthetic cohorts are compared against the real data with the two-sample
Kolmogorov-Smirnov test. The KS statistic is the maximum vertical gap between
the two empirical CDFs (0 = identical distributions):

  1. "Naive" — the ORIGINAL pre-calibration generator parameters documented
     in notebooks/05_nhanes_validation.ipynb: age ~ Uniform(18, 60),
     weight ~ N(88.5, 25). The first validation round found this cohort was
     ~7 kg heavier and ~10 years younger than the real population.

  2. "Calibrated" — the current generator (src/data_generator.py), whose
     demographic parameters are fitted to the NHANES extract above.

Notes on interpretation
-----------------------
* The KS statistic is the practical measure of distribution alignment. With
  n = 500 synthetic vs 5,434 real samples, even small distributional gaps are
  statistically detectable, so a significant p-value does not by itself mean
  the cohort is unusable — the statistic size does.
* The synthetic age/weight marginals are parametric (Gaussian / clipped), so
  a small residual KS gap remains by construction; NHANES weight is
  right-skewed and NHANES age is closer to uniform across adult ranges.
* Comparison is unweighted (NHANES survey weights are not applied).

Run directly to print the report:
    python -m src.calibration
"""

import os

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src.data_generator import generate_dataset

NHANES_EXTRACT_PATH = "data/public/nhanes_adults.csv"

# Original (pre-calibration) generator parameters — kept as the documented
# "before" baseline. Source: notebooks/05_nhanes_validation.ipynb.
NAIVE_PARAMS = {
    "age":    ("uniform", 18, 60),
    "weight": ("normal", 88.5, 25.0),
}


def _resolve_path(path):
    """Find the NHANES extract whether run from the repo root or elsewhere."""
    candidates = [
        path,
        os.path.join(os.path.dirname(__file__), "..", path),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(
        f"NHANES extract not found. Expected it at {NHANES_EXTRACT_PATH}. "
        "This file is committed with the repo — check your working directory."
    )


def load_nhanes_extract(path=NHANES_EXTRACT_PATH):
    """Loads the committed, cleaned NHANES 2017-2018 adult extract."""
    return pd.read_csv(_resolve_path(path))


def _naive_cohort(n_users, rng):
    """Reproduces the ORIGINAL naive parameterization of the generator."""
    age    = rng.uniform(NAIVE_PARAMS["age"][1], NAIVE_PARAMS["age"][2], n_users)
    weight = np.clip(rng.normal(NAIVE_PARAMS["weight"][1], NAIVE_PARAMS["weight"][2], n_users), 32, 220)
    return age, weight


def run_nhanes_calibration_test(n_users=500, seed=42):
    """
    Compares the naive and calibrated synthetic cohorts against the real
    NHANES extract using two-sample KS tests. Returns a dict of results
    (also rendered by the Streamlit app's "Data Realism" panel).
    """
    nhanes = load_nhanes_extract()

    # Calibrated cohort: current generator, seeded for reproducibility.
    df_synthetic = generate_dataset(n_users=n_users, seed=seed)
    user_summary = df_synthetic.groupby("user_id").first()

    # Naive cohort: the original pre-calibration parameters.
    rng = np.random.RandomState(seed)
    naive_age, naive_weight = _naive_cohort(n_users, rng)

    # KS statistics — smaller = closer to the real distribution.
    ks_weight_before, p_w_before = ks_2samp(naive_weight, nhanes["weight_kg"].values)
    ks_weight_after,  p_w_after  = ks_2samp(user_summary["start_weight_kg"].values, nhanes["weight_kg"].values)
    ks_age_before,    p_a_before = ks_2samp(naive_age, nhanes["age"].values)
    ks_age_after,     p_a_after  = ks_2samp(user_summary["age"].values, nhanes["age"].values)

    weight_ks_impr = max(0.0, (ks_weight_before - ks_weight_after) / ks_weight_before * 100)
    age_ks_impr    = max(0.0, (ks_age_before - ks_age_after) / ks_age_before * 100)

    return {
        "n_users":               n_users,
        "n_nhanes":              int(len(nhanes)),
        # Weight (kg)
        "weight_ks_before":      round(float(ks_weight_before), 3),
        "weight_ks_after":       round(float(ks_weight_after), 3),
        "weight_ks_improvement_pct": round(float(weight_ks_impr), 1),
        "weight_p_after":        round(float(p_w_after), 4),
        # Age (years)
        "age_ks_before":         round(float(ks_age_before), 3),
        "age_ks_after":          round(float(ks_age_after), 3),
        "age_ks_improvement_pct":    round(float(age_ks_impr), 1),
        "age_p_after":           round(float(p_a_after), 4),
        # Means (sanity check)
        "mean_synthetic_weight": round(float(user_summary["start_weight_kg"].mean()), 1),
        "mean_nhanes_weight":    round(float(nhanes["weight_kg"].mean()), 1),
        "mean_synthetic_age":    round(float(user_summary["age"].mean()), 1),
        "mean_nhanes_age":       round(float(nhanes["age"].mean()), 1),
    }


if __name__ == "__main__":
    res = run_nhanes_calibration_test()
    print("NHANES calibration report (KS statistic: 0 = identical distributions)")
    print(f"  reference: {res['n_nhanes']} real adults, NHANES 2017-2018 extract")
    print()
    print(f"  weight KS: {res['weight_ks_before']} (naive)  ->  {res['weight_ks_after']} (calibrated)   "
          f"[{res['weight_ks_improvement_pct']}% reduction]")
    print(f"  age    KS: {res['age_ks_before']} (naive)  ->  {res['age_ks_after']} (calibrated)   "
          f"[{res['age_ks_improvement_pct']}% reduction]")
    print()
    print(f"  mean weight: synthetic {res['mean_synthetic_weight']} kg  vs  NHANES {res['mean_nhanes_weight']} kg")
    print(f"  mean age:    synthetic {res['mean_synthetic_age']} yrs  vs  NHANES {res['mean_nhanes_age']} yrs")
    print(f"  p-values (calibrated vs real): weight {res['weight_p_after']}, age {res['age_p_after']}")
