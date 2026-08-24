"""
NHANES calibration & validation for the synthetic population generator.

This module compares the synthetic users produced by ``src/data_generator.py``
against a **real** NHANES sample stored in ``data/public/nhanes_reference.csv``
(7,481 adults aged 18-80 from the CDC NHANES 2009-10 and 2011-12 surveys,
public domain data distributed via the ProjectMOSAIC/NHANES mirror).

The test is the standard two-sample Kolmogorov-Smirnov (KS) test. A smaller
KS statistic means the two distributions are closer. We report the p-value
honestly: p < 0.05 means the distributions are *still significantly
different* even when the means/sds are close. The "before" comparison uses
the pre-calibration constants that were used earlier in the project
(mean weight 88.5 kg, mean age 36.8 yrs) so the improvement numbers are
reproducible and tied to a fixed reference.

Coming from a *different* NHANES cycle than the generator constants
(which were tuned to 2017-18 published summary statistics) is expected:
the point of the check is to show realistic alignment, not identity.
"""
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src.data_generator import generate_dataset

# Pre-calibration (older) generator constants — kept only for the "before" baseline.
UNCALIBRATED = {
    "weight": {"mean": 88.5, "std": 25.0},   # ~7 kg heavier bias
    "age": {"mean": 36.8, "std": 12.0},      # ~10 yrs younger bias
}

REFERENCE_PATH = "data/public/nhanes_reference.csv"


def load_nhanes_reference(path=REFERENCE_PATH):
    """
    Loads the real NHANES sample shipped with the repository.

    Returns a DataFrame with columns: age, weight_kg, bmi, gender.
    Rows with missing weight/age are dropped. Adults 18-80 only,
    matching the synthetic generator's population.
    """
    df = pd.read_csv(path)
    df = df.rename(columns={"Weight": "weight_kg"})
    df = df[
        (df["age"] >= 18) & (df["age"] <= 80)
        & df["weight_kg"].notna() & df["age"].notna()
    ].copy()
    return df


def run_nhanes_calibration_test(n_users=500, reference_path=REFERENCE_PATH):
    """
    Validates the synthetic population against the real NHANES sample.

    Returns a dict with KS statistics before/after calibration, p-values,
    percentage improvement and summary means. This is the single source of
    truth used by the Streamlit app and the docs — no random "reference"
    distributions are invented anywhere.
    """
    reference = load_nhanes_reference(reference_path)
    if reference.empty:
        raise FileNotFoundError(
            f"No valid NHANES reference rows found in {reference_path}. "
            "The validation cannot run without real data."
        )

    df_synthetic = generate_dataset(n_users=n_users)
    summary = df_synthetic.groupby("user_id").first()

    # Pre-calibration baseline (fixed constants, deterministic).
    rng = np.random.RandomState(42)
    uncal_weight = rng.normal(
        loc=UNCALIBRATED["weight"]["mean"], scale=UNCALIBRATED["weight"]["std"], size=n_users
    )
    uncal_age = rng.normal(
        loc=UNCALIBRATED["age"]["mean"], scale=UNCALIBRATED["age"]["std"], size=n_users
    )

    # Real reference samples.
    ref_weight = reference["weight_kg"].values
    ref_age = reference["age"].values

    # KS tests.
    ks_w_before, p_w_before = ks_2samp(uncal_weight, ref_weight)
    ks_w_after, p_w_after = ks_2samp(summary["start_weight_kg"].values, ref_weight)
    ks_a_before, p_a_before = ks_2samp(uncal_age, ref_age)
    ks_a_after, p_a_after = ks_2samp(summary["age"].values, ref_age)

    def _improvement(before, after):
        return 0.0 if before <= 0 else max(0.0, (before - after) / before * 100)

    return {
        "n_users": n_users,
        "n_reference": len(reference),
        "weight_ks_before": round(float(ks_w_before), 4),
        "weight_ks_after": round(float(ks_w_after), 4),
        "weight_ks_improvement_pct": round(_improvement(ks_w_before, ks_w_after), 1),
        "weight_p_value": float(p_w_after),
        "age_ks_before": round(float(ks_a_before), 4),
        "age_ks_after": round(float(ks_a_after), 4),
        "age_ks_improvement_pct": round(_improvement(ks_a_before, ks_a_after), 1),
        "age_p_value": float(p_a_after),
        "mean_synthetic_weight": round(float(summary["start_weight_kg"].mean()), 1),
        "mean_reference_weight": round(float(ref_weight.mean()), 1),
        "mean_synthetic_age": round(float(summary["age"].mean()), 1),
        "mean_reference_age": round(float(ref_age.mean()), 1),
        "distributions_still_significantly_different": bool(
            p_w_after < 0.05 or p_a_after < 0.05
        ),
    }


def format_validation_summary(results):
    """
    Human-readable summary that states exactly what the test proves
    (and what it does not).
    """
    sig = results["distributions_still_significantly_different"]
    return (
        f"Weight KS: {results['weight_ks_before']} -> {results['weight_ks_after']} "
        f"({results['weight_ks_improvement_pct']}% improvement, p={results['weight_p_value']:.4f})\n"
        f"Age    KS: {results['age_ks_before']} -> {results['age_ks_after']} "
        f"({results['age_ks_improvement_pct']}% improvement, p={results['age_p_value']:.4f})\n"
        f"Means: synthetic {results['mean_synthetic_weight']} kg / {results['mean_synthetic_age']} yrs "
        f"vs NHANES {results['mean_reference_weight']} kg / {results['mean_reference_age']} yrs\n"
        + (
            "Note: p < 0.05 — distributions are *closer* but still statistically "
            "different. This is expected and is stated honestly in the docs."
            if sig
            else "Note: p >= 0.05 — no significant difference detected."
        )
    )


if __name__ == "__main__":
    res = run_nhanes_calibration_test()
    print("NHANES Calibration & Validation Results (real NHANES sample):")
    for k, v in res.items():
        print(f"  {k}: {v}")
    print()
    print(format_validation_summary(res))
