"""
Synthetic cohort generator for Simply-Fit.

There is no public dataset of daily weight logs with disease conditions, so
training data is simulated. The simulation has two parts:

1. Demographics (age, gender, height, start weight) — sampled from
   distributions fitted to real CDC NHANES 2017-2018 data. The fitted
   parameters come from data/public/nhanes_adults.csv, a cleaned extract of
   5,434 US adults built directly from the CDC public files (DEMO_J.XPT and
   BMX_J.XPT, https://wwwn.cdc.gov/nchs/nhanes/). See src/calibration.py for
   the distribution alignment check.

2. Physiology and behaviour (BMR via Mifflin-St Jeor, TDEE, goal, adherence,
   metabolic adaptation, daily noise) — simulated from published equations
   and clinical weight-variability assumptions. This is the part the project
   actually contributes: the demographic realism is borrowed from NHANES on
   purpose, so the trained model sees a realistic population.

The original version of this generator used arbitrary parameter choices
(age ~ Uniform(18, 60), weight ~ N(88.5, 25)). The first NHANES validation
round (notebooks/05_nhanes_validation.ipynb) showed that cohort was ~7 kg
heavier and ~10 years younger than the real adult population, so the
demographic parameters were re-fitted to NHANES. The "naive" parameterization
is kept in src/calibration.py as the documented before/after baseline.
"""

import numpy as np
import pandas as pd

KCAL_PER_KG = 7700
DAYS = 90

# Demographic parameters fitted to the cleaned NHANES 2017-2018 adult extract
# (data/public/nhanes_adults.csv, n = 5,434; adults 18-80).
#   * age ~ Normal(49.7, 18.6), clipped to [18, 80]
#   * start weight ~ Lognormal per gender (fitted on log-scale moments).
#     NHANES weight is right-skewed; a lognormal matches the real shape far
#     better than a Gaussian (KS 0.035 vs 0.100 — see src/calibration.py).
#   * height ~ Normal per gender
NHANES_FIT = {
    "p_female": 0.517,
    "age_mean": 49.7,
    "age_std": 18.6,
    "height": {"Male": (173.5, 7.7), "Female": (159.7, 7.0)},
    "weight_lognormal": {"Male": (4.4513, 0.2385), "Female": (4.3068, 0.2703)},
}

DISEASE_OPTIONS = [
    None,
    "Type 2 Diabetes",
    "Hypertension",
    "High Cholesterol",
    "Hypothyroidism",
    "PCOS",
    "NAFLD",
    "Insulin Resistance",
    "CKD"
]


def calculate_bmr(weight, height, age, gender):
    """Mifflin-St Jeor equation (kcal/day)."""
    if gender == "Male":
        return 10 * weight + 6.25 * height - 5 * age + 5
    return 10 * weight + 6.25 * height - 5 * age - 161


def calculate_tdee(bmr, activity_level):
    multipliers = {
        "sedentary":    1.2,
        "light":        1.375,
        "moderate":     1.55,
        "active":       1.725,
        "very_active":  1.9
    }
    return bmr * multipliers[activity_level]


def generate_user(user_id, random_state=None, days=DAYS):
    """
    Generates one synthetic user with `days` daily weight readings.

    Age and start weight are sampled from NHANES-fitted distributions so the
    synthetic population matches real US adult demographics (see module
    docstring). The daily weight trajectory is then simulated from energy
    balance: true weight changes by (daily calorie delta / 7700) per day,
    and the observed scale reading adds glycogen, sodium and measurement
    noise on top. Conditions that cause water retention inflate the
    corresponding noise terms.
    """
    rng = np.random.RandomState(random_state)

    gender = "Female" if rng.rand() < NHANES_FIT["p_female"] else "Male"

    age = int(np.clip(rng.normal(NHANES_FIT["age_mean"], NHANES_FIT["age_std"]), 18, 80))

    h_mean, h_std = NHANES_FIT["height"][gender]
    height = rng.normal(h_mean, h_std)

    w_mu, w_sigma = NHANES_FIT["weight_lognormal"][gender]
    start_weight = np.clip(rng.lognormal(w_mu, w_sigma), 32, 220)

    activity_level = rng.choice(
        ["sedentary", "light", "moderate", "active", "very_active"]
    )
    goal = rng.choice(
        ["lose", "gain", "maintain"],
        p=[0.55, 0.25, 0.20]
    )
    disease = rng.choice(
        DISEASE_OPTIONS,
        p=[0.40, 0.08, 0.08, 0.07, 0.07, 0.07, 0.07, 0.08, 0.08]
    )

    adherence            = rng.uniform(0.4, 1.0)
    metabolic_adapt_rate = rng.uniform(0.02, 0.10)

    bmr  = calculate_bmr(start_weight, height, age, gender)
    tdee = calculate_tdee(bmr, activity_level)

    if goal == "lose":
        target_daily_delta = rng.uniform(-600, -200)
    elif goal == "gain":
        target_daily_delta = rng.uniform(200, 500)
    else:
        target_daily_delta = 0.0

    true_weight = start_weight
    weight_log  = []

    for day in range(days):
        adaptation_factor = 1 - metabolic_adapt_rate * (day / days)
        adapted_delta     = target_daily_delta * adaptation_factor

        if rng.rand() < adherence:
            actual_delta = adapted_delta
        else:
            actual_delta = rng.uniform(-100, 200)

        true_weight += actual_delta / KCAL_PER_KG

        glycogen_noise    = rng.normal(0, 0.30)
        sodium_noise      = rng.normal(0, 0.20)
        measurement_noise = rng.normal(0, 0.12)

        if disease in ["Hypertension", "CKD", "NAFLD"]:
            sodium_noise *= 1.5
        if disease in ["PCOS", "Hypothyroidism"]:
            glycogen_noise *= 1.3

        observed = (
            true_weight
            + glycogen_noise
            + sodium_noise
            + measurement_noise
        )
        weight_log.append(round(observed, 1))

    return {
        "user_id":              user_id,
        "gender":               gender,
        "age":                  age,
        "height_cm":            round(height, 1),
        "start_weight_kg":      round(start_weight, 1),
        "activity_level":       activity_level,
        "goal":                 goal,
        "disease":              disease if disease else "none",
        "adherence":            round(adherence, 2),
        "metabolic_adapt_rate": round(metabolic_adapt_rate, 3),
        "bmr":                  round(bmr, 1),
        "tdee":                 round(tdee, 1),
        "target_daily_delta":   round(target_daily_delta, 1),
        "weight_log":           weight_log
    }


def generate_dataset(n_users=500, days=DAYS, seed=42):
    """
    Generates a full dataset of n_users synthetic users.

    Returns a flat DataFrame with one row per user per day. With a fixed
    `seed` the dataset is fully reproducible.
    """
    users = [
        generate_user(i, random_state=None if seed is None else seed + i, days=days)
        for i in range(n_users)
    ]
    rows = []
    for user in users:
        base = {k: v for k, v in user.items() if k != "weight_log"}
        for day, weight in enumerate(user["weight_log"]):
            row        = base.copy()
            row["day"] = day + 1
            row["weight"] = weight
            rows.append(row)
    return pd.DataFrame(rows)
