import numpy as np
import pandas as pd

KCAL_PER_KG = 7700
DAYS = 90

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
    """Mifflin St Jeor equation."""
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


def generate_user(user_id, random_state=None):
    """
    Generates one synthetic user with 90 days of weight readings.
    Age and weight are sampled from NHANES-calibrated distributions
    to match real population statistics.
    """
    rng = np.random.RandomState(random_state)

    gender         = rng.choice(["Male", "Female"])
    age            = int(np.clip(rng.normal(47.3, 17.2), 18, 80))
    height         = rng.uniform(155, 190)
    start_weight   = np.clip(rng.normal(81.0, 21.5), 32, 220)
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

    for day in range(DAYS):
        adaptation_factor = 1 - metabolic_adapt_rate * (day / DAYS)
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


def generate_dataset(n_users=500, days=DAYS):
    """
    Generates a full dataset of n_users synthetic users.
    Returns a flat DataFrame with one row per user per day.
    """
    users = [generate_user(i) for i in range(n_users)]
    rows  = []
    for user in users:
        base = {k: v for k, v in user.items() if k != "weight_log"}
        for day, weight in enumerate(user["weight_log"]):
            row           = base.copy()
            row["day"]    = day + 1
            row["weight"] = weight
            rows.append(row)
    return pd.DataFrame(rows)