import numpy as np
import pandas as pd
import pytest

from src.data_generator import calculate_bmr, calculate_tdee, generate_dataset, generate_user


def test_mifflin_st_jeor_bmr():
    # male: 10*80 + 6.25*178 - 5*30 + 5 = 800 + 1112.5 - 150 + 5 = 1767.5
    assert calculate_bmr(80, 178, 30, "Male") == pytest.approx(1767.5)
    # female: 10*60 + 6.25*165 - 5*30 - 161 = 600 + 1031.25 - 150 - 161 = 1320.25
    assert calculate_bmr(60, 165, 30, "Female") == pytest.approx(1320.25)


def test_tdee_multipliers():
    assert calculate_tdee(2000, "sedentary") == pytest.approx(2400)
    assert calculate_tdee(2000, "very_active") == pytest.approx(3800)


def test_generate_user_ranges():
    user = generate_user(0, random_state=42)
    assert 18 <= user["age"] <= 80
    assert 32 <= user["start_weight_kg"] <= 220
    assert 140 <= user["height_cm"] <= 200
    assert user["goal"] in {"lose", "gain", "maintain"}
    assert len(user["weight_log"]) == 90
    # observed weight tracks the start weight (no runaway trajectories)
    assert abs(np.mean(user["weight_log"]) - user["start_weight_kg"]) < 8


def test_generate_dataset_reproducible_and_flat():
    df1 = generate_dataset(n_users=5, seed=42)
    df2 = generate_dataset(n_users=5, seed=42)
    pd.testing.assert_frame_equal(df1, df2)
    assert list(df1.columns[:1]) == ["user_id"]
    assert len(df1) == 5 * 90
    assert set(df1["day"]) == set(range(1, 91))


def test_demographics_close_to_nhanes():
    """The NHANES-fitted generator should land near the real population means."""
    df = generate_dataset(n_users=400, seed=7)
    users = df.groupby("user_id").first()
    nhanes = pd.read_csv("data/public/nhanes_adults.csv")
    assert abs(users["age"].mean() - nhanes["age"].mean()) < 3
    assert abs(users["start_weight_kg"].mean() - nhanes["weight_kg"].mean()) < 4
