from src.data_generator import generate_dataset, KCAL_PER_KG


def test_dataset_shapes():
    df = generate_dataset(n_users=10)
    assert len(df) == 10 * 90
    assert {"user_id", "day", "weight"}.issubset(df.columns)
    assert df.groupby("user_id").size().unique().tolist() == [90]


def test_weight_ranges_physical():
    df = generate_dataset(n_users=25)
    assert df["weight"].between(30, 250).all()
    assert df["age"].between(18, 80).all()


def test_goal_covered():
    df = generate_dataset(n_users=50)
    assert set(df["goal"].unique()) <= {"lose", "gain", "maintain"}


def test_constant():
    assert KCAL_PER_KG == 7700
