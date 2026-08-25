import pytest

from src.agent_coach import agent_coach
from src.data_generator import generate_dataset
from src.ml_engine import run_pipeline


@pytest.fixture(scope="module")
def sample_context():
    df = generate_dataset(n_users=3, seed=11)
    weights = df[df.user_id == 0]["weight"].tolist()
    profile = {"age": 40, "gender": "Male", "target_weight": 78.0,
               "tdee": 2400, "disease": "Hypertension"}
    ml_res = run_pipeline(weights)
    return profile, weights, ml_res


INTENT_CASES = [
    ("How long until I reach my goal weight?", "time_to_goal"),
    ("How many calories should I eat every day?", "calorie_nutrition"),
    ("Why does my weight fluctuate so much day to day?", "water_weight"),
    ("I think I have hit a plateau, nothing moves", "plateau"),
    ("What should my doctor know about my hypertension diet?", "medical_guideline"),
    ("How am I doing overall?", "general_progress"),
]


@pytest.mark.parametrize("question,expected", INTENT_CASES)
def test_intent_routing(sample_context, question, expected):
    profile, weights, ml_res = sample_context
    info = agent_coach.analyze_query_intent(
        question, profile, 80.0, ml_res, target_weekly_change=-0.5
    )
    assert info["intent"] == expected


def test_run_agent_without_api_key(sample_context):
    """The deterministic path must produce a grounded answer with no key."""
    profile, weights, _ = sample_context
    res = agent_coach.run_agent(profile, weights,
                                 "How long until I reach my goal weight?",
                                 target_weekly_change=-0.5)
    assert res["intent"]["intent"] == "time_to_goal"
    assert res["response"].strip()
    assert "kg" in res["response"]
    # execution log: intent step + 3 tool observations
    assert len(res["trace"]) >= 4
    assert res["trace"][0]["action"] == "classify_intent"
    # no fabricated 'thought' keys in the trace
    assert all("thought" not in step for step in res["trace"])


def test_time_to_goal_math(sample_context):
    profile, weights, _ = sample_context
    res = agent_coach.run_agent(profile, weights,
                                "How many weeks to reach my goal?",
                                target_weekly_change=-0.5)
    info = res["intent"]
    assert info["days_needed"] == pytest.approx(info["weeks_needed"] * 7, abs=1)
    assert info["remaining_kg"] >= 0
