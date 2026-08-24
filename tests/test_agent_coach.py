import numpy as np

from src.agent_coach import compute_time_to_goal


def test_time_to_goal_math():
    weeks, days, remaining = compute_time_to_goal(80.0, 75.0, -0.5)
    assert weeks == 10.0
    assert days == 70
    assert remaining == 5.0


def test_time_to_goal_at_goal():
    weeks, days, remaining = compute_time_to_goal(75.0, 75.0, -0.5)
    assert weeks == 0.0 and days == 0


def test_time_to_goal_handles_zero_rate():
    weeks, days, _ = compute_time_to_goal(80.0, 75.0, 0.0)
    assert weeks == 10.0  # sane default of 0.5 kg/wk


def test_intent_detection_time_to_goal():
    from src.agent_coach import SimplyFitAgent
    agent = SimplyFitAgent()
    ml_res = {"kcal_per_day": -250.0, "anomalies_detected": 1}
    info = agent.analyze_query_intent(
        "How much time will it take to reach my goal?",
        {"target_weight": 75.0}, 80.0, ml_res, -0.5,
    )
    assert info["intent"] == "time_to_goal"
    assert info["days_needed"] == 70


def test_agent_deterministic_answer_no_key():
    """Without an API key the coach must still answer from real data."""
    from src.agent_coach import SimplyFitAgent
    agent = SimplyFitAgent()
    profile = {
        "age": 30, "gender": "male", "height_cm": 175, "weight": 80,
        "goal": "lose", "disease": None, "target_weight": 75.0, "tdee": 2200,
    }
    log = list(80 - 0.05 * np.arange(40))  # ~ -0.35 kg/wk
    res = agent.run_agent(profile, log, "How long until I reach my goal?",
                          target_weekly_change=-0.35, gemini_api_key=None)
    assert res["used_llm"] is False
    assert "75.0 kg" in res["response"]
    assert "weeks" in res["response"]
    assert res["intent"]["intent"] == "time_to_goal"
