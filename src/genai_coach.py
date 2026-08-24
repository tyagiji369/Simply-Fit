"""
Legacy module — kept for backwards compatibility (notebook 04 / old docs).

The production coach now lives in ``src/agent_coach.py`` (intent-routed,
tool-assisted, with LLM grounding). ``get_response`` simply delegates to
it so there is exactly one source of coaching logic.
"""
from src.agent_coach import agent_coach

KCAL_PER_KG = 7700


def build_context(profile, weight_log, target_weekly_change):
    """
    Deprecated helper — retained for API compatibility.
    Factored out of the LLM prompt in agent_coach.py; the LLM prompt now
    receives structured JSON metrics instead of a template string.
    """
    import numpy as np

    recent = weight_log[-7:]
    weekly_change = round(recent[-1] - recent[0], 2)
    current_weight = round(weight_log[-1], 1)
    start_weight = round(weight_log[0], 1)
    total_change = round(current_weight - start_weight, 2)
    kcal_balance = round((weekly_change * KCAL_PER_KG) / 7, 1)
    on_track = abs(weekly_change - target_weekly_change) < 0.15

    return f"""
USER PROFILE:
- Age: {profile.get('age', 'unknown')}, Gender: {profile.get('gender', 'unknown')}
- Current weight: {current_weight} kg
- Starting weight: {start_weight} kg
- Total change so far: {total_change:+} kg
- Goal: {profile.get('goal', 'unknown')}
- Medical conditions: {', '.join(profile.get('diseases', [])) or 'None'}
- TDEE: {int(profile.get('tdee', 0))} kcal/day

CURRENT WEEK:
- Weight change this week: {weekly_change:+} kg
- Target weekly change: {target_weekly_change:+} kg
- Estimated daily calorie balance: {kcal_balance:+.0f} kcal
- On track: {'Yes' if on_track else 'No'}

RECENT WEIGHT LOG (last 7 days):
{list(recent)}
"""


def get_response(profile, weight_log, question, target_weekly_change):
    """
    Delegates to the production agent coach.
    """
    result = agent_coach.run_agent(
        profile, weight_log, question, target_weekly_change=target_weekly_change
    )
    return result["response"]
