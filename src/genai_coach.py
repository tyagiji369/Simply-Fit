import os
from dotenv import load_dotenv

load_dotenv()

KCAL_PER_KG = 7700


def build_context(profile, weight_log, target_weekly_change):
    """
    Constructs a structured context string from the user's
    real data to inject into the LLM prompt before each response.
    This is the RAG-lite pattern — no vector database needed
    because the relevant data fits directly in the context window.
    """
    recent         = weight_log[-7:]
    weekly_change  = round(recent[-1] - recent[0], 2)
    current_weight = round(weight_log[-1], 1)
    start_weight   = round(weight_log[0], 1)
    total_change   = round(current_weight - start_weight, 2)
    kcal_balance   = round((weekly_change * KCAL_PER_KG) / 7, 1)
    on_track       = abs(weekly_change - target_weekly_change) < 0.15

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
    Returns a personalised coaching response.

    Currently uses a smart rule-based mock that generates
    responses from real user data. To activate the Gemini API,
    uncomment the block below and comment out the mock section.
    """
    context      = build_context(profile, weight_log, target_weekly_change)
    recent       = weight_log[-7:]
    weekly_change = round(recent[-1] - recent[0], 2)
    on_track     = abs(weekly_change - target_weekly_change) < 0.15
    diseases     = profile.get("diseases", [])

    # ── Mock response (rule-based from real data) ──────────────
    if on_track:
        response = (
            f"You are right on track. Your weight changed by "
            f"{weekly_change:+.2f} kg this week against a target of "
            f"{target_weekly_change:+.1f} kg. Keep doing what you are doing. "
            f"Consistency is your biggest asset right now."
        )
    else:
        gap       = round(target_weekly_change - weekly_change, 2)
        food_adj  = abs(int(gap * KCAL_PER_KG / 7 * 0.6))
        act_adj   = abs(int(gap * KCAL_PER_KG / 7 * 0.4))
        response  = (
            f"You are slightly off track — {weekly_change:+.2f} kg this week "
            f"vs target of {target_weekly_change:+.1f} kg. "
            f"Try reducing intake by {food_adj} kcal/day "
            f"and increasing activity by {act_adj} kcal/day."
        )

    if diseases:
        response += (
            f" Given your condition(s), please verify any "
            f"dietary changes with your doctor."
        )
    # ── End mock ───────────────────────────────────────────────

    # ── Gemini API (uncomment when credits available) ──────────
    # from google import genai
    # client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    # system = f"""You are Simply-Fit Coach, a personal health assistant.
    # Answer using the user's actual data below. Be specific and concise.
    # Never diagnose medical conditions. Under 150 words.
    #
    # {context}"""
    # response = client.models.generate_content(
    #     model="gemini-1.5-flash",
    #     contents=f"{system}\n\nUser question: {question}"
    # ).text
    # ── End Gemini ─────────────────────────────────────────────

    return response