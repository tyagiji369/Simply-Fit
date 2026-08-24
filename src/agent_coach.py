"""
Simpl-Fit Conversational Coach.

Architecture (named honestly in the docs): the coach is **intent-routed
and tool-assisted**, not a self-directed ReAct agent. It classifies the
question, computes the relevant metrics with the ML pipeline, retrieves
clinical guidelines and then either (a) answers directly with a
deterministic synthesis, or (b) sends the *already-computed* metrics +
retrieved guidelines to a Gemini model, so the LLM can only reason over
grounded data — it never invents user metrics.

Safety rules enforced in both paths:
  * never diagnose, never give medication advice
  * always attached to the real user data (no generic advice)
  * plateau / diet-break statements are worded with evidence caveats
"""
import os
import json
from datetime import datetime, timedelta
import numpy as np
from dotenv import load_dotenv

from src.ml_engine import run_pipeline as run_ml_pipeline
from src.lstm_forecaster import forecast as forecast_lstm
from src.rag_pipeline import rag_engine

load_dotenv()

GEMINI_MODELS = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]
MAX_RESPONSE_WORDS = 150


def compute_time_to_goal(current_weight, target_weight, weekly_rate):
    """
    Pure helper → returns (weeks, days, remaining_kg). Weekly rate is in
    kg/week (positive in the direction of travel).
    """
    remaining = abs(float(target_weight) - float(current_weight))
    if remaining < 0.05:
        return 0.0, 0, remaining
    rate = abs(float(weekly_rate)) if abs(float(weekly_rate)) > 1e-6 else 0.5
    weeks = remaining / rate
    return weeks, int(round(weeks * 7)), remaining


class SimplyFitAgent:
    """
    Intent-routed, tool-assisted conversational coach.
    """

    def __init__(self):
        self.tools = {
            "infer_calorie_balance": self._tool_infer_calorie_balance,
            "forecast_trajectory": self._tool_forecast_trajectory,
            "retrieve_clinical_guidelines": self._tool_retrieve_clinical_guidelines,
            "check_plateau_status": self._tool_check_plateau_status,
        }

    # ── Tool Definitions ──────────────────────────────────────
    def _tool_infer_calorie_balance(self, weight_log, target_weekly_change):
        return run_ml_pipeline(weight_log, target_weekly_change=target_weekly_change)

    def _tool_forecast_trajectory(self, weight_log):
        try:
            pred = forecast_lstm(weight_log)
            if pred is not None:
                return {
                    "values": [round(float(x), 1) for x in pred["values"]],
                    "method": pred["method"],
                }
        except Exception:
            pass
        from src.lstm_forecaster import linear_baseline
        return {"values": [round(float(x), 1) for x in linear_baseline(weight_log)],
                "method": "linear_fallback"}

    def _tool_retrieve_clinical_guidelines(self, query, disease):
        if not disease or disease in ("none", "None", "No"):
            return []
        return rag_engine.search_guidelines(query, condition=disease, top_k=2)

    def _tool_check_plateau_status(self, weight_log):
        if len(weight_log) < 14:
            return {"is_plateau": False, "change_14d": round(weight_log[-1] - weight_log[0], 2)}
        recent_14 = weight_log[-14:]
        total_delta = recent_14[-1] - recent_14[0]
        is_plateau = abs(total_delta) < 0.2
        return {
            "is_plateau": is_plateau,
            "change_14d": round(total_delta, 2),
            "recommendation": (
                "A short plateau is common. If it persists beyond 3-4 weeks, "
                "review your plan with a professional rather than cutting calories further."
                if is_plateau else "Maintain current deficit."
            ),
        }

    # ── Query Intent & Semantics Analyzer ──────────────────────
    def analyze_query_intent(self, question, profile, current_weight, ml_res, target_weekly_change):
        q_lower = question.lower()
        target_weight = profile.get("target_weight", current_weight - 5.0)

        # 1. Time to goal / duration
        if any(w in q_lower for w in [
            "time", "long", "reach", "achieve", "goal", "when", "weeks", "days",
            "schedule", "finish", "date", "deadline", "eta", "much longer", "how soon",
        ]):
            weeks, days, remaining = compute_time_to_goal(
                current_weight, target_weight, target_weekly_change
            )
            return {
                "intent": "time_to_goal",
                "remaining_kg": round(remaining, 1),
                "target_weight": round(float(target_weight), 1),
                "weeks_needed": round(weeks, 1),
                "days_needed": days,
            }

        # 2. Calorie / Food / Nutrition / Protein
        if any(w in q_lower for w in [
            "calorie", "eat", "intake", "deficit", "macro", "food", "diet",
            "tdee", "protein", "meal", "kcal", "nutrition",
        ]):
            tdee = profile.get("tdee", 2200)
            inferred_intake = tdee + ml_res["kcal_per_day"]
            target_intake = tdee + (target_weekly_change * 7700) / 7.0
            return {
                "intent": "calorie_nutrition",
                "tdee": int(tdee),
                "inferred_intake": int(round(inferred_intake)),
                "target_intake": int(round(target_intake)),
            }

        # 3. Fluctuation / Scale noise / Water weight / Salt
        if any(w in q_lower for w in [
            "fluctuat", "water", "spike", "salt", "sodium", "scale", "anomaly",
            "retention", "bounce", "up and down", "vary", "going up", "went up",
            "higher", "gain but", 
        ]):
            return {"intent": "water_weight", "anomalies": ml_res["anomalies_detected"]}

        # 4. Plateau / Slowdown
        if any(w in q_lower for w in [
            "plateau", "stuck", "slow", "stopped", "not moving", "stagnant",
            "stagnate", "no change", "weeks same",
        ]):
            return {"intent": "plateau", "anomalies": ml_res["anomalies_detected"]}

        # 5. Disease / Medical guideline
        if any(w in q_lower for w in [
            "disease", "hypertension", "diabetes", "pcos", "ckd", "kidney",
            "pressure", "sugar", "condition", "doctor", "health", "safe", "guideline",
        ]):
            return {
                "intent": "medical_guideline",
                "disease": (profile.get("disease")
                            or (profile.get("diseases") or [None])[0]),
            }

        # 6. General progress
        return {"intent": "general_progress"}

    # ── Execution ─────────────────────────────────────────────
    def run_agent(self, profile, weight_log, question, target_weekly_change=-0.5,
                  gemini_api_key=None):
        trace = []
        diseases = profile.get("diseases") or []
        disease = profile.get("disease") or (diseases[0] if diseases else None)
        if disease in (None, "none", "None", "No"):
            disease = None

        # All tools run once; intent then selects what is *used* to answer.
        ml_res = self._tool_infer_calorie_balance(weight_log, target_weekly_change)
        forecast_res = self._tool_forecast_trajectory(weight_log)
        rag_res = self._tool_retrieve_clinical_guidelines(question, disease)
        plateau_res = self._tool_check_plateau_status(weight_log)

        current_weight = round(ml_res["smoothed"][-1], 1)
        intent_info = self.analyze_query_intent(
            question, profile, current_weight, ml_res, target_weekly_change
        )

        trace.append({
            "step": 1,
            "action": "classify_intent",
            "result": intent_info["intent"],
        })
        trace.append({
            "step": 2,
            "tool_call": "infer_calorie_balance",
            "observation": {
                "current_smoothed_weight_kg": current_weight,
                "inferred_kcal_per_day": ml_res["kcal_per_day"],
                "ci_95_kcal_per_day": ml_res["ci_95_kcal_per_day"],
                "weekly_kg_change": ml_res["weekly_kg_change"],
                "anomalies_filtered": ml_res["anomalies_detected"],
            },
        })
        trace.append({"step": 3, "tool_call": "forecast_trajectory",
                      "observation": forecast_res})
        trace.append({
            "step": 4,
            "tool_call": "retrieve_clinical_guidelines",
            "observation": [
                {"condition": g["condition"], "topic": g["topic"],
                 "rule": g["recommendation"]} for g in rag_res
            ] if rag_res else "No condition-specific guidelines retrieved.",
        })

        api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        response_text = None
        llm_error = None
        used_llm = False
        if api_key:
            response_text, llm_error = self._llm_answer(
                api_key, profile, question, current_weight,
                ml_res, forecast_res, rag_res, plateau_res,
                intent_info, disease,
            )
            used_llm = response_text is not None
        if response_text is None:
            response_text = self._smart_intent_synthesis(
                intent_info, question, current_weight, ml_res, target_weekly_change,
                profile, plateau_res, rag_res, disease
            )

        return {
            "trace": trace,
            "response": response_text,
            "ml_output": ml_res,
            "forecast": forecast_res,
            "intent": intent_info,
            "used_llm": used_llm,
            "llm_error": llm_error,
        }

    # ── LLM answer (real API) ─────────────────────────────────
    def _llm_answer(self, api_key, profile, question, current_weight, ml_res,
                    forecast_res, rag_res, plateau_res, intent_info, disease):
        """
        Calls Gemini via the official google-genai SDK.

        Returns (response_text, error). Google currently issues two key
        formats — legacy Standard keys (``AIza…``) and the new Auth keys
        (``AQ.Ab…``); the SDK sends both via the native
        ``x-goog-api-key`` header, so no special handling is needed.
        """
        try:
            from google import genai
        except ImportError:
            return None, "google-genai package is not installed"

        payload = json.dumps({
            "question": question,
            "intent": intent_info["intent"],
            "intent_metrics": intent_info,
            "current_smoothed_weight_kg": current_weight,
            "inferred_kcal_per_day": ml_res["kcal_per_day"],
            "uncertainty_95_ci_kcal": ml_res["ci_95_kcal_per_day"],
            "weekly_kg_change": ml_res["weekly_kg_change"],
            "forecast_7d": forecast_res,
            "plateau_14d": plateau_res,
            "conditions": disease or "none",
            "retrieved_guidelines": rag_res,
            "user_goal": profile.get("goal"),
            "tdee": profile.get("tdee"),
        }, default=str)

        system = (
            "You are Simply-Fit's health coach. Answer the user's exact question first, "
            f"in under {MAX_RESPONSE_WORDS} words, using ONLY the measured data below. "
            "Rules: never diagnose, never prescribe medication, do not invent numbers, "
            "if the retrieved guidelines conflict with general advice say so and recommend "
            "checking with a doctor, be warm and concise.\n\n"
            f"MEASURED DATA (ground truth): {payload}"
        )

        client = genai.Client(api_key=api_key)
        last_error = None
        for model_name in GEMINI_MODELS:
            try:
                res = client.models.generate_content(
                    model=model_name,
                    contents=f"{system}\n\nQuestion: {question}",
                )
                if res and getattr(res, "text", None):
                    return res.text.strip(), None
            except Exception as e:  # noqa: BLE001 - try next model
                last_error = e
        return None, f"{type(last_error).__name__}: {last_error}" if last_error else "No model returned a response"

    # ── Deterministic synthesis ───────────────────────────────
    def _smart_intent_synthesis(self, intent_info, question, current_weight, ml_res,
                                target_weekly_change, profile, plateau_res, rag_res, disease):
        intent = intent_info["intent"]

        if intent == "time_to_goal":
            if intent_info["remaining_kg"] < 0.05:
                return (f"🎯 You have already reached your goal weight of "
                        f"**{intent_info['target_weight']} kg** — great work!")
            target_date = (datetime.now() + timedelta(days=intent_info["days_needed"])).strftime("%d %b %Y")
            return (
                f"🎯 **Estimated Timeline to Goal:**\n\n"
                f"To reach **{intent_info['target_weight']} kg** from your current smoothed "
                f"weight of **{current_weight} kg** ({intent_info['remaining_kg']} kg remaining) at a "
                f"pace of **{abs(target_weekly_change):.2f} kg/week**, it will take about "
                f"**{intent_info['weeks_needed']} weeks** (~**{intent_info['days_needed']} days**).\n\n"
                f"📅 **Estimated date:** **{target_date}** (this assumes your current pace holds)."
            )

        if intent == "calorie_nutrition":
            return (
                f"🥗 **Calorie & Macro Targets:**\n\n"
                f"• **Maintenance (TDEE):** {intent_info['tdee']} kcal/day\n"
                f"• **Inferred current intake:** ~{intent_info['inferred_intake']} kcal/day "
                f"(*{ml_res['kcal_per_day']:+.0f} kcal/day balance, ±{ml_res['ci_95_kcal_per_day']:.0f}*)\n"
                f"• **Target intake:** **{intent_info['target_intake']} kcal/day** for "
                f"{target_weekly_change:+.1f} kg/week pace."
            )

        if intent == "water_weight":
            return (
                f"🌊 **Why Daily Scale Weight Fluctuates:**\n\n"
                f"Normal swings of 0.5–1.5 kg come from water, glycogen and digestion — "
                f"not fat. Your log had **{intent_info['anomalies']} flagged spike(s)** that "
                f"were excluded from the trend. Your smoothed fat-mass trend is "
                f"**{current_weight} kg**. Weigh at the same time each morning and trust "
                f"the 7-day trend, not single days."
            )

        if intent == "plateau":
            change = plateau_res.get("change_14d", 0.0)
            if plateau_res.get("is_plateau"):
                return (
                    f"⚠️ **Plateau Check:**\n\n"
                    f"Your 14-day change is **{change:+.2f} kg** — little movement. Some stalls "
                    f"are temporary water shifts; if it continues 3–4 weeks, the usual cause is "
                    f"that your body now burns fewer calories at the lower weight. Recalculating "
                    f"your deficit is reasonable — a modest diet break or small increase in "
                    f"activity can help re-establish a deficit. If symptoms or uncertainty "
                    f"persist, check with a professional rather than cutting calories further."
                )
            return (
                f"✅ **No plateau detected:** your 14-day change is **{change:+.2f} kg** and "
                f"your smoothed trend is still moving at **{ml_res['weekly_kg_change']:+.2f} kg/week**."
            )

        if intent == "medical_guideline":
            if not disease:
                return (
                    f"🩺 **Guidelines:**\n\n"
                    f"No specific condition is set in your profile. General guidance: stable "
                    f"protein intake, plenty of vegetables, regular movement — and check with "
                    f"your doctor before major dietary changes."
                )
            if rag_res:
                top = rag_res[0]
                return (
                    f"🩺 **Clinical guideline for {disease}:**\n\n"
                    f"• **{top['topic']}:** {top['recommendation']}\n"
                    f"• **Source:** {top['source']}\n\n"
                    f"Please verify any dietary change with your doctor — this guidance is "
                    f"educational, not a prescription."
                )
            return (
                f"🩺 **{disease}:** no guideline matched in the local library — please "
                f"confirm any dietary change with your clinician."
            )

        on_track = abs(ml_res["weekly_kg_change"] - target_weekly_change) < 0.15
        msg = (
            f"📊 **Your progress:**\n\n"
            f"• **Smoothed weight:** {current_weight} kg\n"
            f"• **Rate:** {ml_res['weekly_kg_change']:+.2f} kg/week "
            f"(target {target_weekly_change:+.2f})\n"
            f"• **Inferred balance:** {ml_res['kcal_per_day']:+.0f} kcal/day "
            f"(±{ml_res['ci_95_kcal_per_day']:.0f})\n\n"
        )
        msg += ("You are right on pace — keep your routine consistent." if on_track
                else "You are off pace by a little. A modest daily adjustment (~100–200 kcal) "
                     "and a consistent weigh-in routine usually closes the gap.")
        return msg


# Global Agent instance
agent_coach = SimplyFitAgent()
