import os
import json
import numpy as np
from dotenv import load_dotenv
from src.ml_engine import run_pipeline as run_ml_pipeline
from src.lstm_forecaster import forecast as forecast_lstm
from src.rag_pipeline import rag_engine

load_dotenv()


class SimplyFitAgent:
    """
    Agentic AI Coach using ReAct (Reasoning + Acting) pattern with tool execution.
    Orchestrates ML Signal Processing, LSTM Neural Forecasting, and Clinical RAG.
    """

    def __init__(self):
        self.tools = {
            "infer_calorie_balance": self._tool_infer_calorie_balance,
            "forecast_trajectory": self._tool_forecast_trajectory,
            "retrieve_clinical_guidelines": self._tool_retrieve_clinical_guidelines,
            "check_plateau_status": self._tool_check_plateau_status
        }

    # ── Tool Definitions ──────────────────────────────────────
    def _tool_infer_calorie_balance(self, weight_log, target_weekly_change):
        """Runs EWMA filtering, Isolation Forest anomaly detection, and linear regression."""
        return run_ml_pipeline(weight_log, target_weekly_change=target_weekly_change)

    def _tool_forecast_trajectory(self, weight_log):
        """Runs LSTM neural network forecasting or trend extrapolation fallback."""
        try:
            pred = forecast_lstm(weight_log)
            if pred is not None:
                return [round(float(x), 1) for x in pred]
        except Exception:
            pass

        # Robust analytical forecast fallback if model weights absent
        recent = weight_log[-7:]
        daily_rate = (recent[-1] - recent[0]) / 7.0
        return [round(recent[-1] + daily_rate * i, 1) for i in range(1, 8)]

    def _tool_retrieve_clinical_guidelines(self, query, disease):
        """Retrieves top-k evidence-based clinical nutrition guidelines via vector search."""
        return rag_engine.search_guidelines(query, condition=disease, top_k=2)

    def _tool_check_plateau_status(self, weight_log):
        """Detects metabolic slowdown or weight loss plateau over 14-day window."""
        if len(weight_log) < 14:
            return {"is_plateau": False, "change_14d": round(weight_log[-1] - weight_log[0], 2)}
        recent_14 = weight_log[-14:]
        total_delta = recent_14[-1] - recent_14[0]
        is_plateau = abs(total_delta) < 0.2
        return {
            "is_plateau": is_plateau,
            "change_14d": round(total_delta, 2),
            "recommendation": "Introduce 2-day maintenance refeed" if is_plateau else "Maintain current deficit"
        }

    # ── Agentic Execution Loop ────────────────────────────────
    def run_agent(self, profile, weight_log, question, target_weekly_change=-0.5):
        """
        Executes the ReAct Agent Orchestration loop.
        Returns a dict containing:
        - trace: List of step-by-step reasoning & tool execution logs
        - response: Final grounded coaching response
        """
        trace = []
        disease = profile.get("disease", "none")
        if disease == "none" and profile.get("diseases"):
            disease = profile["diseases"][0] if profile["diseases"] else "none"

        # Step 1: Agent Reasoning & Tool Selection
        trace.append({
            "step": 1,
            "thought": f"Analyzing query: '{question}'. Selecting tools for signal extraction, trajectory forecasting, and medical RAG.",
            "selected_tools": ["infer_calorie_balance", "forecast_trajectory", "retrieve_clinical_guidelines", "check_plateau_status"]
        })

        # Step 2: Tool Execution
        ml_res = self._tool_infer_calorie_balance(weight_log, target_weekly_change)
        forecast_res = self._tool_forecast_trajectory(weight_log)
        rag_res = self._tool_retrieve_clinical_guidelines(question, disease)
        plateau_res = self._tool_check_plateau_status(weight_log)

        trace.append({
            "step": 2,
            "tool_call": "infer_calorie_balance",
            "observation": {
                "inferred_kcal_per_day": ml_res["kcal_per_day"],
                "weekly_kg_change": ml_res["weekly_kg_change"],
                "anomalies_filtered": ml_res["anomalies_detected"]
            }
        })

        trace.append({
            "step": 3,
            "tool_call": "forecast_trajectory",
            "observation": {"7_day_forecast": forecast_res}
        })

        trace.append({
            "step": 4,
            "tool_call": "retrieve_clinical_guidelines",
            "observation": [
                {"condition": g["condition"], "topic": g["topic"], "rule": g["recommendation"]}
                for g in rag_res
            ]
        })

        trace.append({
            "step": 5,
            "tool_call": "check_plateau_status",
            "observation": plateau_res
        })

        # Step 3: Synthesis & Response Generation
        recent_7 = weight_log[-7:]
        weekly_change = round(recent_7[-1] - recent_7[0], 2)
        on_track = abs(weekly_change - target_weekly_change) < 0.15

        # Check if Gemini API key exists
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if gemini_api_key:
            try:
                from google import genai
                client = genai.Client(api_key=gemini_api_key)
                prompt_context = f"""
                AGENT TOOL EXECUTION RESULTS:
                - Current Weight: {weight_log[-1]} kg (7-day change: {weekly_change:+} kg, Target: {target_weekly_change:+} kg/week)
                - Inferred Daily Calorie Balance: {ml_res['kcal_per_day']} kcal/day
                - Anomaly Detection: Filtered {ml_res['anomalies_detected']} transient spikes.
                - 7-Day LSTM Forecast: {forecast_res}
                - Plateau Check: {plateau_res['recommendation']}
                - Medical Condition: {disease}
                - Retrieved Clinical Guidelines: {json.dumps(rag_res)}

                User Question: {question}
                Respond as a supportive, concise medical nutrition AI assistant. Max 120 words.
                """
                response_text = client.models.generate_content(
                    model="gemini-1.5-flash",
                    contents=prompt_context
                ).text
            except Exception as e:
                response_text = self._fallback_synthesis(weekly_change, target_weekly_change, ml_res, plateau_res, rag_res, disease)
        else:
            response_text = self._fallback_synthesis(weekly_change, target_weekly_change, ml_res, plateau_res, rag_res, disease)

        return {
            "trace": trace,
            "response": response_text,
            "ml_output": ml_res,
            "forecast": forecast_res,
            "rag_matches": rag_res
        }

    def _fallback_synthesis(self, weekly_change, target_weekly_change, ml_res, plateau_res, rag_res, disease):
        """Generates deterministic, grounded coaching response from tool outputs."""
        on_track = abs(weekly_change - target_weekly_change) < 0.15

        if on_track:
            msg = (
                f"Your weight changed by {weekly_change:+.2f} kg this week against a target of {target_weekly_change:+.1f} kg. "
                f"Signal processing infers a daily caloric balance of {ml_res['kcal_per_day']:+.0f} kcal/day. "
                f"Your trajectory is steady and on track."
            )
        else:
            msg = (
                f"Your 7-day weight trend shows {weekly_change:+.2f} kg vs target of {target_weekly_change:+.1f} kg. "
                f"Inferred daily balance is {ml_res['kcal_per_day']:+.0f} kcal/day. "
            )
            if "food_adjustment_kcal" in ml_res:
                msg += f"Suggested adjustment: reduce intake by {abs(ml_res['food_adjustment_kcal']):.0f} kcal/day and increase activity by {abs(ml_res['activity_adjustment_kcal']):.0f} kcal/day."

        if plateau_res.get("is_plateau"):
            msg += f" Note: 14-day trend indicates a weight plateau ({plateau_res['change_14d']} kg change). {plateau_res['recommendation']}."

        if rag_res and disease != "none":
            top_rule = rag_res[0]
            msg += f" Clinical Guideline for {disease}: {top_rule['recommendation']}"

        return msg


# Global Agent instance
agent_coach = SimplyFitAgent()


def get_agent_response(profile, weight_log, question, target_weekly_change=-0.5):
    """Convenience wrapper for backward compatibility."""
    return agent_coach.run_agent(profile, weight_log, question, target_weekly_change)
