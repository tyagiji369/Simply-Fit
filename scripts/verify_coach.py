"""
Verifies the coach's REAL Gemini path with your own key.

Usage (from the repo root):

    export GEMINI_API_KEY="..."      # or put it in .env (already supported)
    python scripts/verify_coach.py "How long until I reach my goal?"

The script prints the coach's answer and whether it came from the live
Gemini API or the deterministic fallback (including the API error if the
live call failed). Nothing is logged or stored.

Key formats: Google currently issues two kinds of Gemini API key —
legacy Standard keys (``AIza...``) and the newer Auth keys (``AQ.Ab...``),
which Google AI Studio now issues by default. Both are accepted here and
by the official ``google-genai`` SDK.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from src.agent_coach import agent_coach


def main():
    question = " ".join(sys.argv[1:]) or "How long until I reach my goal?"
    key = os.getenv("GEMINI_API_KEY", "")
    if not key:
        print("No GEMINI_API_KEY found in environment or .env.")
        print("Get a free key at https://aistudio.google.com/apikey")
        return 1

    profile = {
        "age": 30, "gender": "male", "height_cm": 175.0, "weight": 80.0,
        "goal": "lose", "disease": None, "diseases": [],
        "target_weight": 75.0, "tdee": 2200.0,
    }
    log = list(80 - 0.05 * i for i in range(30))  # ~ -0.35 kg/week

    print("Asking the coach…\n")
    res = agent_coach.run_agent(
        profile, log, question,
        target_weekly_change=-0.35, gemini_api_key=key,
    )
    if res["used_llm"]:
        print("PATH: ✅ live Gemini API")
    else:
        print("PATH: ⚠️ deterministic fallback (the live call failed)")
        if res.get("llm_error"):
            print(f"ERROR: {res['llm_error']}")
    print("INTENT:", res["intent"]["intent"])
    print("\nANSWER:\n" + res["response"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
