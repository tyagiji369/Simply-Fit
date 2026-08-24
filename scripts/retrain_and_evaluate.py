"""
Reproducible LSTM training + honest evaluation.

Retrains the forecaster with the causal per-window scaling and
user-grouped splits (see src/lstm_forecaster.py), saves the model, and
compares it against the deterministic linear-extrapolation baseline on
held-out users. Metrics are written to data/synthetic/lstm_evaluation.json

Run from the repo root:
    python scripts/retrain_and_evaluate.py --epochs 30
"""
import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.lstm_forecaster import HAS_TENSORFLOW, train_model, evaluate_model
DATASET = ROOT / "data/synthetic/users_weight_data.csv"
MODEL_PATH = ROOT / "models/simply_fit_lstm.h5"
BACKUP = ROOT / "models/simply_fit_lstm.bak.h5"
METRICS = ROOT / "data/synthetic/lstm_evaluation.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    if not HAS_TENSORFLOW:
        raise SystemExit("TensorFlow is required for training.")

    df = pd.read_csv(DATASET)
    print(f"Dataset: {df.shape}")

    # Keep a backup of the current production model in case training fails.
    if MODEL_PATH.exists():
        shutil.copy(MODEL_PATH, BACKUP)

    model, history = train_model(
        df, epochs=args.epochs, batch_size=args.batch_size
    )
    if model is None:
        raise SystemExit("Training failed.")

    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    metrics = evaluate_model(df, model_path=str(MODEL_PATH), output_path=str(METRICS))
    print("Evaluation (last-7-days forecast per user):")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print(f"Metrics saved to {METRICS}")


if __name__ == "__main__":
    main()
