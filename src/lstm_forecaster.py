"""
LSTM forecaster for Simply-Fit.

Architecture: two stacked LSTM layers with dropout, trained on sliding
windows (14 days input -> 7 days output) of per-user weight series.

Training / evaluation design (leakage-free):
  * Per user, the MinMaxScaler is fit on the FIRST 72 days only, so no
    future information enters training inputs.
  * Train windows: prediction targets lie entirely within days 0-72.
  * Test windows: prediction targets lie entirely after day 72
    (walk-forward split). Windows whose targets straddle day 72 are dropped,
    so no target value is ever seen in both train and test.
  * Two trivial baselines — persistence (repeat last observed weight) and
    linear extrapolation (OLS on the 14 input days) — are evaluated on the
    exact same test windows, so the LSTM's value-add is measurable.

Run `python -m src.lstm_forecaster` to retrain and rewrite
models/simply_fit_lstm.keras and results/model_evaluation.json.
"""

import json
import os
from datetime import datetime, timezone

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

# Safe TensorFlow import block (app must run even without TensorFlow,
# falling back to linear extrapolation).
HAS_TENSORFLOW = False
try:
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.models import Sequential, load_model

    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False

SEQUENCE_LENGTH = 14
FORECAST_LENGTH = 7
TRAIN_DAYS = 72          # first 72 days per user form the training window
MODEL_PATH = os.path.join("models", "simply_fit_lstm.keras")
RESULTS_PATH = os.path.join("results", "model_evaluation.json")


def create_sequences(weight_log, seq_len=SEQUENCE_LENGTH, forecast_len=FORECAST_LENGTH):
    """
    Converts a weight time series into sliding window sequences
    for LSTM training. Input: seq_len days, Output: forecast_len days.
    """
    X, y = [], []
    for i in range(len(weight_log) - seq_len - forecast_len + 1):
        X.append(weight_log[i : i + seq_len])
        y.append(weight_log[i + seq_len : i + seq_len + forecast_len])
    return np.array(X), np.array(y)


def build_model():
    """
    Builds the LSTM architecture.
    Two stacked LSTM layers with dropout regularisation.
    """
    if not HAS_TENSORFLOW:
        return None

    model = Sequential([
        LSTM(64, input_shape=(SEQUENCE_LENGTH, 1), return_sequences=True),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(FORECAST_LENGTH)
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def _walk_forward_split(weights, seq_len=SEQUENCE_LENGTH, forecast_len=FORECAST_LENGTH,
                        train_days=TRAIN_DAYS):
    """
    Splits one user's series into train windows and test windows with no
    overlapping targets (see module docstring). Returns scaled train
    sequences plus raw-kg test cases.

    Returns (X_train_user, y_train_user, test_cases) where each test case is
    a dict with raw input/target values and the user's train-fitted scaler.
    """
    scaler = MinMaxScaler().fit(np.asarray(weights[:train_days], dtype=float).reshape(-1, 1))
    scaled = scaler.transform(np.asarray(weights, dtype=float).reshape(-1, 1)).flatten()

    X_train, y_train, test_cases = [], [], []
    for i in range(len(weights) - seq_len - forecast_len + 1):
        target_start = i + seq_len
        target_end   = target_start + forecast_len
        if target_end <= train_days:                      # target inside training period
            X_train.append(scaled[i : i + seq_len])
            y_train.append(scaled[target_start : target_end])
        elif target_start >= train_days:                  # target fully after training period
            test_cases.append({
                "input_raw":    np.asarray(weights[i : i + seq_len], dtype=float),
                "target_raw":   np.asarray(weights[target_start : target_end], dtype=float),
                "input_scaled": scaled[i : i + seq_len],
                "scaler":       scaler,
            })
        # windows straddling the boundary are dropped by design
    return X_train, y_train, test_cases


def _linear_baseline(input_raw, forecast_len=FORECAST_LENGTH):
    """OLS fit on the input window, extrapolated forward."""
    days = np.arange(len(input_raw)).reshape(-1, 1)
    m = LinearRegression().fit(days, input_raw)
    future = np.arange(len(input_raw), len(input_raw) + forecast_len).reshape(-1, 1)
    return m.predict(future)


def train_model(df, epochs=50, batch_size=64, seed=42,
                model_path=MODEL_PATH, results_path=RESULTS_PATH):
    """
    Trains the LSTM on the synthetic dataset (one row per user-day) with the
    leakage-free walk-forward split, evaluates it against persistence and
    linear baselines on held-out windows, saves the model (native Keras
    format) and writes a metrics report to results/model_evaluation.json.
    """
    if not HAS_TENSORFLOW:
        print("TensorFlow not installed. Skipping training.")
        return None

    tf.keras.utils.set_random_seed(seed)

    X_all, y_all, test_cases = [], [], []
    for uid in df["user_id"].unique():
        weights = df[df["user_id"] == uid]["weight"].values
        if len(weights) < TRAIN_DAYS:
            continue
        X_u, y_u, cases_u = _walk_forward_split(weights)
        X_all.extend(X_u)
        y_all.extend(y_u)
        test_cases.extend(cases_u)

    X = np.array(X_all).reshape(-1, SEQUENCE_LENGTH, 1)
    y = np.array(y_all)

    model = build_model()
    early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    history = model.fit(
        X, y,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=1
    )

    # ── Evaluate on held-out (post-day-72) windows ────────────────
    errs = {"lstm": [], "persistence": [], "linear": []}
    errs_by_day = {k: [[] for _ in range(FORECAST_LENGTH)] for k in errs}
    for case in test_cases:
        target = case["target_raw"]
        preds = {
            "lstm": case["scaler"].inverse_transform(
                model.predict(case["input_scaled"].reshape(1, SEQUENCE_LENGTH, 1),
                              verbose=0)[0].reshape(-1, 1)
            ).flatten(),
            "persistence": np.full(FORECAST_LENGTH, case["input_raw"][-1]),
            "linear": _linear_baseline(case["input_raw"]),
        }
        for name, pred in preds.items():
            abs_err = np.abs(pred - target)
            errs[name].append(abs_err)
            for d in range(FORECAST_LENGTH):
                errs_by_day[name][d].append(abs_err[d])

    mae = {name: float(np.mean(np.concatenate(e))) for name, e in errs.items()}
    mae_by_day = {
        name: [float(np.mean(day)) for day in errs_by_day[name]] for name in errs
    }

    if model_path:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        model.save(model_path)
    if results_path:
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, "w") as f:
            json.dump({
                "trained_at":  datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "n_train_users": int(df["user_id"].nunique()),
                "n_train_windows": int(X.shape[0]),
                "n_test_windows": len(test_cases),
                "split": ("per-user walk-forward: targets <= day 72 train, "
                          "targets > day 72 test; scaler fit on days 0-71 only"),
                "mae_kg": mae,
                "mae_kg_by_horizon_day": mae_by_day,
                "epochs_run": len(history.history["loss"]),
            }, f, indent=2)

    print(f"\nEvaluation on {len(test_cases)} held-out windows "
          f"({df['user_id'].nunique()} users):")
    for name in ["lstm", "persistence", "linear"]:
        print(f"  MAE ({name:12s}): {mae[name]:.3f} kg")
    return model


def forecast(weight_log, model_path=MODEL_PATH):
    """
    Loads the saved LSTM model and forecasts the next
    FORECAST_LENGTH days from the most recent weight readings.
    Returns forecast in original kg scale.
    Falls back to linear trend extrapolation if TensorFlow or the
    model file is unavailable.
    """
    weight_log = list(weight_log)
    if len(weight_log) < SEQUENCE_LENGTH:
        return None

    if not HAS_TENSORFLOW:
        return _linear_extrapolate(weight_log)

    try:
        model = load_model(model_path, compile=False)
    except Exception:
        return _linear_extrapolate(weight_log)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(
        np.array(weight_log).reshape(-1, 1)
    ).flatten()

    sequence = scaled[-SEQUENCE_LENGTH:].reshape(1, SEQUENCE_LENGTH, 1)
    predicted = model.predict(sequence, verbose=0)[0]

    return scaler.inverse_transform(
        predicted.reshape(-1, 1)
    ).flatten()


def _linear_extrapolate(weight_log):
    """Fallback forecast: OLS on the last 7 readings, extended 7 days."""
    recent = weight_log[-7:]
    daily_rate = (recent[-1] - recent[0]) / 7.0
    return np.array([round(recent[-1] + daily_rate * i, 1) for i in range(1, 8)])


if __name__ == "__main__":
    from src.data_generator import generate_dataset

    print("Generating synthetic cohort (500 users x 90 days, seed=42)...")
    df = generate_dataset(n_users=500, seed=42)
    print("Training LSTM with walk-forward split...")
    train_model(df)
