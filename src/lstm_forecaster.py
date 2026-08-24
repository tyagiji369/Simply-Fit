"""
LSTM weight-trajectory forecaster.

Fixes vs the original implementation:
  * **Causal, consistent scaling.** Each window is normalised by its own
    min/max at both train time and inference time. The original fitted a
    MinMaxScaler on the whole 90-day series during training but re-fitted
    it on whatever history existed at inference — train/inference scale
    mismatch (and future info leaking into the scaler).
  * **Group-aware split.** Sequences are split by *user*, not by row, so
    no user's tail appears in both train and validation.
  * **Honest evaluation.** ``evaluate_model`` compares the LSTM to the
    deterministic linear-extrapolation baseline (the fallback the app
    uses) on held-out users and saves metrics to JSON. The model is used
    in production only as long as it actually beats that baseline.
"""
import json
import os
import numpy as np

SEQUENCE_LENGTH = 14
FORECAST_LENGTH = 7

# Safe TensorFlow import block
HAS_TENSORFLOW = False
try:
    from tensorflow.keras.models import load_model, Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False


# ───────────────────────── scaling helpers ─────────────────────────
def _fit_window_scaler(window):
    """Min/max of a single window. Causal: only uses that window."""
    lo, hi = float(np.min(window)), float(np.max(window))
    span = hi - lo
    if span < 1e-6:
        lo, span = lo - 0.5, 1.0
    return lo, span


def _scale_window(window, lo, span):
    return (np.asarray(window, dtype=float) - lo) / span


def _unscale(values, lo, span):
    return np.asarray(values, dtype=float) * span + lo


def create_sequences(weight_log, seq_len=SEQUENCE_LENGTH, forecast_len=FORECAST_LENGTH):
    """
    Sliding windows over a *scaled* series. Each window is normalised by
    its own min/max (the caller provides the scaled log; the mapping is
    applied per window so a single global scaler is never needed).
    Returns X (n, seq_len) and y (n, forecast_len) already scaled per
    window — i.e. every row of X/y is consistent with the same (lo, span).
    """
    X, y, metas = [], [], []
    for i in range(len(weight_log) - seq_len - forecast_len + 1):
        window = np.asarray(weight_log[i: i + seq_len + forecast_len], dtype=float)
        lo, span = _fit_window_scaler(window[:seq_len])
        X.append(_scale_window(window[:seq_len], lo, span))
        y.append(_scale_window(window[seq_len:], lo, span))
        metas.append((lo, span))
    return np.array(X), np.array(y), metas


def build_model():
    """
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
        Dense(FORECAST_LENGTH),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def train_model(df, epochs=30, batch_size=128, val_fraction=0.15, seed=42):
    """
    Trains the LSTM on the synthetic dataset with a user-grouped split.

    Returns (model, history) or (None, None) if TensorFlow is missing.
    """
    if not HAS_TENSORFLOW:
        print("TensorFlow not installed. Skipping training.")
        return None, None

    rng = np.random.RandomState(seed)
    user_ids = np.array(sorted(df["user_id"].unique()))
    rng.shuffle(user_ids)
    n_val = max(1, int(len(user_ids) * val_fraction))
    val_users = set(user_ids[:n_val])
    train_users = user_ids[n_val:]

    X_all, y_all, u_all = [], [], []
    for uid in df["user_id"].unique():
        weight_log = df[df["user_id"] == uid]["weight"].values
        X_user, y_user, metas = create_sequences(weight_log)
        X_all.append(X_user)
        y_all.append(y_user)
        u_all.append(np.full(len(X_user), uid))

    X_all = np.concatenate(X_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)
    u_all = np.concatenate(u_all, axis=0)

    val_mask = np.isin(u_all, list(val_users))
    X_train, y_train = X_all[~val_mask], y_all[~val_mask]
    X_val, y_val = X_all[val_mask], y_all[val_mask]
    # 20% of train users are held out for a final test split.
    test_users = set(train_users[::5])
    test_mask = np.isin(u_all, list(test_users))
    test_mask &= ~val_mask
    X_test, y_test = X_all[test_mask], y_all[test_mask]
    X_train = X_all[~(val_mask | test_mask)]
    y_train = y_all[~(val_mask | test_mask)]

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    print(f"Sequences: train={len(X_train)} val={len(X_val)} test={len(X_test)}")

    model = build_model()
    early_stop = EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=[early_stop],
        verbose=1,
    )
    return model, {"val_loss": history.history.get("val_loss", []),
                   "loss": history.history.get("loss", [])}


# ───────────────────────── baseline & evaluation ─────────────────────────
def linear_baseline(weight_log, horizon=FORECAST_LENGTH):
    """
    Deterministic linear extrapolation from the last 7 days — the fallback
    the app uses when TensorFlow/LSTM is unavailable.
    """
    recent = np.asarray(weight_log[-7:], dtype=float)
    daily_rate = (recent[-1] - recent[0]) / 7.0
    return np.array([round(recent[-1] + daily_rate * i, 4) for i in range(1, horizon + 1)])


def evaluate_model(df, model_path="models/simply_fit_lstm.h5", output_path=None):
    """
    Evaluates the saved LSTM against the linear baseline on held-out users.

    For each user the model sees only the first (N - 7) days and must
    forecast the final 7 days — the same task as in production. Returns
    and optionally saves a metrics dict:
      mae_lstm, mae_linear, n_users, improvement_pct.
    """
    if not HAS_TENSORFLOW:
        raise RuntimeError("TensorFlow required for LSTM evaluation.")
    model = load_model(model_path, compile=False)

    lstm_errors, linear_errors = [], []
    for uid in df["user_id"].unique():
        w = df[df["user_id"] == uid]["weight"].values
        if len(w) < SEQUENCE_LENGTH + FORECAST_LENGTH:
            continue
        history, actual = w[: -FORECAST_LENGTH], w[-FORECAST_LENGTH:]

        lo, span = _fit_window_scaler(history[-SEQUENCE_LENGTH:])
        scaled = _scale_window(history[-SEQUENCE_LENGTH:], lo, span).reshape(1, SEQUENCE_LENGTH, 1)
        pred_scaled = model.predict(scaled, verbose=0)[0]
        pred = _unscale(pred_scaled, lo, span)

        lstm_errors.append(np.abs(np.asarray(pred, dtype=float) - actual).mean())
        linear_errors.append(np.abs(linear_baseline(history) - actual).mean())

    metrics = {
        "n_users": len(lstm_errors),
        "mae_lstm_kg": round(float(np.mean(lstm_errors)), 4),
        "mae_linear_kg": round(float(np.mean(linear_errors)), 4),
        "improvement_pct_vs_linear": round(
            float((np.mean(linear_errors) - np.mean(lstm_errors)) / max(np.mean(linear_errors), 1e-9) * 100), 2
        ) if np.mean(linear_errors) > 0 else 0.0,
        "note": "Last-7-days forecast from first (N-7) days, per user. Lower MAE is better.",
    }
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=2)
    return metrics


# ───────────────────────── production forecast ─────────────────────────
def forecast(weight_log, model_path="models/simply_fit_lstm.h5"):
    """
    Forecasts the next FORECAST_LENGTH days from the most recent readings.

    Returns a dict: {"values": [...], "method": "lstm"|"linear_fallback"}.
    The linear baseline is returned whenever TensorFlow is missing, the
    model cannot be loaded, or the log is too short (also returning it as
    the fallback means the UI can always show something).
    """
    if len(weight_log) < SEQUENCE_LENGTH:
        return None

    fallback = lambda: {
        "values": linear_baseline(weight_log).round(1).tolist(),
        "method": "linear_fallback",
    }

    if not HAS_TENSORFLOW:
        return fallback()

    try:
        model = load_model(model_path, compile=False)
    except Exception:
        return fallback()

    history = np.asarray(weight_log[-SEQUENCE_LENGTH:], dtype=float)
    lo, span = _fit_window_scaler(history)
    scaled = _scale_window(history, lo, span).reshape(1, SEQUENCE_LENGTH, 1)
    pred = _unscale(model.predict(scaled, verbose=0)[0], lo, span)

    return {
        "values": [round(float(x), 1) for x in pred],
        "method": "lstm",
        "scaler": {"lo": round(float(lo), 3), "span": round(float(span), 3)},
    }
