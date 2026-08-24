import numpy as np
from sklearn.preprocessing import MinMaxScaler

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


def train_model(df, epochs=50, batch_size=64):
    """
    Trains the LSTM on a synthetic dataset DataFrame.
    Each user's weight series is scaled independently
    to 0-1 range before sequence creation.
    """
    if not HAS_TENSORFLOW:
        print("TensorFlow not installed. Skipping training.")
        return None

    X_all, y_all = [], []
    scaler = MinMaxScaler()

    for uid in df["user_id"].unique():
        weight_log = df[df["user_id"] == uid]["weight"].values
        scaled = scaler.fit_transform(
            weight_log.reshape(-1, 1)
        ).flatten()
        X_user, y_user = create_sequences(scaled)
        X_all.append(X_user)
        y_all.append(y_user)

    X_all = np.concatenate(X_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)
    X_all = X_all.reshape((X_all.shape[0], X_all.shape[1], 1))

    split = int(len(X_all) * 0.8)
    X_train = X_all[:split]
    y_train = y_all[:split]

    model = build_model()
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=1
    )
    return model


def forecast(weight_log, model_path="models/simply_fit_lstm.h5"):
    """
    Loads the saved LSTM model and forecasts the next
    FORECAST_LENGTH days from the most recent weight readings.
    Returns forecast in original kg scale.
    Fallback to linear trend extrapolation if TensorFlow absent.
    """
    if len(weight_log) < SEQUENCE_LENGTH:
        return None

    if not HAS_TENSORFLOW:
        # Fallback extrapolation if TensorFlow is not installed
        recent = weight_log[-7:]
        daily_rate = (recent[-1] - recent[0]) / 7.0
        return np.array([round(recent[-1] + daily_rate * i, 1) for i in range(1, 8)])

    try:
        model = load_model(model_path, compile=False)
    except Exception:
        recent = weight_log[-7:]
        daily_rate = (recent[-1] - recent[0]) / 7.0
        return np.array([round(recent[-1] + daily_rate * i, 1) for i in range(1, 8)])

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(
        np.array(weight_log).reshape(-1, 1)
    ).flatten()

    sequence = scaled[-SEQUENCE_LENGTH:].reshape(1, SEQUENCE_LENGTH, 1)
    predicted = model.predict(sequence, verbose=0)[0]

    return scaler.inverse_transform(
        predicted.reshape(-1, 1)
    ).flatten()
