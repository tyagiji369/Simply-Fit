import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

SEQUENCE_LENGTH = 14
FORECAST_LENGTH = 7


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
    X_all, y_all = [], []
    scaler = MinMaxScaler()

    for uid in df["user_id"].unique():
        weight_log = df[df["user_id"] == uid]["weight"].values
        scaled     = scaler.fit_transform(
            weight_log.reshape(-1, 1)
        ).flatten()
        X_user, y_user = create_sequences(scaled)
        X_all.append(X_user)
        y_all.append(y_user)

    X_all = np.concatenate(X_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)
    X_all = X_all.reshape((X_all.shape[0], X_all.shape[1], 1))

    split   = int(len(X_all) * 0.8)
    X_train = X_all[:split]
    y_train = y_all[:split]

    model      = build_model()
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
    """
    if len(weight_log) < SEQUENCE_LENGTH:
        return None

    model  = load_model(model_path)
    scaler = MinMaxScaler()

    scaled = scaler.fit_transform(
        np.array(weight_log).reshape(-1, 1)
    ).flatten()

    sequence  = scaled[-SEQUENCE_LENGTH:].reshape(1, SEQUENCE_LENGTH, 1)
    predicted = model.predict(sequence, verbose=0)[0]

    return scaler.inverse_transform(
        predicted.reshape(-1, 1)
    ).flatten()