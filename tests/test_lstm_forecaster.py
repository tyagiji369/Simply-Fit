import numpy as np
import pytest

from src.lstm_forecaster import (
    SEQUENCE_LENGTH,
    FORECAST_LENGTH,
    create_sequences,
    _fit_window_scaler,
    _scale_window,
    _unscale,
    linear_baseline,
    HAS_TENSORFLOW,
)


def test_scaler_roundtrip():
    window = [80.0, 80.2, 79.9, 80.1]
    lo, span = _fit_window_scaler(window)
    scaled = _scale_window(window, lo, span)
    back = _unscale(scaled, lo, span)
    assert np.allclose(back, window)


def test_scaler_handles_flat_window():
    lo, span = _fit_window_scaler([80.0] * 5)
    assert span > 0


def test_create_sequences_shape():
    log = list(80 - 0.05 * np.arange(60))
    X, y, metas = create_sequences(log, seq_len=14, forecast_len=7)
    assert X.shape[1] == 14 and y.shape[1] == 7
    assert len(X) == 60 - 14 - 7 + 1
    # each row is within [0, 1] by construction of per-window scaling
    assert X.min() >= -1e-6 and X.max() <= 1 + 1e-6


def test_linear_baseline():
    log = list(80 - 0.1 * np.arange(30))
    pred = linear_baseline(log)
    assert len(pred) == FORECAST_LENGTH
    # decreasing series → decreasing forecast
    assert np.all(np.diff(pred) < 0)


@pytest.mark.skipif(not HAS_TENSORFLOW, reason="TensorFlow not installed")
def test_forecast_returns_sane_values():
    from src.lstm_forecaster import forecast
    log = list(80 - 0.05 * np.arange(30))
    out = forecast(log)
    assert out is not None
    assert len(out["values"]) == 7
    assert out["method"] in ("lstm", "linear_fallback")
