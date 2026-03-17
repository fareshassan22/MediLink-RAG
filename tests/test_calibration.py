import numpy as np
import os

from app.calibration.calibrator import (
    train_calibrator,
    _expected_calibration_error,
    calibrate_probs,
)


def test_expected_calibration_error():
    probs = np.array([0.1, 0.4, 0.8, 0.9])
    labels = np.array([0, 0, 1, 1])
    ece = _expected_calibration_error(probs, labels, n_bins=2)
    assert ece >= 0


def test_train_and_calibrate():
    np.random.seed(42)
    X = np.random.rand(50, 2)
    y = (X[:, 0] + X[:, 1] > 1).astype(int)

    os.makedirs("models", exist_ok=True)

    res = train_calibrator(X, y)
    params = (res.weights, res.intercept)
    probs = calibrate_probs(params, X)
    assert probs.shape == (50,)
    assert res.ece >= 0
    assert res.brier >= 0
