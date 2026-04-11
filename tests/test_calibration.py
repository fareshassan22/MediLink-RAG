import numpy as np
import os

from app.calibration.calibrator import (
    train_calibrator,
    _expected_calibration_error,
    predict_confidence,
)


def test_expected_calibration_error():
    probs = np.array([0.1, 0.4, 0.8, 0.9])
    labels = np.array([0, 0, 1, 1])
    ece = _expected_calibration_error(probs, labels, n_bins=2)
    assert ece >= 0


def test_train_and_predict():
    np.random.seed(42)
    # 6 features: grounding, retrieval, rerank, ctx_len, ans_len, top_sim
    X = np.random.rand(100, 6)
    y = (X[:, 0] + X[:, 1] > 1).astype(int)

    os.makedirs("models", exist_ok=True)

    res = train_calibrator(X, y)
    assert res.ece >= 0
    assert res.brier >= 0
    assert 0.0 <= res.accuracy <= 1.0
    assert res.weights.shape == (6,)

    # Test predict_confidence uses the saved model
    score = predict_confidence(
        grounding_score=0.9,
        retrieval_score=0.8,
        rerank_score=0.7,
        context_length=500,
        answer_length=100,
        top_similarity=0.85,
    )
    assert 0.0 <= score <= 1.0
