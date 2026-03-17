"""
Confidence Calibration Module for MediLink RAG System.

Uses sklearn Logistic Regression to predict answer correctness based on:
- grounding_score
- retrieval_score
- rerank_score
- context_length
- answer_length
- top_similarity
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from app.core.config import cfg


FEATURE_NAMES = [
    "grounding_score",
    "retrieval_score",
    "rerank_score",
    "context_length",
    "answer_length",
    "top_similarity",
]


@dataclass
class CalibrationResult:
    weights: np.ndarray
    intercept: float
    ece: float
    brier: float
    accuracy: float


def _expected_calibration_error(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 10
) -> float:
    """Compute Expected Calibration Error."""
    if len(probs) == 0:
        return 0.0

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    binids = np.digitize(probs, bins, right=True)
    ece = 0.0

    for i in range(1, len(bins)):
        mask = binids == i
        if mask.sum() == 0:
            continue
        avg_conf = probs[mask].mean()
        avg_acc = labels[mask].mean()
        ece += (mask.sum() / len(probs)) * abs(avg_conf - avg_acc)

    return float(ece)


def collect_features(
    grounding_score: float,
    retrieval_score: float,
    rerank_score: float,
    context_length: int,
    answer_length: int,
    top_similarity: float,
) -> np.ndarray:
    """Collect features for a single prediction."""
    features = np.array(
        [
            grounding_score,
            retrieval_score,
            rerank_score,
            float(context_length),
            float(answer_length),
            top_similarity,
        ],
        dtype=np.float64,
    )

    return features


def train_calibrator(
    features: np.ndarray, labels: np.ndarray, model_path: Path | None = None
) -> CalibrationResult:
    """Train Logistic Regression calibrator using sklearn.

    Args:
        features: 2D array of shape (n_samples, n_features)
        labels: 1D array of binary labels (0 or 1)
        model_path: Optional custom path for saving model

    Returns:
        CalibrationResult with weights, intercept, and metrics
    """
    if features.shape[0] < cfg.CALIBRATION_MIN_SAMPLES:
        raise ValueError(
            f"Not enough samples to train calibrator. Need at least {cfg.CALIBRATION_MIN_SAMPLES}, got {features.shape[0]}"
        )

    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Train Logistic Regression
    model = LogisticRegression(
        random_state=cfg.RANDOM_SEED, max_iter=1000, solver="lbfgs"
    )
    model.fit(features_scaled, labels)

    # Get predictions
    probs = model.predict_proba(features_scaled)[:, 1]
    predictions = model.predict(features_scaled)

    # Compute metrics
    ece = _expected_calibration_error(probs, labels, n_bins=5)
    brier = float(np.mean((probs - labels) ** 2))
    accuracy = float(np.mean(predictions == labels))

    # Save model and scaler
    model_path = model_path or cfg.MODEL_DIR / "confidence_calibrator.pkl"
    cfg.MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model_data = {"model": model, "scaler": scaler, "feature_names": FEATURE_NAMES}

    with open(model_path, "wb") as f:
        pickle.dump(model_data, f)

    print(f"Calibration model saved to: {model_path}")
    print(f"  - ECE: {ece:.4f}")
    print(f"  - Brier Score: {brier:.4f}")
    print(f"  - Accuracy: {accuracy:.4f}")

    return CalibrationResult(
        weights=model.coef_.flatten(),
        intercept=model.intercept_[0],
        ece=ece,
        brier=brier,
        accuracy=accuracy,
    )


def load_calibrator(model_path: Path | None = None) -> Optional[dict]:
    """Load trained calibration model.

    Returns:
        Dict with 'model', 'scaler', 'feature_names' or None if not found
    """
    model_path = model_path or cfg.MODEL_DIR / "confidence_calibrator.pkl"

    if model_path.exists():
        with open(model_path, "rb") as f:
            return pickle.load(f)
    return None


def predict_confidence(
    grounding_score: float,
    retrieval_score: float,
    rerank_score: float = 0.0,
    context_length: int = 0,
    answer_length: int = 0,
    top_similarity: float = 0.0,
    model_data: dict | None = None,
) -> float:
    """Predict confidence score for an answer.

    Args:
        grounding_score: Grounding verification score
        retrieval_score: Average retrieval score
        rerank_score: Reranker score (optional)
        context_length: Length of retrieved context
        answer_length: Length of generated answer
        top_similarity: Top similarity score from retrieval
        model_data: Pre-loaded model dict (optional)

    Returns:
        Confidence score between 0 and 1
    """
    if model_data is None:
        model_data = load_calibrator()

    if model_data is None:
        # Fallback to heuristic if no model
        return _heuristic_confidence(grounding_score, retrieval_score, rerank_score)

    # Collect features
    features = collect_features(
        grounding_score,
        retrieval_score,
        rerank_score,
        context_length,
        answer_length,
        top_similarity,
    ).reshape(1, -1)

    # Scale and predict
    scaler = model_data["scaler"]
    model = model_data["model"]

    features_scaled = scaler.transform(features)
    confidence = model.predict_proba(features_scaled)[0, 1]

    return float(confidence)


def _heuristic_confidence(
    grounding_score: float, retrieval_score: float, rerank_score: float
) -> float:
    """Fallback heuristic confidence if no model is available."""
    confidence = (
        0.4 * grounding_score + 0.3 * retrieval_score + 0.2 * rerank_score + 0.1
    )
    return float(np.clip(confidence, 0.0, 1.0))


def generate_synthetic_training_data(
    n_samples: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic training data for calibration.

    This creates realistic feature combinations with known correctness labels
    for initial model training.

    Returns:
        Tuple of (features, labels)
    """
    np.random.seed(cfg.RANDOM_SEED)

    features = []
    labels = []

    for _ in range(n_samples):
        # Generate realistic values
        grounding = np.random.beta(2, 2)  # Most values around 0.5
        retrieval = np.random.beta(2, 2)
        rerank = np.random.beta(2, 2)
        ctx_len = np.random.randint(100, 2000)
        ans_len = np.random.randint(50, 500)
        top_sim = np.random.beta(2, 2)

        # Label based on rule: high grounding + retrieval = correct
        is_correct = (
            1 if (grounding > 0.5 and retrieval > 0.3) or grounding > 0.7 else 0
        )

        features.append(
            [grounding, retrieval, rerank, float(ctx_len), float(ans_len), top_sim]
        )
        labels.append(is_correct)

    return np.array(features), np.array(labels)


def train_with_synthetic_data(model_path: Path | None = None) -> CalibrationResult:
    """Train calibrator using synthetic data.

    Useful for initial model setup before real data is available.
    """
    features, labels = generate_synthetic_training_data(200)
    return train_calibrator(features, labels, model_path)
