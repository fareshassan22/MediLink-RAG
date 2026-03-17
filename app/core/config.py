from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict


@dataclass(frozen=True)
class Config:
    """Centralized configuration for MediLink.

    Change weights, thresholds, and file paths here.
    """

    # Embedding model (BGE-M3 for multilingual retrieval including Arabic)
    EMBED_MODEL_NAME: str = "BAAI/bge-m3"

    # Reranker — bge-reranker-v2-m3 (multilingual cross-encoder)
    RERANKER_MODEL_NAME: str = "BAAI/bge-reranker-v2-m3"

    # Local model loading (avoid network calls)
    LOCAL_MODEL_ONLY: bool = True

    # Paths
    BASE_DIR: Path = Path.cwd()
    DATA_DIR: Path = BASE_DIR / "data"
    INDEX_DIR: Path = BASE_DIR / "data" / "processed"
    MODEL_DIR: Path = BASE_DIR / "models"
    RESULTS_DIR: Path = BASE_DIR / "results"

    # Fusion weights — dense is primary, BM25 is secondary
    WEIGHT_DENSE: float = 0.8
    WEIGHT_BM25: float = 0.2

    # Retrieval settings
    TOP_K_DENSE: int = 20
    TOP_K_BM25: int = 20
    TOP_K_FINAL: int = 10
    SIMILARITY_THRESHOLD: float = 0.25
    MIN_KEYWORD_OVERLAP: float = 0.1
    MAX_ANSWER_LENGTH: int = 400

    # Grounding thresholds
    GROUNDING_SENT_SIM_THRESHOLD: float = 0.65
    GROUNDING_SCORE_THRESHOLD: float = 0.6

    # Calibration
    CALIBRATION_MIN_SAMPLES: int = 40
    CALIBRATION_MODEL_PATH: Path = MODEL_DIR / "confidence_calibrator.pkl"

    # Random seed
    RANDOM_SEED: int = 42

    # Evaluation dataset (unified ground truth — 99 queries, Arabic + English)
    EVAL_SET_PATH: Path = DATA_DIR / "eval_ground_truth.json"

    # Scripts directories
    SCRIPTS_DIR: Path = BASE_DIR / "scripts"

    # Plots directory
    PLOTS_DIR: Path = RESULTS_DIR / "plots"


cfg = Config()
