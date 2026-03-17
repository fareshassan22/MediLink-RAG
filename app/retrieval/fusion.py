"""Score-level fusion utilities.

Used by hybrid_fusion.py for the main pipeline.
Kept for backward compatibility and direct score-array fusion.
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple, Dict

from app.core.config import cfg


def minmax_scale(arr: np.ndarray) -> np.ndarray:
    """Scale array to 0-1 range."""
    mn = np.nanmin(arr)
    mx = np.nanmax(arr)
    if mx - mn == 0:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


def fuse_scores(
    dense_scores: List[float],
    bm25_scores: List[float],
    weights: Tuple[float, float] | None = None,
) -> List[float]:
    """Fuse dense and BM25 scores into a single score per doc.

    Both score arrays are min-max normalized to [0, 1] before
    applying the configured weights (default: 0.8 dense, 0.2 BM25).
    """
    w_dense, w_bm25 = weights or (cfg.WEIGHT_DENSE, cfg.WEIGHT_BM25)
    d = np.array(dense_scores, dtype=float)
    b = np.array(bm25_scores, dtype=float)

    d_scaled = minmax_scale(d)
    b_scaled = minmax_scale(b)

    fused = w_dense * d_scaled + w_bm25 * b_scaled
    return fused.tolist()
