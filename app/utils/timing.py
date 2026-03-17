import time
import logging
from typing import Dict, Optional
from contextlib import contextmanager

logger = logging.getLogger(__name__)

_latency_stats: Dict[str, list] = {}


@contextmanager
def track_latency(stage: str):
    """Context manager to track latency of each stage."""
    start = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start
        if stage not in _latency_stats:
            _latency_stats[stage] = []
        _latency_stats[stage].append(elapsed)
        logger.debug(f"[LATENCY] {stage}: {elapsed:.3f}s")


def get_latency_stats() -> Dict[str, Dict[str, float]]:
    """Get aggregated latency statistics."""
    stats = {}
    for stage, times in _latency_stats.items():
        if times:
            stats[stage] = {
                "count": len(times),
                "mean": sum(times) / len(times),
                "min": min(times),
                "max": max(times),
                "total": sum(times),
            }
    return stats


def reset_latency_stats():
    """Reset latency statistics."""
    global _latency_stats
    _latency_stats = {}


def get_stage_latency(stage: str) -> Optional[float]:
    """Get mean latency for a specific stage."""
    if stage in _latency_stats and _latency_stats[stage]:
        return sum(_latency_stats[stage]) / len(_latency_stats[stage])
    return None
