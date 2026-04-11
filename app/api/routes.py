from __future__ import annotations

import logging
import os
from typing import Any, Optional

from fastapi import APIRouter, HTTPException

from app.core.config import cfg
from app.indexing.vector_store import VectorStore
from app.indexing.index_pipeline import load_bm25

logger = logging.getLogger(__name__)

router = APIRouter()

_cached_vector_store: Optional[VectorStore] = None
_cached_bm25: Optional[Any] = None


def get_cached_vector_store() -> VectorStore:
    """Get cached vector store or load it."""
    global _cached_vector_store
    if _cached_vector_store is None:
        vs = VectorStore(dim=1024)
        vs.load(str(cfg.INDEX_DIR))
        _cached_vector_store = vs
    return _cached_vector_store


def get_cached_bm25():
    """Get cached BM25 index or load it."""
    global _cached_bm25
    if _cached_bm25 is None:
        _cached_bm25 = load_bm25()
    return _cached_bm25


@router.get("/health")
async def health_check():
    """Health check endpoint with index status."""
    status = {"status": "healthy", "components": {}}

    try:
        vs = get_cached_vector_store()
        status["components"]["vector_store"] = {
            "status": "ok",
            "documents": len(vs.documents),
            "dimension": vs.dim,
        }
    except Exception as e:
        status["components"]["vector_store"] = {"status": "error", "error": str(e)}
        status["status"] = "degraded"

    try:
        bm25 = get_cached_bm25()
        if bm25 is not None:
            status["components"]["bm25"] = {"status": "ok"}
        else:
            status["components"]["bm25"] = {
                "status": "warning",
                "message": "Not loaded",
            }
    except Exception as e:
        status["components"]["bm25"] = {"status": "error", "error": str(e)}
        status["status"] = "degraded"

    status["environment"] = os.getenv("ENV", "development")

    return status


@router.get("/ready")
async def readiness_check():
    """Readiness check - returns 200 if service is ready to handle requests."""
    try:
        vs = get_cached_vector_store()
        if len(vs.documents) == 0:
            raise HTTPException(status_code=503, detail="Vector store is empty")
        return {"status": "ready"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service not ready: {e}")
