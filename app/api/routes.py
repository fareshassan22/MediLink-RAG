from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Tuple, Optional

import time

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.utils.seed import set_seed, DEFAULT_SEED

set_seed(DEFAULT_SEED)

from app.core.config import cfg
from app.indexing.vector_store import VectorStore
from app.indexing.index_pipeline import load_bm25
from app.safety.grounding import grounding_verification
from app.safety.emergency_detector import detect_emergency
from app.safety.content_filter import contains_sensitive_content
from app.indexing.embedder import Embedder
from app.indexing.preprocessing import preprocess_query
from app.services.retrieval import compute_confidence
from app.generation.prompts import build_prompt
from app.generation.groq_client import generate_response
from app.retrieval.query_translator import translate_query

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


class QueryRequest(BaseModel):
    query: str
    mode: str = "hybrid"


def get_page_from_doc_id(doc_id: str) -> Any:
    """Extract page number from document ID."""
    try:
        return int(doc_id.split("_")[-1])
    except (ValueError, IndexError):
        return doc_id


def get_page_from_doc(vs, idx: int) -> Any:
    """Extract page number from document metadata."""
    try:
        doc = vs.get_doc(idx)
        if hasattr(doc, "metadata") and doc.metadata and "page" in doc.metadata:
            return doc.metadata["page"]
        return int(doc.doc_id.split("_")[-1])
    except (ValueError, IndexError, AttributeError):
        return idx


# @router.post("/ask")
# def ask_route(request: QueryRequest) -> Dict[str, Any]:
#     """Simplified ask endpoint - use /ask instead for full functionality"""
#     pass
