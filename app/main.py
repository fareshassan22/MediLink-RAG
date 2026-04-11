from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict
import os
import time

from app.core.config import cfg
from app.utils.seed import set_seed, DEFAULT_SEED

set_seed(DEFAULT_SEED)

from app.api.middleware import RateLimitMiddleware

# Indexing (startup only)
from app.indexing.vector_store import VectorStore
from app.indexing.bm25_index import BM25Index

# Core
from app.core.state import get_state, set_vector_store, set_bm25, set_ready

# Utils
from app.utils.logger import log_request

# Service layer
from app.services.rag_pipeline import rag_pipeline

import logging

logger = logging.getLogger(__name__)


# ==========================
# App Setup
# ==========================

app = FastAPI(title="MediLink API")

# CORS — MUST set MEDILINK_CORS_ORIGINS in production (no default wildcard)
_cors_env = os.getenv("MEDILINK_CORS_ORIGINS", "")
if not _cors_env:
    import logging as _log
    _log.getLogger(__name__).warning(
        "MEDILINK_CORS_ORIGINS not set — defaulting to localhost only. "
        "Set to comma-separated origins for production."
    )
    _allowed_origins = ["http://localhost:3000", "http://localhost:8000"]
else:
    _allowed_origins = [o.strip() for o in _cors_env.split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RateLimitMiddleware, requests_per_minute=60)

from app.api.routes import router as api_router

app.include_router(api_router, prefix="/api")

# Serve frontend static files (mount at /static)

frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.exists(frontend_path):
    app.mount("/static", StaticFiles(directory=frontend_path), name="frontend")

    @app.get("/")
    async def serve_frontend():
        return FileResponse(os.path.join(frontend_path, "index.html"))


# ==========================
# Startup: Load Indexes
# ==========================


@app.on_event("startup")
async def load_models():
    from app.indexing.index_manager import index_manager

    needs_rebuild, reason = index_manager.needs_rebuild(min_documents=100)
    if needs_rebuild:
        logger.warning(f" Index health check: {reason}")
        logger.info("Please rebuild indexes using the indexing pipeline")

    vs = VectorStore(dim=1024)
    try:
        vs.load("data/processed")
        if len(vs.documents) == 0:
            raise RuntimeError("Vector store is empty - please rebuild index")
    except Exception as e:
        logger.error(f" Vector store load failed: {e}")
        raise RuntimeError(f"Failed to load vector store: {e}")

    bm25 = None
    try:
        bm25 = BM25Index.load("data/processed")
    except Exception as e:
        logger.warning(f" BM25 index load failed: {e}")

    set_vector_store(vs)
    set_bm25(bm25)
    set_ready(True)
    logger.info(f" Indexes loaded - {len(vs.documents)} documents")


# ==========================
# Request / Response Models
# ==========================


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=500)
    role: str = Field(default="patient")
    specialty: Optional[str] = None
    mode: str = Field(
        default="hybrid", description="Retrieval mode: hybrid, dense, or bm25"
    )

    @validator("query")
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError("Query must not be empty")
        return v.strip()

    @validator("mode")
    def validate_mode(cls, v):
        allowed = {"hybrid", "dense", "bm25"}
        if v not in allowed:
            raise ValueError(f"mode must be one of {allowed}")
        return v


class ResponseModel(BaseModel):
    answer: str
    confidence: float
    sources: List[str]
    grounding_score: float
    latency_seconds: float
    status: str
    stage_latencies: Optional[Dict[str, float]] = None


# ==========================
# Main RAG Endpoint
# ==========================


@app.post("/ask", response_model=ResponseModel)
def ask(request: QueryRequest):
    start_time = time.time()

    state = get_state()
    state.ensure_ready()

    try:
        result = rag_pipeline.run(
            query=request.query,
            vector_store=state.vector_store,
            bm25=state.bm25,
            role=request.role,
            specialty=request.specialty,
            mode=request.mode,
        )

        latency = round(time.time() - start_time, 3)

        if result.status == "success":
            log_request({
                "query": request.query,
                "grounding_score": result.grounding_score,
                "confidence": result.confidence,
                "latency": latency,
                "status": "success",
            })

        return ResponseModel(
            answer=result.answer,
            confidence=result.confidence,
            sources=result.sources,
            grounding_score=result.grounding_score,
            latency_seconds=latency,
            status=result.status,
            stage_latencies=result.stage_latencies,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Unexpected error: %s", type(e).__name__, exc_info=True)
        raise HTTPException(status_code=500, detail="internal_server_error")
