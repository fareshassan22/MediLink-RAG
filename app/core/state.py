"""Application state management for FastAPI dependency injection."""

from typing import Optional
from dataclasses import dataclass, field

from app.indexing.vector_store import VectorStore
from app.indexing.bm25_index import BM25Index


@dataclass
class AppState:
    """Application state container."""

    vector_store: Optional[VectorStore] = None
    bm25: Optional[BM25Index] = None
    is_ready: bool = False

    def ensure_ready(self):
        """Ensure the application is ready to serve requests."""
        if not self.is_ready:
            raise RuntimeError("Application not ready - indexes not loaded")


# Singleton application state
_state = AppState()


def get_state() -> AppState:
    """Get the application state singleton."""
    return _state


def set_vector_store(vs: VectorStore):
    """Set the vector store."""
    _state.vector_store = vs


def set_bm25(bm25: BM25Index):
    """Set the BM25 index."""
    _state.bm25 = bm25


def set_ready(ready: bool = True):
    """Mark the application as ready."""
    _state.is_ready = ready


def get_vector_store() -> VectorStore:
    """Get the vector store, ensuring it's loaded."""
    _state.ensure_ready()
    if _state.vector_store is None:
        raise RuntimeError("Vector store not initialized")
    return _state.vector_store


def get_bm25() -> Optional[BM25Index]:
    """Get the BM25 index."""
    _state.ensure_ready()
    return _state.bm25
