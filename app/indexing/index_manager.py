"""Index management for health checks and rebuilding."""

import os
import json
from pathlib import Path
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

from app.indexing.vector_store import VectorStore
from app.indexing.bm25_index import BM25Index


INDEX_METADATA_FILE = "index_metadata.json"


@dataclass
class IndexHealth:
    """Health status of an index."""

    is_healthy: bool
    document_count: int
    last_updated: Optional[str]
    issues: list


class IndexManager:
    """Manages index health checks and rebuilding."""

    def __init__(self, index_dir: str = "data/processed"):
        self.index_dir = Path(index_dir)
        self.metadata_file = self.index_dir / INDEX_METADATA_FILE

    def check_health(self) -> Dict[str, IndexHealth]:
        """Check health of all indexes."""
        results = {}

        # Check vector store
        results["vector_store"] = self._check_vector_store()

        # Check BM25
        results["bm25"] = self._check_bm25()

        return results

    def _check_vector_store(self) -> IndexHealth:
        """Check vector store health."""
        issues = []
        doc_count = 0
        last_updated = None

        try:
            vs = VectorStore(dim=1024)
            vs.load(str(self.index_dir))
            doc_count = len(vs.documents)

            # Load metadata
            meta = self._load_metadata()
            if meta:
                last_updated = meta.get("vector_store", {}).get("last_updated")

            if doc_count == 0:
                issues.append("Vector store is empty")
            elif doc_count < 100:
                issues.append(f"Very few documents: {doc_count}")

        except FileNotFoundError:
            issues.append("Vector store files not found")
        except Exception as e:
            issues.append(f"Failed to load: {str(e)}")

        return IndexHealth(
            is_healthy=len(issues) == 0,
            document_count=doc_count,
            last_updated=last_updated,
            issues=issues,
        )

    def _check_bm25(self) -> IndexHealth:
        """Check BM25 index health."""
        issues = []
        doc_count = 0
        last_updated = None

        try:
            bm25 = BM25Index.load(str(self.index_dir))
            if bm25:
                doc_count = len(bm25.corpus)

                # Load metadata
                meta = self._load_metadata()
                if meta:
                    last_updated = meta.get("bm25", {}).get("last_updated")

                if doc_count == 0:
                    issues.append("BM25 index is empty")
            else:
                issues.append("BM25 index is None")

        except FileNotFoundError:
            issues.append("BM25 index files not found")
        except Exception as e:
            issues.append(f"Failed to load: {str(e)}")

        return IndexHealth(
            is_healthy=len(issues) == 0,
            document_count=doc_count,
            last_updated=last_updated,
            issues=issues,
        )

    def _load_metadata(self) -> Optional[Dict]:
        """Load index metadata."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, "r") as f:
                    return json.load(f)
            except Exception:
                pass
        return None

    def save_metadata(self, vector_count: int, bm25_count: int):
        """Save index metadata after building."""
        meta = {
            "vector_store": {
                "document_count": vector_count,
                "last_updated": str(Path(__file__).stat().st_mtime),
            },
            "bm25": {
                "document_count": bm25_count,
                "last_updated": str(Path(__file__).stat().st_mtime),
            },
        }

        self.index_dir.mkdir(parents=True, exist_ok=True)
        with open(self.metadata_file, "w") as f:
            json.dump(meta, f, indent=2)

    def needs_rebuild(self, min_documents: int = 100) -> Tuple[bool, str]:
        """Check if indexes need rebuilding."""
        health = self.check_health()

        vs_health = health.get("vector_store", None)
        bm25_health = health.get("bm25", None)

        if not vs_health or not vs_health.is_healthy:
            return True, "Vector store is unhealthy"

        if not bm25_health or not bm25_health.is_healthy:
            return True, "BM25 index is unhealthy"

        if vs_health.document_count < min_documents:
            return (
                True,
                f"Insufficient documents in vector store ({vs_health.document_count})",
            )

        return False, "All indexes are healthy"


# Global instance
index_manager = IndexManager()
