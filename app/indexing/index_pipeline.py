"""Index pipeline module for loading and managing indexes."""

from app.indexing.bm25_index import BM25Index


def load_bm25(index_dir=None):
    """Load BM25 index from disk.

    Args:
        index_dir: Directory containing the BM25 index files

    Returns:
        BM25Index instance or None if loading fails
    """
    if index_dir is None:
        index_dir = "data/processed"

    try:
        bm25 = BM25Index.load(index_dir)
        return bm25
    except Exception as e:
        print(f"Warning: Failed to load BM25 index: {e}")
        return None
