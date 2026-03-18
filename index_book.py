import sys
import json
from pathlib import Path
import os

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Allow HuggingFace online download if needed
os.environ["HF_HUB_OFFLINE"] = "0"

from app.indexing.vector_store import VectorStore
from app.indexing.bm25_index import BM25Index
from app.indexing.embedder import embed_texts
from app.indexing.preprocessing import clean_text
from app.indexing.chunker import semantic_chunk


# =====================================================
# PDF TEXT EXTRACTION
# =====================================================


def extract_text_from_pdf(pdf_path: str):
    """
    Extract text from PDF while preserving page numbers.
    """
    documents = []
    pdf_name = Path(pdf_path).stem

    try:
        from pypdf import PdfReader

        reader = PdfReader(pdf_path)

        for page_num, page in enumerate(reader.pages, 1):
            text = page.extract_text()

            if text and text.strip():
                documents.append(
                    {
                        "text": text,
                        "page": page_num,
                        "source": pdf_name,
                        "file_path": pdf_path,
                    }
                )

        print(f"✅ Extracted {len(documents)} pages from {pdf_name}")
        return documents

    except Exception as e:
        print(f"❌ Failed to extract PDF text: {str(e)}")
        return []


# =====================================================
# INDEXING PIPELINE
# =====================================================


def index_pdfs(pdf_folder: str = "data/raw", output_folder: str = "data/processed"):

    pdf_folder = Path(pdf_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    pdf_files = list(pdf_folder.glob("*.pdf"))

    if not pdf_files:
        print(f"❌ No PDF files found in {pdf_folder}")
        return

    print(f"\n📚 Found {len(pdf_files)} PDF file(s)\n")

    from app.indexing.embedder import embed_texts

    embed_dim = 1024  # bge-m3 uses 1024 dimensions

    vector_store = VectorStore(dim=embed_dim)
    bm25_index = BM25Index()

    all_chunks = []

    for pdf_path in pdf_files:
        print(f"📄 Processing: {pdf_path.name}")

        pages = extract_text_from_pdf(str(pdf_path))

        if not pages:
            print(f"⚠️ No text extracted from {pdf_path.name}")
            continue

        for page_data in pages:
            text = clean_text(page_data["text"])
            title = page_data.get("source", "")

            if not text or len(text) < 10:
                continue

            chunks = semantic_chunk(
                text, chunk_size=80, overlap=20, min_paragraph_length=30
            )

            for chunk_idx, chunk in enumerate(chunks):
                if len(chunk) < 15:
                    continue

                chunk_with_title = f"Title: {title}. {chunk}"

                metadata = {
                    "page": page_data["page"],
                    "source": page_data["source"],
                    "title": page_data["source"],
                    "chunk_id": chunk_idx,
                    "file_path": page_data["file_path"],
                }

                all_chunks.append((chunk_with_title, metadata))

        print(f"   ✅ Extracted {len(all_chunks)} chunks from {pdf_path.name}")

    if all_chunks:
        print(f"\n🔢 Embedding {len(all_chunks)} chunks in batches...")

        batch_size = 32
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i : i + batch_size]
            texts = [c[0] for c in batch]
            metadata_list = [c[1] for c in batch]

            embeddings = embed_texts(texts)

            for chunk, emb, meta in zip(texts, embeddings, metadata_list):
                try:
                    vector_store.add(text=chunk, embedding=emb, metadata=meta)
                    bm25_index.add_document(
                        doc_id=f"{meta['source']}_p{meta['page']}_c{meta['chunk_id']}",
                        text=chunk,
                        metadata=meta,
                    )
                except Exception as e:
                    import traceback

                    print(f"⚠️ Error indexing chunk: {type(e).__name__}: {e}")
                    traceback.print_exc()
                    continue

            if (i + batch_size) % 500 == 0 or i + batch_size >= len(all_chunks):
                print(
                    f"   Progress: {min(i + batch_size, len(all_chunks))}/{len(all_chunks)}"
                )

    total_chunks = len(all_chunks)

    print("\n💾 Saving indexes...")
    vector_store.save(str(output_folder))
    bm25_index.save(str(output_folder))

    docs_jsonl_path = output_folder / "docs.jsonl"
    with open(docs_jsonl_path, "w", encoding="utf-8") as f:
        for idx, doc in enumerate(vector_store.documents):
            f.write(
                json.dumps(
                    {
                        "doc_id": f"doc_{idx}",
                        "text": doc.text,
                        "metadata": doc.metadata,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"📄 Saved docs.jsonl with {len(vector_store.documents)} documents")

    print("\n✅ Indexing complete!")
    print(f"📊 Total chunks indexed: {total_chunks}")
    print(f"💾 Saved to: {output_folder.absolute()}")


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Index PDFs for MediLink RAG")

    parser.add_argument("--pdf-folder", type=str, default="data/raw")
    parser.add_argument("--output-folder", type=str, default="data/processed")

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("📚 MediLink PDF Indexing System")
    print("=" * 60 + "\n")

    index_pdfs(args.pdf_folder, args.output_folder)
