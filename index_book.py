import sys
import json
import re
from pathlib import Path
import os

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Allow HuggingFace online download if needed
os.environ["HF_HUB_OFFLINE"] = "0"

from app.indexing.vector_store import VectorStore
from app.indexing.bm25_index import BM25Index
from app.indexing.embedder import embed_texts
from app.indexing.preprocessing import preprocess_document
from app.indexing.chunker import semantic_chunk


# =====================================================
# ENCYCLOPEDIA SECTION HEADINGS
# =====================================================

SECTION_HEADINGS = {
    "Definition",
    "Description",
    "Purpose",
    "Treatment",
    "Diagnosis",
    "Causes and symptoms",
    "Prognosis",
    "Prevention",
    "Precautions",
    "Preparation",
    "Risks",
    "Normal results",
    "Aftercare",
    "Alternative treatment",
    "Abnormal results",
    "Side effects",
    "Interactions",
    "Recommended dosage",
    "Causes",
    "Symptoms",
    "Special conditions",
    "Surgery",
}

# Noise patterns to strip from extracted text
NOISE_PATTERN = re.compile(
    r"GALE ENCYCLOPEDIA OF MEDICINE 2\s+\d+|GEM\s*-?\s*\d+.*?Page\s+\d+",
    re.IGNORECASE,
)


def _is_likely_title(line: str) -> bool:
    """Check if a line looks like an encyclopedia article title."""
    words = line.split()
    if len(words) < 1 or len(words) > 6:
        return False
    if line in SECTION_HEADINGS:
        return False
    if "GALE" in line or "GEM" in line:
        return False
    if line.endswith(".") or line.endswith(","):
        return False
    if line.startswith("•") or line.startswith("–"):
        return False
    if not any(w[0].isupper() for w in words if w):
        return False
    if any(x in line for x in ["M.D.", "Ph.D.", "R.N.", "Writer", "Editor"]):
        return False
    return True


# =====================================================
# PDF TEXT EXTRACTION WITH ARTICLE DETECTION
# =====================================================


def extract_pages_with_articles(pdf_path: str):
    """
    Extract text from PDF, detect encyclopedia article titles per page.
    Returns list of {text, page, source, file_path, article_title}.
    """
    documents = []
    pdf_name = Path(pdf_path).stem
    current_article = "Medical Encyclopedia"

    try:
        from pypdf import PdfReader

        reader = PdfReader(pdf_path)

        for page_num, page in enumerate(reader.pages, 1):
            text = page.extract_text()
            if not text or not text.strip():
                continue

            # Detect article title: line before "Definition"
            lines = [l.strip() for l in text.split("\n") if l.strip()]
            for i, l in enumerate(lines):
                if (
                    i + 1 < len(lines)
                    and lines[i + 1] == "Definition"
                    and _is_likely_title(l)
                ):
                    current_article = l
                    break

            documents.append(
                {
                    "text": text,
                    "page": page_num,
                    "source": pdf_name,
                    "file_path": pdf_path,
                    "article_title": current_article,
                }
            )

        print(f"✅ Extracted {len(documents)} pages from {pdf_name}")
        return documents

    except Exception as e:
        print(f"❌ Failed to extract PDF text: {str(e)}")
        return []


def _clean_page_text(text: str) -> str:
    """Light cleaning that preserves sentence-ending punctuation."""
    # Remove encyclopedia footer noise
    text = NOISE_PATTERN.sub("", text)
    # Normalize line breaks to spaces (PDF wraps mid-sentence)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    # Collapse multiple newlines to one
    text = re.sub(r"\n{2,}", "\n\n", text)
    # Collapse whitespace
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _detect_section(text: str) -> str:
    """Detect which section heading this chunk starts in."""
    for heading in SECTION_HEADINGS:
        if text.lstrip().startswith(heading):
            return heading
    return ""


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

    embed_dim = 1024  # bge-m3 uses 1024 dimensions

    vector_store = VectorStore(dim=embed_dim)
    bm25_index = BM25Index()

    all_chunks = []

    for pdf_path in pdf_files:
        print(f"📄 Processing: {pdf_path.name}")

        pages = extract_pages_with_articles(str(pdf_path))

        if not pages:
            print(f"⚠️ No text extracted from {pdf_path.name}")
            continue

        for page_data in pages:
            # Light cleaning — preserves punctuation for sentence splitting
            text = _clean_page_text(page_data["text"])
            article_title = page_data["article_title"]

            if not text or len(text) < 10:
                continue

            # Chunk with intact punctuation so sentence splitter works
            chunks = semantic_chunk(
                text, chunk_size=150, overlap=30, min_paragraph_length=30
            )

            for chunk_idx, chunk in enumerate(chunks):
                if len(chunk.split()) < 10:
                    continue

                section = _detect_section(chunk)

                # Prefix with article title (+ section if detected)
                if section:
                    chunk_with_title = f"{article_title} - {section}. {chunk}"
                else:
                    chunk_with_title = f"{article_title}. {chunk}"

                metadata = {
                    "page": page_data["page"],
                    "source": page_data["source"],
                    "title": article_title,
                    "section": section,
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
