# MediLink RAG

**Bilingual Medical Retrieval-Augmented Generation (RAG) System**

---

## Overview
MediLink RAG is a production-grade, bilingual (Arabic/English) medical question-answering system. It combines dense and hybrid retrieval, LLM-based translation, and LLM-based answer judging for robust, safe, and accurate medical responses.

**Key Features:**
- Dense, BM25, and hybrid retrieval (multilingual, optimized for Arabic)
- LLM-based translation (Groq API, fallback dictionary)
- LLM-based answer judge (Groq API, Qwen2.5-32B-Instruct)
- Emergency detection, content filtering, and confidence calibration
- FastAPI backend, modular codebase, and robust evaluation pipeline

---

## Quickstart

1. **Clone and install dependencies:**
	```bash
	git clone <repo-url>
	cd RAG
	python3 -m venv .venv
	source .venv/bin/activate
	pip install -r requirements.txt
	```

2. **Set up Groq API key:**
	- Create a `.env` file with:
	  ```
	  GROQ_API_KEY=your_groq_api_key_here
	  ```

3. **Prepare data and index:**
	- Place your documents in `data/processed/docs.jsonl` (see example format).
	- Run the indexer:
	  ```bash
	  python3 index_book.py
	  ```

4. **Start the API server:**
	```bash
	python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000
	```

5. **Access the frontend:**
	- Open your browser to `http://localhost:8000` (if frontend is present).

---

## Project Structure

```
├── app/
│   ├── api/              # FastAPI routes & middleware
│   ├── calibration/      # Confidence calibration
│   ├── core/             # Config, messages, state
│   ├── evaluation/       # Evaluation scripts & metrics
│   ├── generation/       # LLM prompt building & Groq client
│   ├── indexing/         # Embedding, BM25, vector store
│   ├── retrieval/        # Hybrid fusion, translation, reranking
│   ├── safety/           # Content filter, emergency, judge
│   ├── services/         # Service orchestration
│   └── utils/            # Logging, timing, seeding
├── data/                 # Raw and processed data
├── results/              # Evaluation outputs, plots, metrics
├── scripts/              # Utility scripts
├── tests/                # Unit and integration tests
```

---

## Core Technologies
- **Retrieval:** BAAI/bge-m3 (dense), BM25, hybrid fusion, reranking (bge-reranker-v2-m3)
- **LLM:** Groq API (llama-3.1-8b-instant), Qwen2.5-32B-Instruct (local eval)
- **Backend:** FastAPI, Python 3.10+
- **Evaluation:** Custom scripts, pandas, matplotlib, seaborn

---

## Evaluation Pipeline
- `evaluate_retrieval.py`: Benchmarks retrieval (recall, MRR, nDCG, per-query CSV)
- `evaluate_plots.py`: Generation evaluation, plotting, and summary
- `run_full_eval.sh`: Orchestrates full evaluation (robust to disconnects)
- Results: See `results/` for metrics, plots, and error analysis

---

## Configuration
- All core settings in `app/core/config.py` (models, thresholds, paths)
- API keys via `.env` file

---

## Testing
Run all tests:
```bash
pytest tests/
```

---

## Citation
If you use MediLink RAG in research or production, please cite the project and models used (BAAI/bge-m3, Groq, Qwen2.5-32B-Instruct).

---

## License
MIT License. See LICENSE file for details.
