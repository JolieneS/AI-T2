# Enterprise Semantic Knowledge Management & Retrieval System

A production-grade five-stage Retrieval-Augmented Generation (RAG) system built with **LangChain**, **ChromaDB**, **HuggingFace Embeddings** (local, no API key), and **Groq** (`llama-3.1-8b-instant`) — deployed via a **FastAPI** endpoint.

---

## 📂 Repository Structure

```text
project/
├── data/                          # Drop PDF or TXT documents here
├── notebooks/
│   └── rag_pipeline_colab.py     # Full Google Colab notebook (14 cells)
├── src/
│   ├── preprocess.py              # Stage 1 — document parsing, cleaning, chunking
│   ├── features.py                # Stage 2 — HuggingFace embeddings + ChromaDB
│   ├── train.py                   # Stage 3 — RAG chain (3 variants)
│   ├── evaluate.py                # Stage 4 — CR, F, AR, L, QR metrics
│   └── utils.py                   # Helpers, logger
├── api/
│   └── app.py                     # Stage 5 — FastAPI production server
├── models/                        # Auto-created: chroma_db, chunks.json, logs
├── README.md
└── requirements.txt
```

---

## 🛠️ Five-Stage Pipeline

### Stage 1 — Document Ingestion & Text Segmentation (`src/preprocess.py`)
- Loads `.pdf` (multi-page via `PyPDFLoader`) and `.txt` files from `./data/`
- Cleans layout noise: collapses whitespace, strips page numbers, removes horizontal rules
- Auto-classifies sections: `SOP | Policy | Compliance | TroubleshootingLog | General`
- Chunks with `RecursiveCharacterTextSplitter` — **chunk_size=400, overlap=50**
- Saves chunks to `./models/chunks.json`

### Stage 2 — Embedding Generation & Indexing (`src/features.py`)
- Embeds chunks using `sentence-transformers/all-MiniLM-L6-v2` — **runs fully local, no API key**
- Persists vector index to `./models/chroma_db/` via ChromaDB

### Stage 3 — LLM Inference & Context Orchestration (`src/train.py`)

Three pipeline variants:

| Variant | Search Type | Config |
|---|---|---|
| `standard` | Similarity | k=3 |
| `reranked` | MMR | k=3, fetch_k=10, λ=0.6 |
| `threshold` | Score threshold | ≥0.35, k=3 |

**Anti-Hallucination Guardrails:**
- System prompt restricts LLM to context-only answers
- `temperature=0.0` — deterministic output
- Out-of-scope queries return: *"I cannot answer this based on the provided corporate documentation."*

### Stage 4 — Pipeline Evaluation (`src/evaluate.py`)

| Metric | Tag | How Measured |
|---|---|---|
| Context Relevance | CR | Is non-empty context retrieved? |
| Faithfulness | F | % of answer tokens found in context |
| Answer Relevance | AR | % of expected keywords in answer |
| Inference Latency | L | Wall-clock time per query (seconds) |
| Query Resolution Rate | QR | 1 = answered, 0 = abstained |

### Stage 5 — API Deployment (`api/app.py`)

| Endpoint | Method | Description |
|---|---|---|
| `/query` | POST | RAG answer + confidence + sources |
| `/health` | GET | Pipeline status |
| `/docs` | GET | Swagger UI |

---

## 🚀 How to Run

### Option A — Google Colab (recommended)
Open `notebooks/rag_pipeline_colab.py`, paste each cell block into Colab in order and run Cell 1 → Cell 14.

### Option B — Local

```bash
# 1. Set API key
export GROQ_API_KEY="your_groq_key_here"

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add documents to ./data/

# 4. Run pipeline
python src/preprocess.py
python src/features.py
python src/evaluate.py

# 5. Launch API
python -m uvicorn api.app:app --reload --port 8000
```

---

## 🔌 API Usage

### POST /query

```json
{
  "query": "What happens in case of a violation?",
  "variant": "standard"
}
```

**Response:**
```json
{
  "answer": "A violation results in immediate termination...",
  "confidence": 0.87,
  "sources": ["IT-Policy.pdf (p.3)"],
  "latency_s": 0.921,
  "variant": "standard"
}
```

---

## ⚙️ Tech Stack

| Component | Library / Model |
|---|---|
| LLM | Groq — `llama-3.1-8b-instant` |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local) |
| Vector Store | ChromaDB (persisted) |
| Orchestration | LangChain |
| API | FastAPI + Uvicorn |
| PDF Parsing | PyPDF |
