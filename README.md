# Compliance Document Assistant

A production-grade RAG (Retrieval-Augmented Generation) application that lets compliance and risk analysts ask natural language questions over regulatory documents and receive cited, accurate answers grounded in the actual source text.

Built as a portfolio project demonstrating end-to-end ML/AI engineering — from document ingestion and vector search to a LangGraph agent and a custom chat interface.

---

## What it does

You upload regulatory PDFs (Basel III, FATF AML guidelines, or any compliance document). The system indexes them into a searchable vector store. You type a question in plain English. The agent retrieves the most relevant paragraphs, classifies your question as simple or complex, and generates a grounded answer — showing exactly which pages it used.

---

## Architecture

```
Browser (Chat UI)
      │
      │  HTTP
      ▼
FastAPI Server (main.py)
      │
      ├── POST /query
      │       │
      │       ▼
      │   LangGraph Agent
      │       │
      │       ├── 1. Route  → simple or complex question?
      │       ├── 2. Retrieve → FAISS vector search
      │       │       └── HuggingFace embeddings (all-MiniLM-L6-v2)
      │       └── 3. Generate → flan-t5-large (local, free)
      │
      ├── POST /upload  → save PDF + re-index
      ├── POST /ingest  → rebuild full index
      └── GET  /health  → system status
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Chat UI | HTML + CSS + Vanilla JS (served by FastAPI) |
| API | FastAPI + Uvicorn |
| Agent / Orchestration | LangGraph |
| LLM | `google/flan-t5-large` — local, free, no API key |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` — local |
| Vector Store | FAISS (Facebook AI Similarity Search) |
| PDF Processing | PyMuPDF + pypdf |
| Data Validation | Pydantic v2 |
| Configuration | pydantic-settings + .env |
| Logging | Loguru |

---

## Project Structure

```
compliance_rag/
├── app/
│   ├── api/
│   │   └── routes.py          # API endpoints: /query /upload /ingest /health
│   ├── core/
│   │   ├── config.py          # All settings via pydantic-settings + .env
│   │   ├── schemas.py         # Pydantic request/response models
│   │   ├── ingestion.py       # PDF loading → chunking → FAISS index
│   │   ├── retriever.py       # Vector store loader + similarity search
│   │   └── agent.py           # LangGraph agent with query routing
│   └── templates/
│       └── index.html         # Chat UI (HTML/CSS/JS)
├── data/
│   ├── docs/                  # PDF documents go here
│   └── vectorstore/           # FAISS index (auto-generated)
├── scripts/
│   ├── download_docs.py       # Downloads sample regulatory PDFs
│   └── evaluate.py            # RAGAS evaluation (faithfulness, relevancy)
├── tests/
│   ├── test_ingestion.py
│   ├── test_retriever.py
│   └── test_routes.py
├── env/                       # Virtual environment (not committed to git)
├── .env                       # Environment variables
├── main.py                    # FastAPI entry point
├── requirements.txt
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.9
- ~3GB free disk space (for AI models downloaded on first run)

### 1. Create and activate a virtual environment

```bash
python -m venv env
env\Scripts\activate        # Windows
source env/bin/activate     # Mac / Linux
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download sample regulatory documents

```bash
python scripts/download_docs.py
```

This downloads three freely available BIS documents into `data/docs/`:
- Basel III: A global regulatory framework
- Basel III: International framework for liquidity risk
- Core Principles for Effective Banking Supervision

### 4. Build the search index

```bash
python -m app.core.ingestion
```

First run downloads the HuggingFace embedding model (~90MB). Takes 2–4 minutes.

### 5. Start the server

```bash
python -m uvicorn main:app --reload
```

Open your browser at **http://localhost:8000**

---

## Usage

### Chat UI
Open `http://localhost:8000` — type any compliance question and press Enter.

### Upload your own PDF
Click the **Upload PDF** button in the top-right corner of the chat. The document is saved and indexed automatically. You can ask questions about it immediately after upload.

### API
The full API is available at `http://localhost:8000/docs`

**Ask a question:**
```http
POST /query
Content-Type: application/json

{
  "question": "What is the minimum CET1 ratio under Basel III?",
  "top_k": 5
}
```

**Response:**
```json
{
  "answer": "4.5% of risk-weighted assets.",
  "sources": [
    {
      "document": "Basel_III_Framework.pdf",
      "page": 36,
      "excerpt": "The minimum Common Equity Tier 1 ratio is set at 4.5%..."
    }
  ],
  "query_type": "simple",
  "faithfulness_score": 0.87
}
```

---

## How the RAG pipeline works

**1. Ingestion** — PDFs are loaded page by page, split into 512-token overlapping chunks, converted into 384-dimensional vectors using a sentence transformer, and stored in a FAISS index on disk.

**2. Query routing** — When a question arrives, the LangGraph agent first classifies it. Single-fact questions ("What is X?") are marked simple. Questions requiring comparison or multi-step reasoning ("Compare X and Y") are marked complex.

**3. Retrieval** — FAISS performs a cosine similarity search to find the top-k most relevant chunks for the question. These chunks become the context window.

**4. Generation** — The LLM receives the question and retrieved context and generates an answer grounded only in that context. If the context doesn't contain the answer, it says so rather than hallucinating.

---

## Configuration

All settings are in `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_MODEL` | `google/flan-t5-large` | HuggingFace model for answer generation |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Model for text embeddings |
| `CHUNK_SIZE` | `512` | Token size of each document chunk |
| `CHUNK_OVERLAP` | `64` | Overlap between consecutive chunks |
| `DEFAULT_TOP_K` | `5` | Paragraphs retrieved per question |
| `LLM_MAX_NEW_TOKENS` | `256` | Maximum length of generated answer |

---

## Evaluation

Run RAGAS evaluation against a built-in test set of compliance Q&A pairs:

```bash
python scripts/evaluate.py
python scripts/evaluate.py --output results/eval_results.json
```

Metrics reported:
- **Faithfulness** — is the answer grounded in the retrieved context?
- **Answer relevancy** — does the answer address the question asked?
- **Context precision** — are the retrieved chunks relevant?

---

## Limitations

- The local LLM (`flan-t5-large`) handles factual questions well but can struggle with complex multi-document synthesis. For production use, swap for an API-based model such as Azure OpenAI GPT-4 by updating `agent.py`.
- Ingestion re-processes all documents on every upload. For large document sets, incremental indexing would be a natural next improvement.
- No authentication on the API endpoints. For any deployment beyond local use, add API key middleware.

---

## Possible Extensions

- Swap local LLM for Azure OpenAI (one function change in `agent.py`)
- Add conversation memory so follow-up questions have context
- Incremental ingestion so only new documents are re-indexed on upload
- User authentication and document access control
- Deploy to Azure App Service with Docker

---

## Documents used for demo

| Document | Source |
|----------|--------|
| Basel III: A global regulatory framework | Bank for International Settlements |
| Basel III: Liquidity risk measurement framework | Bank for International Settlements |
| Core Principles for Effective Banking Supervision | Basel Committee |