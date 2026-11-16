<div align="center">

# 🔬 ResearchLens

### High-Performance Conversational Research Assistant with Two-Stage Retrieval & Real-Time Streaming

[![Python Version](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13-blue.svg?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![LangChain](https://img.shields.io/badge/LangChain-v0.2+-1C3C3C.svg?logo=langchain&logoColor=white)](https://langchain.com)
[![Google Gemini](https://img.shields.io/badge/Google%20Gemini-Flash%20%7C%20Embedding-4285F4.svg?logo=google&logoColor=white)](https://ai.google.dev)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-orange.svg)](https://trychroma.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

<p align="center">
  <b>ResearchLens</b> is a production-grade Retrieval-Augmented Generation (RAG) platform tailored for scientific and academic literature analysis. It combines <b>Structure-Aware Chunking</b>, <b>Two-Stage Hybrid Retrieval with Cross-Encoder Reranking</b>, <b>Real-Time Server-Sent Events (SSE) Streaming</b>, and <b>Thread-Safe Conversational Session Memory</b>.
</p>

[Key Features](#-key-features) •
[System Architecture](#-system-architecture) •
[Benchmark Progression](#-version-wise-benchmark-results) •
[Project Structure](#-project-structure) •
[Quickstart](#-quickstart) •
[API Reference](#-api-reference) •
[Configuration](#-configuration)

</div>

---

## ⚡ Key Features

- **📑 Structure-Aware & Section-Hierarchical Chunking**:
  - Automatically identifies academic headings (`Abstract`, `1. Introduction`, `3.2 Multi-Head Attention`, `References`).
  - Injects contextual breadcrumbs `[Document: <name> | Section: <hierarchy>]` into every chunk to eliminate orphan fragments.
  - Detects and preserves LaTeX mathematical blocks (`$$...$$`, `\[...\]`) and tables as atomic units with zero mid-equation splitting.

- **🎯 Two-Stage Hybrid Retrieval & Reranking**:
  - **Stage 1 (High Recall)**: Dual-channel retrieval fusing **Dense Semantic Vector Search** (Gemini embeddings) with **Sparse Lexical Search** (BM25Okapi) via **Reciprocal Rank Fusion (RRF, k=60)** over top-15 candidates.
  - **Stage 2 (High Precision)**: Cross-Encoder joint attention scoring via Gemini Flash (`temperature=0.0`) evaluates all 15 candidates down to the top-3 highest-signal chunks, reducing prompt token bloat by **~41%** and boosting Context Precision to **0.96**.

- **⚡ Real-Time Server-Sent Events (SSE) Streaming**:
  - Drops **Time-to-First-Token (TTFT)** from ~3,500ms to **~280ms** (-92% perceived latency).
  - Emits retrieved source documents in the initial event payload so researchers can immediately review citations while the response streams.

- **💬 Multi-Turn Conversational Memory**:
  - Thread-safe sliding window session management (`SessionManagerService`) retaining recent conversation turns.
  - Injects structured conversation history into generation prompts, achieving **94% coreference resolution** for natural follow-ups (*"What optimizer did they use?"* → *"What learning rate did they configure for it?"*).
  - Full REST session lifecycle management (`POST/GET/DELETE /api/v1/sessions`).

- **🔭 Enterprise Observability**:
  - Built-in hierarchical tracing with **LangSmith**, logging token consumption, scoring permutations, and execution latency across each pipeline step.

- **🎨 Modern Single-Page Application (SPA)**:
  - Responsive vanilla frontend with drag-and-drop PDF ingestion, live typewriter streaming, turn-by-turn chat thread, and interactive source citations.

---

## 🏗 System Architecture

```mermaid
flowchart TD
    User["Researcher / Client"] -->|"Upload PDF"| IngestAPI["POST /api/v1/ingest"]
    User -->|"Question + Session ID"| StreamAPI["POST /api/v1/query/stream"]

    subgraph SubIngest ["Ingestion Pipeline"]
        IngestAPI --> PyMuPDF["PyMuPDF Text Extraction"]
        PyMuPDF --> StructChunk["Structure-Aware Chunking<br/>Atomic Math and Breadcrumbs"]
        StructChunk --> GeminiEmb["Gemini Embeddings<br/>gemini-embedding-001"]
        GeminiEmb --> Chroma["Persistent ChromaDB Vector Store"]
    end

    subgraph SubRetrieval ["Two-Stage Retrieval Engine"]
        StreamAPI --> SessionMgr["Session Manager<br/>Thread-Safe Sliding Window"]
        SessionMgr --> Stage1["Stage 1: Hybrid Candidate Retrieval<br/>Dense Cosine + In-Memory BM25Okapi"]
        Stage1 --> RRF["Reciprocal Rank Fusion<br/>RRF k=60 to Top-15 Pool"]
        RRF --> Stage2["Stage 2: Cross-Encoder Reranker<br/>Joint Attention Scoring to Top-3"]
    end

    subgraph SubGeneration ["Streaming and Generation"]
        Stage2 --> InstantSources["SSE Event: sources<br/>Immediate Citation Cards"]
        Stage2 --> PromptBuilder["Prompt Formatter<br/>History + Top-3 Grounded Chunks"]
        PromptBuilder --> GeminiGen["Gemini 2.0 Flash<br/>Streaming Completion"]
        GeminiGen --> TokenStream["SSE Event: token<br/>Typewriter Rendering"]
        TokenStream --> DoneEvent["SSE Event: done<br/>Metrics and Turn Saved"]
    end

    InstantSources --> User
    TokenStream --> User
    DoneEvent --> User
```

---

## 📊 Version-Wise Benchmark Results

Every architectural milestone in ResearchLens has been evaluated against a standardized 10-question academic research paper benchmark to measure retrieval precision, groundedness, latency, and token efficiency:

| Evaluation Metric | Target SLA | V1.0 (Baseline) | V2.0 (Hybrid BM25) | V3.0 (Structure-Aware) | V4.0 (Two-Stage Rerank) | V5.0 (Streaming & Memory) | Cumulative Impact |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Retrieval Recall@5** | $\ge 0.80$ | 0.81 | 0.92 | 0.95 | 0.96 | **0.96** | **+18.5%** |
| **Context Precision** | $\ge 0.70$ | 0.65 | 0.78 | 0.89 | 0.96 | **0.96** | **+47.7%** |
| **Generation Faithfulness** | $\ge 0.85$ | 0.84 | 0.88 | 0.94 | 0.97 | **0.97** | **+15.5%** |
| **Mean Reciprocal Rank (MRR)** | $\ge 0.75$ | 0.82 | 0.91 | 0.94 | 0.98 | **0.98** | **+19.5%** |
| **Answer Correctness** | $\ge 0.80$ | 0.85 | 0.90 | 0.94 | 0.97 | **0.97** | **+14.1%** |
| **Hallucination Rate** | $\le 0.15$ | 0.16 | 0.12 | 0.06 | 0.03 | **0.03** | **-81.2%** |
| **Time to First Token (TTFT)** | $\le 500\text{ ms}$ | 3,500 ms | 3,800 ms | 3,850 ms | 3,780 ms | **280 ms** | **-92.6% (Perceived Latency)** |
| **Prompt Token Consumption** | $< 2500$ | ~2,200 | ~2,200 | ~2,200 | ~1,300 | **~1,450** | **-34.1% (Dense Context)** |
| **Multi-Turn Coreference Acc** | $\ge 0.85$ | 0.00 | 0.00 | 0.00 | 0.00 | **0.94** | **Conversational Leap** |
| **End-to-End Latency (p50)** | $\le 2500\text{ ms}$ | 1,850 ms | 1,870 ms | 1,890 ms | 1,940 ms | **1,950 ms** | **Interactive Throughput** |

### 📈 Architectural Progression Breakdown

1. **V1.0 → V2.0 (Hybrid Search with RRF)**:
   - *Challenge*: Dense vector search had blind spots for exact technical terms, acronyms, and hyperparameters (`d_model=512`).
   - *Solution*: Dual-channel retrieval combining Dense Gemini embeddings with Sparse BM25Okapi, fused via Reciprocal Rank Fusion ($k=60$).
   - *Impact*: Retrieval Recall@5 surged from **0.81 to 0.92** (+13.6%) and Context Precision reached **0.78**.

2. **V2.0 → V3.0 (Structure-Aware & Section-Hierarchical Chunking)**:
   - *Challenge*: Naive chunking severed LaTeX equations and tables mid-row while creating orphan chunks without section provenance.
   - *Solution*: Heading parsing, contextual breadcrumb injection (`[Document: ... | Section: ...]`), and atomic preservation of math blocks and tables.
   - *Impact*: Generation Faithfulness jumped to **0.94**, and Hallucination Rate was cut in half (**0.12 → 0.06**).

3. **V3.0 → V4.0 (Two-Stage Retrieval & Cross-Encoder Reranking)**:
   - *Challenge*: Bi-encoders evaluate queries and passages independently, leaving superficial distractor chunks in the candidate pool.
   - *Solution*: Stage 1 retrieves 15 candidates via Hybrid RRF; Stage 2 scores joint query-chunk cross-attention down to the top-3 chunks.
   - *Impact*: Cut prompt token consumption by **~41%** (~2,200 → ~1,300 tokens) while driving Context Precision to **0.96** and Faithfulness to **0.97**.

4. **V4.0 → V5.0 (Real-Time SSE Streaming & Conversational Memory)**:
   - *Challenge*: Batch queries forced users to wait 3–4 seconds before reading any output, and the pipeline was stateless across turns.
   - *Solution*: Server-Sent Events (SSE) streaming delivering instant source previews and token deltas; thread-safe sliding window session memory.
   - *Impact*: Slashed Time-to-First-Token by **92.6%** (**280ms**) and established **94% coreference resolution** on follow-up research questions.

---

## 📁 Project Structure

```
ResearchPilot/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   └── routes.py              # REST API endpoints (query, stream, ingest, sessions)
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   └── schemas.py             # Pydantic v2 request/response schemas
│   │   ├── services/
│   │   │   ├── __init__.py
│   │   │   ├── document_processor.py  # Structure-aware heading parser & chunker
│   │   │   ├── embedding_service.py   # GoogleGenerativeAIEmbeddings wrapper
│   │   │   ├── generator.py           # Answer generation & real-time token streaming
│   │   │   ├── rag_pipeline.py        # Central pipeline orchestrator
│   │   │   ├── reranker.py            # Two-stage cross-encoder joint scoring
│   │   │   ├── retriever.py           # Dual-channel Hybrid RRF retriever
│   │   │   ├── session_manager.py     # Multi-turn conversational memory store
│   │   │   └── vector_store.py        # Persistent ChromaDB collection manager
│   │   ├── utils/
│   │   │   ├── __init__.py
│   │   │   └── logger.py              # Structured logging utility
│   │   ├── config.py                  # Pydantic BaseSettings configuration
│   │   └── main.py                    # FastAPI application initialization & lifespan
│   ├── evaluation/
│   │   ├── datasets/
│   │   │   └── benchmark_v1.json      # Standard evaluation question-answer suite
│   │   ├── benchmark.py               # Benchmark execution harness
│   │   └── metrics.py                 # Retrieval & generation evaluation metrics
│   └── run.py                         # Application entrypoint runner
├── frontend/
│   ├── css/
│   │   └── styles.css                 # Modern CSS styles with chat & streaming layout
│   ├── js/
│   │   └── app.js                     # Vanilla JavaScript SPA & SSE reader
│   └── index.html                     # Web client interface
├── .env.example                       # Template for environment configuration
├── .gitignore                         # Git exclusion rules
├── README.md                          # Project documentation
└── requirements.txt                   # Production Python dependencies
```

---

## 🚀 Quickstart

### 1. Prerequisites
- **Python 3.11+**
- **Google Gemini API Key** ([Get a key here](https://aistudio.google.com/app/apikey))
- **LangSmith API Key** *(Optional, for telemetry)* ([Get a key here](https://smith.langchain.com))

### 2. Clone the Repository
```bash
git clone https://github.com/Adithyan1809/ResearchPilot.git
cd ResearchPilot
```

### 3. Setup Virtual Environment
```bash
# On Linux / macOS
python3 -m venv .venv
source .venv/bin/activate

# On Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 4. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Configure Environment Variables
Create a `.env` file based on `.env.example`:
```bash
cp .env.example .env
```
Fill in your credentials:
```env
GOOGLE_API_KEY=your_gemini_api_key_here

# LangSmith Observability (Optional)
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_api_key_here
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_AGENT_ID=researchpilot
LANGSMITH_AGENT_ENVIRONMENT=local
```

### 6. Launch the Application
```bash
python backend/run.py
```
- **Web Interface**: Open [http://localhost:8000](http://localhost:8000)
- **Interactive OpenAPI Docs**: Open [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🔌 API Reference

### Ingestion Endpoints
- **`POST /api/v1/ingest`**: Upload and ingest a research paper PDF.
  - Body: `multipart/form-data` with `file: UploadFile`.
  - Returns: `doc_id`, `filename`, `total_pages`, `total_chunks`.

### Query Endpoints
- **`POST /api/v1/query/stream`** *(Recommended)*:
  - Server-Sent Events (SSE) real-time stream.
  - Events: `sources` (citations preview) $\rightarrow$ `token` (live deltas) $\rightarrow$ `done` (final response & latency).
- **`POST /api/v1/query`**: Standard JSON query execution.
  - Request body:
    ```json
    {
      "question": "What is the attention mechanism in transformers?",
      "session_id": "optional-uuid-here",
      "top_k": 3,
      "retrieval_mode": "hybrid"
    }
    ```

### Session Endpoints
- **`POST /api/v1/sessions`**: Create a new conversation session.
- **`GET /api/v1/sessions`**: List all active conversation sessions.
- **`GET /api/v1/sessions/{session_id}`**: Retrieve full turn history.
- **`DELETE /api/v1/sessions/{session_id}`**: Clear session history.

### Admin Endpoints
- **`GET /api/v1/health`**: System status, document count, and total vector chunks.
- **`GET /api/v1/documents`**: List indexed documents with page and chunk metadata.
- **`DELETE /api/v1/documents/{doc_id}`**: Remove an indexed document.

---

## ⚙️ Configuration

All settings can be customized via `.env` or environment variables:

| Variable | Default | Description |
| :--- | :--- | :--- |
| `GOOGLE_API_KEY` | `""` | Google Gemini API key |
| `LLM_MODEL` | `gemini-3.5-flash-lite` | Generation LLM model identifier |
| `LLM_TEMPERATURE` | `0.1` | Temperature for generation (low for factual grounding) |
| `EMBEDDING_MODEL` | `models/gemini-embedding-001` | Embedding model identifier |
| `CHUNK_SIZE` | `1000` | Target character window per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap characters between chunks |
| `RETRIEVAL_MODE` | `hybrid` | Strategy: `hybrid`, `dense`, or `bm25` |
| `RRF_K` | `60` | Smoothing constant for Reciprocal Rank Fusion |
| `RERANKER_ENABLED` | `true` | Enable/disable Stage 2 Cross-Encoder reranking |
| `RERANKER_CANDIDATES_K`| `15` | Size of Stage 1 candidate pool |
| `RERANKER_TOP_K` | `3` | Number of chunks injected into context after reranking |
| `CHROMA_PERSIST_DIR` | `./data/chroma_db` | On-disk path for ChromaDB storage |
| `HOST` | `0.0.0.0` | Server host bind address |
| `PORT` | `8000` | Server port |

---

## 📄 License

This project is open-source and licensed under the [MIT License](LICENSE).
