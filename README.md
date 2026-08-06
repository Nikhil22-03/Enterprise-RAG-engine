# 🦙 ML Infrastructure RAG Engine

![Python](https://img.shields.io/badge/Python-3.10-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green) ![Langchain](https://img.shields.io/badge/Langchain-Latest-orange) ![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED)

A production-oriented, containerized Retrieval-Augmented Generation (RAG) pipeline featuring hybrid retrieval, conversational memory, full system observability, and automated CI/CD regression testing.

Purpose-built for deep technical reasoning across a small, high-density corpus of foundational ML systems papers (Attention Is All You Need, Llama 3, LoRA, FlashAttention-2, PyTorch 2.0) — optimized for multi-hop precision over corpus breadth.

## 🚀 Architecture Overview

This system uses a decoupled split-brain architecture: a FastAPI backend manages vector operations and state, while a Streamlit frontend provides a seamless chat experience. The two are packaged into a single Docker container for deployment, with FastAPI running internally and Streamlit exposed on the container's public port.

### Key Engineering Features

* **Environment-Aware Inference:** Automatically selects the best available embedding backend — CUDA GPU, AMD GPU via DirectML (local Windows development), or CPU (containerized/cloud deployment) — with no code changes required between environments.
* **Hybrid Retrieval:** Implements a `DualEnsembleRetriever` combining BM25 keyword search with dense vector semantic search across two chunk granularities (400-token and 1200-token), refined by a `CrossEncoderReranker` to maximize context relevance.
* **Stateful Memory:** Uses `SQLChatMessageHistory` with SQLite to maintain conversation state and context across sessions, with a query rewriter that resolves conversational references (e.g. "does it affect performance?") against prior turns.
* **Observability:** Full telemetry integration with **Langfuse**, tracking P95 latency, token usage, and trace paths for every user query.
* **CI/CD Regression Gate:** Automated quality assurance using **Ragas** and **GitHub Actions**. Any code changes are automatically evaluated against a "Golden Dataset," ensuring system faithfulness never drops below a strict 85% threshold.
* **Offline Indexing / Online Serving Split:** Document parsing and embedding happen once, offline (`build_index.py`), ideally with GPU access. The deployed container only ever *loads* the resulting Chroma stores — it never re-embeds documents at startup, keeping cold boot fast on CPU-only infrastructure.

## 🛠 Tech Stack

* **Orchestration:** LangChain, FastAPI
* **LLM & Compute:** Groq (Llama 3.1 8B Instant)
* **Vector Database:** ChromaDB (local persistent, dual-granularity)
* **Evaluation & CI:** Ragas (LLM-as-a-Judge), Pytest, GitHub Actions
* **Monitoring:** Langfuse
* **Containerization:** Docker (single-container split-brain deployment)

## 📈 Performance Benchmarks

Tested against an evaluation dataset using Ragas metrics:

| Metric | Performance | CI/CD Threshold |
| :--- | :--- | :--- |
| **Average Faithfulness** | > 94% | 0.85 |
| **Average Answer Relevancy** | > 91% | 0.85 |

## 🏗 Setup — Local Development

**1. Clone the repository and set up the environment:**
```bash
python -m venv .venv
source .venv/Scripts/activate  # Windows
pip install -r requirements.txt
```

**2. Configure environment variables:**
Create a `.env` file in the root directory (see `.env.example` for the full list):
```text
GROQ_API_KEY="your_api_key_here"
LANGFUSE_PUBLIC_KEY="your_langfuse_public_key"
LANGFUSE_SECRET_KEY="your_langfuse_secret_key"
LANGFUSE_HOST="https://cloud.langfuse.com"
```

**3. Build the vector index (offline, one-time):**
```bash
python build_index.py
```
This parses `data/*.pdf`, chunks at two granularities, and persists `chroma_db/` and `chroma_db_large/`. Run this locally or on a GPU-equipped machine — embedding is the slow part, so this is intentionally kept separate from serving.

**4. Run the backend API:**
```bash
uvicorn main:app --reload
```

**5. Run the Streamlit frontend:**
```bash
streamlit run frontend.py
```

**6. Run the local regression gate:**
```bash
pytest test_rag.py -v -s
```

## 🐳 Containerized Deployment

The project is fully containerized and deployment-ready — validated end-to-end locally, ready to ship to any Docker-compatible platform (cloud VM, Kubernetes, PaaS).

**1. Build the index offline first** (see step 3 above) — the container never builds the index itself, only loads it.

**2. Build and run the image:**
```bash
docker build -t rag-engine .
docker run --rm -p 7860:7860 --env-file .env rag-engine
```
This runs FastAPI internally on port 8000 and exposes the Streamlit UI on port 7860 (`http://localhost:7860`), orchestrated by `start.sh`, which waits for the backend's `/health` check before starting the frontend.

**3. Deploying to a cloud platform:**
Ship `chroma_db/` and `chroma_db_large/` alongside the code (already committed in this repo for the current 5-paper corpus — for a larger corpus, prefer Git LFS or a mounted volume instead of committing binary DB files directly). Set `GROQ_API_KEY`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, and `LANGFUSE_HOST` as platform secrets — never commit a real `.env`.

### Architecture note: offline indexing vs. online serving

Indexing (parsing + embedding PDFs) is intentionally separated from serving (answering queries). The deployed container only ever *loads* a pre-built Chroma store — it never re-embeds documents at startup. This keeps CPU-only containers fast to boot and avoids re-running an expensive, GPU-friendly job on hardware that doesn't have a GPU.