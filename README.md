---
title: Enterprise RAG Engine
emoji: 📚
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Enterprise RAG Engine — Deployed

Dual-retrieval RAG system (BM25 + dense, two chunk granularities, cross-encoder
reranking) over a small corpus of foundational ML systems papers. FastAPI
backend + Streamlit frontend, packaged into a single container for HF Spaces.

## Deploying this yourself

**1. Build the index offline (NOT inside this container):**
```bash
pip install -r requirements.txt
python build_index.py
```
This needs your `data/*.pdf` files present locally and will produce
`chroma_db/` and `chroma_db_large/`. Run this on your own machine or the HPC
cluster — ideally with GPU access, since embedding is the slow part.

**2. Ship the index with the code:**
Commit `chroma_db/` and `chroma_db_large/` alongside the app code (or, for a
larger corpus, use HF Spaces' persistent storage / Git LFS instead of
committing binary DB files directly — fine for this project's small corpus).

**3. Set secrets:**
In Space Settings → Repository secrets, add `GROQ_API_KEY`,
`LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` (see `.env.example`).

**4. Push:**
```bash
git push hf-space main
```
HF Spaces builds the Dockerfile and serves the Streamlit UI on port 7860,
which internally talks to FastAPI on port 8000 inside the same container.

## Architecture note: offline indexing vs. online serving

Indexing (parsing + embedding PDFs) is intentionally separated from serving
(answering queries). The deployed container only ever *loads* a pre-built
Chroma store — it never re-embeds documents at startup. This keeps the
free-tier CPU-only container fast to boot and avoids re-running an expensive
GPU-friendly job on hardware that doesn't have a GPU.
