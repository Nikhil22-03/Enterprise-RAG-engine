# 🦙 Enterprise RAG Engine: Split-Brain Architecture
![Python](https://img.shields.io/badge/Python-3.10-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi&logoColor=white) ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white) ![Langchain](https://img.shields.io/badge/Langchain-Latest-orange) ![GitHub Actions](https://img.shields.io/badge/CI%2FCD-GitHub_Actions-2088FF?logo=github-actions&logoColor=white)

A production-grade, hardware-accelerated Retrieval-Augmented Generation (RAG) pipeline featuring query optimization, multi-vector ensemble retrieval, conversational memory, full system observability, and automated CI/CD regression testing.

## 🚀 System Architecture
This system abandons the standard monolithic script in favor of a decoupled, scalable **Split-Brain Architecture**:
* **The Backend (FastAPI):** A high-performance REST API that acts as the orchestration engine. It manages all vector database connections, hardware acceleration states, LLM routing, and retrieval logic asynchronously.
* **The Frontend (Streamlit):** A lightweight, reactive chat interface that maintains client-side session states and consumes the FastAPI backend endpoints, ensuring the UI remains perfectly fluid during heavy ML operations.

---

## 🧠 Advanced RAG Pipeline Details

### 1. LLM-Powered Query Rewriting
To handle ambiguous human inputs and coreferences, the pipeline routes the raw user input through a Query Rewriter. Using the user's `SQLChatMessageHistory`, an LLM reformulates the input into a standalone, highly optimized semantic search query before it ever touches the database.

### 2. Multi-Vector Ensemble Retrieval
To balance exact-keyword matching with deep contextual understanding, this engine implements a custom, multi-stage retrieval architecture:
* **Dual Vector Stores:** Maintains two separate persistent ChromaDB vector stores (`chroma_db` for granular 400-token chunks, and `chroma_db_large` for 1200-token chunks).
* **Parallel Ensembling:** The pipeline first creates two independent `EnsembleRetrievers`. It combines BM25 sparse keyword search with the Small Chunk dense vectors, and separately combines BM25 with the Large Chunk dense vectors.
* **Pool Merger:** The results from both independent ensembles are merged into a massive, highly diverse candidate pool.

### 3. Cross-Encoder Re-ranking
Because the merged ensemble pool contains noisy candidates, a HuggingFace `CrossEncoderReranker` scores the user's optimized query against the retrieved documents. It aggressively filters out low-relevance chunks, feeding only the highest-fidelity context to the generation model.

---

## 🗂️ Knowledge Domain & Dataset
This RAG engine is purpose-built to ingest, parse, and reason across highly technical, math-heavy machine learning research papers. 

* **Ingestion Corpus:** The `data/` directory is populated with foundational Deep Learning and LLM architecture documentation:
  * *Attention Is All You Need* (Vaswani et al., 2017)
  * *LoRA: Low-Rank Adaptation of Large Language Models* (Hu et al., 2021)
  * *FlashAttention-2: Faster Attention with Better Parallelism* (Dao, 2023)
  * *The Llama 3 Herd of Models* (Meta, 2024)
  * *PyTorch 2.0 Technical Documentation*
* **Target QA Capabilities:** The evaluation pipeline specifically tests the engine's ability to handle multi-hop reasoning across these specific papers, such as:
  * *"How does the memory complexity of FlashAttention-2 compare to the standard scaled dot-product attention described in the original 2017 Transformer paper?"*
  * *"Explain how LoRA reduces the number of trainable parameters during fine-tuning compared to full parameter updates for architectures like Llama 3."*
  * *"What are the specific compiler-level graph execution optimizations introduced in PyTorch 2.0?"*
  
  ---
  
## 🛡️ Models & Enterprise Engineering Standards

### Model Stack
* **Generation LLM:** Llama 3 8B (Served via Groq for ultra-low latency inference).
* **Embedding Model:** BAAI/bge-large-en-v1.5 (Running locally).
* **Hardware Acceleration:** Native integration with AMD GPUs via DirectML for fast, local embedding generation.

### Observability & Quality Assurance
* **Persistent Memory:** SQLite database implementation ensuring conversational context survives server restarts.
* **Observability:** Full telemetry via **Langfuse**, tracking latency, token usage, and LLM trace paths for every API call.
* **Comprehensive Benchmarking:** The core architecture was rigorously evaluated offline against a full **Golden Dataset of 40 curated Q&A pairs** to establish baseline accuracy.
* **Automated CI/CD Regression Gate:** To balance speed and safety, Pull Requests trigger a GitHub Actions workflow that evaluates system code against a lightweight **20-question CI/CD subset** using **Ragas (LLM-as-a-Judge)**. The pipeline enforces a strict 85% threshold, automatically blocking deployments that degrade AI performance.

## 📈 Evaluation Benchmarks
| Ragas Metric | Full Benchmark (n=40) | CI/CD Action Threshold (n=20) |
| :--- | :--- | :--- |
| **Average Faithfulness** | > 94% | Blocks PR if < 0.85 |
| **Average Answer Relevancy** | > 91% | Blocks PR if < 0.85 |

---

## 🏗 Setup & Deployment

**1. Clone the repository and setup the environment:**
```bash
python -m venv .venv
source .venv/Scripts/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**2. Configure Environment Variables:**
Create a `.env` file in the root directory:
```text
GROQ_API_KEY="your_api_key_here"
LANGFUSE_PUBLIC_KEY="your_langfuse_public_key"
LANGFUSE_SECRET_KEY="your_langfuse_secret_key"
LANGFUSE_HOST="[https://cloud.langfuse.com](https://cloud.langfuse.com)"
```

**3. Boot the Orchestration API:**
```bash
uvicorn main:app --reload
```

**4. Boot the Client Interface:**
Open a separate terminal instance and run:
```bash
streamlit run frontend.py
```

**5. Run the Local Quality Assurance Gate:**
```bash
pytest test_rag.py -v -s
```