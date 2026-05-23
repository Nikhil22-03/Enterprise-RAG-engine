# 🦙 ML Infrastructure RAG Engine
![Python](https://img.shields.io/badge/Python-3.10-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green) ![Langchain](https://img.shields.io/badge/Langchain-Latest-orange) ![DirectML](https://img.shields.io/badge/AMD-DirectML-red)

A production-grade, hardware-accelerated Retrieval-Augmented Generation (RAG) pipeline featuring hybrid retrieval, conversational memory, full system observability, and automated CI/CD regression testing.

## 🚀 Architecture Overview
This system is engineered for low-latency inference and high-accuracy retrieval. It utilizes a decoupled split-brain architecture where a FastAPI backend manages vector operations and state, while a Streamlit frontend provides a seamless chat experience.

### Key Engineering Features
* **Hardware Acceleration:** Integrated AMD GPU via DirectML for high-performance, local embedding generation.
* **Hybrid Retrieval:** Implements a `DualEnsembleRetriever` combining BM25 keyword search with dense vector semantic search, refined by a `CrossEncoderReranker` to maximize context relevance.
* **Stateful Memory:** Uses `SQLChatMessageHistory` with SQLite to maintain conversation state and context across sessions.
* **Observability:** Full telemetry integration with **Langfuse**, tracking P95 latency, token usage, and trace paths for every user query.
* **CI/CD Regression Gate:** Automated quality assurance using **Ragas** and **GitHub Actions**. Any code changes are automatically evaluated against a "Golden Dataset," ensuring system faithfulness never drops below a strict 85% threshold.

## 🛠 Tech Stack
* **Orchestration:** LangChain, FastAPI
* **LLM & Compute:** Groq (Llama 3 8B), AMD GPU (DirectML)
* **Vector Database:** ChromaDB (Local Persistent)
* **Evaluation & CI:** Ragas (LLM-as-a-Judge), Pytest, GitHub Actions
* **Monitoring:** Langfuse

## 📈 Performance Benchmarks
Tested against an extensive evaluation dataset using Ragas metrics:

| Metric | Performance | CI/CD Threshold |
| :--- | :--- | :--- |
| **Average Faithfulness** | > 94% | 0.85 |
| **Average Answer Relevancy** | > 91% | 0.85 |

## 🏗 Setup & Deployment

**1. Clone the repository and setup the environment:**
```bash
python -m venv .venv
source .venv/Scripts/activate  # Windows
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

**3. Run the Backend API:**
```bash
uvicorn main:app --reload
```

**4. Run the Streamlit Frontend:**
```bash
streamlit run frontend.py
```

**5. Run the Local Regression Gate:**
```bash
pytest test_rag.py -v -s
```