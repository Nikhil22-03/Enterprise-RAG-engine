# Enterprise RAG Engine — single-container deploy for HF Spaces (free tier)
# Runs FastAPI (internal, port 8000) + Streamlit (exposed, port 7860) in one
# container via start.sh. Chroma indexes must be pre-built with build_index.py
# and copied into the image (see README for the "ship the index" step).

FROM python:3.10-slim

# System deps: PyMuPDF and chromadb both need a working C toolchain at install time
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY . .

# HF Spaces requires world-writable dirs since the container runs as a
# non-root user by default (uid 1000)
RUN mkdir -p /app/chroma_db /app/chroma_db_large /app/data && \
    chmod -R 777 /app

# HF Spaces free tier only exposes ONE port — Streamlit goes on it.
# FastAPI stays internal on 8000, reached by Streamlit via localhost.
EXPOSE 7860

ENV PYTHONUNBUFFERED=1

COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

CMD ["/app/start.sh"]
