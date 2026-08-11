#!/bin/bash
set -e

echo "[start.sh] Launching FastAPI backend on :8000..."
uvicorn main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

echo "[start.sh] Waiting for backend /health..."
for i in $(seq 1 60); do
  if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
    echo "[start.sh] Backend is healthy."
    break
  fi
  if [ "$i" -eq 60 ]; then
    echo "[start.sh] Backend did not become healthy in time. Exiting."
    kill "$BACKEND_PID" 2>/dev/null || true
    exit 1
  fi
  sleep 5
done

echo "[start.sh] Launching Streamlit frontend on :7860..."
streamlit run frontend.py --server.port=$PORT --server.address=0.0.0.0