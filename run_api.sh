#!/bin/bash
# run_api.sh - inicia a API FastAPI (Psico-RAG)

# interrompe qualquer uvicorn antigo
pkill -f "uvicorn.*api.main:app" 2>/dev/null

# ativa o venv
source .venv/bin/activate

# sobe a API
exec python -m uvicorn api.main:app \
  --app-dir src \
  --host 0.0.0.0 \
  --port 8000 \
  --reload
