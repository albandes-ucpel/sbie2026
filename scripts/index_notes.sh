#!/usr/bin/env bash
# scripts/index_notes.sh
# Indexação das notas (prontuário) no Qdrant com otimizações de memória/CPU.

set -euo pipefail

QUIET=0

for arg in "$@"; do
  case "$arg" in
    --quiet) QUIET=1 ;;
  esac
done

log() {
  if [[ $QUIET -eq 0 ]]; then
    echo "$@"
  fi
}

SWAP_SIZE="${SWAP_SIZE:-2G}"
USE_DOCKER_QDRANT="${USE_DOCKER_QDRANT:-1}"
QDRANT_IMAGE="${QDRANT_IMAGE:-qdrant/qdrant:v1.11.3}"
QDRANT_PORT="${QDRANT_PORT:-6333}"
QDRANT_NAME="${QDRANT_NAME:-qdrant}"
QDRANT_URL="${QDRANT_URL:-http://localhost:${QDRANT_PORT}}"

NOTES_EMBED_MODEL="${NOTES_EMBED_MODEL:-sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2}"
NOTES_EMBED_DIM="${NOTES_EMBED_DIM:-384}"
NOTES_BATCH="${NOTES_BATCH:-16}"

OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-1}"
TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

NOTES_CSV="${NOTES_CSV:-data/raw/notes.csv}"

ensure_swap() {
  if ! sudo swapon --show | grep -q "partition\|file"; then
    log "[swap] criando swap de ${SWAP_SIZE}..."
    sudo fallocate -l "${SWAP_SIZE}" /swapfile || sudo dd if=/dev/zero of=/swapfile bs=1M count=$(( ${SWAP_SIZE%G} * 1024 ))
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile
    echo "/swapfile none swap sw 0 0" | sudo tee -a /etc/fstab >/dev/null || true
  fi
}

ensure_qdrant() {
  if [[ "${USE_DOCKER_QDRANT}" == "1" ]]; then
    if ! docker ps --format '{{.Names}}' | grep -q "^${QDRANT_NAME}\$"; then
      log "[qdrant] iniciando container ${QDRANT_NAME}..."
      docker run -d --name "${QDRANT_NAME}" -p "${QDRANT_PORT}:6333" "${QDRANT_IMAGE}" >/dev/null
    fi
    for i in {1..30}; do
      if curl -fsS "http://localhost:${QDRANT_PORT}/readyz" >/dev/null; then
        break
      fi
      sleep 1
    done
  else
    export QDRANT_URL=":memory:"
    log "[qdrant] usando QDRANT_URL=:memory:"
  fi
}

run_indexer() {
  export NOTES_EMBED_MODEL NOTES_EMBED_DIM NOTES_BATCH
  export OMP_NUM_THREADS TORCH_NUM_THREADS TOKENIZERS_PARALLELISM
  export QDRANT_URL NOTES_CSV

  log "[run] indexando notas do arquivo ${NOTES_CSV}"
  python -m src.rag.load_notes
}

ensure_swap
ensure_qdrant
run_indexer
