#!/usr/bin/env bash
# scripts/reindex_literature.sh
# Reindexa a literatura no Qdrant chamando a API /rag/reindex.

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

API_HOST="${API_HOST:-127.0.0.1}"
API_PORT="${API_PORT:-8000}"
API_URL="http://${API_HOST}:${API_PORT}"
API_KEY="${API_KEY:-dev-key}"

CORPUS_DIR="${CORPUS_DIR:-data/corpus}"

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
  fi
}

call_reindex() {
  if [[ ! -d "${CORPUS_DIR}" ]]; then
    echo "[ERRO] corpus_dir não existe: ${CORPUS_DIR}"; exit 1
  fi
  curl -fsS -X POST "${API_URL}/rag/reindex" \
    -H "X-API-Key: ${API_KEY}" \
    -H "Content-Type: application/json" \
    -d "{\"corpus_dir\":\"${CORPUS_DIR}\"}"
}

ensure_swap
ensure_qdrant
call_reindex
