# src/rag/load_notes.py
import os, re, uuid, time, gc, logging
from typing import Iterable, List

import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

# ---------- Logging ----------
LOG_LEVEL = os.getenv("NOTES_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("load_notes")

# ---------- Config ----------
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
NOTES_COLLECTION = os.getenv("QDRANT_NOTES_COLLECTION", "prontuario_privado")

# Embeddings locais (modelo leve por padrão)
NOTES_EMBED_MODEL = os.getenv(
    "NOTES_EMBED_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)
# Dimensionalidade do vetor (ajuste se trocar de modelo)
NOTES_EMBED_DIM = int(os.getenv("NOTES_EMBED_DIM", "384"))

# Arquivo de entrada e tamanho do lote
NOTES_CSV = os.getenv("NOTES_CSV", "data/raw/notes.csv")
BATCH = int(os.getenv("NOTES_BATCH", "16"))

# Limitar threads (evita travar máquina pequena)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TORCH_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ---------- PII redaction ----------
PII_PATTERNS = [
    re.compile(r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b"),  # CPF
    re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),  # e-mail
    re.compile(r"\+?\d{2}\s?\(?\d{2}\)?\s?\d{4,5}-?\d{4}"),  # telefone BR
]
def redact(text: str) -> str:
    t = text or ""
    for p in PII_PATTERNS:
        t = p.sub("[REDACTED]", t)
    return t

# ---------- Chunking simples ----------
def chunk(text: str, size=800, overlap=120) -> Iterable[str]:
    s = re.sub(r"\s+", " ", (text or "")).strip()
    i = 0
    while i < len(s):
        yield s[i:i + size]
        i += (size - overlap)

# ---------- Embeddings locais ----------
_ENCODER = None
def get_encoder():
    global _ENCODER
    if _ENCODER is None:
        from sentence_transformers import SentenceTransformer
        t0 = time.time()
        _ENCODER = SentenceTransformer(NOTES_EMBED_MODEL)
        log.info(f"Encoder carregado: {NOTES_EMBED_MODEL} (took {time.time()-t0:.2f}s)")
    return _ENCODER

def embed_texts(encoder, texts: List[str]) -> List[List[float]]:
    vecs = encoder.encode(texts, show_progress_bar=False)
    return [v.tolist() for v in vecs]

# ---------- Qdrant helpers ----------
def get_client() -> QdrantClient:
    if QDRANT_URL == ":memory:":
        return QdrantClient(location=":memory:")
    if QDRANT_URL.startswith(("http://", "https://")):
        return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    return QdrantClient(path=QDRANT_URL)

def ensure_collection(client: QdrantClient, name: str, dim: int):
    cols = [c.name for c in client.get_collections().collections]
    if name not in cols:
        client.recreate_collection(
            collection_name=name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        log.info(f"Coleção criada: {name} (dim={dim})")
    else:
        # Não recriar se já existe; apenas informa
        log.info(f"Coleção OK: {name} (dim esperado={dim})")

# ---------- Main ----------
def main():
    start = time.time()

    # 1) Ler CSV
    if not os.path.exists(NOTES_CSV):
        log.error(f"Arquivo não encontrado: {NOTES_CSV}")
        raise FileNotFoundError(NOTES_CSV)

    df = pd.read_csv(NOTES_CSV)
    expected = ["student_pid", "note_ts", "text"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        log.error(f"Colunas ausentes em {NOTES_CSV}: {missing}")
        raise ValueError(f"Colunas ausentes: {missing}")

    rows_total = len(df)
    log.info(f"Lendo CSV: {NOTES_CSV} | linhas={rows_total}")

    # Normaliza e redige PII
    df["student_pid"] = df["student_pid"].astype(str)
    df["note_ts"] = pd.to_datetime(df["note_ts"], errors="coerce")
    antes = len(df)
    df = df.dropna(subset=["note_ts"])
    drop_ts = antes - len(df)
    if drop_ts:
        log.warning(f"{drop_ts} linha(s) descartadas por note_ts inválido")

    df["text"] = df["text"].astype(str).map(redact)

    # 2) Encoder local
    encoder = get_encoder()

    # 3) Qdrant
    client = get_client()
    ensure_collection(client, NOTES_COLLECTION, NOTES_EMBED_DIM)

    # 4) Indexação em lotes
    buffer_points: List[PointStruct] = []
    chunks_total = 0
    upserts_total = 0
    t_last_log = time.time()

    for i, row in df.iterrows():
        pid = row["student_pid"]
        ts_iso = row["note_ts"].isoformat()
        text = row["text"]

        for ch in chunk(text):
            vec = embed_texts(encoder, [ch])[0]
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=vec,
                payload={
                    "pid": pid,
                    "ts": ts_iso,
                    "text": ch,
                    "type": "prontuario",
                },
            )
            buffer_points.append(point)
            chunks_total += 1

            if len(buffer_points) >= BATCH:
                client.upsert(collection_name=NOTES_COLLECTION, points=buffer_points)
                upserts_total += len(buffer_points)
                buffer_points = []
                # limpeza para reduzir pressão de memória
                gc.collect()

        # Log periódico (a cada ~5s)
        if time.time() - t_last_log > 5:
            log.info(f"progresso: linhas={i+1}/{rows_total}, chunks={chunks_total}, upserts={upserts_total}")
            t_last_log = time.time()

    # flush final
    if buffer_points:
        client.upsert(collection_name=NOTES_COLLECTION, points=buffer_points)
        upserts_total += len(buffer_points)
        buffer_points = []
        gc.collect()

    dur = time.time() - start
    thr = chunks_total / dur if dur > 0 else chunks_total
    log.info(
        "FINAL | linhas_lidas=%d | chunks=%d | upserts=%d | tempo=%.2fs | throughput=%.2f chunks/s",
        rows_total, chunks_total, upserts_total, dur, thr
    )

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log.exception(f"Falha na indexação: {e}")
        raise
