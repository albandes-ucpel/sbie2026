# src/rag/ingest_pdfs.py
import os
import re
import uuid
import hashlib
import unicodedata
from typing import List, Dict, Iterable

import fitz  # PyMuPDF
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

# ====== Config ======
load_dotenv()  # carrega .env da raiz (rode a partir da raiz do projeto)

PDF_DIR = os.getenv("PDF_DIR", "data/pdfs")
COLLECTION = os.getenv("QDRANT_COLLECTION", "literatura")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBED_DIM = 1536  # text-embedding-3-small
BATCH = int(os.getenv("EMBED_BATCH", "64"))

# ====== Helpers ======
def normalize_filename(name: str) -> str:
    """Remove acentos e caracteres problemáticos do nome do arquivo."""
    nfkd = unicodedata.normalize("NFKD", name)
    no_accents = "".join(c for c in nfkd if not unicodedata.combining(c))
    no_spaces = no_accents.replace(" ", "_")
    safe = re.sub(r"[^A-Za-z0-9_.\-]", "", no_spaces)
    return safe

def clean_text(t: str) -> str:
    t = t.replace("\r", " ")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{2,}", "\n", t)
    return t.strip()

def chunk_text(text: str, size: int = 900, overlap: int = 150) -> List[str]:
    text = clean_text(text)
    if not text:
        return []
    out, i = [], 0
    while i < len(text):
        out.append(text[i:i + size])
        i += (size - overlap)
    return [c for c in out if c.strip()]

def hash_md5(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()

def chunk_uuid(filepath: str, page: int, idx: int, chunk: str) -> uuid.UUID:
    """
    Gera UUIDv5 determinístico a partir de (arquivo normalizado + página + idx + hash do conteúdo).
    """
    base = normalize_filename(os.path.basename(filepath))
    raw = f"{base}__p{page:04d}__c{idx:04d}__{hash_md5(chunk)}"
    return uuid.uuid5(uuid.NAMESPACE_URL, raw)

def read_pdf_texts(path: str) -> Iterable[tuple[int, str]]:
    """Yield (page_number, text) para cada página não vazia do PDF."""
    with fitz.open(path) as doc:
        for pno in range(len(doc)):
            page = doc.load_page(pno)
            txt = page.get_text("text")
            txt = clean_text(txt)
            if txt:
                yield (pno + 1, txt)

def get_qdrant_client() -> QdrantClient:
    if QDRANT_URL == ":memory:":
        # Evite em produção/hot-reload; prefira HTTP
        return QdrantClient(location=":memory:")
    if QDRANT_URL.startswith(("http://", "https://")):
        return QdrantClient(url=QDRANT_URL, api_key=os.getenv("QDRANT_API_KEY"))
    # caminho de disco local
    return QdrantClient(path=QDRANT_URL)

def ensure_collection(client: QdrantClient, collection: str, dim: int) -> None:
    # versão sem DeprecationWarning
    if not client.collection_exists(collection_name=collection):
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )

class Emb:
    def __init__(self, model: str):
        key = os.getenv("OPENAI_API_KEY")
        if not key or key.startswith("SEU_TOKEN"):
            raise RuntimeError("OPENAI_API_KEY não definido corretamente no .env/ambiente.")
        self.client = OpenAI(api_key=key)
        self.model = model

    def embed(self, texts: List[str]) -> List[List[float]]:
        resp = self.client.embeddings.create(model=self.model, input=texts)
        return [d.embedding for d in resp.data]

# ====== Pipeline ======
def index_pdfs(pdf_dir: str = PDF_DIR) -> dict:
    client = get_qdrant_client()
    ensure_collection(client, COLLECTION, EMBED_DIM)
    emb = Emb(EMBED_MODEL)

    # Valida diretório
    if not os.path.isdir(pdf_dir):
        raise FileNotFoundError(f"Pasta de PDFs não encontrada: {pdf_dir}")

    # Monta pontos (ids determinísticos, payloads) com texto chunkado
    pending: List[PointStruct] = []
    total_chunks = 0

    for fname in sorted(os.listdir(pdf_dir)):
        if not fname.lower().endswith(".pdf"):
            continue
        path = os.path.join(pdf_dir, fname)

        # normaliza nome problemática (mojibake)
        norm = normalize_filename(fname)
        if norm != fname:
            new_path = os.path.join(pdf_dir, norm)
            try:
                os.rename(path, new_path)
                print(f"[INFO] Renomeado: {fname} -> {norm}")
                path = new_path
            except Exception as e:
                print(f"[WARN] Não foi possível renomear {fname}: {e}")

        try:
            for page_no, text in read_pdf_texts(path):
                chunks = chunk_text(text)
                for idx, ch in enumerate(chunks):
                    pid = chunk_uuid(path, page_no, idx, ch)  # UUIDv5 determinístico
                    payload: Dict = {
                        "text": ch,
                        "source": os.path.relpath(path),
                        "meta": {"page": page_no, "chunk": idx},
                    }
                    pending.append((pid, ch, payload))
                    total_chunks += 1
        except Exception as e:
            print(f"[WARN] Falha ao processar '{path}': {e}")

    if total_chunks == 0:
        print("[INFO] Nenhum chunk gerado a partir dos PDFs.")
        return {"ok": True, "chunks_indexed": 0}

    # Embeddings + upsert em lote
    inserted = 0
    for i in range(0, len(pending), BATCH):
        batch = pending[i:i + BATCH]
        ids = [pid for (pid, _ch, _pl) in batch]
        texts = [_ch for (_pid, _ch, _pl) in batch]
        payloads = [_pl for (_pid, _ch, _pl) in batch]

        vecs = emb.embed(texts)
        qpoints = [
            PointStruct(id=str(pid), vector=vec, payload=pl)
            for pid, vec, pl in zip(ids, vecs, payloads)
        ]
        client.upsert(collection_name=COLLECTION, points=qpoints)
        inserted += len(qpoints)
        print(f"[INFO] Upsert: {inserted}/{total_chunks}")

    return {"ok": True, "chunks_indexed": inserted}

if __name__ == "__main__":
    result = index_pdfs(PDF_DIR)
    print(result)
