# src/rag/common.py
import os, re, json, uuid
from typing import List, Dict, Optional, Iterable, Tuple
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from openai import OpenAI

load_dotenv()

COLLECTION = os.getenv("QDRANT_COLLECTION", "literatura")
EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

client = OpenAI()
qdrant = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))

def ensure_collection():
    try:
        qdrant.get_collection(COLLECTION)
    except Exception:
        qdrant.recreate_collection(
            collection_name=COLLECTION,
            vectors_config={
                "vec_abstract": VectorParams(size=1536, distance=Distance.COSINE),
                "vec_fulltext": VectorParams(size=1536, distance=Distance.COSINE),
            },
        )

def embed(text: str) -> List[float]:
    text = (text or "").strip()
    if not text:
        return []
    r = client.embeddings.create(model=EMBED_MODEL, input=text)
    return r.data[0].embedding

def chunk_text(s: str, max_tokens=700, overlap=100) -> List[str]:
    # Heurística simples por comprimento (caracteres) como proxy de tokens
    # Ajuste fino se desejar usar tiktoken
    n = max_tokens * 4
    o = overlap * 4
    s = re.sub(r'\s+', ' ', s).strip()
    out = []
    i = 0
    while i < len(s):
        j = min(len(s), i + n)
        out.append(s[i:j])
        if j == len(s): break
        i = max(0, j - o)
    return out
