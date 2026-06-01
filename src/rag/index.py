# src/rag/index.py
import os
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding

# Dimensões por modelo
_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
_DIM_MAP = {"text-embedding-3-small": 1536, "text-embedding-3-large": 3072}
DIM = _DIM_MAP.get(_MODEL, 1536)

def _make_client(url: str) -> QdrantClient:
    if url == ":memory:":
        return QdrantClient(location=":memory:")
    if url.startswith(("http://", "https://")):
        return QdrantClient(url=url, api_key=os.getenv("QDRANT_API_KEY"))
    return QdrantClient(path=url)  # disco local

def _ensure_collection(client: QdrantClient, name: str, dim: int = DIM) -> None:
    names = [c.name for c in client.get_collections().collections]
    if name not in names:
        client.recreate_collection(
            collection_name=name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )

def get_client() -> QdrantClient:
    url = os.getenv("QDRANT_URL", "http://localhost:6333")
    return _make_client(url)

def ensure_collection() -> QdrantClient:
    client = get_client()
    _ensure_collection(client, os.getenv("QDRANT_COLLECTION", "literatura"), DIM)
    return client

def get_index() -> VectorStoreIndex:
    """VectorStoreIndex com Qdrant + OpenAI Embeddings; coleção garantida."""
    collection = os.getenv("QDRANT_COLLECTION", "literatura")
    model = os.getenv("EMBEDDING_MODEL", _MODEL)

    embed = OpenAIEmbedding(model=model)
    client = ensure_collection()
    vector_store = QdrantVectorStore(client=client, collection_name=collection)
    storage = StorageContext.from_defaults(vector_store=vector_store)
    return VectorStoreIndex.from_documents([], storage_context=storage, embed_model=embed)
