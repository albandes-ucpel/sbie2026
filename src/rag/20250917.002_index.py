# src/rag/index.py
import os
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.embeddings.openai import OpenAIEmbedding
from qdrant_client import QdrantClient
from llama_index.vector_stores.qdrant import QdrantVectorStore

def get_index():
    """
    Cria/abre um índice usando Qdrant + OpenAI Embeddings.
    - QDRANT_URL=":memory:" → índice em memória (ótimo para dev)
    - QDRANT_URL="http://localhost:6333" → Qdrant local
    - QDRANT_COLLECTION → nome da coleção (default: "literatura")
    - EMBEDDING_MODEL → nome do embedding da OpenAI (default: "text-embedding-3-small")
    """
    url = os.getenv("QDRANT_URL", ":memory:")                # ":memory:" para dev sem docker
    collection = os.getenv("QDRANT_COLLECTION", "literatura")
    model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

    # Embeddings via OpenAI (requer OPENAI_API_KEY no .env)
    embed = OpenAIEmbedding(model=model)

    # Qdrant client (memória ou HTTP)
    client = QdrantClient(url=url)
    vector_store = QdrantVectorStore(client=client, collection_name=collection)

    storage = StorageContext.from_defaults(vector_store=vector_store)

    # Índice vazio por ora; você deve alimentá-lo em outro passo (ex.: /rag/reindex)
    index = VectorStoreIndex.from_documents(
        [], storage_context=storage, embed_model=embed
    )
    return index

#Perfeito — seu rag/index.py ainda cria o embedder do HuggingFace. 
#Abaixo vai o arquivo corrigido para usar OpenAIEmbedding e com suporte a Qdrant em memória se você quiser.
