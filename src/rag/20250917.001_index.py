import os
from llama_index.core import VectorStoreIndex, StorageContext
# Alterado para usar o embedding na openAI em vez de local
#from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
#
from qdrant_client import QdrantClient
from llama_index.vector_stores.qdrant import QdrantVectorStore

def get_index():
    url = os.getenv("QDRANT_URL", "http://localhost:6333")
    collection = os.getenv("QDRANT_COLLECTION", "literatura")
    model = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")

    embed = HuggingFaceEmbedding(model_name=model)
    client = QdrantClient(url=url)
    vs = QdrantVectorStore(client=client, collection_name=collection)
    storage = StorageContext.from_defaults(vector_store=vs)
    index = VectorStoreIndex.from_documents([], storage_context=storage, embed_model=embed)
    return index



# de:
#from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# para:
#from llama_index.embeddings.openai import OpenAIEmbedding
