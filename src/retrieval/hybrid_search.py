# src/retrieval/hybrid_search.py
from src.retrieval.bm25_search import BM25Retriever
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer


# src/retrieval/hybrid_search.py
class HybridRetriever:
    def __init__(self, qdrant_store):
        self.qdrant_store = qdrant_store  # Only Qdrant in rollback

    def search(self, query, k=10):
        # Perform only semantic search from Qdrant
        return self.qdrant_store.similarity_search(query, k=k)



