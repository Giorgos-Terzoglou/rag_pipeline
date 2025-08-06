from langchain.schema import Document
import numpy as np

class HybridRetriever:
    def __init__(self, qdrant_store, bm25_retriever, faiss_weight=0.7, bm25_weight=0.3):
        self.qdrant_store = qdrant_store
        self.bm25_retriever = bm25_retriever
        self.faiss_weight = faiss_weight
        self.bm25_weight = bm25_weight

    def search(self, query, k=10):
        # --- Semantic (FAISS/Qdrant) Search ---
        faiss_results = self.qdrant_store.similarity_search_with_score(query, k=k)
        
        # --- BM25 Search ---
        bm25_results = self.bm25_retriever.search(query, k=k)

        # --- Normalize scores ---
        faiss_scores = np.array([score for _, score in faiss_results])
        bm25_scores = np.array([score for _, score in bm25_results])

        if len(faiss_scores) > 0:
            faiss_scores = (faiss_scores - faiss_scores.min()) / (faiss_scores.max() - faiss_scores.min() + 1e-9)
        if len(bm25_scores) > 0:
            bm25_scores = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-9)

        # --- Combine results ---
        combined_docs = {}
        for i, (doc, score) in enumerate(faiss_results):
            combined_docs[doc.page_content] = self.faiss_weight * faiss_scores[i]

        for i, (doc, score) in enumerate(bm25_results):
            if doc.page_content in combined_docs:
                combined_docs[doc.page_content] += self.bm25_weight * bm25_scores[i]
            else:
                combined_docs[doc.page_content] = self.bm25_weight * bm25_scores[i]

        # --- Sort by combined score ---
        ranked_results = sorted(combined_docs.items(), key=lambda x: x[1], reverse=True)[:k]
        ranked_docs = [Document(page_content=doc, metadata={}) for doc, _ in ranked_results]

        # --- Debug log ---
        print("\n📊 [DEBUG] Hybrid Scores (FAISS + BM25 fusion):")
        for doc, score in ranked_results:
            print(f"Score: {score:.4f} | Content: {doc[:80]}...")

        return ranked_docs
