# src/retrieval/reranker.py
from sentence_transformers import CrossEncoder
import numpy as np

class Reranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        print(f"🧠 Loading Cross-Encoder model: {model_name}")
        self.model = CrossEncoder(model_name)

    def rerank(self, query, docs, top_k=5):
        if not docs:
            print("⚠️ No documents to rerank.")
            return []

        # Prepare (query, doc) pairs
        pairs = [(query, doc.page_content) for doc in docs]

        # Predict relevance scores
        scores = self.model.predict(pairs)

        # Normalize to 0–1
        norm_scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
        scored_docs = list(zip(docs, norm_scores))

        # Combine docs with scores
        scored_docs = list(zip(docs, norm_scores))

        # Sort by score (descending)
        reranked = sorted(scored_docs, key=lambda x: x[1], reverse=True)

        print("\n📊 [DEBUG] Reranker Scores:")
        for rank, (doc, score) in enumerate(reranked[:top_k], 1):
            print(f"Rank {rank} | Score: {score:.4f} | Content: {doc.page_content[:100]}...")

        # Return top_k docs
        return [doc for doc, score in reranked[:top_k]]
